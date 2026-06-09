"""
Stage 2 preprocessing: nuScenes mini → JSONL for latent pretraining.

For each consecutive CAM_FRONT frame pair (t, t+3):
  - Run LAQ inference (PyTorch)  → 1 delta token (code 0–3)
  - Run VQGAN encoding (JAX)     → 256 vision tokens (code 0–8191)
  - Attach scene description from nuScenes scene.json

Output: /home/andy/LAPA/data/nuscenes_stage2.jsonl
Each line:
  {"instruction": "...", "vision": [256 ints], "delta": [1 int], "fields": "[instruction],[vision],delta"}

Run from LAPA root:
    conda activate lapa
    python data/prepare_nuscenes_stage2.py

Implementation note: LAQ (PyTorch) and VQGAN (JAX) both need the full GPU.
PyTorch's CUDA context persists even after del + empty_cache, so we run the
LAQ pass in a subprocess that fully exits before JAX initialises.
"""

import os
os.environ["WANDB_MODE"] = "offline"

import json
import sys
import subprocess
from pathlib import Path

# Add LAPA root to path so latent_pretraining is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
NUSCENES_ROOT  = Path("/home/andy/Dataset/Nuscenes/v1.0-mini")
ANNOT_DIR      = NUSCENES_ROOT / "v1.0-mini"
LAQ_CHECKPOINT = Path(__file__).parent.parent / "laq/results_nuscenes_smoke_v3/vae.2000.pt"
VQGAN_CKPT     = Path(__file__).parent.parent / "lapa_checkpoints/vqgan"
OUTPUT_JSONL   = Path("/home/andy/LAPA/data/nuscenes_stage2.jsonl")
DELTA_CACHE    = Path("/home/andy/LAPA/data/nuscenes_stage2_delta.json")   # written by subprocess, read by main

CAMERA       = "CAM_FRONT"
OFFSET       = 3
LAQ_BATCH    = 8
VQGAN_BATCH  = 8


# ═══════════════════════════════════════════════════════════════════════════════
# Shared: nuScenes data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_nuscenes_pairs_with_descriptions():
    """Return list of (path_t, path_t+3, description) for all CAM_FRONT pairs."""
    with open(ANNOT_DIR / "sensor.json") as f:
        sensors = json.load(f)
    cam_token = next(s["token"] for s in sensors if s["channel"] == CAMERA)

    with open(ANNOT_DIR / "calibrated_sensor.json") as f:
        cal_sensors = json.load(f)
    cal_tokens = {cs["token"] for cs in cal_sensors if cs["sensor_token"] == cam_token}

    with open(ANNOT_DIR / "sample_data.json") as f:
        sample_data = json.load(f)
    cam_frames = [sd for sd in sample_data if sd["calibrated_sensor_token"] in cal_tokens]

    with open(ANNOT_DIR / "sample.json") as f:
        samples = json.load(f)
    sample_by_token = {s["token"]: s for s in samples}

    with open(ANNOT_DIR / "scene.json") as f:
        scenes = json.load(f)
    scene_by_token = {sc["token"]: sc for sc in scenes}

    by_token = {sd["token"]: sd for sd in cam_frames}
    roots = [sd for sd in cam_frames if not sd["prev"]]

    pairs = []
    for root in roots:
        chain = []
        curr = root
        while curr:
            chain.append(curr)
            curr = by_token.get(curr["next"]) if curr["next"] else None

        for i in range(len(chain) - OFFSET):
            f_t  = chain[i]
            f_t3 = chain[i + OFFSET]

            sample_token = f_t["sample_token"]
            scene_token  = sample_by_token[sample_token]["scene_token"]
            description  = scene_by_token[scene_token]["description"]

            pairs.append((
                NUSCENES_ROOT / f_t["filename"],
                NUSCENES_ROOT / f_t3["filename"],
                description,
            ))
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 (runs in a subprocess): LAQ delta tokens via PyTorch
# ═══════════════════════════════════════════════════════════════════════════════

def _phase_laq():
    """Run entirely in a child process; PyTorch CUDA context released on exit."""
    import torch
    torch.backends.cudnn.enabled = False
    from torchvision import transforms as T

    sys.path.insert(0, str(Path(__file__).parent.parent / "laq"))
    from laq_model import LatentActionQuantization  # type: ignore[import]

    laq_transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),
    ])

    assert LAQ_CHECKPOINT.exists(), f"LAQ checkpoint not found: {LAQ_CHECKPOINT}"
    laq = LatentActionQuantization(
        dim=256, quant_dim=32, codebook_size=4,
        image_size=256, patch_size=32,
        spatial_depth=2, temporal_depth=2,
        dim_head=64, heads=4, code_seq_len=1,
    ).cuda().eval()
    laq.load_state_dict(torch.load(str(LAQ_CHECKPOINT), map_location="cuda"))
    print(f"  Loaded LAQ checkpoint: {LAQ_CHECKPOINT}", flush=True)

    pairs = load_nuscenes_pairs_with_descriptions()
    total = len(pairs)
    print(f"  {total} pairs", flush=True)

    # {json_serialisable_key: code_int}  — key = "path_t|||path_t3"
    delta_tokens = {}
    for start in range(0, total, LAQ_BATCH):
        batch = pairs[start : start + LAQ_BATCH]
        imgs = []
        for path_t, path_t3, _ in batch:
            img_t  = laq_transform(Image.open(path_t)).unsqueeze(1)
            img_t3 = laq_transform(Image.open(path_t3)).unsqueeze(1)
            imgs.append(torch.cat([img_t, img_t3], dim=1))
        video = torch.stack(imgs).cuda()
        with torch.no_grad():
            indices = laq.inference(video, return_only_codebook_ids=True)
            codes = indices[:, 0].cpu().tolist()
        for code, (path_t, path_t3, _) in zip(codes, batch):
            key = f"{path_t}|||{path_t3}"
            delta_tokens[key] = int(code)

        if (start // LAQ_BATCH) % 20 == 0:
            print(f"  LAQ: {start + len(batch)}/{total}", flush=True)

    DELTA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(DELTA_CACHE, "w") as f:
        json.dump(delta_tokens, f)
    print(f"  Saved delta tokens → {DELTA_CACHE}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 (runs in main process after subprocess exits): VQGAN vision tokens
# ═══════════════════════════════════════════════════════════════════════════════

def run_vqgan_encoding(unique_paths):
    """Returns {str(path): [256 ints]}.

    JAX 0.4.23 cuDNN bindings are incompatible with the RTX 5080 (Blackwell).
    Force CPU mode — JIT-compiled JAX on CPU takes ~3s/batch, ~14 min total.
    """
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    from latent_pretraining.vqgan import VQGAN

    assert VQGAN_CKPT.exists(), f"VQGAN checkpoint not found: {VQGAN_CKPT}"
    vqgan = VQGAN(str(VQGAN_CKPT), replicate=False)
    print(f"  Loaded VQGAN checkpoint: {VQGAN_CKPT}")
    print(f"  JAX devices: {jax.devices()}")

    vision_tokens = {}
    paths = list(unique_paths)
    total = len(paths)
    for start in range(0, total, VQGAN_BATCH):
        batch_paths = paths[start : start + VQGAN_BATCH]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB").resize((256, 256), Image.BILINEAR)
            imgs.append((np.array(img) / 127.5 - 1.0).astype(np.float32))
        batch = np.stack(imgs)  # (B, 256, 256, 3) channels-last, [-1, 1]

        _, indices = vqgan.encode(batch)
        indices_np = jax.device_get(indices)
        for path, idx in zip(batch_paths, indices_np):
            vision_tokens[str(path)] = idx.flatten().tolist()

        if (start // VQGAN_BATCH) % 10 == 0:
            print(f"  VQGAN: {start + len(batch_paths)}/{total}")

    return vision_tokens


# ═══════════════════════════════════════════════════════════════════════════════
# Entry points
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--phase=laq" in sys.argv:
        # Running as the LAQ subprocess
        _phase_laq()
        sys.exit(0)

    # ── Main process ──────────────────────────────────────────────────────────
    print("Loading nuScenes pairs + scene descriptions...")
    pairs = load_nuscenes_pairs_with_descriptions()
    print(f"  {len(pairs)} pairs across {OFFSET}-frame offset")

    # Phase 1: spawn a subprocess for PyTorch/LAQ so its CUDA context is
    # completely released before JAX starts.
    if DELTA_CACHE.exists():
        print(f"\nFound cached delta tokens at {DELTA_CACHE}, skipping LAQ pass.")
        print("  (Delete the file to rerun LAQ inference.)")
    else:
        print("\nPass 1: LAQ delta tokens (PyTorch subprocess)...")
        result = subprocess.run(
            [sys.executable, __file__, "--phase=laq"],
            check=True,
        )

    with open(DELTA_CACHE) as f:
        raw = json.load(f)
    # Restore tuple keys
    delta_tokens = {tuple(k.split("|||")): v for k, v in raw.items()}
    print(f"  {len(delta_tokens)} delta tokens loaded")
    from collections import Counter
    code_counts = Counter(delta_tokens.values())
    print(f"  Code distribution: {dict(sorted(code_counts.items()))}")

    # Phase 2: VQGAN in main process (PyTorch subprocess already exited)
    unique_frame_t = {str(p) for p, _, _ in pairs}
    print(f"\nPass 2: VQGAN vision tokens (JAX) for {len(unique_frame_t)} unique frames...")
    vision_tokens = run_vqgan_encoding(unique_frame_t)
    print(f"  {len(vision_tokens)} frames encoded")

    # Write JSONL
    print(f"\nWriting JSONL to {OUTPUT_JSONL}...")
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with open(OUTPUT_JSONL, "w") as f:
        for path_t, path_t3, description in pairs:
            key = (str(path_t), str(path_t3))
            if key not in delta_tokens or str(path_t) not in vision_tokens:
                skipped += 1
                continue
            record = {
                "instruction": description,
                "vision": vision_tokens[str(path_t)],
                "delta": [delta_tokens[key]],
                "fields": "[instruction],[vision],delta",
            }
            f.write(json.dumps(record) + "\n")
            written += 1

    print(f"  Written: {written} records, skipped: {skipped}")
    print(f"\nSanity check first record:")
    with open(OUTPUT_JSONL) as f:
        first = json.loads(f.readline())
    print(f"  vision tokens: {len(first['vision'])} (expect 256)")
    print(f"  delta:         {first['delta']} (expect [0-3])")
    print(f"  instruction:   {first['instruction'][:60]}")
    print(f"\nDone → {OUTPUT_JSONL}")

"""
Stage 3 preprocessing: CARLA episodes → JSONL for action fine-tuning.

For each recorded step in each episode:
  - VQGAN (JAX CPU) encodes the 256×256 front-camera frame → 256 vision tokens
  - 7D raw actions [steer, throttle, brake, hand_brake, reverse, vel_x, vel_y]
    are discretized into 256 bins each via pd.qcut (same logic as finetune_preprocess.py)

Output schema (per JSONL line):
  {"instruction": "...", "raw_actions": [float×7], "vision": [int×256],
   "action": [int×7 in 0-255], "fields": "[instruction],[vision],action"}

Usage (from LAPA root, lapa env active):
    python data/prepare_carla_stage3.py \\
        --episodes-dir carla_episodes \\
        --output-jsonl /home/andy/data/carla_stage3.jsonl \\
        --action-bins carla_action_bins.csv

Implementation note: VQGAN (JAX) runs on CPU to avoid Blackwell cuDNN issues.
"""

import os
os.environ["WANDB_MODE"] = "offline"

import json
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Action discretization ──────────────────────────────────────────────────────

BINARY_DIMS = {3, 4}  # hand_brake, reverse — binary float {0.0, 1.0}
BINARY_BINS = [-0.5, 0.5, 1.5]
N_BINS = 256
DIM_NAMES = ["steer", "throttle", "brake", "hand_brake", "reverse", "vel_x", "vel_y"]


def _assign_bin(value: float, bins: list) -> int:
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return i
    if value >= bins[-1]:
        return len(bins) - 2
    return 0


def compute_bins(raw_actions_by_dim: list) -> list:
    """Return list of 7 bin-edge arrays, one per action dimension."""
    all_bins = []
    for dim_idx, values in enumerate(raw_actions_by_dim):
        if dim_idx in BINARY_DIMS:
            all_bins.append(BINARY_BINS)
        else:
            series = pd.Series(values)
            _, edges = pd.qcut(series, N_BINS, labels=False, retbins=True, duplicates="drop")
            all_bins.append(edges.tolist())
    return all_bins


# ── Episode loading ────────────────────────────────────────────────────────────

def load_episodes(episodes_dir: Path):
    """Yield (metadata_dict, steps_list, frames_dir) for each completed episode."""
    ep_dirs = sorted(d for d in (episodes_dir / "episodes").iterdir() if d.is_dir())
    for ep_dir in ep_dirs:
        meta_path = ep_dir / "metadata.json"
        ep_path = ep_dir / "episode.json"
        frames_dir = ep_dir / "frames"
        if not meta_path.exists() or not ep_path.exists() or not frames_dir.exists():
            print(f"  [SKIP] Incomplete episode: {ep_dir.name}")
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        with open(ep_path) as f:
            steps = json.load(f)
        yield meta, steps, frames_dir


# ── VQGAN encoding ─────────────────────────────────────────────────────────────

def run_vqgan_encoding(frame_paths: list, vqgan_ckpt: str, batch_size: int = 8) -> dict:
    """Returns {str(path): [256 ints]}.

    JAX forced to CPU (JAX 0.4.23 incompatible with Blackwell GPU).
    """
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    from latent_pretraining.vqgan import VQGAN

    vqgan = VQGAN(vqgan_ckpt, replicate=False)
    print(f"  Loaded VQGAN: {vqgan_ckpt}")
    print(f"  JAX devices: {jax.devices()}")

    vision_tokens = {}
    total = len(frame_paths)
    for start in range(0, total, batch_size):
        batch_paths = frame_paths[start: start + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB").resize((256, 256), Image.BILINEAR)
            imgs.append((np.array(img) / 127.5 - 1.0).astype(np.float32))
        batch = np.stack(imgs)  # (B, 256, 256, 3), channels-last, [-1, 1]

        _, indices = vqgan.encode(batch)
        indices_np = jax.device_get(indices)
        for path, idx in zip(batch_paths, indices_np):
            vision_tokens[str(path)] = idx.flatten().tolist()

        if (start // batch_size) % 20 == 0:
            print(f"  VQGAN: {start + len(batch_paths)}/{total}")

    return vision_tokens


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    episodes_dir = Path(args.episodes_dir)
    output_jsonl = Path(args.output_jsonl)
    action_bins_csv = Path(args.action_bins)
    vqgan_ckpt = str(Path(__file__).parent.parent / "lapa_checkpoints" / "vqgan")

    # ── Pass 0: Collect all episodes + raw actions for bin fitting ─────────────
    print("Pass 0: Scanning episodes...")
    all_meta = []
    all_steps_by_episode = []
    all_frame_paths = []
    raw_by_dim = [[] for _ in range(7)]

    for meta, steps, frames_dir in load_episodes(episodes_dir):
        for step in steps:
            frame_path = frames_dir / f"{step['step']:04d}.jpg"
            if not frame_path.exists():
                continue
            raw = [
                step["steer"],
                step["throttle"],
                step["brake"],
                step["hand_brake"],
                step["reverse"],
                step["vel_x"],
                step["vel_y"],
            ]
            for dim_idx, v in enumerate(raw):
                raw_by_dim[dim_idx].append(v)
            all_frame_paths.append(str(frame_path))
        all_meta.append(meta)
        all_steps_by_episode.append((meta, steps, frames_dir))

    total_steps = len(all_frame_paths)
    print(f"  Found {len(all_meta)} episodes, {total_steps} valid steps")
    if total_steps == 0:
        print("ERROR: No valid steps found. Run the data collector first.")
        sys.exit(1)

    # ── Pass 1: Compute discretization bins ────────────────────────────────────
    print("\nPass 1: Computing action bins...")
    bins_per_dim = compute_bins(raw_by_dim)
    for dim_idx, (name, edges) in enumerate(zip(DIM_NAMES, bins_per_dim)):
        print(f"  dim {dim_idx} ({name}): {len(edges)-1} bins, "
              f"range [{edges[0]:.4f}, {edges[-1]:.4f}]")

    # Save bins CSV for inference-time de-tokenization
    action_bins_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(bins_per_dim)
    df.to_csv(str(action_bins_csv), index=False)
    print(f"  Saved action bins → {action_bins_csv}")

    # ── Pass 2: VQGAN encoding ─────────────────────────────────────────────────
    print(f"\nPass 2: VQGAN encoding {total_steps} frames (JAX CPU)...")
    unique_paths = list(dict.fromkeys(all_frame_paths))  # preserve order, dedupe
    vision_tokens = run_vqgan_encoding(unique_paths, vqgan_ckpt, batch_size=args.vqgan_batch)
    print(f"  Encoded {len(vision_tokens)} frames")

    # ── Pass 3: Write JSONL ────────────────────────────────────────────────────
    print(f"\nPass 3: Writing JSONL to {output_jsonl}...")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0

    with open(output_jsonl, "w") as out:
        for meta, steps, frames_dir in all_steps_by_episode:
            instruction = meta.get("instruction",
                                   f"Drive on {meta.get('map','unknown')}. Follow traffic rules.")
            for step in steps:
                frame_path = str(frames_dir / f"{step['step']:04d}.jpg")
                if frame_path not in vision_tokens:
                    skipped += 1
                    continue
                raw = [
                    step["steer"],
                    step["throttle"],
                    step["brake"],
                    step["hand_brake"],
                    step["reverse"],
                    step["vel_x"],
                    step["vel_y"],
                ]
                action_tokens = [_assign_bin(v, bins_per_dim[i]) for i, v in enumerate(raw)]
                record = {
                    "instruction": instruction,
                    "raw_actions": raw,
                    "vision": vision_tokens[frame_path],
                    "action": action_tokens,
                    "fields": "[instruction],[vision],action",
                }
                out.write(json.dumps(record) + "\n")
                written += 1

    print(f"  Written: {written} records, skipped: {skipped}")

    # Sanity check
    print("\nSanity check first record:")
    with open(output_jsonl) as f:
        first = json.loads(f.readline())
    print(f"  vision tokens : {len(first['vision'])} (expect 256)")
    print(f"  action tokens : {first['action']} (expect 7 ints in 0-255)")
    print(f"  raw_actions   : {[round(v,4) for v in first['raw_actions']]}")
    print(f"  instruction   : {first['instruction'][:70]}")
    print(f"\nDone → {output_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CARLA episodes for Stage 3 fine-tuning")
    parser.add_argument("--episodes-dir", required=True,
                        help="Root directory of collected episodes (contains episodes/ subdir)")
    parser.add_argument("--output-jsonl", default="/home/andy/data/carla_stage3.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--action-bins", default="carla_action_bins.csv",
                        help="CSV file to save action bin edges for deployment")
    parser.add_argument("--vqgan-batch", type=int, default=8,
                        help="VQGAN encoding batch size (default: 8)")
    main(parser.parse_args())

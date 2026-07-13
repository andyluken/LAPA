"""N-way comparison across LAQ checkpoints on nuScenes-mini (Phases 1 & 2).

Generalizes the old eval_heatmap_vs_baseline.py (2-way only) into a runner
over a list of named (checkpoint, use_heatmap) configs, so adding a new
checkpoint to compare is a list entry, not a new near-duplicate script.

For each checkpoint, runs inference over the same nuScenes-mini frame pairs
and reports:
  - codebook usage / perplexity
  - the maneuver x code distribution table (as in eval_latent_clusters.py)
  - normalized mutual information (NMI) between codebook index and
    ground-truth maneuver label
  - mean intra-codebook-cluster cost-map cosine similarity: the direct test
    of whether Phase 2's cost-map regularization pulled similar-risk scenes
    into the same code (computed for every config, regardless of whether
    that checkpoint was trained with cost-map regularization, since it only
    depends on cached cost maps + the codes a model assigns)

Usage (from laq/ directory):
    python eval_compare_checkpoints.py

Requires the checkpoints listed in CONFIGS below, and the heatmap/costmap
caches from data/prepare_nuscenes_heatmaps.py and
data/prepare_nuscenes_costmaps.py.
"""

import math
import os
os.environ["WANDB_MODE"] = "offline"

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

torch.backends.cudnn.enabled = False

sys.path.insert(0, str(Path(__file__).parent))
from laq_model import LatentActionQuantization
from laq_model.eval_utils import (
    IMAGE_TRANSFORM,
    MANEUVERS,
    build_symlink_index,
    load_cached_array,
    load_nuscenes_pairs,
    maneuver_label,
    mean_intra_cluster_costmap_similarity,
    normalized_mutual_information,
)

# ── Config ────────────────────────────────────────────────────────────────────
NUSCENES_ROOT = Path("/home/andy/Dataset/Nuscenes/v1.0-mini")
FRAMES_DIR = Path("/home/andy/Dataset/nuscenes_laq_frames")
OFFSET = 3          # must match all training runs
BATCH_SIZE = 8
CAMERA = "CAM_FRONT"
PATCH_GRID = (8, 8)        # image_size=256, patch_size=32
COSTMAP_GRID = (32, 32)
N_CODES = 8

MODEL_CONFIG = dict(
    dim=512, quant_dim=32, codebook_size=N_CODES,
    image_size=256, patch_size=32,
    spatial_depth=2, temporal_depth=2,
    dim_head=64, heads=16, code_seq_len=1,
)

REPO_ROOT = Path(__file__).parent.parent
LAQ_DIR = Path(__file__).parent

# (display name, checkpoint path, use_heatmap_at_inference)
# Baseline/heatmap point at the 5000-step runs the user extended at the repo
# root. No Phase 2 (cost-map) checkpoint is listed here: an extensive ablation
# (see README's "Phase 2" section and laq_model/costmap_loss.py's docstring)
# found that cost-map regularization conflicts with NSVQ codebook formation
# at nuScenes-mini's data/codebook scale under every configuration tried, so
# no checkpoint trained with it is fit to ship as a comparison point. To
# compare a Phase 2 checkpoint you train yourself, add a third tuple here,
# e.g.: ("Heatmap + Cost-map", LAQ_DIR / "results_nuscenes_costmap_smoke_w<W>/vae.5000.pt", True).
CONFIGS = [
    ("Baseline", REPO_ROOT / "results_nuscenes_base_8codebook_smoke_/vae.3000.pt", False),
    ("Ego-motion (Exp 1)", REPO_ROOT / "results_nuscenes_egomotion_mag8_smoke/vae.3000.pt", True),
    ("FSQ+balanced_sampler (Exp 2)", REPO_ROOT / "results_laq_ad_mini/laq_ad.5001.pt", True),
   
]


def load_model(checkpoint: Path) -> LatentActionQuantization:
    assert checkpoint.exists(), f"Checkpoint not found: {checkpoint}"
    laq = LatentActionQuantization(**MODEL_CONFIG).cuda().eval()
    laq.load_state_dict(torch.load(str(checkpoint), map_location="cuda"))
    print(f"Loaded checkpoint: {checkpoint}")
    return laq


def run_inference(laq, pairs, symlink_index, use_heatmap: bool):
    """Returns list of (code, maneuver_label, aggregated_costmap) per pair."""
    results = []
    for batch_start in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[batch_start: batch_start + BATCH_SIZE]

        imgs, heatmaps, costmaps = [], [], []
        for path_t, path_t3, _, _ in batch:
            img_t = IMAGE_TRANSFORM(Image.open(path_t)).unsqueeze(1)
            img_t3 = IMAGE_TRANSFORM(Image.open(path_t3)).unsqueeze(1)
            imgs.append(torch.cat([img_t, img_t3], dim=1))

            if use_heatmap:
                h_t = load_cached_array(path_t, symlink_index, ".heatmap.npy", PATCH_GRID)
                h_t3 = load_cached_array(path_t3, symlink_index, ".heatmap.npy", PATCH_GRID)
                heatmaps.append(torch.from_numpy(np.stack([h_t, h_t3])))

            c_t = load_cached_array(path_t, symlink_index, ".costmap.npy", COSTMAP_GRID)
            c_t3 = load_cached_array(path_t3, symlink_index, ".costmap.npy", COSTMAP_GRID)
            costmaps.append(np.maximum(c_t, c_t3))  # matches training-time elementwise-max aggregation

        video = torch.stack(imgs).cuda()
        heatmap_batch = torch.stack(heatmaps).cuda() if use_heatmap else None

        with torch.no_grad():
            indices = laq.inference(video, heatmap=heatmap_batch, return_only_codebook_ids=True)
            codes = indices[:, 0].cpu().tolist()

        for code, costmap, (_, _, pose_t, pose_t3) in zip(codes, costmaps, batch):
            results.append((int(code), maneuver_label(pose_t, pose_t3), costmap))

    return results


def summarize(name: str, results: list) -> dict:
    codes = [r[0] for r in results]
    labels = [r[1] for r in results]
    costmaps = [r[2] for r in results]

    usage = Counter(codes)
    n = len(codes)
    probs = [c / n for c in usage.values()]
    perplexity = math.exp(-sum(p * math.log(p) for p in probs if p > 0))
    nmi = normalized_mutual_information(codes, labels)
    intra_sim = mean_intra_cluster_costmap_similarity(codes, costmaps)

    print(f"\n── {name} ──────────────────────────────────────────────")
    print(f"  codebook usage:           {dict(sorted(usage.items()))}")
    print(f"  perplexity:                {perplexity:.3f}  (max possible = {N_CODES})")
    print(f"  maneuver NMI:              {nmi:.4f}  (0 = independent, 1 = perfectly aligned)")
    intra_sim_str = f"{intra_sim:.4f}" if intra_sim is not None else "n/a (no cluster with 2+ members)"
    print(f"  intra-cluster costmap sim: {intra_sim_str}")

    counts = defaultdict(lambda: defaultdict(int))
    for code, label, _ in results:
        counts[code][label] += 1

    header = f"  {'Code':>4} | " + " | ".join(f"{m:>11}" for m in MANEUVERS) + " | total"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for code in range(N_CODES):
        row = counts[code]
        total = sum(row.values())
        cells = " | ".join(f"{row[m]:>11}" for m in MANEUVERS)
        print(f"  {code:>4} | {cells} | {total}")

    return {"perplexity": perplexity, "nmi": nmi, "intra_cluster_costmap_sim": intra_sim}


if __name__ == "__main__":
    print("Loading nuScenes frame pairs...")
    pairs = load_nuscenes_pairs(NUSCENES_ROOT, OFFSET, camera=CAMERA)
    print(f"  {len(pairs)} pairs across {OFFSET}-frame offset")

    symlink_index = build_symlink_index(FRAMES_DIR)
    print(f"  {len(symlink_index)} cached frame symlinks indexed")

    summaries = {}
    for name, checkpoint, use_heatmap in CONFIGS:
        print(f"\nRunning {name}...")
        model = load_model(checkpoint)
        results = run_inference(model, pairs, symlink_index, use_heatmap)
        summaries[name] = summarize(name, results)
        del model
        torch.cuda.empty_cache()

    print("\n── Summary ──────────────────────────────────────────────────")
    for name, s in summaries.items():
        intra = f"{s['intra_cluster_costmap_sim']:.4f}" if s["intra_cluster_costmap_sim"] is not None else "n/a"
        print(f"  {name:<32} perplexity={s['perplexity']:.3f}  NMI={s['nmi']:.4f}  intra-costmap-sim={intra}")

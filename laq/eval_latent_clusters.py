"""
Verify that LAQ latent action codes cluster by driving maneuver type.

Usage (from laq/ directory):
    python eval_latent_clusters.py

Requires a trained checkpoint at results_nuscenes_smoke_v3/vae.2000.pt.
Outputs:
  - latent_cluster_results.png  — grouped bar chart
  - printed table of code × maneuver counts
"""

import os
os.environ["WANDB_MODE"] = "offline"

import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.backends.cudnn.enabled = False

sys.path.insert(0, str(Path(__file__).parent))
from laq_model import LatentActionQuantization
from laq_model.eval_utils import IMAGE_TRANSFORM, MANEUVERS, load_nuscenes_pairs, maneuver_label

# ── Config ────────────────────────────────────────────────────────────────────
NUSCENES_ROOT   = Path("/home/andy/Dataset/Nuscenes/v1.0-mini")
CHECKPOINT      = Path("results_nuscenes_egomotion_smoke/vae.3000.pt")
OFFSET          = 3        # must match training
BATCH_SIZE      = 8
CAMERA          = "CAM_FRONT"

transform = IMAGE_TRANSFORM


def load_model():
    assert CHECKPOINT.exists(), f"Checkpoint not found: {CHECKPOINT}\nRun training first."
    laq = LatentActionQuantization(
        dim=512, quant_dim=32, codebook_size=4,
        image_size=256, patch_size=32,
        spatial_depth=2, temporal_depth=2,
        dim_head=64, heads=16, code_seq_len=1,
    ).cuda().eval()
    laq.load_state_dict(torch.load(str(CHECKPOINT), map_location="cuda"))
    print(f"Loaded checkpoint: {CHECKPOINT}")
    return laq


def run_inference(laq, pairs):
    """Return list of (code_index, maneuver_label) for every pair."""
    results = []
    for batch_start in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[batch_start : batch_start + BATCH_SIZE]
        imgs = []
        for path_t, path_t3, _, _ in batch:
            img_t  = transform(Image.open(path_t)).unsqueeze(1)   # (3,1,256,256)
            img_t3 = transform(Image.open(path_t3)).unsqueeze(1)
            imgs.append(torch.cat([img_t, img_t3], dim=1))         # (3,2,256,256)
        video = torch.stack(imgs).cuda()                            # (B,3,2,256,256)

        with torch.no_grad():
            indices = laq.inference(video, return_only_codebook_ids=True)  # (B,1)
            codes = indices[:, 0].cpu().tolist()

        for code, (_, _, pose_t, pose_t3) in zip(codes, batch):
            label = maneuver_label(pose_t, pose_t3)
            results.append((int(code), label))

        if (batch_start // BATCH_SIZE) % 20 == 0:
            print(f"  {batch_start + len(batch)}/{len(pairs)} pairs processed")

    return results


def plot_and_print(results, n_codes=4):
    counts = defaultdict(lambda: defaultdict(int))
    for code, label in results:
        counts[code][label] += 1

    print("\n── Latent Code × Maneuver Distribution ─────────────────────────")
    header = f"{'Code':>4} | " + " | ".join(f"{m:>11}" for m in MANEUVERS) + " | total"
    print(header)
    print("-" * len(header))
    for code in range(n_codes):
        row = counts[code]
        total = sum(row.values())
        cells = " | ".join(f"{row[m]:>11}" for m in MANEUVERS)
        print(f"{code:>4} | {cells} | {total}")
    print()

    # grouped bar chart
    x = np.arange(n_codes)
    width = 0.2
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (maneuver, color) in enumerate(zip(MANEUVERS, colors)):
        vals = [counts[c][maneuver] for c in range(n_codes)]
        ax.bar(x + i * width, vals, width, label=maneuver, color=color)

    ax.set_xlabel("Codebook index")
    ax.set_ylabel("Frame pair count")
    ax.set_title("LAQ latent action codes vs driving maneuver (nuScenes mini)")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"Code {c}" for c in range(n_codes)])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = Path(__file__).parent / "latent_cluster_results_egomotion.png"
    plt.savefig(out, dpi=150)
    print(f"Chart saved → {out}")


if __name__ == "__main__":
    print("Loading nuScenes frame pairs...")
    pairs = load_nuscenes_pairs(NUSCENES_ROOT, OFFSET, camera=CAMERA)
    print(f"  {len(pairs)} pairs across {OFFSET}-frame offset")

    laq = load_model()

    print("Running LAQ inference...")
    results = run_inference(laq, pairs)

    plot_and_print(results)

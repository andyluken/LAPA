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

import json
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms as T
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.backends.cudnn.enabled = False

sys.path.insert(0, str(Path(__file__).parent))
from laq_model import LatentActionQuantization

# ── Config ────────────────────────────────────────────────────────────────────
NUSCENES_ROOT   = Path("/home/andy/Dataset/Nuscenes/v1.0-mini")
ANNOT_DIR       = NUSCENES_ROOT / "v1.0-mini"
CHECKPOINT      = Path(__file__).parent / "results_nuscenes_smoke_v3/vae.2000.pt"
OFFSET          = 3        # must match training
BATCH_SIZE      = 8
CAMERA          = "CAM_FRONT"

# Maneuver thresholds for OFFSET=3 frames at ~12 Hz (≈250ms between pair)
YAW_THRESH      = 0.06     # rad (~3.4°): |yaw_delta| above this → turning
DIST_THRESH     = 0.2      # m: distance below this → stationary
MANEUVERS       = ["straight", "turn_left", "turn_right", "stationary"]


# ── Preprocessing (matches data.py training fix) ──────────────────────────────
transform = T.Compose([
    T.Lambda(lambda img: img.convert("RGB")),
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),
])


def yaw_delta(pose_t, pose_t3):
    """Signed yaw change (radians) from pose_t to pose_t3.
    nuScenes quaternion = [w, x, y, z]; scipy expects [x, y, z, w].
    """
    def to_rot(p):
        q = p["rotation"]           # [w, x, y, z]
        return Rotation.from_quat([q[1], q[2], q[3], q[0]])

    rel = to_rot(pose_t3) * to_rot(pose_t).inv()
    return float(rel.as_rotvec()[2])  # z-component ≈ yaw for flat road


def maneuver_label(pose_t, pose_t3):
    t, t3 = pose_t["translation"], pose_t3["translation"]
    dist = ((t3[0] - t[0]) ** 2 + (t3[1] - t[1]) ** 2) ** 0.5
    if dist < DIST_THRESH:
        return "stationary"
    yaw = yaw_delta(pose_t, pose_t3)
    if yaw > YAW_THRESH:
        return "turn_left"
    if yaw < -YAW_THRESH:
        return "turn_right"
    return "straight"


def load_nuscenes_pairs():
    """Build list of (img_path_t, img_path_t3, ego_pose_t, ego_pose_t3)."""
    with open(ANNOT_DIR / "sensor.json") as f:
        sensors = json.load(f)
    cam_token = next(s["token"] for s in sensors if s["channel"] == CAMERA)

    with open(ANNOT_DIR / "calibrated_sensor.json") as f:
        cal_sensors = json.load(f)
    cal_tokens = {cs["token"] for cs in cal_sensors if cs["sensor_token"] == cam_token}

    with open(ANNOT_DIR / "sample_data.json") as f:
        sample_data = json.load(f)
    cam_frames = [sd for sd in sample_data if sd["calibrated_sensor_token"] in cal_tokens]

    with open(ANNOT_DIR / "ego_pose.json") as f:
        ego_poses = json.load(f)
    ego_by_token = {ep["token"]: ep for ep in ego_poses}

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
            pairs.append((
                NUSCENES_ROOT / f_t["filename"],
                NUSCENES_ROOT / f_t3["filename"],
                ego_by_token[f_t["ego_pose_token"]],
                ego_by_token[f_t3["ego_pose_token"]],
            ))
    return pairs


def load_model():
    assert CHECKPOINT.exists(), f"Checkpoint not found: {CHECKPOINT}\nRun training first."
    laq = LatentActionQuantization(
        dim=256, quant_dim=32, codebook_size=4,
        image_size=256, patch_size=32,
        spatial_depth=2, temporal_depth=2,
        dim_head=64, heads=4, code_seq_len=1,
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
    out = Path(__file__).parent / "latent_cluster_results.png"
    plt.savefig(out, dpi=150)
    print(f"Chart saved → {out}")


if __name__ == "__main__":
    print("Loading nuScenes frame pairs...")
    pairs = load_nuscenes_pairs()
    print(f"  {len(pairs)} pairs across {OFFSET}-frame offset")

    laq = load_model()

    print("Running LAQ inference...")
    results = run_inference(laq, pairs)

    plot_and_print(results)

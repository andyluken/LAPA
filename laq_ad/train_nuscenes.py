"""LAQ-AD training on nuScenes (trainval or mini).

Default config:
  - FSQ levels=[3,3] → 9 codes, 2 independent axes (yaw × speed)
  - Independent projection heads per FSQ dimension (prevents correlation collapse)
  - spread_reg_weight=2.0: prevents FSQ posterior/saturation collapse
  - entropy_reg_weight=0.0: boundary-attraction collapses NMI mid-training
  - Egomotion magnitude heatmap conditioning (best from laq/ experiments)
  - Maneuver-balanced sampler (oversamples rare turns)
  - CAN bus auxiliary loss (weight=0.05; skips gracefully if CAN bus unavailable)
  - 3-frame offset at 2 Hz keyframes = 1.5 s motion window

For quick smoke-test on nuScenes-mini, set:
    NUSCENES_VERSION = "v1.0-mini"
    NUSCENES_DATAROOT = Path("/home/andy/Dataset/Nuscenes/v1.0-mini")
    NUM_TRAIN_STEPS = 5_000

Run:
    conda activate lapa
    cd laq_ad
    pip install -e .
    python train_nuscenes.py
"""

import os
os.environ["WANDB_MODE"] = "offline"

import sys
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False

# ── Config ────────────────────────────────────────────────────────────────────
NUSCENES_DATAROOT = Path("/media/andy/Samsung_T7/nuScenes/Trainval")
NUSCENES_VERSION = "v1.0-trainval"
FRAMES_DIR = Path("/home/andy/Dataset/nuscenes_laq_frames_improved")  # precomputed frames for egomotion heatmap

# For nuScenes-mini smoke-test, uncomment:
# NUSCENES_DATAROOT = Path("/home/andy/Dataset/Nuscenes/v1.0-mini")
# NUSCENES_VERSION = "v1.0-mini"

OFFSET = 3          # 3 frames ≈ 250ms at 12 Hz
BATCH_SIZE = 16
NUM_TRAIN_STEPS = 50_000
WARMUP_STEPS = 500
RESULTS_FOLDER = "results_laq_ad_trainval_3x3_canStrong2"
RESUME_FROM = None       # fresh run

# FSQ levels: codebook_size = prod(levels)
# [3,3] → 9 codes across 2 independent axes (yaw × speed).
# Each FSQ dimension gets its own independent project_in head (512→1), so the
# two axes are decoupled and can capture orthogonal motion features.
# [8] (1D single head) collapsed to 2 codes mid-training — bimodal z_pre.
LEVELS = [3, 3]

# ── Build components ──────────────────────────────────────────────────────────
from nuscenes.nuscenes import NuScenes
from laq_ad_model import (
    LAQADModel,
    LAQADTrainer,
    NuScenesLAQDataset,
    maneuver_balanced_sampler,
    collate_fn,
)

print(f"Loading nuScenes {NUSCENES_VERSION}…")
nusc = NuScenes(version=NUSCENES_VERSION, dataroot=str(NUSCENES_DATAROOT), verbose=False)

print(f"Building dataset (offset={OFFSET})…")
dataset = NuScenesLAQDataset(
    nusc=nusc,
    frames_dir=FRAMES_DIR,
    offset=OFFSET,
    use_heatmap=True,
    use_can_bus=True,
)
print(f"  {len(dataset)} frame pairs | maneuver dist: {dataset.class_counts}")

sampler = maneuver_balanced_sampler(dataset)

model = LAQADModel(
    dim=512,
    levels=LEVELS,
    image_size=256,
    patch_size=32,
    spatial_depth=2,
    temporal_depth=2,
    dim_head=64,
    heads=8,
    heatmap_alpha=1.0,
    can_bus_weight=1.0,            # Doubled from 0.5 — stronger yaw bootstrap to reliably separate left/right
    spread_reg_weight=0.05,       # CRITICAL FIX: spread_reg was 9× stronger than flow, forcing
                                  # std→1 and overriding natural maneuver-based code assignment
    covariance_reg_weight=1.0,    # VICReg: keeps axes orthogonal — prevents correlated bimodal
    entropy_reg_weight=0.0,
    flow_prediction_weight=5.0,   # dominant signal: flow MSE from z_q — K-means baseline NMI=0.37
    recon_loss_weight=0.0,
)
print(f"  LAQ-AD codebook size: {model.codebook_size}")

trainer = LAQADTrainer(
    model=model,
    dataset=dataset,
    sampler=sampler,
    results_folder=RESULTS_FOLDER,
    num_train_steps=NUM_TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    grad_accum_every=1,
    lr=1e-4,
    warmup_steps=WARMUP_STEPS,
    save_model_every=2_000,
    save_results_every=200,
    use_wandb=False,
    collate_fn=collate_fn,
    resume_from=RESUME_FROM,
)

trainer.train()

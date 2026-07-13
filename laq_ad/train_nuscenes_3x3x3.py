"""LAQ-AD trainval experiment: FSQ levels=[3,3,3] → 27 codes.

Same perAxisCAN config as the best [3,3] run (NMI=0.4007) but with 3 levels
per axis instead of 2, giving a 3×3×3=27-code grid.  The extra resolution should
allow the model to separate gentle vs. sharp turns within each direction axis
rather than collapsing them into mixed straight/turn codes.

Baseline for comparison:
    [3,3] perAxisCAN, 50K steps → NMI=0.4007 (results_laq_ad_trainval_3x3_perAxisCAN)

Run:
    conda activate lapa
    cd laq_ad
    pip install -e .
    python train_nuscenes_3x3x3.py
"""

import os
os.environ["WANDB_MODE"] = "offline"

from pathlib import Path
import torch
torch.backends.cudnn.enabled = False

# ── Config ────────────────────────────────────────────────────────────────────
NUSCENES_DATAROOT = Path("/media/andy/Samsung_T7/nuScenes/Trainval")
NUSCENES_VERSION  = "v1.0-trainval"
FRAMES_DIR        = Path("/home/andy/Dataset/nuscenes_laq_frames")

OFFSET            = 3
BATCH_SIZE        = 16
NUM_TRAIN_STEPS   = 50_000
WARMUP_STEPS      = 500
RESULTS_FOLDER    = "results_laq_ad_trainval_3x3x3_canStrong_ifsq"
RESUME_FROM       = None       # fresh run

#  3×3×3 grid → 27 codes; per-axis heads force axis0=yaw, axis1=speed
LEVELS = [8,6,5,5]  # 4×3×3=36 codes; axis0=yaw, axis1=speed, axis2=turn magnitude

# ── Build ──────────────────────────────────────────────────────────────────────
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
    can_bus_weight=1.0,         # 0.5 → 1.0: reliable yaw bootstrapping (proven in canStrong2)
    spread_reg_weight=0.2,      # 0.05 → 0.2: 3 dims + iFSQ drift needs stronger mean penalty
    covariance_reg_weight=1.0,
    flow_prediction_weight=5.0,
    recon_loss_weight=0.2,      # 0 → 0.1: gives axis 2 an indirect signal (no CAN target for axis 2)
)
print(f"  LAQ-AD codebook size: {model.codebook_size}")  # 27

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

"""LAQ-AD smoke-test on nuScenes-mini (10 scenes, ~2300 pairs).

Tuned config for the mini split:
  - FSQ levels=[8] → 8 codes (direct comparison with laq/ 8-code baseline)
  - spread_reg_weight=2.0 forces z_pre ~ N(0,1): std_penalty prevents posterior
    collapse (all z_pre → 0) by requiring unit variance across the batch
  - NO entropy_reg: entropy_reg on z_bounded causes boundary-attraction (soft
    entropy near max while hard perplexity drops), degrading NMI over training
  - 5000 steps with frequent logging so quality is visible quickly

Compare against:
  laq/results_nuscenes_base_8codebook_smoke_/     baseline NSVQ NMI=0.2150
  laq/results_nuscenes_egomotion_mag8_smoke/      best     NSVQ NMI=0.2415

Run:
    cd laq_ad
    pip install -e .
    python train_nuscenes_mini.py
"""

import os
os.environ["WANDB_MODE"] = "offline"

import torch
torch.backends.cudnn.enabled = False

from pathlib import Path
from nuscenes.nuscenes import NuScenes
from laq_ad_model import (
    LAQADModel,
    LAQADTrainer,
    NuScenesLAQDataset,
    maneuver_balanced_sampler,
    collate_fn,
)

NUSCENES_DATAROOT = Path("/home/andy/Dataset/Nuscenes/v1.0-mini")
FRAMES_DIR = Path("/home/andy/Dataset/nuscenes_laq_frames")

nusc = NuScenes(version="v1.0-mini", dataroot=str(NUSCENES_DATAROOT), verbose=False)

dataset = NuScenesLAQDataset(
    nusc=nusc,
    frames_dir=FRAMES_DIR,
    offset=3,
    use_heatmap=True,
    use_can_bus=True,
)
print(f"{len(dataset)} pairs | {dataset.class_counts}")

model = LAQADModel(
    dim=512,
    levels=[3, 3],    # 9 codes, 2 independent axes (yaw × speed)
    image_size=256,
    patch_size=32,
    spatial_depth=2,
    temporal_depth=2,
    dim_head=64,
    heads=8,
    heatmap_alpha=1.0,
    can_bus_weight=0.05,
    spread_reg_weight=2.0,
    covariance_reg_weight=1.0,   # VICReg: forces axes orthogonal — prevents correlated bimodal
    entropy_reg_weight=0.0,
    flow_prediction_weight=0.5,  # direct motion supervision from z_q; bypasses decoder shortcut
)

trainer = LAQADTrainer(
    model=model,
    dataset=dataset,
    sampler=maneuver_balanced_sampler(dataset),
    results_folder="results_laq_ad_mini",
    num_train_steps=5_001,
    batch_size=16,
    lr=1e-4,
    warmup_steps=500,
    save_model_every=1_000,
    save_results_every=100,
    use_wandb=False,
    collate_fn=collate_fn,
)

trainer.train()

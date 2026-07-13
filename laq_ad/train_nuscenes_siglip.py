"""LAQ-AD trainval experiment: SigLIP NaFlex spatial encoder.

Replaces the simple patch embedding (3×32²=3072 → 512 linear layer) with a
pretrained NaFlex SigLIP ViT-B/16 encoder (768-dim, pooled 16×16 → 8×8 tokens).
The SigLIP backbone is frozen; only the projection head and the rest of the
LAQ-AD architecture are trained.  This gives the model access to rich semantic
and motion features from the SigLIP pretraining.

The rest of the config is identical to the best [3,3] perAxisCAN run (NMI=0.40):
  - FSQ levels=[3,3] → 9 codes
  - Per-axis CAN bus heads (axis0=yaw, axis1=speed)
  - Flow prediction from z_q → egomotion heatmap
  - ManeuverBalancedSampler

Two-phase training recommended:
  Phase 1 (this script): freeze_backbone=True → train projection + quantizer
  Phase 2: unfreeze backbone with lower lr (set freeze_backbone=False, lr=1e-5)

Run:
    conda activate lapa
    cd laq_ad
    pip install -e .
    python train_nuscenes_siglip.py
"""

import os
os.environ["WANDB_MODE"] = "offline"

from pathlib import Path
import torch
torch.backends.cudnn.enabled = False

# ── Config ────────────────────────────────────────────────────────────────────
NUSCENES_DATAROOT = Path("/media/andy/Samsung_T7/nuScenes/Trainval")
NUSCENES_VERSION  = "v1.0-trainval"
FRAMES_DIR        = Path("/home/andy/Dataset/nuscenes_laq_frames_improved")

OFFSET            = 3
BATCH_SIZE        = 8          # smaller batch: SigLIP encoder uses more VRAM
NUM_TRAIN_STEPS   = 50_000
WARMUP_STEPS      = 500
RESULTS_FOLDER    = "results_laq_ad_trainval_3x3_siglip"
RESUME_FROM       = None

LEVELS            = [3, 3]     # same as best [3,3] perAxisCAN run
FREEZE_BACKBONE   = True       # Phase 1: freeze SigLIP, train the rest

# ── Build ──────────────────────────────────────────────────────────────────────
from nuscenes.nuscenes import NuScenes
from laq_ad_model import (
    LAQADTrainer,
    NuScenesLAQDataset,
    maneuver_balanced_sampler,
    collate_fn,
)
from laq_ad_model.model_siglip import LAQADModelSigLIP

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

print("Loading SigLIP NaFlex backbone (downloads on first run ~300MB)…")
model = LAQADModelSigLIP(
    dim=512,
    levels=LEVELS,
    siglip_model_name="naflexvit_base_patch16_siglip",
    freeze_backbone=FREEZE_BACKBONE,
    image_size=256,
    patch_size=32,            # determines 8×8 grid size; SigLIP pools to this
    spatial_depth=2,
    temporal_depth=2,
    dim_head=64,
    heads=8,
    heatmap_alpha=1.0,
    can_bus_weight=1.0,
    spread_reg_weight=0.05,
    covariance_reg_weight=1.0,
    entropy_reg_weight=0.0,
    flow_prediction_weight=5.0,
    recon_loss_weight=0.0,
)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"  LAQ-AD codebook size: {model.codebook_size}")
print(f"  Parameters: {trainable:,} trainable / {total:,} total "
      f"({100*trainable/total:.1f}% — backbone {'frozen' if FREEZE_BACKBONE else 'unfrozen'})")

trainer = LAQADTrainer(
    model=model,
    dataset=dataset,
    sampler=sampler,
    results_folder=RESULTS_FOLDER,
    num_train_steps=NUM_TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    grad_accum_every=2,       # effective batch = 16 despite smaller physical batch
    lr=1e-4,
    warmup_steps=WARMUP_STEPS,
    save_model_every=2_000,
    save_results_every=200,
    use_wandb=False,
    collate_fn=collate_fn,
    resume_from=RESUME_FROM,
)

trainer.train()

"""Experiment 1: 8-code ego-motion, magnitude-only gate.

Isolated test of whether ego-motion magnitude heatmaps help a larger codebook.
Direction channels (fx, fy) are loaded from the 3-channel .egomotion.npy caches
but zeroed out via heatmap_dir_alpha=0.0, so only the flow-magnitude gate is
active: gate = 1 + alpha * mag.

Compare against:
  results_nuscenes_base_8codebook_smoke_/    (8-code, no heatmap)
  results_nuscenes_egomotion_mag8_smoke/     (this run)

Run data prep first if not already done:
  python ../data/prepare_nuscenes_laq.py
  python ../data/prepare_nuscenes_egomotion.py --overwrite  # regenerates 3-channel caches

Then:
  python train_nuscenes_egomotion_mag8.py
"""

import os
os.environ["WANDB_MODE"] = "offline"

import torch
torch.backends.cudnn.enabled = False

from laq_model import LAQTrainer, LatentActionQuantization
from laq_model.data import EgoMotionVideoDataset

laq = LatentActionQuantization(
    dim=512,
    quant_dim=32,
    codebook_size=8,
    image_size=256,
    patch_size=32,
    spatial_depth=2,
    temporal_depth=2,
    dim_head=64,
    heads=16,
    code_seq_len=1,
    heatmap_alpha=1.0,
    heatmap_dir_alpha=0.0,    # direction channels disabled — magnitude gate only
    heatmap_abs_dir=False,    # n/a when heatmap_dir_alpha=0
    heatmap_additive_dir=False,
).cuda()

trainer = LAQTrainer(
    laq,
    folder="/home/andy/Dataset/nuscenes_laq_frames",
    offsets=3,
    batch_size=16,
    grad_accum_every=1,
    train_on_images=False,
    use_ema=False,
    num_train_steps=3001,
    results_folder="results_nuscenes_egomotion_mag8_smoke",
    lr=1e-4,
    save_model_every=1000,
    save_results_every=200,
    dataset_cls=EgoMotionVideoDataset,
    dataset_returns_heatmap=True,
)

trainer.train()

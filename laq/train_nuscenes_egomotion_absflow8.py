"""Experiment 2: 8-code ego-motion, absolute-value direction gate.

Fixes the codebook-collapse problem from Exp (direction signed) by using
|fx| + |fy| instead of fx + fy in the gate:

    gate = 1 + alpha * mag + alpha_dir * (|fx| + |fy|)

Gate is now >= 1 always (no patch suppression), so NSVQ training stays
stable. The absolute-value direction still distinguishes:
  turns (high |fx| throughout image) vs
  straight (antisymmetric fx cancels in the spatial average, so lower |fx|)
  stationary (near-zero flow, both mag and |fx|/|fy| ≈ 0)

Loses signed left-vs-right distinction; use Exp 3 (additive) for that.

Compare against:
  results_nuscenes_base_8codebook_smoke_/          (8-code, no heatmap)
  results_nuscenes_egomotion_mag8_smoke/           (Exp 1)
  results_nuscenes_egomotion_absflow8_smoke/       (this run)

Run data prep first:
  python ../data/prepare_nuscenes_egomotion.py --overwrite  # 3-channel caches required

Then:
  python train_nuscenes_egomotion_absflow8.py
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
    heatmap_dir_alpha=0.25,
    heatmap_abs_dir=True,       # |fx|+|fy|: no suppression, stable gate
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
    results_folder="results_nuscenes_egomotion_absflow8_smoke",
    lr=1e-4,
    save_model_every=1000,
    save_results_every=200,
    dataset_cls=EgoMotionVideoDataset,
    dataset_returns_heatmap=True,
)

trainer.train()

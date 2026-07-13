"""Experiment 3: 8-code ego-motion, additive direction embedding.

Combines a magnitude gate with a learned additive direction embedding:

    patch_tokens = (1 + alpha * mag) * patch_tokens   # magnitude gate
    patch_tokens = patch_tokens + Linear([fx, fy])    # additive direction

The Linear projection (2 → dim) is learned end-to-end so the model weights
fx vs fy optimally. Critically, signed fx/fy are preserved — the encoder
sees opposite embeddings for left turns (fx>0) and right turns (fx<0) —
without any patch suppression or scale distortion.

heatmap_dir_alpha=0.0 disables the multiplicative direction terms so the
gate is pure magnitude-only, and only the additive embedding carries the
direction signal.

Compare against:
  results_nuscenes_base_8codebook_smoke_/              (8-code, no heatmap)
  results_nuscenes_egomotion_mag8_smoke/               (Exp 1: mag gate only)
  results_nuscenes_egomotion_absflow8_smoke/           (Exp 2: |fx| gate)
  results_nuscenes_egomotion_additive8_smoke/          (this run)

Run data prep first:
  python ../data/prepare_nuscenes_egomotion.py --overwrite  # 3-channel caches required

Then:
  python train_nuscenes_egomotion_additive8.py
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
    heatmap_dir_alpha=0.0,      # multiplicative direction disabled — additive carries it
    heatmap_abs_dir=False,      # n/a when heatmap_dir_alpha=0
    heatmap_additive_dir=True,  # Linear([fx, fy]) → dim added to patch tokens
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
    results_folder="results_nuscenes_egomotion_additive8_smoke",
    lr=1e-4,
    save_model_every=1000,
    save_results_every=200,
    dataset_cls=EgoMotionVideoDataset,
    dataset_returns_heatmap=True,
)

trainer.train()

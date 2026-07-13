"""Phase-1 A/B counterpart to train_nuscenes.py: saliency-conditioned encoder.

Identical model/training config to train_nuscenes.py so the two runs are a
controlled comparison — the only difference is HeatmapVideoDataset (which
loads the saliency caches written by data/prepare_nuscenes_heatmaps.py) and
dataset_returns_heatmap=True.

Run data prep first:
  python ../data/prepare_nuscenes_laq.py
  python ../data/prepare_nuscenes_heatmaps.py

Then train:
  python train_nuscenes_heatmap.py

Results (reconstruction PNGs + checkpoint) land in results_nuscenes_heatmap_smoke/.
Compare against results_nuscenes_smoke_v3/ (the baseline, no-heatmap run) with
eval_compare_checkpoints.py.
"""

import os
os.environ["WANDB_MODE"] = "offline"  # no wandb account needed

import torch
torch.backends.cudnn.enabled = False  # work around cuDNN version mismatch on this system

from laq_model import LAQTrainer, LatentActionQuantization
from laq_model.data import HeatmapVideoDataset

# Same reduced config as train_nuscenes.py for a fair, controlled comparison.
laq = LatentActionQuantization(
    dim=512,
    quant_dim=32,
    codebook_size=4,   # 4 codes: batch_size=16 can activate all 4; 8 was too many
    image_size=256,
    patch_size=32,
    spatial_depth=2,
    temporal_depth=2,
    dim_head=64,
    heads=16,
    code_seq_len=1,
    heatmap_alpha=1.0,
).cuda()

trainer = LAQTrainer(
    laq,
    folder="/home/andy/Dataset/nuscenes_laq_frames",
    offsets=3,          # ~250ms between frames at 12Hz sweep rate
    batch_size=16,
    grad_accum_every=1,
    train_on_images=False,
    use_ema=False,
    num_train_steps=3001,
    results_folder="results_nuscenes_heatmap_smoke",
    lr=1e-4,
    save_model_every=1000,
    save_results_every=200,
    dataset_cls=HeatmapVideoDataset,
    dataset_returns_heatmap=True,
)

trainer.train()

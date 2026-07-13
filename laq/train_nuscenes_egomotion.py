"""Ego-motion saliency conditioned LAQ training (controlled comparison).

Identical model/training config to train_nuscenes_heatmap.py so the two
runs are a controlled comparison — the only difference is EgoMotionVideoDataset,
which loads DIS optical-flow magnitude heatmaps (data/prepare_nuscenes_egomotion.py)
instead of YOLO detection heatmaps.

Ego-motion heatmaps highlight WHERE the camera's own movement is most visible
(road markings, nearby structures), steering the encoder toward action-relevant
patches rather than other-agent positions.

Run data prep first:
  python ../data/prepare_nuscenes_laq.py
  python ../data/prepare_nuscenes_egomotion.py

Then train:
  python train_nuscenes_egomotion.py

Results land in results_nuscenes_egomotion_smoke/.
Compare against:
  results_nuscenes_smoke_v3/           (baseline, no heatmap)
  results_nuscenes_heatmap_smoke/      (detection-based heatmap)
with eval_compare_checkpoints.py and eval_latent_clusters.py.
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
    results_folder="results_nuscenes_egomotion_flow_direction_smoke",
    lr=1e-4,
    save_model_every=1000,
    save_results_every=200,
    dataset_cls=EgoMotionVideoDataset,
    dataset_returns_heatmap=True,
)

trainer.train()

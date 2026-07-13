"""Phase-2 A/B counterpart to train_nuscenes_heatmap.py: + cost-map latent regularization.

Identical model/training config to train_nuscenes.py and
train_nuscenes_heatmap.py (same 5001-step budget) so all runs are a
controlled comparison. Adds CostmapVideoDataset (which also loads the BEV
cost-map caches written by data/prepare_nuscenes_costmaps.py) and a nonzero
costmap_loss_weight, pulling together latent actions of scenes with a
similar collision-risk profile (see laq_model/costmap_loss.py).

This went through several iterations during development (see
laq_model/costmap_loss.py's docstring for the full story):
  1. Pull-only loss on raw latent distance: collapsed the codebook to a
     single entry at every nonzero weight tried (0.001-0.1).
  2. Pull-only loss on L2-normalized latent distance: fixed the uniform
     collapse, but still degraded codebook health vs. Phase 1 at every
     weight, without improving cost-map cluster alignment.
  3. Geometry-matching (Gram-matrix MSE) loss: has built-in repulsion, but
     *still* collapsed the codebook -- within the first ~60 steps, before
     the codebook had even stabilized via reconstruction alone.
That last finding pointed at training dynamics, not loss math: every
healthy run (baseline, heatmap-only, and a costmap_loss_weight=0 control on
this same dataset pipeline) shows transient collapse for several hundred
steps that self-corrects via NSVQ's replace_unused_codebooks; every
costmap-loss-enabled run never recovers. --costmap-warmup-steps holds the
regularizer's effective weight at 0 until the codebook has had a chance to
stabilize on its own, then linearly ramps it in.

Run data prep first:
  python ../data/prepare_nuscenes_laq.py
  python ../data/prepare_nuscenes_heatmaps.py
  python ../data/prepare_nuscenes_costmaps.py

Then train (results folder defaults to results_nuscenes_costmap_smoke_w<weight>):
  python train_nuscenes_costmap.py --costmap-loss-weight 0.3 --costmap-warmup-steps 1000

Compare against results_nuscenes_smoke_v3/ (baseline) and
results_nuscenes_heatmap_smoke/ (Phase 1) with eval_compare_checkpoints.py.
"""

import argparse
import os
os.environ["WANDB_MODE"] = "offline"  # no wandb account needed

import torch
torch.backends.cudnn.enabled = False  # work around cuDNN version mismatch on this system

from laq_model import LAQTrainer, LatentActionQuantization
from laq_model.data import CostmapVideoDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--costmap-loss-weight", type=float, default=0.3)
    parser.add_argument("--costmap-warmup-steps", type=int, default=1000)
    parser.add_argument("--num-train-steps", type=int, default=5001)
    parser.add_argument("--results-folder", type=str, default=None,
                         help="Defaults to results_nuscenes_costmap_smoke_w<weight>.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_folder = args.results_folder or f"results_nuscenes_costmap_smoke_w{args.costmap_loss_weight}"

    # Same reduced config as train_nuscenes.py / train_nuscenes_heatmap.py for
    # a fair, controlled comparison -- only costmap_loss_weight/warmup vary.
    laq = LatentActionQuantization(
        dim=256,
        quant_dim=32,
        codebook_size=4,
        image_size=256,
        patch_size=32,
        spatial_depth=2,
        temporal_depth=2,
        dim_head=64,
        heads=4,
        code_seq_len=1,
        heatmap_alpha=1.0,
        costmap_loss_weight=args.costmap_loss_weight,
        costmap_warmup_steps=args.costmap_warmup_steps,
    ).cuda()

    trainer = LAQTrainer(
        laq,
        folder="/tmp/nuscenes_laq_frames",
        offsets=3,
        batch_size=16,
        grad_accum_every=1,
        train_on_images=False,
        use_ema=False,
        num_train_steps=args.num_train_steps,
        results_folder=results_folder,
        lr=1e-4,
        save_model_every=1000,
        save_results_every=200,
        dataset_cls=CostmapVideoDataset,
        dataset_returns_heatmap=True,
        dataset_returns_costmap=True,
    )

    trainer.train()


if __name__ == "__main__":
    main()

"""Evaluate a trained LAQ-AD checkpoint on nuScenes (mini or trainval).

Reports:
  - Codebook usage and perplexity
  - NMI (latent code vs ground-truth maneuver label)
  - Per-code maneuver distribution table
  - CAN bus prediction MAE (yaw_rate, speed) — if CAN bus available

Usage (from laq_ad/ directory):
    python eval_nuscenes.py --checkpoint results_laq_ad_mini/laq_ad.5000.pt \\
                            --version v1.0-mini \\
                            --levels 8 8
"""

import argparse
import os
os.environ["WANDB_MODE"] = "offline"

from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

from nuscenes.nuscenes import NuScenes
from laq_ad_model import LAQADModel, NuScenesLAQDataset, collate_fn, MANEUVERS
from laq_ad_model.eval_utils import print_summary
from laq_ad_model.model import CANBusHead


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--dataroot", default="/home/andy/Dataset/Nuscenes/v1.0-mini",
                   help="nuScenes dataroot — the directory that contains the version subdir "
                        "(e.g. /media/andy/Samsung_T7/nuScenes/Trainval for trainval)")
    p.add_argument("--frames-dir", default="/home/andy/Dataset/nuscenes_laq_frames", type=Path)
    p.add_argument("--offset", type=int, default=3)
    p.add_argument("--levels", nargs="+", type=int, default=None,
                   help="FSQ levels — auto-detected from config.json if omitted")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--no-heatmap", action="store_true")
    return p.parse_args()


def load_config(checkpoint: Path) -> dict:
    """Load config.json from the checkpoint's parent directory, if present."""
    import json
    cfg_path = checkpoint.parent / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        print(f"Loaded config from {cfg_path}: {cfg}")
        return cfg
    return {}


def main():
    args = parse_args()

    # Auto-detect architecture from config.json saved alongside the checkpoint
    cfg = load_config(args.checkpoint)
    levels = args.levels or cfg.get("levels", [8])
    heatmap_alpha = 0.0 if args.no_heatmap else cfg.get("heatmap_alpha", 1.0)
    can_bus_weight = cfg.get("can_bus_weight", 0.1)
    entropy_reg_weight = cfg.get("entropy_reg_weight", 0.0)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    dataset = NuScenesLAQDataset(
        nusc=nusc,
        frames_dir=args.frames_dir,
        offset=args.offset,
        use_heatmap=not args.no_heatmap,
        use_can_bus=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = LAQADModel(
        dim=512,
        levels=levels,
        image_size=256,
        patch_size=32,
        spatial_depth=2,
        temporal_depth=2,
        dim_head=64,
        heads=8,
        heatmap_alpha=heatmap_alpha,
        can_bus_weight=can_bus_weight,
        entropy_reg_weight=entropy_reg_weight,
    ).cuda().eval()

    model.load(args.checkpoint)
    print(f"Loaded {args.checkpoint} | codebook_size={model.codebook_size}")

    all_codes, all_labels = [], []
    yr_errors, sp_errors = [], []

    with torch.no_grad():
        for video, heatmap, can_bus, maneuver_idx in loader:
            video = video.cuda()
            if heatmap is not None:
                heatmap = heatmap.cuda()

            indices = model.inference(video, heatmap=heatmap, return_only_codebook_ids=True)
            codes = indices.cpu().tolist()
            labels = [MANEUVERS[i] for i in maneuver_idx.tolist()]
            all_codes.extend(codes)
            all_labels.extend(labels)

            # CAN bus prediction error
            if can_bus is not None and model.can_bus_head is not None:
                can_bus = can_bus.cuda()
                # Re-run quantize step to get z_q
                first_frame = video[:, :, :1]
                rest_frames = video[:, :, 1:]
                first_tokens = model.to_patch_emb(first_frame)
                rest_tokens = model.to_patch_emb(rest_frames)
                if heatmap is not None:
                    from laq_ad_model.heatmap import apply_patch_heatmap
                    first_tokens = apply_patch_heatmap(first_tokens, heatmap[:, :1], model.heatmap_alpha)
                    rest_tokens = apply_patch_heatmap(rest_tokens, heatmap[:, 1:], model.heatmap_alpha)
                tokens = torch.cat([first_tokens, rest_tokens], dim=1)
                _, last_enc = model._encode(tokens)
                z_q, *_ = model._quantize(last_enc)
                pred = model.can_bus_head(z_q).cpu().numpy()
                gt = can_bus.cpu().numpy()
                yr_errors.extend(np.abs(pred[:, 0] - gt[:, 0]).tolist())
                sp_errors.extend(np.abs(pred[:, 1] - gt[:, 1]).tolist())

    print_summary(args.checkpoint.stem, all_codes, all_labels, model.codebook_size)

    if yr_errors:
        print(f"\n  CAN bus MAE (normalized):")
        print(f"    yaw_rate: {np.mean(yr_errors):.4f}  (×{CANBusHead.YAW_SCALE} rad/s scale)")
        print(f"    speed:    {np.mean(sp_errors):.4f}  (×{CANBusHead.SPEED_SCALE} m/s scale)")


if __name__ == "__main__":
    main()

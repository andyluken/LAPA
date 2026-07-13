"""Sweep all checkpoints in a results folder and report NMI / perplexity.

For each checkpoint saved by LAQADTrainer, computes:
  - NMI (latent code vs ground-truth maneuver label)
  - Perplexity and number of active codes
Prints a table sorted by step, then shows the per-code maneuver distribution
for the best-NMI checkpoint.

Model architecture is auto-detected from results_folder/config.json.

Usage (from laq_ad/ directory):
    python eval_sweep_checkpoints.py --results results_laq_ad_trainval_3x3_flowStrong
    python eval_sweep_checkpoints.py --results results_laq_ad_trainval_3x3_flowStrong \\
        --version v1.0-mini \\
        --dataroot /home/andy/Dataset/Nuscenes/v1.0-mini
"""

import argparse
import json
import os
os.environ["WANDB_MODE"] = "offline"

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from nuscenes.nuscenes import NuScenes
from laq_ad_model import LAQADModel, LAQADModelSigLIP, NuScenesLAQDataset, collate_fn, MANEUVERS
from laq_ad_model.eval_utils import normalized_mutual_information, print_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, type=Path,
                   help="Results folder containing laq_ad.*.pt checkpoints and config.json")
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--dataroot", default="/media/andy/Samsung_T7/nuScenes/Trainval")
    p.add_argument("--frames-dir", default="/home/andy/Dataset/nuscenes_laq_frames", type=Path)
    p.add_argument("--offset", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-heatmap", action="store_true")
    return p.parse_args()


def load_config(results_folder: Path) -> dict:
    cfg_path = results_folder / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        print(f"  config: {cfg}")
        return cfg
    print("  WARNING: no config.json found — using defaults")
    return {}


def build_model(cfg: dict, use_heatmap: bool):
    common = dict(
        dim=cfg.get("dim", 512),
        levels=cfg.get("levels", [3, 3]),
        image_size=tuple(cfg.get("image_size", [256, 256])),
        patch_size=tuple(cfg.get("patch_size", [32, 32])),
        spatial_depth=2,
        temporal_depth=2,
        dim_head=64,
        heads=8,
        heatmap_alpha=0.0 if not use_heatmap else cfg.get("heatmap_alpha", 1.0),
        can_bus_weight=cfg.get("can_bus_weight", 0.05),
        #entropy_reg_weight=cfg.get("entropy_reg_weight", 0.0),
        spread_reg_weight=cfg.get("spread_reg_weight", 0.05),
        covariance_reg_weight=cfg.get("covariance_reg_weight", 1.0),
        flow_prediction_weight=cfg.get("flow_prediction_weight", 5.0),
        recon_loss_weight=cfg.get("recon_loss_weight", 0.0),
    )
    model_type = cfg.get("model_type", "LAQADModel")
    if model_type == "LAQADModelSigLIP":
        model = LAQADModelSigLIP(
            **common,
            siglip_model_name=cfg.get("siglip_model_name",
                                      "naflexvit_base_patch16_siglip"),
            freeze_backbone=True,
        )
    else:
        model = LAQADModel(**common)
    return model.cuda().eval()


def eval_checkpoint(model, loader, codebook_size):
    all_codes, all_labels = [], []
    with torch.no_grad():
        for video, heatmap, _, maneuver_idx in loader:
            video = video.cuda()
            if heatmap is not None:
                heatmap = heatmap.cuda()
            indices = model.inference(video, heatmap=heatmap, return_only_codebook_ids=True)
            all_codes.extend(indices.cpu().tolist())
            all_labels.extend([MANEUVERS[i] for i in maneuver_idx.tolist()])

    nmi = normalized_mutual_information(all_codes, all_labels)
    n_unique = len(set(all_codes))
    counts = np.bincount(all_codes, minlength=codebook_size).astype(float)
    probs = counts / counts.sum()
    log_probs = np.where(probs > 0, np.log(probs), 0.0)
    perplexity = np.exp(-(probs * log_probs).sum())
    return nmi, perplexity, n_unique, all_codes, all_labels


def main():
    args = parse_args()

    ckpts = sorted(args.results.glob("laq_ad.*.pt"),
                   key=lambda p: int(p.stem.split(".")[1]))
    if not ckpts:
        print(f"No checkpoints found in {args.results}")
        return

    print(f"Loading nuScenes {args.version}…", flush=True)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    cfg = load_config(args.results)
    use_heatmap = not args.no_heatmap

    dataset = NuScenesLAQDataset(
        nusc=nusc,
        frames_dir=args.frames_dir,
        offset=args.offset,
        use_heatmap=use_heatmap,
        use_can_bus=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=4)
    print(f"  {len(dataset)} pairs", flush=True)

    model = build_model(cfg, use_heatmap)
    codebook_size = model.codebook_size
    print(f"  codebook_size={codebook_size}  checkpoints={len(ckpts)}\n")

    print(f"{'Step':>8}  {'NMI':>8}  {'Perplexity':>10}  {'Codes':>6}")
    print("-" * 40)

    best_nmi, best_ckpt, best_codes, best_labels = 0.0, None, None, None
    for ckpt in ckpts:
        step = int(ckpt.stem.split(".")[1])
        model.load(ckpt)
        nmi, perplexity, n_unique, all_codes, all_labels = eval_checkpoint(
            model, loader, codebook_size)
        flag = " ◄ best" if nmi > best_nmi else ""
        print(f"{step:>8}  {nmi:>8.4f}  {perplexity:>10.2f}  {n_unique:>6}{flag}", flush=True)
        if nmi > best_nmi:
            best_nmi, best_ckpt = nmi, ckpt
            best_codes, best_labels = all_codes, all_labels

    if best_ckpt is not None:
        print_summary(best_ckpt.name, best_codes, best_labels, codebook_size)


if __name__ == "__main__":
    main()

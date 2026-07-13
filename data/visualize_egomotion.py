"""Visualize egomotion heatmaps overlaid on original frames.

Creates a 3×4 figure: one column per maneuver (right-turn, left-turn, straight,
stationary), three rows: original frame | magnitude overlay | flow-direction quiver.

Usage (from repo root):
    python data/visualize_egomotion.py
    python data/visualize_egomotion.py --output figures/egomotion_viz.png --dpi 150
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import FancyArrowPatch


# ── Representative frames (one per maneuver) ──────────────────────────────────
# Right-turn: high flow on LEFT patches (world moves left as car turns right)
# Left-turn:  high flow on RIGHT patches
# Straight:   uniform forward flow across all patches
# Stationary: near-zero flow everywhere
EXAMPLES = {
    "right turn":  Path("/home/andy/Dataset/nuscenes_laq_frames/01e4fcbe6e49483293ce45727152b36e/frame_0001"),
    "left turn":   Path("/home/andy/Dataset/nuscenes_laq_frames/034dee1695304630b0692da8c1f153fc/frame_0001"),
    "straight":    Path("/home/andy/Dataset/nuscenes_laq_frames/034256c9639044f98da7562ef3de3646/frame_0002"),
    "stationary":  Path("/home/andy/Dataset/nuscenes_laq_frames/080a52cb8f59489b9cddc7b721808088/frame_0002"),
}


def load_pair(stem: Path):
    """Return (img_rgb uint8 H×W×3, egomotion float32 3×8×8)."""
    img = cv2.cvtColor(cv2.imread(str(stem.with_suffix(".jpg"))), cv2.COLOR_BGR2RGB)
    ego = np.load(str(stem) + ".egomotion.npy")   # (3, 8, 8)
    return img, ego


def heatmap_to_overlay(img_rgb: np.ndarray, mag: np.ndarray,
                       cmap="inferno", alpha: float = 0.55) -> np.ndarray:
    """Upscale 8×8 magnitude map and alpha-blend onto the image."""
    h, w = img_rgb.shape[:2]
    mag_up = cv2.resize(mag, (w, h), interpolation=cv2.INTER_LINEAR)
    mag_norm = np.clip(mag_up, 0, 1)
    cm = plt.get_cmap(cmap)
    heatmap_rgba = cm(mag_norm)                           # H×W×4, float [0,1]
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    overlay = (img_rgb * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
    return overlay


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=None,
                   help="Save path (e.g. figures/egomotion_viz.png). "
                        "If omitted, opens an interactive window.")
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--alpha", type=float, default=0.55,
                   help="Heatmap blend alpha (0=invisible, 1=opaque)")
    return p.parse_args()


def main():
    args = parse_args()

    labels = list(EXAMPLES.keys())
    n_cols = len(labels)
    n_rows = 3    # original | overlay | quiver

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3.2 * n_rows),
                             constrained_layout=True)
    fig.suptitle("Egomotion heatmaps — nuScenes front camera", fontsize=13, y=1.01)

    row_labels = ["original frame", "magnitude overlay", "flow direction (8×8)"]
    for row, rlabel in enumerate(row_labels):
        axes[row, 0].set_ylabel(rlabel, fontsize=10, labelpad=6)

    for col, label in enumerate(labels):
        stem = EXAMPLES[label]
        img, ego = load_pair(stem)

        mag = ego[0]   # (8,8) in [0,1]
        fx  = ego[1]   # (8,8) in [-1,1]
        fy  = ego[2]   # (8,8) in [-1,1]

        # ── Row 0: original frame ────────────────────────────────────────────
        ax = axes[0, col]
        ax.imshow(img)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axis("off")

        # ── Row 1: magnitude overlay ─────────────────────────────────────────
        ax = axes[1, col]
        overlay = heatmap_to_overlay(img, mag, cmap="inferno", alpha=args.alpha)
        ax.imshow(overlay)
        # Colorbar tickmarks
        sm = plt.cm.ScalarMappable(cmap="inferno",
                                   norm=mcolors.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, label="mag/peak")
        ax.axis("off")

        # ── Row 2: flow direction quiver ─────────────────────────────────────
        # Use normalized [0,1]×[0,1] coords so quiver params are image-size agnostic.
        ax = axes[2, col]
        ax.imshow(img, extent=[0, 1, 1, 0])  # display in unit square

        xs = np.linspace(1/16, 15/16, 8)
        ys = np.linspace(1/16, 15/16, 8)
        xg, yg = np.meshgrid(xs, ys)

        u = fx * mag   # in normalized image units, positive → rightward
        v = -fy * mag  # flip y: fy convention follows DIS optical flow (down=positive)

        ax.quiver(xg, yg, u, v,
                  mag.flatten(),
                  cmap="inferno", clim=[0, 1],
                  pivot="mid",
                  scale=10,        # larger → shorter arrows; 10 ≈ 10% width per unit
                  width=0.006,     # shaft width as fraction of axes width
                  headwidth=4, headlength=4)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.axis("off")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved → {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

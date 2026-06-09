"""
Visual inspection of LAQ codebook codes.

For each code (0–3), shows 8 example frame pairs the model assigned to it.
Each cell = side-by-side [frame_t | frame_t+3] so you can see what motion
the code captured.  Maneuver label printed under each cell.

Usage (from laq/ directory):
    python visualize_codes.py
Output:
    laq/code_examples.png
"""

import os, sys, random
os.environ["WANDB_MODE"] = "offline"
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.backends.cudnn.enabled = False

# Reuse all shared logic from eval_latent_clusters
from eval_latent_clusters import (
    load_nuscenes_pairs, load_model, maneuver_label,
    transform, BATCH_SIZE, NUSCENES_ROOT, MANEUVERS
)

N_CODES       = 4
EXAMPLES_PER_CODE = 8
DISPLAY_SIZE  = 128   # px per frame in the output grid


def run_inference_with_paths(laq, pairs):
    """Returns list of (code, maneuver_label, path_t, path_t3)."""
    results = []
    for start in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[start : start + BATCH_SIZE]
        imgs = []
        for path_t, path_t3, _, _ in batch:
            img_t  = transform(Image.open(path_t)).unsqueeze(1)
            img_t3 = transform(Image.open(path_t3)).unsqueeze(1)
            imgs.append(torch.cat([img_t, img_t3], dim=1))
        video = torch.stack(imgs).cuda()
        with torch.no_grad():
            indices = laq.inference(video, return_only_codebook_ids=True)
            codes = indices[:, 0].cpu().tolist()
        for code, (path_t, path_t3, pose_t, pose_t3) in zip(codes, batch):
            label = maneuver_label(pose_t, pose_t3)
            results.append((int(code), label, path_t, path_t3))
    return results


def diverse_sample(items, n):
    """Sample up to n items, preferring diversity across maneuver types."""
    by_maneuver = defaultdict(list)
    for item in items:
        by_maneuver[item[0]].append(item)  # item[0] = label

    selected = []
    quota = max(1, n // len(MANEUVERS))
    for m in MANEUVERS:
        selected.extend(random.sample(by_maneuver[m], min(quota, len(by_maneuver[m]))))

    # fill remainder randomly if we haven't hit n yet
    remaining = [x for x in items if x not in selected]
    random.shuffle(remaining)
    selected.extend(remaining[:max(0, n - len(selected))])
    return selected[:n]


def load_display_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR)
    return np.array(img)


def build_grid(by_code, n_codes=N_CODES, cols=EXAMPLES_PER_CODE):
    """Return figure with 4 rows × cols side-by-side cells."""
    cell_w = DISPLAY_SIZE * 2   # two frames side by side
    cell_h = DISPLAY_SIZE

    fig_w = cols * cell_w / 100 + 1.5   # inches (+margin for row labels)
    fig_h = n_codes * (cell_h / 100 + 0.55)
    fig, axes = plt.subplots(n_codes, cols,
                             figsize=(fig_w, fig_h),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.6})

    total_by_code = {c: len(by_code[c]) for c in range(n_codes)}

    for row, code in enumerate(range(n_codes)):
        items = by_code[code]
        samples = diverse_sample([(lbl, pt, pt3) for lbl, pt, pt3 in items],
                                 min(cols, len(items)))

        for col in range(cols):
            ax = axes[row][col]
            ax.axis("off")
            if col < len(samples):
                label, path_t, path_t3 = samples[col]
                try:
                    left  = load_display_image(path_t)
                    right = load_display_image(path_t3)
                    cell  = np.hstack([left, right])
                    ax.imshow(cell)
                    ax.set_xlabel(label, fontsize=6, labelpad=2)
                except Exception:
                    pass   # broken symlink or corrupt image — skip

        # row label on the leftmost cell
        axes[row][0].set_ylabel(
            f"Code {code}\n({total_by_code[code]} pairs)",
            fontsize=8, rotation=0, labelpad=60, va="center"
        )

    fig.suptitle(
        "LAQ codebook codes — example frame pairs [frame_t | frame_t+3]",
        fontsize=10, y=1.01
    )
    return fig


if __name__ == "__main__":
    random.seed(42)

    print("Loading nuScenes frame pairs...")
    pairs = load_nuscenes_pairs()
    print(f"  {len(pairs)} pairs")

    laq = load_model()

    print("Running inference on all pairs...")
    results = run_inference_with_paths(laq, pairs)

    by_code = defaultdict(list)
    for code, label, path_t, path_t3 in results:
        by_code[code].append((label, path_t, path_t3))

    print("Code counts:")
    for c in range(N_CODES):
        print(f"  Code {c}: {len(by_code[c])} pairs")

    print("Building grid...")
    fig = build_grid(by_code)

    out = Path(__file__).parent / "code_examples.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved → {out}")

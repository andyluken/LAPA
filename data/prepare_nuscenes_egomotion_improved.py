"""Precomputes ego-motion saliency heatmaps for nuScenes LAQ frames.

Uses dense optical flow (DIS) between consecutive frames to measure where
the ego camera's motion is most salient in the image.  The flow magnitude
at each pixel is dominated by the global camera motion (not individual agents)
once area-averaged down to the 8×8 LAQ patch grid, because other-agent
pixels are a small fraction of each 32×32-pixel patch.

Flow pattern semantics at patch scale:
  straight   → radial outward flow from the vanishing point; road / lane
                markings in the lower half have high magnitude
  turn left  → rotation component; right side of image moves right-to-left
  turn right → opposite rotation
  stationary → near-zero flow everywhere → flat (all-zero) heatmap

For every frameNNNN.jpg it writes a sibling frameNNNN.egomotion.npy: an
(8, 8) float32 array normalized to [0, 1].  The LAST frame in each scene
has no successor; it is skipped (no .egomotion.npy written).

Idempotent: frames whose cache already exists are skipped unless
--overwrite is passed.

Usage:
    python data/prepare_nuscenes_egomotion.py [--overwrite]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

FRAMES_DIR = Path("/home/andy/Dataset/nuscenes_laq_frames_improved")
PATCH_GRID = (8, 8)   # (h, w) for image_size=256, patch_size=32
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# Flow below this pixel magnitude (after resize to compute scale) is treated
# as essentially stationary; the heatmap is set to all-zeros.
STATIONARY_THRESHOLD = 0.5  # pixels at native resolution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    parser.add_argument("--patch-h", type=int, default=PATCH_GRID[0])
    parser.add_argument("--patch-w", type=int, default=PATCH_GRID[1])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sorted_frames(scene_dir: Path) -> list[Path]:
    files = [
        f for f in scene_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda p: int(''.join(filter(str.isdigit, p.stem))))


def compute_egomotion_heatmap(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    patch_h: int,
    patch_w: int,
) -> np.ndarray:
    """Return a (3, patch_h, patch_w) float32 ego-motion feature map.

    Channel layout (all share the same peak normalization so they are
    geometrically consistent):
      [0] mag / peak  ∈ [0,  1]   — how much each patch moved
      [1] fx  / peak  ∈ [-1, 1]   — signed horizontal flow (+ve = rightward)
      [2] fy  / peak  ∈ [-1, 1]   — signed vertical flow   (+ve = downward)

    Maneuver signatures at patch scale:
      straight   → [1] antisymmetric (left<0, right>0), [2] positive in lower half
      turn left  → [1] uniformly positive across image
      turn right → [1] uniformly negative across image
      stationary → all channels zero

    Sharing the peak across channels preserves the geometric relationship
    between components: |fx|, |fy| ≤ mag ≤ peak by construction.
    """
    # Convert to grayscale for optical flow computation
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    # Compute dense optical flow (DIS) between the two frames
    dis = cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_MEDIUM)
    flow = dis.calc(gray_a, gray_b, None)  # (H, W, 2): [..., 0]=fx, [..., 1]=fy

    fx_raw = flow[..., 0]
    fy_raw = flow[..., 1]
    
    # Downsample the raw pixel displacements first using area averaging
    # This accurately preserves the mean pixel displacement in each patch, which is what we want for
    fx_patch = cv2.resize(fx_raw, (patch_w, patch_h), interpolation=cv2.INTER_AREA)
    fy_patch = cv2.resize(fy_raw, (patch_w, patch_h), interpolation=cv2.INTER_AREA)

    # Compute the magnitude of the flow vectors at each pixel
    mag_patch, _ = cv2.cartToPolar(fx_patch, fy_patch)
    #mag    = np.sqrt(fx_raw ** 2 + fy_raw ** 2)

    #extract peak magnitude from the localized patch scale to avoid single-pixel noise spikes
    patch_peak = float(mag_patch.max())

    # Apply stationary threshold check
    if patch_peak < STATIONARY_THRESHOLD:
        return np.zeros((3, patch_h, patch_w), dtype=np.float32)

    # Normalize safely [0, 1] for mag, [-1, 1} for direction fx, fy
    mag_norm = mag_patch / patch_peak
    fx_norm = fx_patch / patch_peak
    fy_norm = fy_patch / patch_peak

    return np.stack([mag_norm, fx_norm, fy_norm], axis=0).astype(np.float32)
    
def egomotion_cache_path(frame_path: Path) -> Path:
    return frame_path.with_suffix('').with_suffix('.egomotion.npy')


def main() -> None:
    args = parse_args()
    assert args.frames_dir.exists(), (
        f"Run data/prepare_nuscenes_laq.py first: {args.frames_dir} not found"
    )

    scene_dirs = sorted(d for d in args.frames_dir.iterdir() if d.is_dir())
    print(f"Found {len(scene_dirs)} scenes under {args.frames_dir}")

    written = skipped = 0

    for scene_dir in tqdm(scene_dirs, desc="scenes"):
        frames = sorted_frames(scene_dir)

        # Consecutive pairs: (frame_n, frame_{n+1}); last frame has no successor.
        for frame_a_path, frame_b_path in zip(frames[:-1], frames[1:]):
            cache_path = egomotion_cache_path(frame_a_path)
            if cache_path.exists() and not args.overwrite:
                skipped += 1
                continue

            img_a = cv2.imread(str(frame_a_path))
            img_b = cv2.imread(str(frame_b_path))
            if img_a is None or img_b is None:
                tqdm.write(f"  warning: could not read {frame_a_path} or {frame_b_path}, skipping")
                continue

            heatmap = compute_egomotion_heatmap(img_a, img_b, args.patch_h, args.patch_w)
            np.save(cache_path, heatmap)
            written += 1

    print(f"\nWrote {written} egomotion caches, skipped {skipped} already-cached frames.")
    print("Note: the last frame of each scene has no .egomotion.npy (no successor frame).")
    print("      EgoMotionVideoDataset falls back to an all-zero heatmap for those frames.")


if __name__ == "__main__":
    main()

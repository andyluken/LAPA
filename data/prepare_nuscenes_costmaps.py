"""Precomputes BEV collision-risk cost maps for nuScenes LAQ frames (Phase 2).

Run data/prepare_nuscenes_laq.py first (this script reproduces its exact
CAM_FRONT chain-walking algorithm via nuscenes-devkit, so frameNNNN.jpg
numbering lines up exactly without needing to reverse-resolve symlinks).
For every frameNNNN.jpg it writes a sibling frameNNNN.costmap.npy: a 32x32
float32 ego-centric BEV grid (40m forward x +/-20m lateral), built from
YOLO detections projected onto the ground plane and weighted by per-class
collision risk (see heatmap_lapa/costmap.py).

Idempotent: frames whose cache already exists are skipped unless
--overwrite is passed.

Usage:
    python data/prepare_nuscenes_costmaps.py
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heatmap_lapa.detector import YoloDetector
from heatmap_lapa.costmap import BevCostMapGenerator

NUSCENES_ROOT = Path("/home/andy/Dataset/Nuscenes/v1.0-mini")
FRAMES_DIR = Path("/tmp/nuscenes_laq_frames")
CAMERA = "CAM_FRONT"
GRID_SIZE = (32, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuscenes-root", type=Path, default=NUSCENES_ROOT)
    parser.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def costmap_cache_path(frame_path: Path) -> Path:
    return frame_path.with_suffix("").with_suffix(".costmap.npy")


def build_map_mask_by_log_token(nusc, dataroot: Path) -> dict:
    """log_token -> MapMask, lazily built once per distinct map file."""
    from nuscenes.utils.map_mask import MapMask

    mask_by_filename = {}
    filename_by_log_token = {}
    for map_record in nusc.map:
        for log_token in map_record["log_tokens"]:
            filename_by_log_token[log_token] = map_record["filename"]

    masks = {}
    for log_token, filename in filename_by_log_token.items():
        if filename not in mask_by_filename:
            mask_by_filename[filename] = MapMask(str(dataroot / filename))
        masks[log_token] = mask_by_filename[filename]
    return masks


def chain_cam_front_frames(nusc):
    """Reproduces prepare_nuscenes_laq.py's exact per-scene CAM_FRONT chain.

    sample_data records don't carry a "channel" field directly; resolve it
    via calibrated_sensor -> sensor, same as prepare_nuscenes_laq.py and
    laq_model/eval_utils.py do.
    """
    cam_tokens = {cs["token"] for cs in nusc.calibrated_sensor
                  if nusc.get("sensor", cs["sensor_token"])["channel"] == CAMERA}
    cam_frames = [sd for sd in nusc.sample_data if sd["calibrated_sensor_token"] in cam_tokens]

    by_token = {sd["token"]: sd for sd in cam_frames}
    roots = [sd for sd in cam_frames if not sd["prev"]]

    chains = []
    for root in roots:
        chain = []
        curr = root
        while curr:
            chain.append(curr)
            curr = by_token.get(curr["next"]) if curr["next"] else None
        chains.append(chain)
    return chains


def main() -> None:
    args = parse_args()
    assert args.frames_dir.exists(), f"Run data/prepare_nuscenes_laq.py first: {args.frames_dir} not found"

    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version="v1.0-mini", dataroot=str(args.nuscenes_root), verbose=False)

    masks_by_log_token = build_map_mask_by_log_token(nusc, args.nuscenes_root)
    detector = YoloDetector()
    generator = BevCostMapGenerator(grid_size=GRID_SIZE)

    chains = chain_cam_front_frames(nusc)
    print(f"{len(chains)} scenes")

    written, skipped = 0, 0
    for scene_idx, chain in enumerate(chains):
        scene_dir = args.frames_dir / f"scene_{scene_idx:03d}"
        if not scene_dir.exists():
            print(f"  warning: {scene_dir} not found (did prepare_nuscenes_laq.py produce a different scene order?), skipping")
            continue

        scene_record = nusc.scene[scene_idx]
        map_mask = masks_by_log_token.get(scene_record["log_token"])
        generator.map_mask = map_mask

        for n, sample_data in enumerate(tqdm(chain, desc=f"scene_{scene_idx:03d}"), start=1):
            frame_path = scene_dir / f"frame{n:04d}.jpg"
            cache_path = costmap_cache_path(frame_path)
            if cache_path.exists() and not args.overwrite:
                skipped += 1
                continue
            if not frame_path.exists():
                continue

            calibrated_sensor = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
            ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])

            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            detections = detector.detect(frame)
            grid = generator.generate(
                detections,
                camera_intrinsic=calibrated_sensor["camera_intrinsic"],
                cam_to_ego_rotation=Quaternion(calibrated_sensor["rotation"]),
                cam_to_ego_translation=calibrated_sensor["translation"],
                ego_to_global_rotation=Quaternion(ego_pose["rotation"]) if map_mask is not None else None,
                ego_to_global_translation=ego_pose["translation"] if map_mask is not None else None,
            )

            np.save(cache_path, grid)
            written += 1

    print(f"\nWrote {written} cost-map caches, skipped {skipped} already-cached frames.")


if __name__ == "__main__":
    main()

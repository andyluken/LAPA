"""
Prepares nuScenes mini dataset for LAQ training.

Creates a folder structure compatible with ImageVideoDataset:
  OUTPUT_DIR/
    scene_000/
      frame0001.jpg -> .../sweeps/CAM_FRONT/xxx.jpg  (symlink)
      frame0002.jpg -> ...
    scene_001/
      ...

Uses symlinks so no data is copied. Run once before training.
"""

import json
import os
from pathlib import Path

NUSCENES_ROOT = Path("/home/andy/Dataset/Nuscenes/v1.0-mini")
ANNOT_DIR = NUSCENES_ROOT / "v1.0-mini"
OUTPUT_DIR = Path("/tmp/nuscenes_laq_frames")
CAMERA = "CAM_FRONT"


def main():
    with open(ANNOT_DIR / "sensor.json") as f:
        sensors = json.load(f)
    cam_token = next(s["token"] for s in sensors if s["channel"] == CAMERA)

    with open(ANNOT_DIR / "calibrated_sensor.json") as f:
        cal_sensors = json.load(f)
    cal_tokens = {cs["token"] for cs in cal_sensors if cs["sensor_token"] == cam_token}

    with open(ANNOT_DIR / "sample_data.json") as f:
        sample_data = json.load(f)
    cam_frames = [sd for sd in sample_data if sd["calibrated_sensor_token"] in cal_tokens]

    by_token = {sd["token"]: sd for sd in cam_frames}
    # roots = first frame in each scene's CAM_FRONT stream (no predecessor)
    roots = [sd for sd in cam_frames if not sd["prev"]]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    for i, root in enumerate(roots):
        scene_dir = OUTPUT_DIR / f"scene_{i:03d}"
        scene_dir.mkdir(exist_ok=True)

        n = 1
        curr = root
        while curr:
            dst = scene_dir / f"frame{n:04d}.jpg"
            if not dst.exists():
                os.symlink(NUSCENES_ROOT / curr["filename"], dst)
            n += 1
            curr = by_token.get(curr["next"]) if curr["next"] else None

        frames_in_scene = n - 1
        total_frames += frames_in_scene
        print(f"  scene_{i:03d}: {frames_in_scene} frames")

    print(f"\n{len(roots)} scenes, {total_frames} total frames, "
          f"~{total_frames - len(roots)} consecutive pairs available")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

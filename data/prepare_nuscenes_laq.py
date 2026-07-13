"""Prepares nuScenes dataset for LAQ-AD training.

Exports only KEYFRAME images (annotated samples at ~2 Hz, not 12 Hz sweeps).
This matches NuScenesLAQDataset._build_index which iterates sample tokens and
maps sample_index_i → frame_{i+1:04d}.jpg within each scene directory.

Directory structure:
  OUTPUT_DIR/
    <scene_token>/
      frame_0001.jpg → .../samples/CAM_FRONT/<keyframe>.jpg  (symlink)
      frame_0002.jpg → ...
      ...

Uses symlinks so no data is copied. Run once before training.

Usage:
    # mini (default)
    python data/prepare_nuscenes_laq.py

    # trainval
    python data/prepare_nuscenes_laq.py \\
        --dataroot /media/andy/Samsung_T7/nuScenes/Trainval \\
        --version v1.0-trainval \\
        --output /home/andy/Dataset/nuscenes_laq_frames
"""

import argparse
import os
from pathlib import Path

DEFAULT_DATAROOT = Path("/home/andy/Dataset/Nuscenes/v1.0-mini")
DEFAULT_VERSION = "v1.0-mini"
DEFAULT_OUTPUT = Path("/home/andy/Dataset/nuscenes_laq_frames_improved")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataroot", type=Path, default=DEFAULT_DATAROOT,
                   help="nuScenes dataset root (contains the version subdir)")
    p.add_argument("--version", default=DEFAULT_VERSION,
                   help="nuScenes version string, e.g. v1.0-mini or v1.0-trainval")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Output directory for frame symlinks")
    p.add_argument("--overwrite", action="store_true",
                   help="Recreate symlinks even if destination already exists")
    return p.parse_args()


def main():
    args = parse_args()

    # Use nuScenes API to iterate scenes and samples correctly
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)

    print(f"nuScenes root : {args.dataroot}")
    print(f"Version       : {args.version}  ({len(nusc.scene)} scenes, {len(nusc.sample)} keyframes)")
    print(f"Output        : {args.output}")

    args.output.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    for scene in nusc.scene:
        scene_dir = args.output / scene["token"]
        scene_dir.mkdir(exist_ok=True)

        # Walk the sample (keyframe) chain for this scene
        sample = nusc.get("sample", scene["first_sample_token"])
        i = 1
        while True:
            cam_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            dst = scene_dir / f"frame_{i:04d}.jpg"
            if args.overwrite and dst.exists():
                dst.unlink()
            if not dst.exists():
                os.symlink(Path(args.dataroot) / cam_data["filename"], dst)
            i += 1
            if sample["next"] == "":
                break
            sample = nusc.get("sample", sample["next"])

        frames_in_scene = i - 1
        total_frames += frames_in_scene

    print(f"\n{len(nusc.scene)} scenes, {total_frames} keyframes exported")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

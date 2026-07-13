"""Interactive CLI demo: YOLO detection + accumulated saliency heatmap overlay.

Refactor of the original Heatmap_detection/heatmap_people_crowd.py prototype,
built on the reusable heatmap_lapa.detector / heatmap_lapa.saliency modules
instead of inline duplicated logic, with configurable video/weights/classes
instead of hardcoded paths, and with the COCO class-id bug fixed (the
original mislabeled class 5/6 as truck/bus; correct order is 5=bus, 7=truck).

Usage:
    python -m heatmap_lapa.cli_demo --video heatmap_lapa/demo_videos/traffic.mp4
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from heatmap_lapa.detector import DEFAULT_CLASSES, DEFAULT_WEIGHTS, YoloDetector
from heatmap_lapa.saliency import SaliencyHeatmapGenerator

BOX_COLORS = {
    "person": (0, 255, 0),
    "car": (255, 0, 0),
    "motorcycle": (0, 255, 255),
    "bicycle": (255, 255, 0),
    "bus": (0, 165, 255),
    "truck": (0, 100, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Path to input video file.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="YOLO checkpoint path.")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=list(DEFAULT_CLASSES),
        help="COCO class ids to track (default: person, bicycle, car, motorcycle, bus, truck).",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--radius", type=int, default=30, help="Heat splat radius in pixels.")
    parser.add_argument("--blur-sigma", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    detector = YoloDetector(weights=args.weights, classes=args.classes, confidence_threshold=args.confidence)
    saliency = SaliencyHeatmapGenerator(radius=args.radius, blur_sigma=args.blur_sigma)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    accumulated_heat: np.ndarray | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        if accumulated_heat is None:
            accumulated_heat = np.zeros((h, w), dtype=np.float32)

        detections = detector.detect(frame)
        accumulated_heat += saliency.raw_heat((h, w), detections)

        counts = Counter(det.cls_name for det in detections)
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = BOX_COLORS.get(det.cls_name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, det.center, 4, (0, 0, 255), -1)
            cv2.putText(frame, det.cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        heat_norm = (saliency.smooth_and_normalize(accumulated_heat) * 255).astype(np.uint8)
        heat_color = saliency.colorize(heat_norm)
        output = cv2.addWeighted(frame, 0.7, heat_color, 0.3, 0)

        for i, (cls_name, count) in enumerate(sorted(counts.items())):
            cv2.putText(
                output,
                f"{cls_name.capitalize()} Count: {count}",
                (20, 50 + i * 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Traffic Flow + Heatmap", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Thin wrapper around an Ultralytics YOLO model for traffic-scene detection.

Centralizes the COCO class-id -> name mapping used for heatmap generation.
The original prototype (Heatmap_detection/heatmap_people_crowd.py) mislabeled
this mapping (it treated class 5 as "truck" and class 6 as "bus", and class 4
as "lane markings"). The actual COCO80 order is:
  0 person, 1 bicycle, 2 car, 3 motorcycle, 4 airplane, 5 bus, 6 train, 7 truck
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from ultralytics import YOLO

# Correct COCO80 class ids for the categories relevant to traffic/driving scenes.
COCO_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

DEFAULT_CLASSES = tuple(COCO_CLASS_NAMES.keys())
DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "yolo11n.pt"


@dataclass(frozen=True)
class Detection:
    cls_id: int
    cls_name: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def bottom_center(self) -> tuple[int, int]:
        """Approximate ground-contact pixel, used for BEV ground-plane projection."""
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, y2


class YoloDetector:
    """Loads a YOLO checkpoint once and detects objects from a fixed class set."""

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        classes: Sequence[int] = DEFAULT_CLASSES,
        confidence_threshold: float = 0.25,
    ):
        self.model = YOLO(str(weights))
        self.classes = set(classes)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single BGR frame, returns boxes for tracked classes only."""
        results = self.model(frame, verbose=False)
        detections: list[Detection] = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if cls_id not in self.classes or confidence < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(
                    Detection(
                        cls_id=cls_id,
                        cls_name=COCO_CLASS_NAMES.get(cls_id, str(cls_id)),
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                    )
                )

        return detections

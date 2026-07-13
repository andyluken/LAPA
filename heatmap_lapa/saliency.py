"""Detection-based saliency heatmap generation.

Splats a Gaussian-ish blob of heat at the center of each detection box, blurs
and normalizes the result, and (optionally) downsamples it to the LAQ encoder's
patch grid resolution so it can condition patch embeddings.

Two use sites with different temporal semantics rely on this module:
  - The CLI demo accumulates raw heat across an entire video before blurring,
    to visualize where activity concentrates over time.
  - The offline nuScenes cache (data/prepare_nuscenes_heatmaps.py) calls
    `generate` per individual frame, since each LAQ training sample is an
    independent (frame_t, frame_t+offset) pair, not a continuous stream.
`raw_heat` and `smooth_and_normalize` are exposed separately so callers can
choose either accumulation strategy.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from heatmap_lapa.detector import Detection


class SaliencyHeatmapGenerator:
    def __init__(self, radius: int = 30, blur_sigma: float = 15.0):
        self.radius = radius
        self.blur_sigma = blur_sigma

    def raw_heat(self, frame_shape: tuple[int, int], detections: Sequence[Detection]) -> np.ndarray:
        """Unnormalized heat splat: one filled circle per detection center."""
        h, w = frame_shape
        heat = np.zeros((h, w), dtype=np.float32)
        for det in detections:
            cv2.circle(heat, det.center, self.radius, 1.0, -1)
        return heat

    def smooth_and_normalize(self, raw_heat: np.ndarray) -> np.ndarray:
        """Gaussian-blur and min-max normalize to [0, 1]. All-zero input stays zero."""
        blurred = cv2.GaussianBlur(raw_heat, (0, 0), self.blur_sigma)
        peak = blurred.max()
        if peak <= 0:
            return blurred.astype(np.float32)
        return (blurred / peak).astype(np.float32)

    def generate(self, frame: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
        """Full single-frame pipeline: raw splat -> blur -> normalize to [0, 1]."""
        h, w = frame.shape[:2]
        return self.smooth_and_normalize(self.raw_heat((h, w), detections))

    @staticmethod
    def downsample_to_patch_grid(heatmap: np.ndarray, patch_h: int, patch_w: int) -> np.ndarray:
        """Area-average downsample a full-resolution heatmap to the encoder's patch grid."""
        resized = cv2.resize(heatmap, (patch_w, patch_h), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32)

    @staticmethod
    def colorize(heatmap_uint8: np.ndarray) -> np.ndarray:
        """JET colormap for visualization. Expects a [0, 255] uint8 heatmap."""
        return cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

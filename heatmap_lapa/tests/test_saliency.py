import numpy as np
import pytest

from heatmap_lapa.detector import Detection
from heatmap_lapa.saliency import SaliencyHeatmapGenerator


def make_detection(cx: int, cy: int, size: int = 10) -> Detection:
    return Detection(
        cls_id=0,
        cls_name="person",
        bbox=(cx - size, cy - size, cx + size, cy + size),
        confidence=0.9,
    )


def test_generate_shape_and_range():
    gen = SaliencyHeatmapGenerator(radius=10, blur_sigma=5.0)
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    detections = [make_detection(60, 50)]

    heatmap = gen.generate(frame, detections)

    assert heatmap.shape == (100, 120)
    assert heatmap.dtype == np.float32
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_no_detections_yields_zero_heatmap():
    gen = SaliencyHeatmapGenerator()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    heatmap = gen.generate(frame, detections=[])

    assert np.allclose(heatmap, 0.0)


def test_heat_concentrated_near_detection():
    gen = SaliencyHeatmapGenerator(radius=8, blur_sigma=4.0)
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    detections = [make_detection(100, 20)]

    heatmap = gen.generate(frame, detections)

    near_box = heatmap[15:25, 95:105].mean()
    far_from_box = heatmap[100:110, 5:15].mean()

    assert near_box > far_from_box
    assert near_box > 0.5


def test_downsample_to_patch_grid_shape():
    gen = SaliencyHeatmapGenerator()
    heatmap = np.random.rand(256, 256).astype(np.float32)

    patch_grid = gen.downsample_to_patch_grid(heatmap, patch_h=8, patch_w=8)

    assert patch_grid.shape == (8, 8)
    assert patch_grid.dtype == np.float32

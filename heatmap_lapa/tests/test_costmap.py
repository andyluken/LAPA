import numpy as np
import pytest
from pyquaternion import Quaternion

from heatmap_lapa.costmap import BevCostMapGenerator, ground_plane_intersection
from heatmap_lapa.detector import Detection

# A simple synthetic forward-facing camera. Camera convention is x-right,
# y-down, z-forward (the optical-axis direction); the synthetic ego frame
# here is x-forward, y-left, z-up (matching nuScenes' real ego convention),
# with the camera mounted at height 1.5m, pointed straight ahead with no tilt.
_CAM_TO_EGO_ROTATION_MATRIX = np.array([
    [0.0, 0.0, 1.0],   # ego-x (forward) <- camera-z (forward)
    [-1.0, 0.0, 0.0],  # ego-y (left)    <- -camera-x (right)
    [0.0, -1.0, 0.0],  # ego-z (up)      <- -camera-y (down)
])
ROTATION = Quaternion(matrix=_CAM_TO_EGO_ROTATION_MATRIX)
TRANSLATION = [0.0, 0.0, 1.5]
SIMPLE_INTRINSIC = [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]


def make_detection(bbox, cls_name="person") -> Detection:
    return Detection(cls_id=0, cls_name=cls_name, bbox=bbox, confidence=0.9)


def test_ground_plane_intersection_above_horizon_returns_none():
    # A pixel at (or above) the principal point looks at/above the horizon.
    result = ground_plane_intersection((50, 50), SIMPLE_INTRINSIC, ROTATION, TRANSLATION)
    assert result is None

    result_above = ground_plane_intersection((50, 10), SIMPLE_INTRINSIC, ROTATION, TRANSLATION)
    assert result_above is None


def test_ground_plane_intersection_below_horizon_hits_ground_ahead():
    result = ground_plane_intersection((50, 90), SIMPLE_INTRINSIC, ROTATION, TRANSLATION)
    assert result is not None
    x, y = result
    assert x > 0  # in front of the camera
    assert abs(y) < 1e-6  # centered pixel -> no lateral offset


def test_closer_to_bottom_of_image_is_closer_to_camera():
    far = ground_plane_intersection((50, 60), SIMPLE_INTRINSIC, ROTATION, TRANSLATION)
    near = ground_plane_intersection((50, 95), SIMPLE_INTRINSIC, ROTATION, TRANSLATION)

    assert far is not None and near is not None
    assert near[0] < far[0]  # pixel nearer the bottom edge -> closer on the ground


def test_bev_generator_shape_and_range():
    gen = BevCostMapGenerator(grid_size=(16, 16), forward_range_m=20.0, lateral_range_m=10.0)

    detections = [make_detection((40, 80, 60, 98), cls_name="person")]
    grid = gen.generate(detections, SIMPLE_INTRINSIC, ROTATION, TRANSLATION)

    assert grid.shape == (16, 16)
    assert grid.dtype == np.float32
    assert grid.min() >= 0.0
    assert grid.max() <= 1.0 + 1e-6
    assert grid.max() > 0.0  # the detection should have produced some heat


def test_bev_generator_no_detections_yields_zero_grid():
    gen = BevCostMapGenerator(grid_size=(8, 8))
    grid = gen.generate([], SIMPLE_INTRINSIC, ROTATION, TRANSLATION)
    assert np.allclose(grid, 0.0)


def test_pedestrian_outweighs_vehicle_in_the_same_frame():
    # Two non-overlapping detections in one call: after normalization (which
    # divides by the global max), the higher-risk class should retain a
    # higher peak than the lower-risk class, reflecting RISK_WEIGHTS.
    gen = BevCostMapGenerator(grid_size=(16, 16), forward_range_m=20.0, lateral_range_m=10.0, blur_sigma=0.5)

    person_box = (20, 80, 30, 98)
    car_box = (70, 80, 80, 98)
    grid = gen.generate(
        [make_detection(person_box, cls_name="person"), make_detection(car_box, cls_name="car")],
        SIMPLE_INTRINSIC, ROTATION, TRANSLATION,
    )

    person_xy = ground_plane_intersection(make_detection(person_box).bottom_center, SIMPLE_INTRINSIC, ROTATION, TRANSLATION)
    car_xy = ground_plane_intersection(make_detection(car_box).bottom_center, SIMPLE_INTRINSIC, ROTATION, TRANSLATION)
    person_row, person_col = gen._to_grid_indices(*person_xy)
    car_row, car_col = gen._to_grid_indices(*car_xy)

    person_peak = grid[int(round(person_row)), int(round(person_col))]
    car_peak = grid[int(round(car_row)), int(round(car_col))]

    assert person_peak > car_peak

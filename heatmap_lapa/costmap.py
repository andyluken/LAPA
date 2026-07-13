"""BEV (bird's-eye-view) collision-risk cost-map generation (Phase 2).

Projects detections onto the ego-frame ground plane via a camera-ray /
z=0-plane intersection (the IPM-style geometry is unavoidable regardless of
tooling), weights them by per-class collision risk, optionally boosts risk
for objects on the drivable area (queried from nuScenes' raster
`MapMask`), and splats the result into a small ego-centric grid — used by
laq_model/costmap_loss.py to pull together latent actions of scenes with a
similar risk profile.

Uses `nuscenes.utils.map_mask.MapMask`, which works directly off the raster
PNGs already present in the nuScenes-mini download, rather than
`nuscenes.map_expansion.map_api.NuScenesMap` (the vector polygon API),
which needs the separate map-expansion pack that is not present here.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np
from pyquaternion import Quaternion

from heatmap_lapa.detector import Detection

# Per-class collision-risk weight: vulnerable road users outweigh vehicles,
# per the design doc's "high intensity = high collision risk" framing.
RISK_WEIGHTS = {
    "person": 3.0,
    "bicycle": 2.5,
    "motorcycle": 2.0,
    "car": 1.0,
    "bus": 1.0,
    "truck": 1.0,
}
DEFAULT_RISK_WEIGHT = 1.0


def ground_plane_intersection(
    pixel_xy: tuple[float, float],
    camera_intrinsic: Sequence[Sequence[float]],
    cam_to_ego_rotation: Quaternion,
    cam_to_ego_translation: Sequence[float],
) -> Optional[np.ndarray]:
    """Intersects the camera ray through `pixel_xy` with the ego-frame z=0 plane.

    Returns the (x, y) ego-frame ground coordinates, or None if the ray
    points at or above the horizon (no ground intersection in front of the
    vehicle).
    """
    intrinsic = np.asarray(camera_intrinsic, dtype=np.float64)
    u, v = pixel_xy
    ray_cam = np.linalg.inv(intrinsic) @ np.array([u, v, 1.0])
    ray_cam = ray_cam / np.linalg.norm(ray_cam)

    ray_ego = cam_to_ego_rotation.rotation_matrix @ ray_cam
    origin_ego = np.asarray(cam_to_ego_translation, dtype=np.float64)

    if ray_ego[2] >= -1e-6:  # parallel to or pointing above the ground plane
        return None

    scale = -origin_ego[2] / ray_ego[2]
    if scale <= 0:
        return None

    ground_point = origin_ego + scale * ray_ego
    return ground_point[:2]


class BevCostMapGenerator:
    def __init__(
        self,
        map_mask=None,  # nuscenes.utils.map_mask.MapMask, or None to skip drivable-area weighting
        grid_size: tuple[int, int] = (32, 32),
        forward_range_m: float = 40.0,
        lateral_range_m: float = 20.0,
        risk_weights: dict = RISK_WEIGHTS,
        drivable_area_bonus: float = 1.5,
        splat_radius_px: int = 1,
        blur_sigma: float = 1.0,
    ):
        self.map_mask = map_mask
        self.grid_size = grid_size
        self.forward_range_m = forward_range_m
        self.lateral_range_m = lateral_range_m
        self.risk_weights = risk_weights
        self.drivable_area_bonus = drivable_area_bonus
        self.splat_radius_px = splat_radius_px
        self.blur_sigma = blur_sigma

    def _to_grid_indices(self, x: float, y: float) -> tuple[float, float]:
        """Ego-frame (x=forward, y=left) meters -> (row, col) in the BEV grid.

        row=grid_h-1 is at the ego vehicle (x=0), row=0 is forward_range_m
        ahead. col=0 is the leftmost extent (y=+lateral_range_m), col=grid_w-1
        is the rightmost extent (y=-lateral_range_m).
        """
        grid_h, grid_w = self.grid_size
        row = (1.0 - np.clip(x / self.forward_range_m, 0.0, 1.0)) * (grid_h - 1)
        col = (1.0 - np.clip((y + self.lateral_range_m) / (2 * self.lateral_range_m), 0.0, 1.0)) * (grid_w - 1)
        return row, col

    def _risk_weight(self, detection: Detection, ego_to_global_rotation: Optional[Quaternion],
                      ego_to_global_translation: Optional[Sequence[float]], ground_xy: np.ndarray) -> float:
        risk = self.risk_weights.get(detection.cls_name, DEFAULT_RISK_WEIGHT)

        if self.map_mask is not None and ego_to_global_rotation is not None:
            point_ego = np.array([ground_xy[0], ground_xy[1], 0.0])
            point_global = ego_to_global_rotation.rotation_matrix @ point_ego + np.asarray(ego_to_global_translation)
            if self.map_mask.is_on_mask([point_global[0]], [point_global[1]])[0]:
                risk *= self.drivable_area_bonus

        return risk

    def generate(
        self,
        detections: Sequence[Detection],
        camera_intrinsic: Sequence[Sequence[float]],
        cam_to_ego_rotation: Quaternion,
        cam_to_ego_translation: Sequence[float],
        ego_to_global_rotation: Optional[Quaternion] = None,
        ego_to_global_translation: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Builds one (grid_h, grid_w) BEV cost map, normalized to [0, 1]."""
        grid = np.zeros(self.grid_size, dtype=np.float32)

        for det in detections:
            ground_xy = ground_plane_intersection(
                det.bottom_center, camera_intrinsic, cam_to_ego_rotation, cam_to_ego_translation
            )
            if ground_xy is None:
                continue

            x, y = ground_xy
            if not (0 <= x <= self.forward_range_m and -self.lateral_range_m <= y <= self.lateral_range_m):
                continue

            risk = self._risk_weight(det, ego_to_global_rotation, ego_to_global_translation, ground_xy)
            row, col = self._to_grid_indices(x, y)
            cv2.circle(grid, (int(round(col)), int(round(row))), self.splat_radius_px, float(risk), -1)

        blurred = cv2.GaussianBlur(grid, (0, 0), self.blur_sigma)
        peak = blurred.max()
        if peak <= 0:
            return blurred.astype(np.float32)
        return (blurred / peak).astype(np.float32)

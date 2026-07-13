"""nuScenes data pipeline for LAQ-AD.

Key features over the upstream laq/laq_model/data.py:
  - Uses nuScenes API directly for robustness (not a flat frame export structure)
  - Maneuver-balanced sampler: oversamples rare turns so every maneuver has
    equal representation in each batch, despite the 65%/20%/7%/7% distribution
  - CAN bus loading: interpolates yaw_rate and speed to frame timestamps
    (requires nuScenes CAN bus download; degrades gracefully if unavailable)
  - Egomotion heatmap loading: reads precomputed (3, H_p, W_p) .egomotion.npy
    caches (magnitude channel used as patch gate)

Dataset index layout:
    Each sample = (scene_token, cam_sample_token, next_cam_sample_token, offset)
    Maneuver labels are computed from ego pose delta at construction time.

Usage:
    dataset = NuScenesLAQDataset(
        nusc=NuScenes('v1.0-trainval', dataroot='/path/nuscenes'),
        frames_dir=Path('/path/nuscenes_laq_frames'),
        offset=3,
        use_heatmap=True,
        use_can_bus=True,
    )
    sampler = ManeuverBalancedSampler(dataset)
    loader = DataLoader(dataset, batch_size=16, sampler=sampler)
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    NUSCENES_AVAILABLE = True
except ImportError:
    NUSCENES_AVAILABLE = False

try:
    from pyquaternion import Quaternion
    QUATERNION_AVAILABLE = True
except ImportError:
    QUATERNION_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

IMAGE_TRANSFORM = T.Compose([
    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),
])

PATCH_GRID = (8, 8)   # for image_size=256, patch_size=32
CAMERA = "CAM_FRONT"

# Maneuver thresholds (derived from ego pose delta over `offset` frames)
YAW_THRESHOLD = 0.03   # rad — below this → straight or stationary
SPEED_THRESHOLD = 0.3  # m/s — below this → stationary

MANEUVERS = ["straight", "turn_left", "turn_right", "stationary"]


# ── Maneuver labeling ─────────────────────────────────────────────────────────

def maneuver_label(pose_t: dict, pose_t_next: dict, offset_secs: float = 0.5) -> str:
    """Compute maneuver from two ego pose dicts (translation + rotation).

    pose = {'translation': [x,y,z], 'rotation': [w,x,y,z]}
    """
    if not QUATERNION_AVAILABLE:
        return "straight"

    q_t = Quaternion(pose_t["rotation"])
    q_tn = Quaternion(pose_t_next["rotation"])

    tx, ty = pose_t["translation"][:2]
    nx, ny = pose_t_next["translation"][:2]
    dist = ((nx - tx) ** 2 + (ny - ty) ** 2) ** 0.5
    speed_approx = dist / max(offset_secs, 0.1)

    if speed_approx < SPEED_THRESHOLD:
        return "stationary"

    # Yaw delta: angle of rotation about z-axis
    q_delta = q_tn * q_t.inverse
    yaw_delta = q_delta.yaw_pitch_roll[0]   # radians, per-offset interval

    if abs(yaw_delta) < YAW_THRESHOLD:
        return "straight"
    return "turn_left" if yaw_delta > 0 else "turn_right"


# ── CAN bus interpolation ─────────────────────────────────────────────────────

class CANBusCache:
    """Loads and interpolates nuScenes CAN bus signals for a dataset split.

    Falls back to None gracefully when CAN bus data is unavailable.
    """

    def __init__(self, nusc: "NuScenes", dataroot: str | Path):
        self._cache: dict[str, list[dict]] = {}
        self._available = False
        if not NUSCENES_AVAILABLE:
            return
        dataroot = Path(dataroot)
        # Some distributions nest the CAN bus data as <dataroot>/can_bus/can_bus/*.json
        # instead of the expected <dataroot>/can_bus/*.json.  Try the nested path first.
        can_bus_root = dataroot / "can_bus" if (dataroot / "can_bus" / "can_bus").is_dir() else dataroot
        try:
            self._can = NuScenesCanBus(dataroot=str(can_bus_root))
            self._available = True
        except Exception:
            pass

    def get(self, scene_name: str, utime: int) -> Optional[tuple[float, float]]:
        """Return (yaw_rate rad/s, speed m/s) at `utime` (µs) or None."""
        if not self._available:
            return None
        if scene_name not in self._cache:
            try:
                msgs = self._can.get_messages(scene_name, "vehicle_monitor")
                self._cache[scene_name] = msgs
            except Exception:
                self._cache[scene_name] = []
        msgs = self._cache[scene_name]
        if not msgs:
            return None
        # find surrounding messages and interpolate
        utimes = [m["utime"] for m in msgs]
        idx = np.searchsorted(utimes, utime)
        if idx == 0:
            m = msgs[0]
        elif idx >= len(msgs):
            m = msgs[-1]
        else:
            m0, m1 = msgs[idx - 1], msgs[idx]
            t0, t1 = m0["utime"], m1["utime"]
            alpha = (utime - t0) / max(t1 - t0, 1)
            yr = m0["yaw_rate"] * (1 - alpha) + m1["yaw_rate"] * alpha
            sp = m0["vehicle_speed"] * (1 - alpha) + m1["vehicle_speed"] * alpha
            return float(yr), float(sp)
        return float(m["yaw_rate"]), float(m["vehicle_speed"])


# ── Main dataset ──────────────────────────────────────────────────────────────

class NuScenesLAQDataset(Dataset):
    """nuScenes frame-pair dataset for LAQ-AD training.

    Each sample returns:
        video:    (3, 2, 256, 256)  — pair [frame_t, frame_{t+offset}]
        heatmap:  (2, 3, 8, 8) or None  — egomotion heatmap for each frame
        can_bus:  (2,) float32 [yaw_rate_norm, speed_norm] or None
        maneuver: int index in MANEUVERS list (for evaluation; not used in loss)
    """

    def __init__(
        self,
        nusc: "NuScenes",
        frames_dir: str | Path,
        offset: int = 3,
        use_heatmap: bool = True,
        use_can_bus: bool = True,
        patch_grid: tuple[int, int] = PATCH_GRID,
        camera: str = CAMERA,
    ):
        self.nusc = nusc
        self.frames_dir = Path(frames_dir)
        self.offset = offset
        self.use_heatmap = use_heatmap
        self.patch_grid = patch_grid
        self.camera = camera

        self.can_cache = CANBusCache(nusc, nusc.dataroot) if use_can_bus else None

        self.samples: list[dict] = []
        self.maneuver_indices: list[int] = []
        self._build_index()

    def _build_index(self):
        """Walk all scenes and collect (frame_t, frame_{t+offset}) pairs.

        Also builds self._token_to_path: sample_token → Path in frames_dir,
        so egomotion caches (frame_NNNN.egomotion.npy siblings) can be found.
        """
        from collections import Counter
        counts: Counter = Counter()
        self._token_to_path: dict[str, Path] = {}

        for scene in self.nusc.scene:
            tokens = self._scene_cam_tokens(scene)
            scene_name = scene["name"]
            scene_dir = self.frames_dir / scene["token"]

            # Map each sample token to its exported frame path (1-indexed, zero-padded)
            for i, tok in enumerate(tokens):
                self._token_to_path[tok] = scene_dir / f"frame_{i + 1:04d}.jpg"

            for i in range(len(tokens) - self.offset):
                tok_t = tokens[i]
                tok_tn = tokens[i + self.offset]
                pose_t = self._ego_pose(tok_t)
                pose_tn = self._ego_pose(tok_tn)
                dt = (self.nusc.get("sample", tok_tn)["timestamp"] -
                      self.nusc.get("sample", tok_t)["timestamp"]) / 1e6
                maneuver = maneuver_label(pose_t, pose_tn, offset_secs=dt)
                m_idx = MANEUVERS.index(maneuver)
                self.samples.append({
                    "scene_name": scene_name,
                    "tok_t": tok_t,
                    "tok_tn": tok_tn,
                    "utime_t": self.nusc.get("sample", tok_t)["timestamp"],
                })
                self.maneuver_indices.append(m_idx)
                counts[maneuver] += 1

        self.class_counts = {m: counts[m] for m in MANEUVERS}

    def _scene_cam_tokens(self, scene: dict) -> list[str]:
        """Return ordered list of sample tokens for `self.camera` in `scene`."""
        tokens = []
        sample = self.nusc.get("sample", scene["first_sample_token"])
        while True:
            tokens.append(sample["token"])
            if sample["next"] == "":
                break
            sample = self.nusc.get("sample", sample["next"])
        return tokens

    def _ego_pose(self, sample_token: str) -> dict:
        """Return ego pose dict {'translation': [...], 'rotation': [...]}."""
        sample = self.nusc.get("sample", sample_token)
        cam_data = self.nusc.get("sample_data", sample["data"][self.camera])
        ep = self.nusc.get("ego_pose", cam_data["ego_pose_token"])
        return {"translation": ep["translation"], "rotation": ep["rotation"]}

    def _frame_path(self, sample_token: str) -> Path:
        """Resolve path to the exported frame JPEG in frames_dir.

        Uses the pre-built _token_to_path index from _build_index().
        Falls back to native nuScenes path if the export doesn't exist
        (e.g. when running without prepare_nuscenes_laq.py first).
        """
        exported = self._token_to_path.get(sample_token)
        if exported is not None and exported.exists():
            return exported
        # Fallback: native nuScenes image path (egomotion cache won't be found
        # alongside it, so _load_egomotion will return zeros)
        sample = self.nusc.get("sample", sample_token)
        cam_data = self.nusc.get("sample_data", sample["data"][self.camera])
        return Path(self.nusc.dataroot) / cam_data["filename"]

    def _load_egomotion(self, frame_path: Path) -> np.ndarray:
        """Load (3, H_p, W_p) egomotion heatmap, zeros if missing."""
        cache = frame_path.with_suffix("").with_suffix(".egomotion.npy")
        if not cache.exists():
            return np.zeros((3,) + self.patch_grid, dtype=np.float32)
        arr = np.load(cache)
        if arr.ndim == 2:
            # backward-compat: old (H_p, W_p) magnitude-only caches
            arr = np.stack([arr, np.zeros_like(arr), np.zeros_like(arr)], axis=0)
        return arr.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        entry = self.samples[idx]
        tok_t, tok_tn = entry["tok_t"], entry["tok_tn"]

        path_t = self._frame_path(tok_t)
        path_tn = self._frame_path(tok_tn)

        # --- Video tensor (3, 2, 256, 256) ----------------------------------
        img_t = IMAGE_TRANSFORM(Image.open(path_t))    # (3, 256, 256)
        img_tn = IMAGE_TRANSFORM(Image.open(path_tn))
        video = torch.stack([img_t, img_tn], dim=1)    # (3, 2, 256, 256)

        # --- Heatmap (2, 3, H_p, W_p) ----------------------------------------
        heatmap = None
        if self.use_heatmap:
            h_t = self._load_egomotion(path_t)
            h_tn = self._load_egomotion(path_tn)
            heatmap = torch.from_numpy(np.stack([h_t, h_tn], axis=0))  # (2, 3, H_p, W_p)

        # --- CAN bus (2,) float32 --------------------------------------------
        # Raw nuScenes vehicle_monitor units: yaw_rate in deg/s, vehicle_speed in km/h.
        # Normalize to [-1, 1] using rad/s and m/s scales matching CANBusHead.
        can_bus = None
        if self.can_cache is not None:
            result = self.can_cache.get(entry["scene_name"], entry["utime_t"])
            if result is not None:
                yr_degs, sp_kmh = result
                yr_rads = yr_degs * (math.pi / 180.0)   # deg/s → rad/s
                sp_ms   = sp_kmh / 3.6                   # km/h  → m/s
                yr_n = float(np.clip(yr_rads / CANBusHead_YAW_SCALE, -1.0, 1.0))
                sp_n = float(np.clip(sp_ms   / CANBusHead_SPEED_SCALE, 0.0, 1.0)) * 2.0 - 1.0
                can_bus = torch.tensor([yr_n, sp_n], dtype=torch.float32)

        # --- Maneuver index (for eval) ----------------------------------------
        maneuver_idx = self.maneuver_indices[idx]

        return video, heatmap, can_bus, maneuver_idx


# Reference normalization constants (must match CANBusHead)
CANBusHead_YAW_SCALE = 1.5
CANBusHead_SPEED_SCALE = 15.0


# ── Balanced sampler ──────────────────────────────────────────────────────────

def maneuver_balanced_sampler(dataset: NuScenesLAQDataset) -> WeightedRandomSampler:
    """WeightedRandomSampler that equalizes maneuver class frequency.

    Without this, the natural nuScenes-trainval distribution (~65% straight,
    ~20% stationary, ~7% each turn) means the codebook allocates almost all
    capacity to straight and stationary driving.
    """
    counts = dataset.class_counts
    total = sum(counts.values())
    n_classes = sum(1 for v in counts.values() if v > 0)

    # weight per sample = 1 / class_count (so rarer classes are sampled more)
    class_weights = {
        m: (total / (n_classes * max(counts[m], 1)))
        for m in MANEUVERS
    }
    sample_weights = [class_weights[MANEUVERS[m_idx]] for m_idx in dataset.maneuver_indices]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """Collate that handles optional None heatmap/can_bus tensors.

    can_bus: returned as (b, 2) only if ALL samples in the batch have CAN bus
    data; otherwise None. This avoids a shape mismatch between can_pred (b, 2)
    and a partial-batch can_bus tensor.
    """
    videos, heatmaps, can_buses, maneuvers = zip(*batch)

    videos = torch.stack(videos)                                     # (b, 3, 2, 256, 256)
    heatmaps = torch.stack(heatmaps) if heatmaps[0] is not None else None

    if all(c is not None for c in can_buses):
        can_buses_out = torch.stack(can_buses)                       # (b, 2)
    else:
        can_buses_out = None

    maneuvers = torch.tensor(maneuvers, dtype=torch.long)
    return videos, heatmaps, can_buses_out, maneuvers

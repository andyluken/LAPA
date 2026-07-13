"""Shared nuScenes evaluation helpers: frame-pair loading + maneuver labeling.

Lifted out of eval_latent_clusters.py so eval_compare_checkpoints.py (and any
future nuScenes evaluation script) doesn't have to copy-paste the ego-pose
pairing and yaw/distance maneuver-classification logic.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from torchvision import transforms as T

MANEUVERS = ["straight", "turn_left", "turn_right", "stationary"]

# Preprocessing must match laq_model/data.py's ImageVideoDataset transform.
IMAGE_TRANSFORM = T.Compose([
    T.Lambda(lambda img: img.convert("RGB")),
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),
])


def yaw_delta(pose_t: dict, pose_t3: dict) -> float:
    """Signed yaw change (radians) from pose_t to pose_t3.

    nuScenes quaternion = [w, x, y, z]; scipy expects [x, y, z, w].
    """
    def to_rot(p):
        q = p["rotation"]
        return Rotation.from_quat([q[1], q[2], q[3], q[0]])

    rel = to_rot(pose_t3) * to_rot(pose_t).inv()
    return float(rel.as_rotvec()[2])  # z-component ≈ yaw for flat road


def maneuver_label(
    pose_t: dict,
    pose_t3: dict,
    yaw_thresh: float = 0.06,
    dist_thresh: float = 0.2,
) -> str:
    t, t3 = pose_t["translation"], pose_t3["translation"]
    dist = ((t3[0] - t[0]) ** 2 + (t3[1] - t[1]) ** 2) ** 0.5
    if dist < dist_thresh:
        return "stationary"
    yaw = yaw_delta(pose_t, pose_t3)
    if yaw > yaw_thresh:
        return "turn_left"
    if yaw < -yaw_thresh:
        return "turn_right"
    return "straight"


def load_nuscenes_pairs(
    nuscenes_root: Path,
    offset: int,
    camera: str = "CAM_FRONT",
) -> list[tuple[Path, Path, dict, dict]]:
    """Build (img_path_t, img_path_t3, ego_pose_t, ego_pose_t3) tuples for every
    consecutive `offset`-frame pair across all scenes' `camera` stream.
    """
    annot_dir = nuscenes_root / "v1.0-mini"

    with open(annot_dir / "sensor.json") as f:
        sensors = json.load(f)
    cam_token = next(s["token"] for s in sensors if s["channel"] == camera)

    with open(annot_dir / "calibrated_sensor.json") as f:
        cal_sensors = json.load(f)
    cal_tokens = {cs["token"] for cs in cal_sensors if cs["sensor_token"] == cam_token}

    with open(annot_dir / "sample_data.json") as f:
        sample_data = json.load(f)
    cam_frames = [sd for sd in sample_data if sd["calibrated_sensor_token"] in cal_tokens]

    with open(annot_dir / "ego_pose.json") as f:
        ego_poses = json.load(f)
    ego_by_token = {ep["token"]: ep for ep in ego_poses}

    by_token = {sd["token"]: sd for sd in cam_frames}
    roots = [sd for sd in cam_frames if not sd["prev"]]

    pairs = []
    for root in roots:
        chain = []
        curr = root
        while curr:
            chain.append(curr)
            curr = by_token.get(curr["next"]) if curr["next"] else None
        for i in range(len(chain) - offset):
            f_t = chain[i]
            f_t3 = chain[i + offset]
            pairs.append((
                nuscenes_root / f_t["filename"],
                nuscenes_root / f_t3["filename"],
                ego_by_token[f_t["ego_pose_token"]],
                ego_by_token[f_t3["ego_pose_token"]],
            ))
    return pairs


def _entropy(counts: Counter, total: int) -> float:
    return -sum((n / total) * math.log(n / total) for n in counts.values() if n > 0)


def normalized_mutual_information(labels_a: list, labels_b: list) -> float:
    """NMI(A, B) = I(A; B) / sqrt(H(A) * H(B)), in [0, 1].

    0 means the two labelings are independent; 1 means one is a deterministic
    function of the other. Used to score how well latent codebook indices
    align with ground-truth driving maneuvers, without depending on sklearn.
    """
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return 0.0

    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)

    h_a = _entropy(counts_a, n)
    h_b = _entropy(counts_b, n)
    if h_a == 0 or h_b == 0:
        return 0.0
    
    joint_counts = Counter(zip(labels_a, labels_b))

    mutual_info = 0.0
    for (a, b), n_ab in joint_counts.items():
        mutual_info += n_ab * math.log((n_ab * n) / (counts_a[a] * counts_b[b]))
    mutual_info /= n

    nmi = mutual_info / math.sqrt(h_a * h_b)
    return min(max(nmi, 0.0), 1.0)  # clamp to [0, 1] in case of floating-point error

def build_symlink_index(frames_dir: Path) -> dict[Path, Path]:
    """Maps each symlinked frame's real target path -> the symlink path itself.

    load_nuscenes_pairs() returns original nuScenes sweep paths, but cached
    per-frame arrays (*.heatmap.npy, *.costmap.npy) live next to the
    *symlinks* in frames_dir (written by data/prepare_nuscenes_heatmaps.py
    and data/prepare_nuscenes_costmaps.py). This index bridges the two.
    """
    index = {}
    for symlink_path in frames_dir.glob("scene_*/frame*.jpg"):
        if symlink_path.is_symlink():
            index[symlink_path.resolve()] = symlink_path
    return index


def load_cached_array(
    original_path: Path,
    symlink_index: dict[Path, Path],
    suffix: str,
    shape: tuple[int, int],
) -> np.ndarray:
    """Loads a cached `frameNNNN<suffix>` array for a frame given its original
    (non-symlinked) nuScenes path, falling back to zeros if uncached.
    """
    symlink_path = symlink_index.get(original_path.resolve())
    if symlink_path is None:
        return np.zeros(shape, dtype=np.float32)

    cache_path = symlink_path.with_suffix("").with_suffix(suffix)
    if not cache_path.exists():
        return np.zeros(shape, dtype=np.float32)

    return np.load(cache_path)


def mean_intra_cluster_costmap_similarity(codes: list[int], costmaps: list[np.ndarray]) -> float | None:
    """Mean cosine similarity between cost maps of pairs sharing a codebook index.

    Pooled across all clusters (codebook entries) with at least 2 members.
    This is the direct test of whether cost-map latent regularization
    (laq_model/costmap_loss.py) actually pulled similar-risk scenes into the
    same code: higher = scenes sharing a code also share a risk profile.
    Returns None if no cluster has 2+ members (metric undefined).
    """
    by_code = defaultdict(list)
    for code, costmap in zip(codes, costmaps):
        by_code[code].append(np.asarray(costmap, dtype=np.float64).flatten())

    pair_similarities = []
    for members in by_code.values():
        if len(members) < 2:
            continue
        matrix = np.stack(members)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = matrix / norms
        sim = normalized @ normalized.T
        i_upper, j_upper = np.triu_indices(len(members), k=1)
        pair_similarities.extend(sim[i_upper, j_upper].tolist())

    if not pair_similarities:
        return None
    return float(np.mean(pair_similarities))

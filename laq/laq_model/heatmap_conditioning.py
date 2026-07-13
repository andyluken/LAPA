"""Saliency-conditioning of LAQ encoder patch embeddings (Phase 1).

Implements the highest-priority recommendation from the heatmap-integration
design note: multiply patch embeddings by a per-patch saliency weight before
they enter the encoder transformer, so the latent-action codebook learns to
represent dynamic agents (pedestrians, vehicles) rather than static
background.

Uses a residual gate `1 + alpha * heatmap` rather than a raw multiplicative
weight `heatmap`, because frames with zero detections produce an all-zero
heatmap — multiplying directly by that would zero out the entire frame's
embedding instead of leaving it unweighted. `alpha=0` or `heatmap=None` is
an exact identity, which is what laq/tests/test_latent_action_quantization_
backward_compat.py verifies.

Three conditioning strategies are supported (see apply_patch_heatmap):
  1. Single-channel magnitude gate  (heatmap ndim=4)
  2. Three-channel gate: mag + signed fx/fy  (heatmap ndim=5, abs_dir=False)
  3. Three-channel gate: mag + |fx|/|fy|     (heatmap ndim=5, abs_dir=True)
  4. Additive direction embedding             (EgoMotionDirectionEmbedding)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


def exists(val) -> bool:
    return val is not None


def apply_patch_heatmap(
    patch_tokens: Tensor,
    heatmap: Optional[Tensor],
    alpha: float,
    alpha_dir: float = 0.25,
    abs_dir: bool = False,
) -> Tensor:
    """Gate patch embeddings by a per-patch ego-motion heatmap.

    Single-channel heatmap  (b, t, h, w):
        gate = 1 + alpha * mag
        Backward-compatible with detection-based and magnitude-only egomotion
        heatmaps.

    Three-channel heatmap  (b, t, 3, h, w)  — channels [mag, fx, fy]:
        abs_dir=False  (signed — UNSTABLE, caused codebook collapse):
            gate = 1 + alpha * mag + alpha_dir * (fx + fy)
            Gate can go below 1.0, suppressing patches. This destabilises
            NSVQ early training: suppressed right-turn patches collapse to
            the same codebook entry.

        abs_dir=True  (absolute — recommended):
            gate = 1 + alpha * mag + alpha_dir * (|fx| + |fy|)
            Gate ≥ 1 always. Loses signed left/right distinction but
            highlights patches with strong horizontal motion (turns) vs
            antisymmetric/zero motion (straight/stationary).

    For left/right distinction without suppression use EgoMotionDirectionEmbedding
    (additive, not multiplicative).

    Args:
        patch_tokens: (b, t, h, w, d) patch embeddings.
        heatmap: (b, t, h, w) or (b, t, 3, h, w), or None.
        alpha: strength of the magnitude gate. alpha=0 is a no-op.
        alpha_dir: strength of the direction channels (3-channel only).
        abs_dir: if True, use absolute value of direction channels.

    Returns:
        (b, t, h, w, d) gated patch embeddings.
    """
    if not exists(heatmap) or alpha == 0:
        return patch_tokens

    heatmap = heatmap.to(dtype=patch_tokens.dtype, device=patch_tokens.device)

    if heatmap.ndim == 5:  # (b, t, 3, h, w) — mag + direction channels
        mag = heatmap[:, :, 0]  # (b, t, h, w)
        fx  = heatmap[:, :, 1]  # (b, t, h, w)
        fy  = heatmap[:, :, 2]  # (b, t, h, w)

        if abs_dir:
            # Gate >= 1 always: no suppression, no collapse risk.
            dir_term = fx.abs() + fy.abs()
        else:
            # Signed: gate can drop to 0.5 (min), causing patch suppression.
            # Kept for reference — led to 115x codebook imbalance in practice.
            dir_term = fx + fy

        gate = 1.0 + alpha * mag + alpha_dir * dir_term
    else:                   # (b, t, h, w) — single magnitude channel
        gate = 1.0 + alpha * heatmap

    return patch_tokens * gate.unsqueeze(-1)


class EgoMotionDirectionEmbedding(nn.Module):
    """Additive per-patch direction embedding from ego-motion flow channels.

    Projects [fx, fy] per-patch direction features into token space and ADDS
    the result to patch tokens, rather than gating (multiplying) them.

    Advantages over the multiplicative gate:
    - Signed fx/fy are preserved: left turns (fx>0) and right turns (fx<0)
      produce opposite embeddings the encoder can distinguish.
    - Gate value is always 1.0 (pure additive): no patch suppression, no
      scale distortion, no NSVQ instability risk.
    - The projection is learned, so the model can weight fx vs fy optimally.

    Heatmap must be (b, t, 3, h, w); a 4D (magnitude-only) heatmap or None
    is a no-op (returns patch_tokens unchanged).
    """

    def __init__(self, dim: int):
        super().__init__()
        # bias=False: the patch embedding already has positional bias terms
        self.proj = nn.Linear(2, dim, bias=False)

    def forward(self, patch_tokens: Tensor, heatmap: Optional[Tensor]) -> Tensor:
        """
        Args:
            patch_tokens: (b, t, h, w, d)
            heatmap: (b, t, 3, h, w) or None

        Returns:
            (b, t, h, w, d) — tokens with direction embedding added.
        """
        if not exists(heatmap) or heatmap.ndim != 5:
            return patch_tokens

        fx = heatmap[:, :, 1]  # (b, t, h, w)
        fy = heatmap[:, :, 2]  # (b, t, h, w)

        # Stack into (b, t, h, w, 2) then project to (b, t, h, w, dim)
        dir_features = torch.stack([fx, fy], dim=-1).to(
            dtype=patch_tokens.dtype, device=patch_tokens.device
        )
        return patch_tokens + self.proj(dir_features)

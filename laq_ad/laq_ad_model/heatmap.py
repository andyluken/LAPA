"""Egomotion heatmap conditioning for LAQ-AD.

Only the magnitude gate is exposed here — the signed/abs-direction variants
were proven no better (or harmful) compared to magnitude-only in the nuScenes
smoke-test experiments (see laq/ experiment results in CLAUDE.md).

The gate is always >= 1.0, so patch activations are only boosted, never
suppressed. This means the heatmap has zero effect on reconstruction when
the ego is stationary (near-zero flow → near-zero heatmap → gate ≈ 1).
"""

import torch


def apply_patch_heatmap(
    patch_tokens: torch.Tensor,
    heatmap: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Scale patch token vectors by a per-patch magnitude gate.

    Works for both single-channel (4D) and 3-channel egomotion (5D) heatmaps:
      4D (b, t, H_p, W_p):       gate = 1 + alpha * heatmap
      5D (b, t, 3, H_p, W_p):    gate = 1 + alpha * heatmap[:, :, 0]  (magnitude ch)

    The direction channels (ch 1, 2) are deliberately ignored here — magnitude
    alone contains the left/right discriminability signal via spatial pattern.

    Args:
        patch_tokens: (b, t, H_p, W_p, d)
        heatmap:      (b, t, H_p, W_p) or (b, t, 3, H_p, W_p)
        alpha:        gate strength (0 = off)
    Returns:
        patch_tokens scaled by gate, same shape as input
    """
    if alpha == 0.0:
        return patch_tokens

    if heatmap.ndim == 5:
        # 3-channel egomotion: extract magnitude (channel 0)
        mag = heatmap[:, :, 0, :, :]   # (b, t, H_p, W_p)
    else:
        mag = heatmap                   # (b, t, H_p, W_p)

    gate = 1.0 + alpha * mag           # always >= 1
    gate = gate.unsqueeze(-1)          # (b, t, H_p, W_p, 1)
    return patch_tokens * gate

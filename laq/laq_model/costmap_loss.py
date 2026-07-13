"""Cost-map latent regularization loss (Phase 2).

Implements the design doc's "Latent Distance Penalty" (if two different
driving scenes in a batch share a similar collision-risk cost map, their
continuous latent action vectors should be pulled closer together in
latent space) as a geometry-matching loss: the pairwise cosine-similarity
structure ("Gram matrix") of the latent vectors is pushed to match the
pairwise cosine-similarity structure of the cost maps, via MSE between the
two similarity matrices.

This went through two design iterations during development, both verified
empirically on nuScenes-mini, not just in theory:

1. First version: pull-only, raw (unnormalized) squared Euclidean distance,
   weighted by cost-map similarity. Collapsed the entire LAQ codebook to a
   single entry at every nonzero weight tried (0.001-0.1): unnormalized
   distance has a free degenerate optimum (shrink every latent toward the
   same point), which NSVQ's codebook/decoder can absorb almost for free.
2. Second version: pull-only, but with L2-normalized latent vectors (fixes
   the scale-collapse exploit -- verified via a clean monotonic weight-vs-
   perplexity dose-response curve instead of uniform collapse). But still
   only ever pulls; with no repulsive pressure, real driving cost maps turn
   out similar enough to each other on average that "pull when similar"
   still degrades codebook diversity at every tested weight, without
   improving cost-map cluster alignment.
3. This version: matches the *full* similarity geometry (attraction where
   cost-map similarity is high, repulsion where it's low), which removes
   the residual "align everything since most pairs have some positive
   similarity" failure mode of version 2 -- collapsing every latent
   together now actively hurts the loss for every pair whose cost maps
   aren't actually similar.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-8


def costmap_similarity_matrix(costmaps: Tensor) -> Tensor:
    """Pairwise cosine similarity between flattened per-sample cost maps.

    Args:
        costmaps: (B, H, W) per-sample cost maps.
    Returns:
        (B, B) similarity matrix, negative similarity clipped to 0 (cost
        maps are non-negative, so this only guards near-zero-norm pairs).
    """
    flat = costmaps.flatten(start_dim=1)
    normalized = F.normalize(flat, dim=-1, eps=EPS)
    sim = normalized @ normalized.t()
    return sim.clamp(min=0.0)


def latent_similarity_matrix(z: Tensor) -> Tensor:
    """Pairwise cosine similarity between L2-normalized per-sample latent vectors.

    Args:
        z: (B, D) continuous latent vectors.
    Returns:
        (B, B) similarity matrix in [-1, 1].
    """
    normalized = F.normalize(z, dim=-1, eps=EPS)
    return normalized @ normalized.t()


def costmap_regularization_loss(z: Tensor, costmaps: Tensor) -> Tensor:
    """Matches the latent geometry's pairwise similarity to the cost maps'.

    loss = mean_{i != j} (cos(z_i, z_j) - cos(costmap_i, costmap_j))^2

    Minimizing this pulls latent vectors of similar-risk scenes together
    (target cosine ~1) *and* pushes latent vectors of dissimilar-risk scenes
    apart (target cosine ~0), unlike a pull-only loss. Collapsing every
    latent together (cos=1 for all pairs) is no longer a free optimum: it's
    heavily penalized for every pair whose cost-map similarity isn't also
    ~1, which on real driving data is most pairs.
    """
    batch_size = z.shape[0]
    z_sim = latent_similarity_matrix(z)
    cost_sim = costmap_similarity_matrix(costmaps)

    off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=z.device)
    diff_sq = (z_sim - cost_sim).square()

    return diff_sq[off_diagonal].mean()

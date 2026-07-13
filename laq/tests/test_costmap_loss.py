import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from laq_model.costmap_loss import (
    costmap_regularization_loss,
    costmap_similarity_matrix,
    latent_similarity_matrix,
)


def test_matching_geometry_yields_zero_loss():
    """If the latent similarity structure already matches the cost-map
    similarity structure exactly, the loss should be ~0."""
    batch_size = 4
    torch.manual_seed(0)
    base = torch.rand(batch_size, 8)  # non-negative, valid as a "cost map" too
    # Cost maps are literally the same vectors (just reshaped), so their
    # cosine-similarity Gram matrix matches z's exactly.
    z = base
    costmaps = base.reshape(batch_size, 2, 4)

    loss = costmap_regularization_loss(z, costmaps)

    z_sim = latent_similarity_matrix(z)
    cost_sim = costmap_similarity_matrix(costmaps)
    assert torch.allclose(z_sim, cost_sim, atol=1e-5)
    assert loss.item() < 1e-4


def test_uniform_collapse_is_penalized_when_costmaps_are_dissimilar():
    """Regression guard: this is what versions 1 and 2 of the loss got wrong.

    If every latent collapses to the same direction (cos=1 for every pair),
    but the cost maps are mostly dissimilar (low pairwise cosine), the loss
    must be large -- collapse is no longer a free way to minimize it.
    """
    batch_size = 5
    shared_direction = torch.ones(8)
    z = shared_direction.unsqueeze(0).repeat(batch_size, 1)  # full collapse

    # One-hot cost maps with disjoint supports -> mutual cosine similarity ~0.
    costmaps = torch.zeros(batch_size, batch_size, batch_size)
    for i in range(batch_size):
        costmaps[i, i, i] = 1.0

    loss = costmap_regularization_loss(z, costmaps)

    assert loss.item() > 0.9  # target sim ~0, actual sim ~1 -> (1-0)^2 ~1


def test_disjoint_costmaps_push_latents_toward_orthogonality():
    """Gradient sanity check: with target similarity ~0 for all pairs, the
    gradient should push z away from a fully-aligned (collapsed) state."""
    batch_size = 4
    shared_direction = torch.ones(8)
    z = (shared_direction.unsqueeze(0).repeat(batch_size, 1) + 1e-3 * torch.randn(batch_size, 8))
    z.requires_grad_(True)

    costmaps = torch.zeros(batch_size, batch_size, batch_size)
    for i in range(batch_size):
        costmaps[i, i, i] = 1.0

    loss = costmap_regularization_loss(z, costmaps)
    loss.backward()

    assert z.grad is not None
    assert torch.any(z.grad != 0)


def test_loss_is_invariant_to_uniform_latent_scale():
    """The loss depends only on direction (via cosine similarity), not
    magnitude -- scaling z by any positive constant must not change it."""
    batch_size = 5
    torch.manual_seed(3)
    z = torch.randn(batch_size, 8)
    costmaps = torch.rand(batch_size, 6, 6)

    loss_unit_scale = costmap_regularization_loss(z, costmaps)
    loss_shrunk = costmap_regularization_loss(z * 1e-3, costmaps)
    loss_grown = costmap_regularization_loss(z * 1e3, costmaps)

    assert torch.allclose(loss_unit_scale, loss_shrunk, atol=1e-5)
    assert torch.allclose(loss_unit_scale, loss_grown, atol=1e-5)


def test_gradient_flows_into_z():
    batch_size = 4
    torch.manual_seed(2)
    z = torch.randn(batch_size, 8, requires_grad=True)
    costmaps = torch.rand(batch_size, 6, 6)

    loss = costmap_regularization_loss(z, costmaps)
    loss.backward()

    assert z.grad is not None
    assert torch.any(z.grad != 0)

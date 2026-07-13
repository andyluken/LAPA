"""Regression guard for Phase 1 heatmap conditioning.

Saliency conditioning must be a strict additive feature: with no heatmap
(or alpha=0, or an all-zero heatmap), LatentActionQuantization.forward()
and .inference() must behave exactly as before the change, so existing
Sthv2/baseline training is provably unaffected.

NSVQ hardcodes its codebook buffers on 'cuda' (laq_model/nsvq.py), so this
test requires a CUDA device; it skips cleanly otherwise.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from laq_model import LatentActionQuantization

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="NSVQ codebook buffers require CUDA")

TINY_CONFIG = dict(
    dim=32,
    quant_dim=8,
    codebook_size=4,
    image_size=64,
    patch_size=8,  # 64 / 8 = 8x8 patch grid; NSVQ's code_seq_len=1 CNN head needs >= 8x8 input
    spatial_depth=1,
    temporal_depth=1,
    dim_head=8,
    heads=2,
    code_seq_len=1,
)


def make_model(**overrides) -> LatentActionQuantization:
    torch.manual_seed(42)
    config = {**TINY_CONFIG, **overrides}
    return LatentActionQuantization(**config).cuda()


def make_video(batch_size=2) -> torch.Tensor:
    torch.manual_seed(123)
    return torch.rand(batch_size, 3, 2, 64, 64).cuda()


def patch_grid_shape(model: LatentActionQuantization) -> tuple[int, int]:
    return model.patch_height_width


@requires_cuda
def test_none_heatmap_matches_explicit_zero_heatmap():
    model = make_model()
    video = make_video()
    h, w = patch_grid_shape(model)
    zero_heatmap = torch.zeros(video.shape[0], 2, h, w).cuda()

    torch.manual_seed(0)
    recon_none = model(video, heatmap=None, return_recons_only=True)

    torch.manual_seed(0)
    recon_zero = model(video, heatmap=zero_heatmap, return_recons_only=True)

    assert torch.allclose(recon_none, recon_zero)


@requires_cuda
def test_alpha_zero_ignores_nonzero_heatmap():
    model = make_model(heatmap_alpha=0.0)
    video = make_video()
    h, w = patch_grid_shape(model)
    torch.manual_seed(7)
    random_heatmap = torch.rand(video.shape[0], 2, h, w).cuda()

    torch.manual_seed(0)
    recon_none = model(video, heatmap=None, return_recons_only=True)

    torch.manual_seed(0)
    recon_random = model(video, heatmap=random_heatmap, return_recons_only=True)

    assert torch.allclose(recon_none, recon_random)


@requires_cuda
def test_nonzero_heatmap_with_default_alpha_changes_output():
    model = make_model()  # default heatmap_alpha=1.0
    video = make_video()
    h, w = patch_grid_shape(model)
    torch.manual_seed(7)
    random_heatmap = torch.rand(video.shape[0], 2, h, w).cuda()

    torch.manual_seed(0)
    recon_none = model(video, heatmap=None, return_recons_only=True)

    torch.manual_seed(0)
    recon_conditioned = model(video, heatmap=random_heatmap, return_recons_only=True)

    assert not torch.allclose(recon_none, recon_conditioned)


@requires_cuda
def test_inference_none_matches_zero_heatmap():
    model = make_model().eval()
    video = make_video()
    h, w = patch_grid_shape(model)
    zero_heatmap = torch.zeros(video.shape[0], 2, h, w).cuda()

    with torch.no_grad():
        torch.manual_seed(0)
        recon_none = model.inference(video, heatmap=None)

        torch.manual_seed(0)
        recon_zero = model.inference(video, heatmap=zero_heatmap)

    assert torch.allclose(recon_none, recon_zero)


# ── Phase 2: cost-map regularization must also be a strict additive feature ──

@requires_cuda
def test_default_costmap_loss_weight_ignores_nonzero_costmap():
    model = make_model()  # default costmap_loss_weight=0.0
    video = make_video()
    torch.manual_seed(9)
    random_costmap = torch.rand(video.shape[0], 2, 32, 32).cuda()

    torch.manual_seed(0)
    loss_none, n_none, logs_none = model(video, costmap=None)

    torch.manual_seed(0)
    loss_costmap, n_costmap, logs_costmap = model(video, costmap=random_costmap)

    assert torch.allclose(loss_none, loss_costmap)
    assert n_none == n_costmap
    assert logs_none['costmap_loss'] == 0.0
    assert logs_costmap['costmap_loss'] == 0.0
    assert logs_none['recon_loss'] == logs_costmap['recon_loss']


@requires_cuda
def test_no_costmap_passed_yields_zero_costmap_loss_log():
    model = make_model()
    video = make_video()

    loss, num_unique_indices, aux_logs = model(video)

    assert set(aux_logs.keys()) == {'recon_loss', 'costmap_loss'}
    assert aux_logs['costmap_loss'] == 0.0
    assert torch.allclose(loss, torch.tensor(aux_logs['recon_loss']).cuda())


@requires_cuda
def test_nonzero_costmap_loss_weight_changes_loss():
    model = make_model(costmap_loss_weight=0.5)
    video = make_video(batch_size=4)  # need >1 distinct sample for a nonzero pairwise loss
    torch.manual_seed(9)
    random_costmap = torch.rand(video.shape[0], 2, 32, 32).cuda()

    torch.manual_seed(0)
    loss_disabled, _, logs_disabled = model(video, costmap=None)

    torch.manual_seed(0)
    loss_enabled, _, logs_enabled = model(video, costmap=random_costmap)

    assert logs_disabled['costmap_loss'] == 0.0
    assert logs_enabled['costmap_loss'] != 0.0
    assert not torch.allclose(loss_disabled, loss_enabled)
    # recon_loss itself must be identical -- the costmap term is purely additive.
    assert logs_disabled['recon_loss'] == logs_enabled['recon_loss']


@requires_cuda
def test_costmap_warmup_is_a_hard_gate_not_a_ramp():
    """A gradual ramp (nonzero from step 1) was tried first and still
    collapsed the codebook identically to a constant weight -- isolated to
    the gradient itself, not its magnitude. The gate must therefore be hard:
    exactly 0 below costmap_warmup_steps, exactly the target weight at and
    above it -- no partial values in between.
    """
    model = make_model(costmap_loss_weight=1.0, costmap_warmup_steps=100)
    video = make_video(batch_size=4)
    torch.manual_seed(9)
    random_costmap = torch.rand(video.shape[0], 2, 32, 32).cuda()

    torch.manual_seed(0)
    loss_at_step0, _, logs_at_step0 = model(video, step=0, costmap=random_costmap)

    torch.manual_seed(0)
    loss_no_costmap, _, logs_no_costmap = model(video, step=0, costmap=None)

    # Below the gate, the effective weight must be exactly 0 -- same as if
    # costmap were never passed at all.
    assert torch.allclose(loss_at_step0, loss_no_costmap)
    assert logs_at_step0['costmap_loss'] == 0.0

    assert model._effective_costmap_loss_weight(0) == 0.0
    assert model._effective_costmap_loss_weight(50) == 0.0
    assert model._effective_costmap_loss_weight(99) == 0.0
    assert model._effective_costmap_loss_weight(100) == 1.0
    assert model._effective_costmap_loss_weight(200) == 1.0


def test_costmap_warmup_disabled_by_default_is_constant_weight():
    model = LatentActionQuantization(**TINY_CONFIG, costmap_loss_weight=0.7)
    assert model.costmap_warmup_steps == 0
    assert model._effective_costmap_loss_weight(0) == 0.7
    assert model._effective_costmap_loss_weight(99999) == 0.7

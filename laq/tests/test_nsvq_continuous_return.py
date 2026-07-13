"""Regression guard for NSVQ's return_continuous flag (Phase 2).

return_continuous=False must reproduce the exact original 4-tuple; setting
it True must only add a 5th element without perturbing the first 4, so any
existing caller of NSVQ.forward() is unaffected.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from laq_model.nsvq import NSVQ

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="NSVQ codebook buffers require CUDA")


def make_vq() -> NSVQ:
    torch.manual_seed(0)
    return NSVQ(dim=16, num_embeddings=4, embedding_dim=8, device="cuda", code_seq_len=1, patch_size=8, image_size=64).cuda()


def make_inputs(batch_size=3):
    torch.manual_seed(1)
    # encode() expects (batch, h*w, dim) flattened patch tokens for an 8x8 grid.
    first = torch.randn(batch_size, 64, 16).cuda()
    last = torch.randn(batch_size, 64, 16).cuda()
    return first, last


@requires_cuda
def test_default_signature_unchanged():
    vq = make_vq()
    first, last = make_inputs()

    torch.manual_seed(42)
    out = vq(first, last)

    assert len(out) == 4


@requires_cuda
def test_return_continuous_adds_fifth_element_without_changing_first_four():
    vq = make_vq()
    first, last = make_inputs()

    torch.manual_seed(42)
    out_default = vq(first, last)

    vq2 = make_vq()
    torch.manual_seed(42)
    out_continuous = vq2(first, last, return_continuous=True)

    assert len(out_continuous) == 5
    for a, b in zip(out_default[:2], out_continuous[:2]):  # quantized_input, perplexity
        assert torch.allclose(a, b)
    assert (out_default[2] == out_continuous[2]).all()  # codebooks_used
    assert torch.equal(out_default[3], out_continuous[3])  # indices

    continuous = out_continuous[4]
    batch_size = first.shape[0]
    assert continuous.shape[0] == batch_size
    assert continuous.ndim == 2

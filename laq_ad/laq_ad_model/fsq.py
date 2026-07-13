"""Finite Scalar Quantization (Mentzer et al., 2023).

Replaces NSVQ throughout laq_ad. Key advantages over NSVQ:
  - No codebook collapse: quantization is a deterministic rounding op per dimension
  - No commitment loss, no EMA, no replace_unused_codebooks schedule
  - Codebook size = product(levels) — fully controlled by the levels list

Level design rules:
  - Prefer ODD levels (3, 5, 7, 9) — the symmetric tanh bound gives clean rounding
    without special-casing. With even L the range is asymmetric and you must
    include the even-level atanh shift (handled automatically here).
  - Minimum level per dimension: 3 (L=2 degenerates because tanh*0.5 rounds to 0).
  - Recommended configs:
      [8]         → 8 codes    (direct baseline comparison, single dim, even-corrected)
      [5, 5]      → 25 codes
      [8, 6, 5]   → 240 codes  (paper default)
      [8, 5, 5, 5] → 1000 codes (paper large)

Usage:
    fsq = FSQ(levels=[8, 5, 5])   # 200 codes, 3-dim latent
    z_q, indices = fsq(z)         # z: (..., len(levels))

Reference: https://arxiv.org/abs/2309.15505
"""

from math import prod

import torch
import torch.nn as nn


class FSQ(nn.Module):
    """Finite Scalar Quantization over a fixed list of per-dimension levels.

    Each dimension i of the input is independently bounded with a tanh-based
    function and rounded to one of L_i values.

    For ODD L_i: symmetric bound → values in {-(L-1)/2, …, (L-1)/2}
    For EVEN L_i: asymmetric bound (atanh shift) → values in {-L/2, …, L/2-1}
    Both give exactly L_i distinct integers after rounding.

    Flat codebook index (mixed-radix):
        idx = sum_i  digit_i * prod(levels[:i])
    where digit_i = z_q_int_i + L_i // 2  (shifted to [0, L_i-1]).
    """

    def __init__(self, levels: list[int]):
        super().__init__()
        assert all(l >= 3 for l in levels), \
            "Each level must be >= 3. L=2 collapses because tanh*0.5 always rounds to 0."
        self.levels = levels
        self._d = len(levels)
        self._codebook_size = prod(levels)

        levels_t = torch.tensor(levels, dtype=torch.float32)
        half_l = (levels_t - 1) / 2.0   # e.g. 3.5 for L=8, 2.0 for L=5

        # For EVEN levels: shift the tanh input so the output range has L distinct
        # integers after rounding. Without this, tanh*3.5 for L=8 reaches at most
        # 3 (never 4) giving only 7 values instead of 8.
        #   offset = 0.5 for even L (asymmetric range [-L/2, L/2-1])
        #   offset = 0.0 for odd  L (symmetric range [-(L-1)/2, (L-1)/2])
        offset = torch.where(
            levels_t % 2 == 0,
            torch.full_like(levels_t, 0.5),
            torch.zeros_like(levels_t),
        )
        # shift = atanh(offset / half_l) — the input bias that achieves this range
        shift = (offset / half_l).atanh()

        self.register_buffer("_half_l", half_l)
        self.register_buffer("_offset", offset)
        self.register_buffer("_shift", shift)

        # Integer half-width for digit extraction: L//2
        #   even L=8: digits = z_q_int + 4 ∈ {0,..,7}
        #   odd  L=5: digits = z_q_int + 2 ∈ {0,..,4}
        half_l_int = torch.tensor([l // 2 for l in levels], dtype=torch.long)
        self.register_buffer("_half_l_int", half_l_int)

        # Mixed-radix basis for flat index computation
        basis = torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.long), dim=0)
        self.register_buffer("_basis", basis)

        levels_long = torch.tensor(levels, dtype=torch.long)
        self.register_buffer("_levels_long", levels_long)

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @property
    def dim(self) -> int:
        return self._d

    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        """Map R^D to the valid quantization range via shifted tanh.

        For even L: range ≈ (-L/2, L/2-1)  (asymmetric, L integers reachable)
        For odd  L: range ≈ (-(L-1)/2, (L-1)/2)  (symmetric, L integers reachable)
        """
        return (z + self._shift).tanh() * self._half_l - self._offset

    def _quantize_st(self, z_bounded: torch.Tensor) -> torch.Tensor:
        """Round to nearest integer with a straight-through gradient estimator."""
        z_q = torch.round(z_bounded)
        return z_bounded + (z_q - z_bounded).detach()

    def _to_indices(self, z_q: torch.Tensor) -> torch.Tensor:
        """Quantized float values → flat codebook indices.

        z_q values are (or should be) integers after _quantize_st in forward.
        We add L//2 to shift them into [0, L-1] per dimension, then compute
        the mixed-radix flat index.
        """
        z_int = z_q.round().long()
        digits = (z_int + self._half_l_int).clamp(
            torch.zeros_like(self._half_l_int),
            self._levels_long - 1,
        )
        return (digits * self._basis).sum(dim=-1)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (..., D)  pre-quantization encoder output
        Returns:
            z_q:     (..., D) quantized, straight-through gradient
            indices: (...,)  flat codebook index in [0, codebook_size)
        """
        z_bounded = self._bound(z)
        z_q = self._quantize_st(z_bounded)
        indices = self._to_indices(z_q.detach())
        return z_q, indices, z_bounded   # z_bounded returned for entropy reg

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Flat indices → quantized code vectors (for inference / decoding)."""
        codes = []
        idx = indices
        for l_i, half_i in zip(self.levels, self._half_l_int.tolist()):
            z_int = (idx % l_i).long() - half_i
            codes.append(z_int.float())
            idx = idx // l_i
        return torch.stack(codes, dim=-1)

    def perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        flat = indices.reshape(-1)
        counts = torch.bincount(flat, minlength=self._codebook_size).float()
        probs = counts / counts.sum().clamp(min=1)
        log_probs = torch.where(probs > 0, probs.log(), torch.zeros_like(probs))
        return (-(probs * log_probs).sum()).exp()


def soft_entropy_reg(z_bounded: torch.Tensor, levels: list[int], temperature: float = 0.5) -> torch.Tensor:
    """Differentiable joint entropy regularization on the continuous FSQ embeddings.

    Computes soft assignments to ALL K = prod(levels) joint codebook entries via
    an outer product of per-dimension soft assignments (independence assumption).
    Maximises the entropy of the JOINT marginal distribution over the batch.

    This correctly penalises the anti-correlated diagonal pattern that per-dimension
    independent entropy would miss: e.g. for levels=[3,3], codes (1,-1), (0,0),
    (-1,1) each getting 1/3 of samples gives perfect per-dimension marginal entropy
    but only 3/9 joint codes used.  The joint entropy is log(3) << log(9).

    Args:
        z_bounded: (b, D) bounded continuous values from FSQ._bound()
        levels:    list[int] per-dimension level counts
        temperature: softmax temperature (lower = harder assignments)
    Returns:
        neg_joint_entropy: scalar ≤ 0.  Add `-weight * result` to the main loss
                           (negative because we maximise entropy = minimise -entropy).
    """
    b = z_bounded.shape[0]
    per_dim_probs = []
    for i, L in enumerate(levels):
        if L % 2 == 0:
            centers = torch.arange(-L // 2, L // 2, dtype=torch.float32, device=z_bounded.device)
        else:
            centers = torch.arange(-(L - 1) // 2, (L - 1) // 2 + 1, dtype=torch.float32, device=z_bounded.device)
        z_d = z_bounded[:, i:i + 1]                        # (b, 1)
        log_soft = -(z_d - centers).pow(2) / temperature   # (b, L)
        per_dim_probs.append(log_soft.softmax(dim=-1))     # (b, L_i)

    # Joint soft assignment via outer product (independence approximation).
    # Shape builds as (b, L0) → (b, L0*L1) → (b, L0*L1*L2) → ...
    joint = per_dim_probs[0]                               # (b, L0)
    for probs in per_dim_probs[1:]:
        # broadcast outer product: (b, K_so_far, 1) * (b, 1, L_i) → (b, K_so_far*L_i)
        joint = joint.unsqueeze(-1) * probs.unsqueeze(-2)  # (b, K_so_far, L_i)
        joint = joint.reshape(b, -1)                       # (b, K_so_far * L_i)

    marginal = joint.mean(0)                               # (K,) joint marginal over batch
    neg_entropy = (marginal * marginal.clamp(min=1e-10).log()).sum()   # -H(joint)
    return neg_entropy   # add as `-weight * this` to the total loss

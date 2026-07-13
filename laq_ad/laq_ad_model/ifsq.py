"""Improved Finite Scalar Quantization(IFSQ)
    A Tokenizer enhancement method"

Traditional FSQ suffers from 'activation collapse ' where 
many quantization buckets go unused because neural network 
outputs follow a Gaussian distribution.

The Solution: It replaces the tanh-based activation function 
(y = 2.0 ⋅ σ(1.6x) - 1) that flattens the distribution.

With just this one line of code change, it achieves upto 100% 
odebook usage, eliminating representation waste and significantly 
improving image reconstruction fidelity
"""
from math import prod
import torch
import torch.nn as nn

class iFSQ(nn.Module):
    """improved Finite Scalar Quantization (iFSQ).
    
    Replaces the traditional tanh mapping with a distribution-matching activation
    function (y = 2.0 * sigmoid(1.6 * x) - 1.0) to eliminate activation collapse
    and achieve 100% codebook utilization without complex entropy tracking.
    """
    def __init__(self, levels: list[int]):
        super().__init__()
        # iFSQ generalizes effortlessly across both ODD and EVEN layers.
        # Highly recommended to stick to a configuration totaling ~4 bits per dimension.
        assert all(l >= 2 for l in levels), "iFSQ supports any levels >= 2."
        
        self.levels = torch.tensor(levels)
        self._d = len(levels)
        self._codebook_size = prod(levels)

        levels_t = torch.tensor(levels, dtype=torch.float32)

        #Scaling half-width for digit extraction: L//2
        # iFSQ scales symmetrically from [-1, 1] mapped perfectly to your output grids
        half_l = (levels_t - 1) / 2.0
        self.register_buffer("_half_l", half_l)

        # Integer half-width offset for flat-digit index tracking
        half_l_int = torch.tensor([l // 2 for l in levels], dtype=torch.long)
        self.register_buffer("_half_l_int", half_l_int)

        # Mixed radix basis tracking for flat codebook lookup indices
        basis = torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.long), dim=0)
        self.register_buffer("basis", basis)

        levels_long = torch.tensor(levels, dtype=torch.long)
        self.register_buffer("levels_long", levels_long)

    @property
    def codebook_size(self) -> int:
        return self._codebook_size
    
    @property
    def dim(self) -> int:
        return self._d
    
    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        """The core iFSQ improvement.
        
        Maps standard Gaussian outputs from the encoder into a perfect uniform
        distribution bounded tightly within (-half_l, half_l).
        """
        # iFSQ Activation Function: y = 2.0 * sigmoid(1.6 * x) - 1.0
        z_uniform = 2.0 * torch.sigmoid(1.6 * z) - 1.0
        return z_uniform * self._half_l  # Scale to (-half_l, half_l)
    
    def _quantize_st(self, z_bouded:torch.tensor) -> torch.Tensor:
        """Round values using a straight -through gradient estimator.
        """
        z_q = torch.round(z_bouded)
        return z_bouded + (z_q - z_bouded).detach()
    
    def _to_indices(self, z_q: torch.Tensor) -> torch.Tensor:
        """Maps quantized float grid values into absolute codebook indices.
        """
        z_int = z_q.round().long()
        digits = (z_int + self._half_l_int).clamp(
            torch.zeros_like(self._half_l_int),
            self.levels_long - 1,
        )
        return (digits * self.basis).sum(dim=-1)
    
    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (..., D) Continuous pre-quantization latents from the encoder
        Returns:
            z_q:     (..., D) Discrete quantized outputs containing straight-through gradient
            indices: (...,)  flat indexing vector matching codebook index [0, codebook_size)
        """
        z_bounded = self._bound(z)
        z_q = self._quantize_st(z_bounded)
        indices = self._to_indices(z_q.detach())

        # No extra soft-entropy returns needed; codebook usage naturally stayes at 100%
        return z_q, indices
    
    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Decodes flat indexes back into discrete latent dimensions.
        """
        codes = []
        idx = indices.clone()
        for l_i, half_i in zip(self.levels, self._half_l_int.tolist()):
            z_int = (idx % l_i).long() - half_i
            codes.append(z_int)
            idx = idx // l_i

            # De-scale back to the target bounded grid matrix
        return torch.stack(codes, dim=-1)
    
    def perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        flat = indices.reshape(-1)
        counts = torch.bincount(flat, minlength=self._codebook_size).float()
        probs = counts / counts.sum().clamp(min=1)
        log_probs = torch.where(probs > 0, probs.log(), torch.zeros_like(probs))
        return (-(probs * log_probs).sum()).exp()

    
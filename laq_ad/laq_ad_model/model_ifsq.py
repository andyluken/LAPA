"""LAQ-AD: clean autonomous-driving Latent Action Quantization model.

Key differences from the upstream laq/ implementation:
  - FSQ replaces NSVQ: no collapse risk, no commitment loss, no codebook schedule
  - CAN bus auxiliary head: predicts normalized [yaw_rate, speed] from latent code
  - Egomotion heatmap conditioning (magnitude-only gate, proven best in experiments)
  - Clean separation of encoder/quantizer/decoder for easy ablation

Codebook size = product(levels). Default levels=[8,6,5,5] → 1200 codes.
"""

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, pack
from einops.layers.torch import Rearrange

# Reuse attention modules from the sibling laq/ directory.
# Add its path so imports work whether or not `pip install -e laq/` was run.
import sys as _sys
from pathlib import Path as _Path
_LAQ_DIR = _Path(__file__).resolve().parents[2] / "laq"
if str(_LAQ_DIR) not in _sys.path:
    _sys.path.insert(0, str(_LAQ_DIR))

from laq_model.attention import Transformer, ContinuousPositionBias

#from .fsq import FSQ, soft_entropy_reg
from .ifsq import iFSQ
from .heatmap import apply_patch_heatmap


def exists(val):
    return val is not None


def pair(val):
    return (val, val) if not isinstance(val, tuple) else val


class CANBusHead(nn.Module):
    """Per-axis CAN bus prediction: each iFSQ axis independently predicts one signal.

        axis 0 → yaw_rate_norm  (signed: left=−1, straight≈0, right=+1)
        axis 1 → speed_norm     (stationary=−1, moving≈+1)

    Using one Linear(1,1) per axis eliminates the symmetry in a shared
    Linear(ifsq_dim, 2) head where swapping which axis predicts yaw vs. speed
    gives the same loss. That symmetry causes ~50% of runs to converge to
    the wrong assignment (speed in axis 0), producing no left/right turn
    separation. With per-axis heads the assignment is fixed by construction.
    """

    YAW_SCALE: float = 1.5    # rad/s: covers typical urban + highway turns
    SPEED_SCALE: float = 15.0  # m/s ≈ 54 km/h highway cruising

    N_CAN_SIGNALS: int = 2  # always yaw_rate + speed

    def __init__(self, ifsq_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Bind only the first N_CAN_SIGNALS axes to CAN predictions.
        # Extra iFSQ axes (ifsq_dim > 2) are left free — shaped by the flow loss.
        # hidden_dim is ignored (kept for API compatibility).
        n = min(ifsq_dim, self.N_CAN_SIGNALS)
        self.heads = nn.ModuleList([nn.Linear(1, 1) for _ in range(n)])

    @classmethod
    def normalize_targets(cls, yaw_rate: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """Return (b, 2) normalized CAN bus targets in [-1, 1]."""
        yr_n = torch.clamp(yaw_rate / cls.YAW_SCALE, -1.0, 1.0)
        sp_n = torch.clamp(speed / cls.SPEED_SCALE, 0.0, 1.0) * 2.0 - 1.0
        return torch.stack([yr_n, sp_n], dim=-1)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """z_q: (b, ifsq_dim) → (b, N_CAN_SIGNALS) predicted [yaw_rate_norm, speed_norm].

        Only the first N_CAN_SIGNALS iFSQ axes are used; remaining axes are free.
        """
        return torch.cat([h(z_q[:, i:i+1]) for i, h in enumerate(self.heads)], dim=-1)


class LAQADModel(nn.Module):
    """Latent Action Quantization for Autonomous Driving.

    Architecture:
        patch_embed → [heatmap gate] → spatial_transformer → temporal_transformer
        → CNN compress (8×8 → 1) → project_in → iFSQ → project_out
        → expand → cross-attn decoder → pixel reconstruction
                                ↘ CAN bus head → [yaw_rate_pred, speed_pred]

    Args:
        dim:              Transformer hidden dimension.
        levels:           iFSQ per-dimension level counts. Codebook size = prod(levels).
        image_size:       Input image resolution (square or (H, W)).
        patch_size:       Patch size for patch embedding (square or (H, W)).
        spatial_depth:    Number of spatial transformer layers.
        temporal_depth:   Number of temporal transformer layers.
        dim_head:         Attention head dimension.
        heads:            Number of attention heads.
        heatmap_alpha:    Egomotion gate strength (0 = off, 1 = full gate).
        can_bus_weight:   Loss weight for CAN bus auxiliary prediction (0 = off).
    """

    def __init__(
        self,
        *,
        dim: int = 512,
        levels: list[int] = (8, 6, 5, 5),
        image_size: int | tuple[int, int] = 256,
        patch_size: int | tuple[int, int] = 32,
        spatial_depth: int = 2,
        temporal_depth: int = 2,
        dim_head: int = 64,
        heads: int = 8,
        channels: int = 3,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        heatmap_alpha: float = 1.0,
        can_bus_weight: float = 0.1,
        #entropy_reg_weight: float = 0.1,
        spread_reg_weight: float = 1.0,
        covariance_reg_weight: float = 0.0,
        flow_prediction_weight: float = 0.0,
        recon_loss_weight: float = 1.0,
    ):
        super().__init__()

        self.covariance_reg_weight = covariance_reg_weight
        self.flow_prediction_weight = flow_prediction_weight
        self.recon_loss_weight = recon_loss_weight
        self.heatmap_alpha = heatmap_alpha
        self.can_bus_weight = can_bus_weight
        #self.entropy_reg_weight = entropy_reg_weight
        self.spread_reg_weight = spread_reg_weight

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        ph, pw = self.patch_size
        ih, iw = self.image_size
        assert ih % ph == 0 and iw % pw == 0
        self._patch_h = ih // ph
        self._patch_w = iw // pw

        ifsq_dim = len(levels)
        self.ifsq = iFSQ(list(levels))

        # ── Patch embedding ────────────────────────────────────────────────────
        self.to_patch_emb = nn.Sequential(
            Rearrange("b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)", p1=ph, p2=pw),
            nn.LayerNorm(channels * ph * pw),
            nn.Linear(channels * ph * pw, dim),
            nn.LayerNorm(dim),
        )

        # ── Transformers ───────────────────────────────────────────────────────
        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)

        attn_kw = dict(dim=dim, dim_head=dim_head, heads=heads,
                       attn_dropout=attn_dropout, ff_dropout=ff_dropout,
                       peg=True, peg_causal=True)
        xattn_kw = dict(**attn_kw, has_cross_attn=True, dim_context=dim)

        self.enc_spatial = Transformer(depth=spatial_depth, **attn_kw)
        self.enc_temporal = Transformer(depth=temporal_depth, **attn_kw)
        self.dec_spatial = Transformer(depth=spatial_depth, **xattn_kw)

        # ── Quantizer bridge ──────────────────────────────────────────────────
        # CNN mirrors NSVQ's compress: (embed_dim, 8, 8) → (embed_dim, 1, 1)
        embed_dim = dim  # keep same dim to avoid info bottleneck before FSQ
        self.cnn_compress = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),  # 8→4
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=4, stride=1, padding=0),  # 4→1
        )
        # Independent projection head per FSQ dimension.  Using a shared linear
        # (512→D) causes all D dimensions to be correlated (they live on a 1D
        # manifold in D-space), so only ~8 of 1200 codes are ever visited.
        # Each dimension gets its own 512→1 head, allowing orthogonal axes.
        self.project_in = nn.ModuleList(
            [nn.Linear(embed_dim, 1) for _ in range(ifsq_dim)]
        )
        self.project_out = nn.Linear(ifsq_dim, dim)

        # Small init keeps z_pre near 0 at step 0 so tanh is unsaturated
        # and all codes are initially reachable.  The spread_reg loss then
        # prevents the weights from growing large during training.
        for head in self.project_in:
            nn.init.normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

        # ── CAN bus head ──────────────────────────────────────────────────────
        self.can_bus_head = CANBusHead(ifsq_dim) if can_bus_weight > 0 else None

        # ── Flow prediction head ───────────────────────────────────────────────
        # Predicts the (3, 8, 8) egomotion heatmap directly from z_q.  This
        # bypasses the decoder shortcut: z_q must encode motion regardless of
        # whether the decoder can reconstruct from first-frame appearance alone.
        # Only active when heatmap is 5D (egomotion format) and weight > 0.
        if flow_prediction_weight > 0:
            self.flow_head = nn.Sequential(
                nn.Linear(ifsq_dim, 64),
                nn.GELU(),
                nn.Linear(64, 3 * 8 * 8),
            )
        else:
            self.flow_head = None

        # ── Decoder ───────────────────────────────────────────────────────────
        self.to_pixels = nn.Sequential(
            nn.Linear(dim, channels * ph * pw),
            Rearrange("b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)", p1=ph, p2=pw),
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def patch_height_width(self) -> tuple[int, int]:
        return self._patch_h, self._patch_w

    @property
    def codebook_size(self) -> int:
        return self.ifsq.codebook_size

    # ── Encode ────────────────────────────────────────────────────────────────

    def _encode(self, tokens: torch.Tensor):
        """Spatial + temporal transformer. Returns first_tokens, last_tokens."""
        b = tokens.shape[0]
        h, w = self._patch_h, self._patch_w
        video_shape = tuple(tokens.shape[:-1])

        # spatial
        tokens = rearrange(tokens, "b t h w d -> (b t) (h w) d")
        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)
        tokens = self.enc_spatial(tokens, attn_bias=attn_bias, video_shape=video_shape)
        tokens = rearrange(tokens, "(b t) (h w) d -> b t h w d", b=b, h=h, w=w)

        # temporal
        tokens = rearrange(tokens, "b t h w d -> (b h w) t d")
        tokens = self.enc_temporal(tokens, video_shape=video_shape)
        tokens = rearrange(tokens, "(b h w) t d -> b t h w d", b=b, h=h, w=w)

        return tokens[:, :1], tokens[:, 1:]  # first_tokens, last_tokens

    # ── Quantize ──────────────────────────────────────────────────────────────

    def _quantize(self, last_tokens: torch.Tensor):
        """Compress last_tokens (b,1,h,w,d) → quantized action vector.

        Returns:
            z_q:      (b, ifsq_dim) quantized values (straight-through grad)
            indices:  (b,) flat codebook indices
            z_pre:    (b, ifsq_dim) pre-tanh projection (for spread reg)
        """
        b = last_tokens.shape[0]

        x = rearrange(last_tokens, "b 1 h w d -> b d h w")
        x = self.cnn_compress(x)   # (b, d, 1, 1)
        x = x.reshape(b, -1)       # (b, d)

        z_pre = torch.cat([h(x) for h in self.project_in], dim=-1)  # (b, ifsq_dim)
        z_q, indices = self.ifsq(z_pre)
        return z_q, indices, z_pre

    # ── Decode ────────────────────────────────────────────────────────────────

    def _decode(self, first_tokens: torch.Tensor, action_context: torch.Tensor):
        """Cross-attention decoder. Reconstructs the rest frame from first+action.

        Args:
            first_tokens:   (b, 1, h, w, d) — first frame patch tokens (detached)
            action_context: (b, 1, h, w, d) — action embedding broadcast spatially
        Returns:
            recon: (b, c, 1, H, W)
        """
        b = first_tokens.shape[0]
        h, w = self._patch_h, self._patch_w
        video_shape = tuple(first_tokens.shape[:-1])

        q = rearrange(first_tokens, "b t h w d -> (b t) (h w) d")
        ctx = rearrange(action_context, "b t h w d -> (b t) (h w) d")
        attn_bias = self.spatial_rel_pos_bias(h, w, device=q.device)

        out = self.dec_spatial(q, attn_bias=attn_bias, video_shape=video_shape, context=ctx)
        out = rearrange(out, "(b t) (h w) d -> b t h w d", b=b, h=h, w=w)
        return self.to_pixels(out)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        video: torch.Tensor,
        heatmap: Optional[torch.Tensor] = None,
        can_bus: Optional[torch.Tensor] = None,
        return_recons_only: bool = False,
        return_only_codebook_ids: bool = False,
    ):
        """
        Args:
            video:    (b, c, 2, H, W) — frame pair (t, t+offset)
            heatmap:  (b, 2, H_p, W_p) or (b, 2, 3, H_p, W_p)  optional
            can_bus:  (b, 2) — normalized [yaw_rate, speed] targets  optional
        Returns (training):
            loss, num_unique_indices, aux_logs
        """
        assert video.ndim == 5 and video.shape[2] == 2
        b = video.shape[0]

        first_frame = video[:, :, :1]
        rest_frames = video[:, :, 1:]

        # 1. Patch embed
        first_tokens = self.to_patch_emb(first_frame)   # (b, 1, h, w, d)
        rest_tokens = self.to_patch_emb(rest_frames)    # (b, 1, h, w, d)

        # 2. Egomotion heatmap gate (magnitude-only; heatmap=None → identity)
        if exists(heatmap):
            first_tokens = apply_patch_heatmap(first_tokens, heatmap[:, :1], self.heatmap_alpha)
            rest_tokens = apply_patch_heatmap(rest_tokens, heatmap[:, 1:], self.heatmap_alpha)

        # 3. Encode
        tokens = torch.cat([first_tokens, rest_tokens], dim=1)   # (b, 2, h, w, d)
        first_enc, last_enc = self._encode(tokens)

        # 4. Quantize
        z_q, indices, z_pre = self._quantize(last_enc)

        if return_only_codebook_ids:
            return indices

        # 5. Perplexity (for logging)
        n_unique = indices.unique().numel()
        perplexity = self.ifsq.perplexity(indices).item()

        # 6. Project quantized action → decoder context
        action_emb = self.project_out(z_q)           # (b, d)
        h, w = self._patch_h, self._patch_w
        action_ctx = action_emb[:, None, None, None, :].expand(b, 1, h, w, -1)

        # 7. Decode (first_tokens.detach() as query — same convention as upstream laq)
        recon = self._decode(first_tokens.detach(), action_ctx)  # (b, c, 1, H, W)

        if return_recons_only:
            return rearrange(recon, "b c 1 h w -> b c h w")

        # 8. Losses
        recon_loss = F.mse_loss(recon, rest_frames)
        loss = self.recon_loss_weight * recon_loss

        can_loss_val = 0.0
        if exists(self.can_bus_head) and exists(can_bus):
            can_pred = self.can_bus_head(z_q)
            can_loss = F.mse_loss(can_pred, can_bus)
            loss = loss + self.can_bus_weight * can_loss
            can_loss_val = can_loss.item()

        # Flow prediction: predict (3, 8, 8) egomotion from z_q.
        # Forces z_q to encode motion patterns — bypasses the decoder shortcut.
        # Only applies when heatmap is the 5D egomotion format (b, 2, 3, h, w).
        flow_loss_val = 0.0
        if self.flow_head is not None and exists(heatmap) and heatmap.ndim == 5:
            flow_target = heatmap[:, 0].detach()   # (b, 3, 8, 8) first frame egomotion
            flow_pred = self.flow_head(z_q).reshape(b, 3, self._patch_h, self._patch_w)
            flow_loss = F.mse_loss(flow_pred, flow_target)
            loss = loss + self.flow_prediction_weight * flow_loss
            flow_loss_val = flow_loss.item()

        # 9. Spread regularization on z_pre (BEFORE tanh — gradient never vanishes).
        #    Penalises deviation from N(0,1): zero batch-mean and unit batch-std.
        #    This is the primary anti-collapse mechanism.  The entropy reg below
        #    adds extra spreading pressure in the bounded space once z_pre is
        #    well-behaved.
        spread_loss_val = 0.0
        if self.spread_reg_weight > 0 and z_pre.shape[0] > 1:
            mean_penalty = z_pre.mean(0).pow(2).mean()
            std_penalty = (z_pre.std(0) - 1.0).pow(2).mean()
            spread_loss = mean_penalty + std_penalty
            loss = loss + self.spread_reg_weight * spread_loss
            spread_loss_val = spread_loss.item()

        # 10. Covariance regularization (VICReg-style): penalise off-diagonal elements
        #     of the z_pre batch covariance matrix.  Forces FSQ dimensions to capture
        #     orthogonal motion features (yaw vs speed) rather than collapsing to a
        #     correlated bimodal along a single shared encoder direction.
        #     Only meaningful when fsq_dim > 1.
        cov_loss_val = 0.0
        if self.covariance_reg_weight > 0 and z_pre.shape[1] > 1 and z_pre.shape[0] > 1:
            b, d = z_pre.shape
            z_c = (z_pre - z_pre.mean(0)) / (z_pre.std(0) + 1e-6)   # (b, d) standardised
            cov = (z_c.T @ z_c) / (b - 1)                             # (d, d) batch cov
            off_diag = cov - torch.diag(cov.diag())
            cov_loss = off_diag.pow(2).sum() / d
            loss = loss + self.covariance_reg_weight * cov_loss
            cov_loss_val = cov_loss.item()

        # 11. Entropy regularization on z_bounded (secondary; differentiable soft
        #     assignments across levels — less effective when tanh is saturated,
        #     but works well in combination with spread reg above).
        #entropy_loss_val = 0.0
        #if self.entropy_reg_weight > 0:
        #    neg_entropy = soft_entropy_reg(z_bounded, self.ifsq.levels)
        #    loss = loss - self.entropy_reg_weight * neg_entropy
        #    entropy_loss_val = neg_entropy.item()

        aux_logs = {
            "recon_loss": recon_loss.item(),
            "can_bus_loss": can_loss_val,
            "flow_loss": flow_loss_val,
            "spread_loss": spread_loss_val,
            "cov_loss": cov_loss_val,
            #"entropy_neg": entropy_loss_val,
            "perplexity": perplexity,
            "n_unique_codes": n_unique,
        }
        return loss, n_unique, aux_logs

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def inference(
        self,
        video: torch.Tensor,
        heatmap: Optional[torch.Tensor] = None,
        return_only_codebook_ids: bool = False,
    ):
        assert video.ndim == 5 and video.shape[2] == 2
        b = video.shape[0]

        first_frame = video[:, :, :1]
        rest_frames = video[:, :, 1:]

        first_tokens = self.to_patch_emb(first_frame)
        rest_tokens = self.to_patch_emb(rest_frames)

        if exists(heatmap):
            first_tokens = apply_patch_heatmap(first_tokens, heatmap[:, :1], self.heatmap_alpha)
            rest_tokens = apply_patch_heatmap(rest_tokens, heatmap[:, 1:], self.heatmap_alpha)

        tokens = torch.cat([first_tokens, rest_tokens], dim=1)
        first_enc, last_enc = self._encode(tokens)
        #z_q, indices, _zb, _zp = self._quantize(last_enc)
        z_q, indices, z_pre = self._quantize(last_enc)

        if return_only_codebook_ids:
            return indices

        action_emb = self.project_out(z_q)
        h, w = self._patch_h, self._patch_w
        action_ctx = action_emb[:, None, None, None, :].expand(b, 1, h, w, -1)
        recon = self._decode(first_tokens, action_ctx)
        return rearrange(recon, "b c 1 h w -> b c h w")

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, strict=False, **kwargs)

    def save(self, path: str | Path):
        torch.save(self.state_dict(), str(path))

    def load(self, path: str | Path):
        self.load_state_dict(torch.load(str(path), map_location="cpu"))

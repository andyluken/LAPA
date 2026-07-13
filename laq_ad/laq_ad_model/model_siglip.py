"""LAQ-AD model variant with SigLIP NaFlex spatial encoder.

Identical to LAQADModel except the simple patch-embedding layer is replaced by a
pretrained SigLIP NaFlex ViT encoder.  The rest of the architecture — spatial and
temporal transformers, CNN compress, FSQ, CAN bus head, flow head — is unchanged.

The SigLIP backbone is frozen by default; only the projection head and the rest of
the network are trained.  This keeps memory cost low and avoids catastrophic forgetting
of the rich SigLIP visual features.  Set freeze_backbone=False for full fine-tuning
(recommended only after the model converges with the frozen backbone).

Usage:
    from laq_ad_model.model_siglip import LAQADModelSigLIP

    model = LAQADModelSigLIP(
        dim=512,
        levels=[3, 3],
        siglip_model_name="naflexvit_base_patch16_siglip",
        freeze_backbone=True,
        can_bus_weight=0.5,
        flow_prediction_weight=5.0,
        spread_reg_weight=0.05,
        covariance_reg_weight=1.0,
        entropy_reg_weight=0.0,
        recon_loss_weight=0.0,
    )
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, pack
from einops.layers.torch import Rearrange

import sys as _sys
_LAQ_DIR = Path(__file__).resolve().parents[2] / "laq"
if str(_LAQ_DIR) not in _sys.path:
    _sys.path.insert(0, str(_LAQ_DIR))

from laq_model.attention import Transformer, ContinuousPositionBias

from .fsq import FSQ, soft_entropy_reg
from .heatmap import apply_patch_heatmap
from .model import CANBusHead
from .siglip_encoder import SigLIPNaFlexEncoder


def exists(val):
    return val is not None


def pair(val):
    return (val, val) if not isinstance(val, tuple) else val


class LAQADModelSigLIP(nn.Module):
    """LAQ-AD with pretrained SigLIP NaFlex patch encoder.

    The SigLIP backbone extracts rich visual features at 16×16 → pooled to 8×8
    tokens.  The temporal transformer, FSQ quantizer, CAN bus head, flow head,
    and decoder are identical to LAQADModel.

    Args:
        dim:               Transformer hidden dimension.
        levels:            FSQ per-dimension level counts. Codebook size = prod(levels).
        siglip_model_name: timm model ID (default: naflexvit_base_patch16_siglip).
        freeze_backbone:   Freeze SigLIP weights during training (recommended initially).
        image_size:        Input image resolution (used for grid_size computation).
        patch_size:        Used only to compute the 8×8 grid size (keep 32 for 256×256).
        spatial_depth:     Spatial transformer layers.
        temporal_depth:    Temporal transformer layers.
        can_bus_weight:    CAN bus auxiliary loss weight (0 = off).
        flow_prediction_weight: Flow prediction loss weight (0 = off).
        spread_reg_weight: Anti-collapse spread regularization.
        covariance_reg_weight: VICReg covariance penalty.
        entropy_reg_weight: Soft entropy regularization (keep 0.0).
        recon_loss_weight: Reconstruction loss weight (0.0 confirmed best).
        heatmap_alpha:     Egomotion gate strength.
    """

    def __init__(
        self,
        *,
        dim: int = 512,
        levels: list[int] = (3, 3),
        siglip_model_name: str = SigLIPNaFlexEncoder.MODEL_NAME,
        freeze_backbone: bool = True,
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
        can_bus_weight: float = 0.5,
        entropy_reg_weight: float = 0.0,
        spread_reg_weight: float = 0.05,
        covariance_reg_weight: float = 1.0,
        flow_prediction_weight: float = 5.0,
        recon_loss_weight: float = 0.0,
    ):
        super().__init__()

        self.covariance_reg_weight = covariance_reg_weight
        self.flow_prediction_weight = flow_prediction_weight
        self.recon_loss_weight = recon_loss_weight
        self.heatmap_alpha = heatmap_alpha
        self.can_bus_weight = can_bus_weight
        self.entropy_reg_weight = entropy_reg_weight
        self.spread_reg_weight = spread_reg_weight
        self.siglip_model_name = siglip_model_name

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        ph, pw = self.patch_size
        ih, iw = self.image_size
        assert ih % ph == 0 and iw % pw == 0
        self._patch_h = ih // ph   # 8 for image_size=256, patch_size=32
        self._patch_w = iw // pw

        fsq_dim = len(levels)
        self.fsq = FSQ(list(levels))

        # ── SigLIP NaFlex patch encoder ───────────────────────────────────────
        self.patch_encoder = SigLIPNaFlexEncoder(
            model_name=siglip_model_name,
            out_dim=dim,
            grid_size=self._patch_h,
            freeze_backbone=freeze_backbone,
            img_size=self.image_size[0],
        )

        # ── Transformers ──────────────────────────────────────────────────────
        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)

        attn_kw = dict(dim=dim, dim_head=dim_head, heads=heads,
                       attn_dropout=attn_dropout, ff_dropout=ff_dropout,
                       peg=True, peg_causal=True)
        xattn_kw = dict(**attn_kw, has_cross_attn=True, dim_context=dim)

        self.enc_spatial = Transformer(depth=spatial_depth, **attn_kw)
        self.enc_temporal = Transformer(depth=temporal_depth, **attn_kw)
        self.dec_spatial = Transformer(depth=spatial_depth, **xattn_kw)

        # ── Quantizer bridge ──────────────────────────────────────────────────
        embed_dim = dim
        self.cnn_compress = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),  # 8→4
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=4, stride=1, padding=0),  # 4→1
        )
        self.project_in = nn.ModuleList(
            [nn.Linear(embed_dim, 1) for _ in range(fsq_dim)]
        )
        self.project_out = nn.Linear(fsq_dim, dim)

        for head in self.project_in:
            nn.init.normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

        # ── CAN bus head (per-axis) ───────────────────────────────────────────
        self.can_bus_head = CANBusHead(fsq_dim) if can_bus_weight > 0 else None

        # ── Flow prediction head ──────────────────────────────────────────────
        if flow_prediction_weight > 0:
            self.flow_head = nn.Sequential(
                nn.Linear(fsq_dim, 64),
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
    def codebook_size(self) -> int:
        return self.fsq.codebook_size

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _embed_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Embed one frame via SigLIP encoder.

        Args:
            frame: (b, c, 1, H, W)
        Returns:
            tokens: (b, 1, h, w, dim)
        """
        return self.patch_encoder(frame.squeeze(2))   # squeeze time dim

    def _encode(self, tokens: torch.Tensor):
        b = tokens.shape[0]
        h, w = self._patch_h, self._patch_w
        video_shape = tuple(tokens.shape[:-1])

        tokens = rearrange(tokens, "b t h w d -> (b t) (h w) d")
        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)
        tokens = self.enc_spatial(tokens, attn_bias=attn_bias, video_shape=video_shape)
        tokens = rearrange(tokens, "(b t) (h w) d -> b t h w d", b=b, h=h, w=w)

        tokens = rearrange(tokens, "b t h w d -> (b h w) t d")
        tokens = self.enc_temporal(tokens, video_shape=video_shape)
        tokens = rearrange(tokens, "(b h w) t d -> b t h w d", b=b, h=h, w=w)

        return tokens[:, :1], tokens[:, 1:]

    def _quantize(self, last_tokens: torch.Tensor):
        b = last_tokens.shape[0]
        x = rearrange(last_tokens, "b 1 h w d -> b d h w")
        x = self.cnn_compress(x)
        x = x.reshape(b, -1)
        z_pre = torch.cat([h(x) for h in self.project_in], dim=-1)
        z_q, indices, z_bounded = self.fsq(z_pre)
        return z_q, indices, z_bounded, z_pre

    def _decode(self, first_tokens: torch.Tensor, action_context: torch.Tensor):
        b = first_tokens.shape[0]
        h, w = self._patch_h, self._patch_w
        video_shape = tuple(first_tokens.shape[:-1])

        q   = rearrange(first_tokens, "b t h w d -> (b t) (h w) d")
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
        assert video.ndim == 5 and video.shape[2] == 2
        b = video.shape[0]

        first_frame = video[:, :, :1]
        rest_frames  = video[:, :, 1:]

        first_tokens = self._embed_frame(first_frame)
        rest_tokens  = self._embed_frame(rest_frames)

        if exists(heatmap):
            first_tokens = apply_patch_heatmap(first_tokens, heatmap[:, :1], self.heatmap_alpha)
            rest_tokens  = apply_patch_heatmap(rest_tokens,  heatmap[:, 1:], self.heatmap_alpha)

        tokens = torch.cat([first_tokens, rest_tokens], dim=1)
        first_enc, last_enc = self._encode(tokens)

        z_q, indices, z_bounded, z_pre = self._quantize(last_enc)

        if return_only_codebook_ids:
            return indices

        n_unique   = indices.unique().numel()
        perplexity = self.fsq.perplexity(indices).item()

        action_emb = self.project_out(z_q)
        h, w = self._patch_h, self._patch_w
        action_ctx = action_emb[:, None, None, None, :].expand(b, 1, h, w, -1)

        recon = self._decode(first_tokens.detach(), action_ctx)

        if return_recons_only:
            return rearrange(recon, "b c 1 h w -> b c h w")

        recon_loss = F.mse_loss(recon, rest_frames)
        loss = self.recon_loss_weight * recon_loss

        can_loss_val = 0.0
        if exists(self.can_bus_head) and exists(can_bus):
            can_pred = self.can_bus_head(z_q)
            can_loss = F.mse_loss(can_pred, can_bus)
            loss = loss + self.can_bus_weight * can_loss
            can_loss_val = can_loss.item()

        flow_loss_val = 0.0
        if self.flow_head is not None and exists(heatmap) and heatmap.ndim == 5:
            flow_target = heatmap[:, 0].detach()
            flow_pred = self.flow_head(z_q).reshape(b, 3, self._patch_h, self._patch_w)
            flow_loss = F.mse_loss(flow_pred, flow_target)
            loss = loss + self.flow_prediction_weight * flow_loss
            flow_loss_val = flow_loss.item()

        spread_loss_val = 0.0
        if self.spread_reg_weight > 0 and z_pre.shape[0] > 1:
            mean_penalty = z_pre.mean(0).pow(2).mean()
            std_penalty  = (z_pre.std(0) - 1.0).pow(2).mean()
            spread_loss  = mean_penalty + std_penalty
            loss = loss + self.spread_reg_weight * spread_loss
            spread_loss_val = spread_loss.item()

        cov_loss_val = 0.0
        if self.covariance_reg_weight > 0 and z_pre.shape[1] > 1 and z_pre.shape[0] > 1:
            b_sz, d = z_pre.shape
            z_c = (z_pre - z_pre.mean(0)) / (z_pre.std(0) + 1e-6)
            cov = (z_c.T @ z_c) / (b_sz - 1)
            off_diag = cov - torch.diag(cov.diag())
            cov_loss = off_diag.pow(2).sum() / d
            loss = loss + self.covariance_reg_weight * cov_loss
            cov_loss_val = cov_loss.item()

        entropy_loss_val = 0.0
        if self.entropy_reg_weight > 0:
            neg_entropy = soft_entropy_reg(z_bounded, self.fsq.levels)
            loss = loss - self.entropy_reg_weight * neg_entropy
            entropy_loss_val = neg_entropy.item()

        aux_logs = {
            "recon_loss":   recon_loss.item(),
            "can_bus_loss": can_loss_val,
            "flow_loss":    flow_loss_val,
            "spread_loss":  spread_loss_val,
            "cov_loss":     cov_loss_val,
            "entropy_neg":  entropy_loss_val,
            "perplexity":   perplexity,
            "n_unique_codes": n_unique,
        }
        return loss, n_unique, aux_logs

    # ── Inference ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def inference(
        self,
        video: torch.Tensor,
        heatmap: Optional[torch.Tensor] = None,
        return_only_codebook_ids: bool = False,
    ):
        assert video.ndim == 5 and video.shape[2] == 2
        b = video.shape[0]

        first_tokens = self._embed_frame(video[:, :, :1])
        rest_tokens  = self._embed_frame(video[:, :, 1:])

        if exists(heatmap):
            first_tokens = apply_patch_heatmap(first_tokens, heatmap[:, :1], self.heatmap_alpha)
            rest_tokens  = apply_patch_heatmap(rest_tokens,  heatmap[:, 1:], self.heatmap_alpha)

        tokens = torch.cat([first_tokens, rest_tokens], dim=1)
        first_enc, last_enc = self._encode(tokens)
        z_q, indices, _zb, _zp = self._quantize(last_enc)

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

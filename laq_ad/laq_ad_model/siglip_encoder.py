"""SigLIP NaFlex spatial encoder for LAQ-AD.

Replaces the simple patch-embedding layer with a pretrained NaFlex ViT encoder.
NaFlex (Natural Flexible resolution) accepts any input size; patches are extracted
at patch_size=16 and pooled to the target 8×8 grid expected by the rest of the model.

Model used: naflexvit_base_patch16_siglip  (timm ≥ 1.0)
  embed_dim : 768
  patch_size: 16
  At 256×256: 16×16 = 256 patch tokens (no CLS token)
  Pooled to 8×8 = 64 tokens, projected to LAQ-AD dim (512)
  Normalization: mean=std=0.5  →  pixel [0,1] maps to [-1,+1]

Requires timm ≥ 1.0:
    pip install "timm>=1.0"
"""

import torch
import torch.nn as nn
from einops import rearrange

SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD  = (0.5, 0.5, 0.5)


class SigLIPNaFlexEncoder(nn.Module):
    """Pretrained NaFlex SigLIP encoder → 8×8 patch token grid.

    Wraps a timm NaFlex ViT, extracts patch tokens (no CLS), pools spatially
    to `grid_size × grid_size`, and projects to `out_dim`.

    The output tensor shape is (b, 1, grid_size, grid_size, out_dim), which
    matches the output of `LAQADModel.to_patch_emb` and drops in directly.

    Args:
        model_name:      timm model identifier.
        out_dim:         output feature dimension after projection.
        grid_size:       target spatial grid (default 8 → 8×8 = 64 tokens).
        freeze_backbone: freeze all SigLIP weights; only projection is trainable.
                         Set False for full fine-tuning (needs lower lr).
        img_size:        image resolution passed to timm (must match training data).
    """

    MODEL_NAME = "naflexvit_base_patch16_siglip"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        out_dim: int = 512,
        grid_size: int = 8,
        freeze_backbone: bool = True,
        img_size: int = 256,
    ):
        super().__init__()

        try:
            import timm
        except ImportError:
            raise ImportError("SigLIPNaFlexEncoder requires timm ≥ 1.0: pip install 'timm>=1.0'")

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            img_size=img_size,
        )
        backbone_dim = self.backbone.embed_dim  # 768 for ViT-B

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.grid_size = grid_size
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.project = nn.Sequential(
            nn.Linear(backbone_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode one video frame.

        Args:
            frame: (b, 3, H, W) float tensor in [0, 1]
        Returns:
            tokens: (b, 1, grid_size, grid_size, out_dim)
        """
        # NaFlex normalization: pixel [0,1] → [-1,+1]
        x = frame * 2.0 - 1.0

        # forward_features → (b, n_patches, backbone_dim); no CLS token in NaFlex
        feats = self.backbone.forward_features(x)   # (b, 256, 768) at 256×256

        # Infer spatial grid from token count (should be square)
        n = feats.shape[1]
        h = w = int(n ** 0.5)
        feats = rearrange(feats, "b (h w) d -> b d h w", h=h, w=w)

        feats = self.pool(feats)                     # (b, backbone_dim, grid_size, grid_size)
        feats = rearrange(feats, "b d h w -> b h w d")
        feats = self.project(feats)                  # (b, grid_size, grid_size, out_dim)

        return feats.unsqueeze(1)                    # (b, 1, grid_size, grid_size, out_dim)

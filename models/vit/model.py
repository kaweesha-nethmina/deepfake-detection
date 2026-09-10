"""ViT-B/16 — owner: C.

Includes the experimental **frequency-hybrid** variant (DCT-based high-frequency
branch) to investigate cross-generator robustness. Run with:

    python models/vit/train.py -c configs/vit.yaml

Outputs land in `results/vit/`; cross-generator comparison writes to
`results/comparison/` (owner-C only).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class FrequencyBranch(nn.Module):
    """Small 2D-DCT high/low frequency branch fused with ViT features.

    Purpose (Owner C): give the transformer an explicit gradient-aligned
    frequency signal, testing the hypothesis that generator-agnostic cues live
    in high-frequency residuals (e.g., scaling artefacts).

    NOTE: this is experimental. The `use_frequency_hybrid` flag in config
    switches it off for the baseline ViT-B/16 comparison.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 256,
        patch_size: int = 16,
        high_freq: bool = True,
    ) -> None:
        super().__init__()
        self.high_freq = high_freq
        self.patch_size = patch_size
        # Small conv stack operating on the DCT magnitude image.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((patch_size, patch_size)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(64 * patch_size * patch_size, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Block-wise 2D-DCT of stride=patch_size then magnitude.
        dct = _block_dct(x, self.patch_size, high=self.high_freq)  # (B,C,p,p)
        dct = torch.log1p(torch.abs(dct))                         # >= 0 magnitude
        dct = (dct - dct.amin(dim=(2, 3), keepdim=True)) / (
            dct.amax(dim=(2, 3), keepdim=True) + 1e-6
        )
        feats = self.conv(dct)
        feats = feats.flatten(1)
        return self.head(feats)


def _block_dct(x: torch.Tensor, patch: int = 16, high: bool = True) -> torch.Tensor:
    """Extract per-patch DCT, returning the low (DC) or high-freq magnitude map."""
    b, c, h, w = x.shape
    hp, wp = h // patch, w // patch
    x = x[:, :, : hp * patch, : wp * patch]
    blocks = x.reshape(b, c, hp, patch, wp, patch).permute(0, 1, 2, 4, 3, 5)
    blocks = blocks.reshape(b * c * hp * wp, patch, patch)

    n = patch
    u = torch.arange(n, device=x.device).float()
    v = u[:, None]
    base = math.sqrt(2.0 / n)
    cos_mu = (torch.cos(math.pi * u[None, :] * (2 * torch.arange(n, device=x.device).float()[:, None] + 1) / (2 * n)) * base)
    cos_mu[0] = cos_mu[0] / math.sqrt(2.0)
    cos_nu = cos_mu.clone()

    # 2D-DCT: F = C_u^T x C
    dct = (cos_mu @ blocks) @ cos_nu.T          # (K, n, n), K = b*c*hp*wp
    if high:
        # High-frequency → retain only mid/upper bands: zero out low bands.
        mask = torch.ones(dct.shape[-2:], device=dct.device, dtype=torch.bool)
        low = n // 4
        mask[:low, :] = False
        mask[:, :low] = False
        dct = dct * mask.to(dct.dtype)
    else:
        mask = torch.ones(dct.shape[-2:], device=dct.device, dtype=torch.bool)
        high = n // 2
        mask[high:, :] = False
        mask[:, high:] = False
        dct = dct * mask.to(dct.dtype)

    return dct.view(b, c, hp, wp, n, n).permute(0, 1, 2, 4, 3, 5).reshape(b, c, hp * n, wp * n)


class ViTBinary(nn.Module):
    """ViT-B/16 binary classifier with optional frequency-hybrid branch."""

    def __init__(
        self,
        num_classes: int = 2,
        variant: str = "vit_base_patch16_224",
        weights: str | bool = "imagenet",
        use_frequency_hybrid: bool = False,
        patch_size: int = 16,
        drop_rate: float = 0.1,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover
            raise ImportError("timm is required for ViT.") from exc

        self.backbone = timm.create_model(
            variant,
            pretrained=bool(weights),
            num_classes=0,          # no head
            drop_rate=drop_rate,
            attn_drop_rate=0.0,
        )
        in_features = self.backbone.num_features

        self.freq_branch = (
            FrequencyBranch(in_channels=3, out_dim=256, patch_size=patch_size, high_freq=True)
            if use_frequency_hybrid
            else None
        )
        if self.freq_branch is not None:
            self.head = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Dropout(p=drop_rate),
                nn.Linear(in_features + 256, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Dropout(p=drop_rate),
                nn.Linear(in_features, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)          # (B, D)
        if self.freq_branch is not None:
            feats = torch.cat([feats, self.freq_branch(x)], dim=1)
        return self.head(feats.view(feats.size(0), -1))


def build_model(cfg: dict) -> nn.Module:
    m = cfg.get("model", {})
    return ViTBinary(
        num_classes=m.get("num_classes", 2),
        variant=m.get("variant", "vit_base_patch16_224"),
        weights=m.get("weights", "imagenet"),
        use_frequency_hybrid=m.get("use_frequency_hybrid", False),
        patch_size=m.get("patch_size", 16),
        drop_rate=m.get("drop_rate", 0.1),
    )


if __name__ == "__main__":
    dummy = torch.randn(2, 3, 224, 224)
    net = build_model({"model": {"variant": "vit_base_patch16_224", "use_frequency_hybrid": True}})
    print("out:", net(dummy).shape)
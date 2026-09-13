"""
Model 4: ViT-B/16 (Transfer Learning) + optional frequency-hybrid branch.
Owner: Member C

This is the ONLY place the network architecture is defined. Hyperparameters
(learning rate, epochs, whether the frequency branch is on, etc.) all come
from configs/vit.yaml — do not hardcode them here.
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required for the ViT backbone. Add it to requirements.txt "
        "and `pip install timm`."
    ) from e


class FrequencyBranch(nn.Module):
    """
    Lightweight frequency-domain branch (novelty add-on).

    Takes the 2D FFT magnitude spectrum of the input image and runs it through
    a small CNN. GAN / diffusion upsampling artifacts often show up more
    clearly in frequency space than in pixel space, so fusing this signal
    with the ViT's spatial features can help cross-generator generalization.
    """

    def __init__(self, in_channels: int = 3, conv_channels=(16, 32, 64), fusion_dim: int = 128):
        super().__init__()
        layers = []
        prev = in_channels
        for ch in conv_channels:
            layers += [
                nn.Conv2d(prev, ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ]
            prev = ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(prev, fusion_dim)

    @staticmethod
    def to_fft_magnitude(x: torch.Tensor) -> torch.Tensor:
        """Convert a batch of RGB images (B, C, H, W) to log-magnitude FFT spectra."""
        fft = torch.fft.fft2(x, norm="ortho")
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        magnitude = torch.log(torch.abs(fft_shifted) + 1e-8)
        return magnitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = self.to_fft_magnitude(x)
        feat = self.conv(freq)
        feat = self.pool(feat).flatten(1)
        return self.proj(feat)


class ViTDeepfakeDetector(nn.Module):
    """
    ViT-B/16 backbone (pretrained, via timm) with a binary classification head.
    If `use_frequency_hybrid` is enabled, the FFT branch's features are
    concatenated with the ViT's pooled output before the final linear layer.
    """

    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.1,
        use_frequency_hybrid: bool = False,
        freq_branch_cfg: dict | None = None,
    ):
        super().__init__()
        self.use_frequency_hybrid = use_frequency_hybrid

        # num_classes=0 -> timm returns pooled features instead of its own head
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        vit_feat_dim = self.backbone.num_features

        if use_frequency_hybrid:
            freq_branch_cfg = freq_branch_cfg or {}
            self.freq_branch = FrequencyBranch(
                in_channels=3,
                conv_channels=freq_branch_cfg.get("conv_channels", [16, 32, 64]),
                fusion_dim=freq_branch_cfg.get("fusion_dim", 128),
            )
            head_in_dim = vit_feat_dim + freq_branch_cfg.get("fusion_dim", 128)
        else:
            self.freq_branch = None
            head_in_dim = vit_feat_dim

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(head_in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vit_feat = self.backbone(x)  # (B, vit_feat_dim)

        if self.use_frequency_hybrid:
            freq_feat = self.freq_branch(x)  # (B, fusion_dim)
            fused = torch.cat([vit_feat, freq_feat], dim=1)
        else:
            fused = vit_feat

        fused = self.dropout(fused)
        logits = self.classifier(fused)  # (B, num_classes)
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits


def build_model(cfg: dict) -> nn.Module:
    """Factory used by train.py — reads the `model:` section of configs/vit.yaml."""
    m_cfg = cfg["model"]
    return ViTDeepfakeDetector(
        backbone=m_cfg.get("backbone", "vit_base_patch16_224"),
        pretrained=m_cfg.get("pretrained", True),
        num_classes=m_cfg.get("num_classes", 1),
        dropout=m_cfg.get("dropout", 0.1),
        use_frequency_hybrid=m_cfg.get("use_frequency_hybrid", False),
        freq_branch_cfg=m_cfg.get("freq_branch", {}),
    )

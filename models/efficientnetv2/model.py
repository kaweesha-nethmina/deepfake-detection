"""EfficientNetV2 — owner: B.

Defines the architecture on top of timm's `efficientnetv2_*` zoo.
Run with:

    python models/efficientnetv2/train.py -c configs/efficientnetv2.yaml

Outputs land in `results/efficientnetv2/`.
"""

from __future__ import annotations

import torch
from torch import nn


class EfficientNetV2Classifier(nn.Module):
    """timm EfficientNetV2 backbone + tunable classification head."""

    def __init__(
        self,
        num_classes: int = 2,
        variant: str = "efficientnetv2_s",
        weights: str | bool = "imagenet",
        freeze_backbone: bool = False,
        drop_rate: float = 0.2,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover
            raise ImportError("timm is required for EfficientNetV2.") from exc

        self.backbone = timm.create_model(
            variant,
            pretrained=bool(weights),
            num_classes=0,          # drop the default classifier head
            drop_rate=drop_rate,    # stochastic depth/classifier dropout applied by timm
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate * 0.5),
            nn.Linear(256, num_classes),
        )
        self.depth = getattr(self.backbone, "depth", 4)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)      # (B, num_features) — global pooled
        return self.head(feats)


def build_model(cfg: dict) -> nn.Module:
    m = cfg.get("model", {})
    return EfficientNetV2Classifier(
        num_classes=m.get("num_classes", 2),
        variant=m.get("variant", "efficientnetv2_s"),
        weights=m.get("weights", "imagenet"),
        freeze_backbone=m.get("freeze_backbone", False),
        drop_rate=m.get("drop_rate", 0.2),
    )


if __name__ == "__main__":
    dummy = torch.randn(2, 3, 224, 224)
    net = build_model({"model": {"variant": "efficientnetv2_s"}})
    print("out:", net(dummy).shape)
"""ResNet50 — owner: B.

Defines the architecture (torchvision pretrained backbone, optional
progressive unfreezing). Run with:

    python models/resnet50/train.py -c configs/resnet50.yaml

Outputs land in `results/resnet50/`.
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from torchvision.models import resnet50, ResNet50_Weights
except ImportError:  # torchvision missing — let the import error surface clearly
    resnet50 = None
    ResNet50_Weights = None


class ResNet50DeeplabHead(nn.Module):
    """ResNet50 with a 2-class head swapped in for binary deepfake classification."""

    def __init__(
        self,
        num_classes: int = 2,
        weights: str = "IMAGENET1K_V1",
        freeze_backbone: bool = False,
        unfreeze_from_layer: int | None = None,
        drop_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if resnet50 is None:
            raise ImportError("torchvision is required for ResNet50.")

        weights_enum = ResNet50_Weights[weights] if weights in ResNet50_Weights.__members__ else None
        self.backbone = resnet50(weights=weights_enum)
        # Replace the FC head with a small MLP + dropout.
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate * 0.5),
            nn.Linear(256, num_classes),
        )

        self.freeze_backbone = freeze_backbone
        self.unfreeze_from_layer = unfreeze_from_layer
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.backbone.fc.parameters():
                p.requires_grad = True  # always train the classification head

    def freeze_until(self, layer: int) -> None:
        """Freeze all backbone params below `layer` (1=layer1 .. 4=layer4)."""
        self.freeze_backbone = False
        names = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc")
        for p in self.parameters():
            p.requires_grad = False
        for name in names[max(1, layer) :]:
            for p in getattr(self.backbone, name).parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(cfg: dict) -> nn.Module:
    m = cfg.get("model", {})
    net = ResNet50DeeplabHead(
        num_classes=m.get("num_classes", 2),
        weights=m.get("weights", "IMAGENET1K_V1"),
        freeze_backbone=m.get("freeze_backbone", False),
        drop_rate=m.get("drop_rate", 0.1),
    )
    if m.get("unfreeze_from_layer"):
        net.freeze_until(m["unfreeze_from_layer"])
    return net


if __name__ == "__main__":
    dummy = torch.randn(2, 3, 224, 224)
    net = build_model({"model": {"num_classes": 2}})
    print("out:", net(dummy).shape)

"""Custom CNN — owner: A.

Defines the architecture. Keep this file self-contained (imports `src` for
utils, but no model-specific config). Run with:

    python models/custom_cnn/train.py -c configs/custom_cnn.yaml

Outputs land in `results/custom_cnn/`.
"""

from __future__ import annotations

import torch
from torch import nn


class CustomCNN(nn.Module):
    """Small bespoke CNN designed to generalise across generators.

    Design choices (Owner A to extend):
      * moderate width, deep stem, residual-free "VGG-style" blocks + BN,
      * global-average pooling + single linear head (fewer params -> less
        overfitting to generator-specific texture),
      * optional DropBlock/rand-aug variants can be toggled via config.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_channels: int = 32,
        num_blocks: int = 4,
        dropout: float = 0.3,
        spatial_input: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.dropout = dropout

        layers: list[nn.Module] = []
        cin = in_channels
        for i in range(num_blocks):
            channels = base_channels * (2**i)
            layers += [
                nn.Conv2d(cin, channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            cin = channels

        self.features = nn.Sequential(*layers)

        # GAP gives an input-size-independent feature vector.
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(cin, num_classes),
        )

        # Logger-free init is fine here; weight init done in train.py.
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape (B, num_classes)."""
        x = self.features(x)
        x = x.mean(dim=(2, 3))          # GAP
        return self.head(x)


def get_model(num_classes: int = 2) -> nn.Module:
    """Factory used by demo_app.py — mirrors the default config architecture."""
    return CustomCNN(
        in_channels=3,
        num_classes=num_classes,
        base_channels=32,
        num_blocks=4,
        dropout=0.3,
        spatial_input=128,
    )


def build_model(cfg: dict) -> nn.Module:
    """Factory used by train.py — reads only the ``model:`` block of the yaml."""
    m = cfg.get("model", {})
    return CustomCNN(
        in_channels=m.get("in_channels", 3),
        num_classes=m.get("num_classes", 2),
        base_channels=m.get("base_channels", 32),
        num_blocks=m.get("num_blocks", 4),
        dropout=m.get("dropout", 0.3),
        spatial_input=cfg.get("image", {}).get("size", 128),
    )


if __name__ == "__main__":
    dummy = torch.randn(2, 3, 128, 128)
    net = build_model({"model": {"base_channels": 32, "num_blocks": 3}})
    print(net)
    print("out:", net(dummy).shape)
"""
Model 2: ResNet50, transfer learning from ImageNet weights.
Two-phase training strategy handled in train.py:
  Phase 1: base frozen, only new head trains
  Phase 2: last few residual blocks unfrozen, fine-tuned at a low LR
"""
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def get_model(num_classes: int = 2) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module):
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False


def unfreeze_last_blocks(model: nn.Module, n_blocks: int = 2):
    """Unfreezes the last `n_blocks` ResNet layer groups (layer3, layer4, ...)
    plus the classifier head, for the fine-tuning phase."""
    unfreeze_names = ["fc"]
    all_layer_groups = ["layer1", "layer2", "layer3", "layer4"]
    unfreeze_names += all_layer_groups[-n_blocks:]

    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in unfreeze_names):
            param.requires_grad = True

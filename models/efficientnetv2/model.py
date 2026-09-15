"""
Model 3: EfficientNetV2-S, transfer learning from ImageNet weights.
Same two-phase strategy as ResNet50 (warmup then fine-tune), so the two
transfer-learning models are directly comparable under identical rules.
"""
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


def get_model(num_classes: int = 2) -> nn.Module:
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module):
    for name, param in model.named_parameters():
        if not name.startswith("classifier."):
            param.requires_grad = False


def unfreeze_last_blocks(model: nn.Module, n_blocks: int = 2):
    """Unfreezes the last `n_blocks` feature stages plus the classifier head."""
    total_stages = len(model.features)
    unfreeze_stage_indices = set(range(total_stages - n_blocks, total_stages))

    for name, param in model.named_parameters():
        if name.startswith("classifier."):
            param.requires_grad = True
            continue
        if name.startswith("features."):
            stage_idx = int(name.split(".")[1])
            if stage_idx in unfreeze_stage_indices:
                param.requires_grad = True

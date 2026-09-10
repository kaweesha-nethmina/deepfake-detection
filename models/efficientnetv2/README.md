# EfficientNetV2 (Owner B)

Pretrained MBConv-scaling net — the "efficient ImageNet prior" baseline. Compared
against ResNet50 as an ablation of architecture efficiency vs accuracy under the
same data budget.

## Architecture

`timm.create_model("efficientnetv2_s", pretrained=True, num_classes=0)` with a
replacement head:

```
Dropout → Linear(features → 256) → ReLU → Linear(256 → 2)
```

Variants `efficientnetv2_m` / `_l` are available via `configs/efficientnetv2.yaml`
(`model.variant`) if compute allows.

## Hyperparameters

| Name | Value (default) | Config key |
|---|---|---|
| image size | 224 | `image.size` |
| variant | efficientnetv2_s | `model.variant` |
| learning rate | 1e-4 | `training.learning_rate` |
| batch size | 32 | `training.batch_size` |
| epochs | 35 | `training.epochs` |
| seed | 42 | `seed` |

## Run

```bash
python models/efficientnetv2/train.py -c configs/efficientnetv2.yaml
```

## Outputs

```
results/efficientnetv2/<run_name>/
├── checkpoints/best.pt
├── checkpoints/last.pt
└── metrics.csv
```

Same notebook ownership as ResNet50 (`03_resnet_efficientnet_experiments.ipynb`).
Kick-off with the ResNet50 results copied into that notebook so B owns one file
for both backbones.

## Note on timm versions

`timm` occasionally changes model keys / pretrained tags. Pin `timm==1.0.7`
(see `requirements.txt`) and re-run only after a reviewed bump — never silently.
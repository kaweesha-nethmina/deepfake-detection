# ResNet50 (Owner B)

Pretrained ImageNet backbone, full or progressive fine-tune — the "strong
ImageNet prior" baseline for the cross-generator comparison.

## Architecture

`torchvision.resnet50` (IMAGENET1K_V1 weights) with the FC head replaced by:

```
Dropout → Linear(2048 → 256) → ReLU → Linear(256 → 2)
```

`freeze_backbone: false` trains everything; `unfreeze_from_layer: N`
progressive-unfreezes from `layerN` upward (use for low-compute runs).

## Hyperparameters

| Name | Value (default) | Config key |
|---|---|---|
| image size | 224 | `image.size` |
| pretrained | IMAGENET1K_V1 | `model.weights` |
| learning rate | 1e-4 | `training.learning_rate` |
| batch size | 32 | `training.batch_size` |
| epochs | 35 | `training.epochs` |
| seed | 42 | `seed` |

High LR on a fully-fine-tuned pretrained net destroys the prior — keep ≤1e-4
for full fine-tune, ≤5e-5 for unfreeze-from-layer4.

## Run

```bash
python models/resnet50/train.py -c configs/resnet50.yaml
```

## Outputs

```
results/resnet50/<run_name>/
├── checkpoints/best.pt
├── checkpoints/last.pt
└── metrics.csv
```

Summarised experiments live in `notebooks/03_resnet_efficientnet_experiments.ipynb`,
owned by B. Final comparison tables (all models) are assembled by C into
`results/comparison/` — do **not** write there yourself.
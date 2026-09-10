# Custom CNN (Owner A)

A small, bespoke convolution network designed to test how much inductive bias
(end-to-end conv, moderate capacity) generalises across generators when trained
on a single deepfake family.

## Architecture

```
Conv stem (3x3 + BN + ReLU) * N blocks, each:
    Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU → MaxPool2x2
→ Global Average Pool
→ Dropout → Linear(num_classes)
```

Key decisions:
- **No pre-training** — evaluates the "learn from scratch on this domain only"
  baseline.
- **GAP + small head** keeps the param count low to fight generator-specific
  overfitting.
- Base channels / depth / dropout are read from `configs/custom_cnn.yaml`
  (`model:` block) — never hardcoded.

## Hyperparameters

| Name | Value (default) | Config key |
|---|---|---|
| image size | 128 | `image.size` |
| base channels | 32 | `model.base_channels` |
| blocks | 4 | `model.num_blocks` |
| dropout | 0.30 | `model.dropout` |
| learning rate | 3e-4 | `training.learning_rate` |
| batch size | 64 | `training.batch_size` |
| epochs | 80 | `training.epochs` |
| seed | 42 | `seed` |

## Run

```bash
python models/custom_cnn/train.py -c configs/custom_cnn.yaml
```

## Outputs

Everything lands in `results/custom_cnn/` (gitignored):

```
results/custom_cnn/<run_name>/
├── checkpoints/best.pt     # best model by val F1
├── checkpoints/last.pt     # last-epoch state + cfg
└── metrics.csv             # per-epoch train/val curves
```

Plotting the curves and recording val/test F1/AUC is done in
`notebooks/02_custom_cnn_experiments.ipynb` (this notebook is owned by A and is
the one place running experiments are summarised).
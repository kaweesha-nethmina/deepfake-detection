# ViT-B/16 + frequency-hybrid variant (Owner C)

The transformer baseline plus the cross-generator evaluation. This is the
member who owns the repo's central research question: *does a model trained on
one generator's fakes still detect fakes from unseen generators?*

## Architecture

`timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)` with:

```
LayerNorm → Dropout → Linear(768 → 2)
```

**Frequency-hybrid variant** (`model.use_frequency_hybrid: true`): a small
2D-DCT branch (block-wise, patch=16, high-frequency band mask) is concatenated
onto the class token before the head. This tests whether explicit high-frequency
signals (scaling artefacts) improve unseen-generator AUC.

```
features = concat( ViT_cls, FrequencyBranch(DCT-mag) )
head(features)
```

## Hyperparameters

| Name | Value (default) | Config key |
|---|---|---|
| image size | 224 | `image.size` |
| variant | vit_base_patch16_224 | `model.variant` |
| learning rate | 5e-5 | `training.learning_rate` |
| weight decay | 0.05 (AdamW) | `training.weight_decay` |
| warmup | 5 epochs | `training.warmup_epochs` |
| batch size | 16 | `training.batch_size` |
| epochs | 30 | `training.epochs` |
| seed | 42 | `seed` |

ViTs need a *lower* LR + warmup than CNNs; do not copy the conv-net settings.

## Run

```bash
python models/vit/train.py -c configs/vit.yaml
```

## Outputs

```
results/vit/<run_name>/
├── checkpoints/best.pt
├── checkpoints/last.pt
└── metrics.csv
```

Plus (the key deliverable) cross-generator results:

```
results/comparison/crossgen_<run_name>.csv     # per-generator ACC / F1 / AUC
```

The `crossgen` block in the yaml controls which generators are evaluated
(`data/cross_gen_test/<generator>/`). Per-generator ROC/confusion figures and
the final all-models comparison are produced in
`notebooks/04_vit_crossgen_evaluation.ipynb`. **This folder (`results/comparison/`)
is the only shared aggregation point and is written only by C** to avoid two
members clobbering the same table.
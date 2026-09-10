# Final Report — Draft (SE4050)

> Team members: A (custom CNN + data pipeline), B (ResNet50 / EfficientNetV2), C (ViT + cross-generator eval)
> If the team prefers Overleaf/Google Docs, replace this file with a one-line link.

## 1. Introduction & problem statement

(TODO) Cross-generator generalizability of deepfake detection …

## 2. Related work

- Dataset / generator families
- Prior deepfake-detection work (frequency-domain, transformer-based)

## 3. Methodology

- Data pipeline & split (train / val / test / cross-gen) — reference `data/README.md`
- Models: Custom CNN, ResNet50, EfficientNetV2, ViT-B/16 (+ frequency-hybrid variant)
- Training configs in `configs/*.yaml`; metrics contract in `src/metrics.py`

## 4. Experiments

| Model | Config | Val ACC | Val F1 | Val AUC | Test ACC | Unseen-Gen ACC/AUC |
|---|---|---|---|---|---|---|
| Custom CNN | `custom_cnn.yaml` | — | — | — | — | — |
| ResNet50 | `resnet50.yaml` | — | — | — | — | — |
| EfficientNetV2 | `efficientnetv2.yaml` | — | — | — | — | — |
| ViT-B/16 | `vit.yaml` | — | — | — | — | — |

(Fill from `results/*/metrics.csv` + `results/comparison/crossgen_*.csv` at the
final freeze. Only Owner C writes the aggregation in `results/comparison/`.)

## 5. Discussion

(TODO) Why does each model generalise (or not) to unseen generators?

## 6. Conclusions & future work

## 7. Individual contributions

See [`docs/contribution_log.md`](../docs/contribution_log.md).
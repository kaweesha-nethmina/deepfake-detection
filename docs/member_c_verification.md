# Verification Record

Date: 23 September 2026. Branch: feature/kalana.

## Checked locally

- `python -m pytest -q`: 35 tests passed.
- Offline forward passes: Custom CNN, ResNet50, EfficientNetV2, ViT-B/16 and ViT+FFT.
- Synthetic pipeline: audit -> training -> checkpoint freeze -> all-four evaluation -> plot regeneration.
- Interrupted/resumed CPU training produces identical model parameters to an uninterrupted run.
- Partial gradient accumulation matches full-batch updates for the tested fixture.
- The training command does not read primary/cross-generator image files.
- Missing/changed checkpoints, smoke checkpoints, incompatible configs and split leakage fail explicitly.
- Predictions from the CLI inference helper and evaluator agree; evaluation does not update weights.
- Notebook schema/cell compilation, command help, dependency consistency and Git whitespace checks pass.
- Generated confusion matrix/ROC and F1 chart layouts were visually inspected using synthetic fixture outputs.

These checks prove software behavior, not detector performance. Synthetic images,
labels, tiny training models and plot values are test fixtures, NOT assignment results.
Full-sized architecture tests use random weights and no downloaded pretrained weights.

## Local environment

Python 3.12.14, macOS ARM64; tests used CPU. Primary installed versions:
torch 2.14.0, torchvision 0.29.0, timm 1.0.30, numpy 2.5.3, pandas 3.0.6,
Pillow 12.3.0, PyYAML 6.0.3, scikit-learn 1.9.1, matplotlib 3.11.2,
pytest 9.1.1 and nbformat 5.11.1. Each real training run records its own environment.
These Mac versions are not a claim that a Colab/Kaggle CUDA environment was tested.

## Not completed / external dependencies

- No actual Wish images or Member A manifest are present locally.
- No pretrained-weight download, real-data smoke run, full training or CUDA AMP run has been performed.
- No real checkpoints, final metrics, failure images, completed report PDF or final submission exist yet.
- The requested dev merge was declined. Newer dev model interfaces have adapter coverage, but actual
  post-merge checkpoints and the Streamlit GUI are not integrated/verified on this branch.
- A/B must provide protocol-compatible artifacts or run the common manifest trainer; histories and
  preprocessing from different training protocols must not be silently mixed into the final comparison.
- Turnitin, team details, video upload and Gradescope/CourseWeb submission require team action.

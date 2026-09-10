# Deepfake Cross-Generator Detection

**Supervised Deep Learning project (SE4050 assignment)**

Cross-generator generalizable deepfake face detection: we train on one deepfake
generator family and evaluate whether the model generalizes to faces produced by
*unseen* generators. Four architectures are compared:

| Model | Hardware-friendly variant | Owner |
|---|---|---|
| Custom CNN | Small bespoke classifier, hypothesis on inductive bias | A |
| ResNet50 | Pretrained ImageNet backbone, full/partial fine-tune | B |
| EfficientNetV2 | Pretrained backbone, MBConv scaling | B |
| ViT-B/16 | Pretrained visual transformer | C |

## Team

| Member | Role | Primary files |
|---|---|---|
| **A** | Data pipeline, preprocessing, EDA, Custom CNN | `src/*`, `notebooks/01_eda.ipynb`, `notebooks/02_custom_cnn_experiments.ipynb`, `models/custom_cnn/` |
| **B** | ResNet50, EfficientNetV2 | `notebooks/03_resnet_efficientnet_experiments.ipynb`, `models/resnet50/`, `models/efficientnetv2/` |
| **C** | ViT, frequency-hybrid variant, cross-generator evaluation | `notebooks/04_vit_crossgen_evaluation.ipynb`, `models/vit/` |

## Repository layout

```
├── configs/                 # YAML configs (hyperparameters, paths, seeds)
├── data/                    # raw/, processed/, cross_gen_test/ — gitignored
├── src/                     # shared module (imported by all members)
├── models/<model>/          # model.py + train.py + README per architecture
├── notebooks/               # one notebook owner per file
├── results/<model>/         # per-model outputs, gitignored contents
├── reports/                 # final report draft
├── docs/contribution_log.md # weekly who-did-what log (viva support)
```

### Working rules (read before you commit)

1. **One owner per file.** Each member works inside their own `models/<name>/`
   and notebook. Never edit a file owned by someone else.
2. **Shared code lives in `src/`** — imported, not copy-pasted. Changes there must
   be *additive and backward-compatible*. A breaking change must be discussed and
   flagged, never silently merged.
3. **No data / checkpoints / images in git.** Everything binary lives in
   gitignored folders. Use the download scripts / `data/README.md` to reproduce.
4. **No heavy notebook outputs.** Strip outputs before committing
   (`Edit → Clear All Outputs`, or a `jupyter nbconvert --clear-output` hook).
5. **Config over hardcode.** Never bake a path, LR, seed, or batch size into a
   script — put it in `configs/<model>.yaml`.
6. **Ask before touching shared code.** Coordinate on `docs/contribution_log.md`.

## Setup

```bash
# 1) Create + activate environment (recommend Python 3.10+)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2) Install dependencies (adjust pinned versions to your hardware)
pip install -r requirements.txt

# 3) Fetch the datasets (never commit binaries)
#    Follow data/README.md or run the download scripts.
```

Import the shared module with (from the repo root):

```python
import sys; sys.path.insert(0, ".")
from src.config import ...   # after setup, or:
from src import utils
```

## How to run each model

All trainers share the same contract: `python models/<name>/train.py -c configs/<name>.yaml`.
Each writes outputs to `results/<name>/` and print a summary on completion.

```bash
python models/custom_cnn/train.py      -c configs/custom_cnn.yaml
python models/resnet50/train.py        -c configs/resnet50.yaml
python models/efficientnetv2/train.py  -c configs/efficientnetv2.yaml
python models/vit/train.py             -c configs/vit.yaml
```

Cross-generator evaluation scripts live under `models/vit/` (Owner C) and write to
`results/comparison/` the final comparison tables/plots used in the report.

## Results summary

| Model | Val ACC | Val AUC | Test ACC | Unseen-gen ACC | Notes |
|---|---|---|---|---|---|
| Custom CNN | — | — | — | — | — |
| ResNet50 | — | — | — | — | — |
| EfficientNetV2 | — | — | — | — | — |
| ViT-B/16 | — | — | — | — | — |

Fill this table at the end of the project (do not commit live updates mid-training —
keep it for the final freeze).

## References

- Dataset links and citations: [`data/README.md`](data/README.md).
- Contribution log (viva / individual-contribution requirement): [`docs/contribution_log.md`](docs/contribution_log.md).
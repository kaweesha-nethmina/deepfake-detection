# Your Folder Guide

Each member has their **own folder**. You only work inside yours. This guide explains it model by model.

## The simple idea

```
Your folder  →  you edit here forever
Other folders → you never touch
```

| Member | Your model folder | Your notebook | Your config |
|---|---|---|---|
| A | `models/custom_cnn/` | `01_eda.ipynb`, `02_custom_cnn_experiments.ipynb` | `configs/custom_cnn.yaml` |
| B | `models/resnet50/` + `models/efficientnetv2/` | `03_resnet_efficientnet_experiments.ipynb` | `configs/resnet50.yaml`, `configs/efficientnetv2.yaml` |
| C | `models/vit/` | `04_vit_crossgen_evaluation.ipynb` | `configs/vit.yaml` |

---

## For Member A — Custom CNN

**Your files:** everything inside `models/custom_cnn/` and both notebooks

What each file is for:

| File | What it does |
|---|---|
| `model.py` | Your neural network design. Change the layers here. |
| `train.py` | Runs the training. You usually **don't** touch this — settings live in the config. |
| `README.md` | Your notes about the network and how to run it. |

**How you work:**

1. Edit `model.py` to change your network.
2. Edit `configs/custom_cnn.yaml` to change learning rate, batch size, epochs, seed.
3. Run it:
   ```bash
   python models/custom_cnn/train.py -c configs/custom_cnn.yaml
   ```
4. Results appear in `results/custom_cnn/<run_name>/` — your checkpoint and `metrics.csv`.
5. Load those results in `02_custom_cnn_experiments.ipynb` to make plots.

---

## For Member B — ResNet50 + EfficientNetV2

**Your files:** everything inside `models/resnet50/`, `models/efficientnetv2/`, and notebook `03`

Each model folder is the same shape:

| File | What it does |
|---|---|
| `model.py` | Uses the pretrained backbone. You adjust the head/freezing here. |
| `train.py` | Runs training, reads settings from config. Usually untouched. |
| `README.md` | Your notes for that model. |

**How you work (same for both models):**

1. Edit your configs to pick pretrained variant / freeze settings / hyperparameters.
2. Run each model separately:
   ```bash
   python models/resnet50/train.py -c configs/resnet50.yaml
   python models/efficientnetv2/train.py -c configs/efficientnetv2.yaml
   ```
3. Results go to `results/resnet50/<run_name>/` and `results/efficientnetv2/<run_name>/`.
4. Compare both in notebook `03` (your one notebook covers both).

---

## For Member C — ViT + cross-generator

**Your files:** everything inside `models/vit/` and notebook `04`

| File | What it does |
|---|---|
| `model.py` | ViT backbone, plus your frequency-hybrid variant (turned on in config). |
| `train.py` | Trains, then runs the cross-generator evaluation at the end. |
| `README.md` | Your notes. |

**How you work:**

1. Turn the frequency branch on/off in `configs/vit.yaml` → `model.use_frequency_hybrid`.
2. Run it:
   ```bash
   python models/vit/train.py -c configs/vit.yaml
   ```
3. Training results go to `results/vit/<run_name>/`.
4. Cross-generator results go to `results/comparison/crossgen_<run_name>.csv` — the only shared results folder, and **you** are the only one who writes to it.
5. Build the final all-model comparison tables in notebook `04` (reading the other members' folders read-only).

---

## What everyone shares — but carefully

- **`src/`** has ready-made helpers. Import them:
  ```python
  from src import set_seed, load_config, evaluate_binary, build_dataset_paths
  ```
- If `src/` needs a new function: **add it, don't rewrite what's there**, and tell the team first.
- **`data/`** is only for downloaded datasets — don't hand-edit the files inside.
- **`docs/contribution_log.md`** — add your own weekly row at the bottom. Edit only yours.

## Golden rules

1. Change settings in **your config** — never hardcode them in code.
2. Give each experiment a new `run_name` so runs don't overwrite each other.
3. Keep the same `seed` so you can compare fairly.
4. Not sure? Ask in the team chat first.
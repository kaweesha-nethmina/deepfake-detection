# Model 4 — ViT-B/16 (+ Frequency-Hybrid) & Cross-Generator Harness — Owner: Member C

Implements the strict experimental protocol: train/select on **StyleGAN** only,
evaluate once on the **StyleGAN primary test set**, then — without touching the
model again — evaluate on **Stable Diffusion** (never seen before). Label
convention throughout: **REAL = 0, FAKE = 1** (fake is the positive class).

## Pipeline order

```bash
# 1) Classify the raw download, split StyleGAN 70/15/15, hold out Stable Diffusion,
#    run leakage checks, and write the dataset summary table.
python models/vit/prepare_data.py -c configs/dataset_prep.yaml

# 2) Experiment A — ViT-B/16, no frequency branch
python models/vit/train.py -c configs/vit.yaml

# 3) Experiment B — ViT-B/16 + frequency-hybrid branch (identical config otherwise)
python models/vit/train.py -c configs/vit_freq_hybrid.yaml

# 4) Score all 4 team models (+ both my experiments) with identical evaluation code
python models/vit/evaluate_crossgen.py -c configs/crossgen_harness.yaml

# 5) Build the report figures
jupyter notebook notebooks/04_vit_crossgen_evaluation.ipynb
```

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Scans the raw Kaggle download, classifies every file as real/fake + generator, reserves real images for the cross-gen test **before** splitting (zero leakage by construction), stratified-splits the StyleGAN pool 70/15/15, runs leakage checks, materializes `data/processed/{train,val,test}` and `data/cross_gen_test`, writes the dataset verification + leakage reports and the summary table. |
| `common.py` | `set_seed`, `load_config`, `build_dataset_paths`, `DeepfakeImageDataset`, augmentation transforms. Tries the shared `src/` module first, falls back to a local implementation. Only imports torch lazily where actually needed, so `prepare_data.py` runs even without a working torch install. |
| `model.py` | `ViTDeepfakeDetector` — ViT-B/16 (`timm`) + optional `FrequencyBranch` (2D FFT → small CNN), toggled by `model.use_frequency_hybrid`. |
| `train.py` | Trains, selects on val, evaluates primary test once, then cross-gen once (no model changes in between). Writes `metrics.csv` (incl. `accuracy_drop` and `relative_accuracy_drop_pct`) and the shared comparison row. |
| `evaluate_crossgen.py` | The **cross-generator evaluation harness** — identical evaluation code scores any teammate's checkpoint, so the 4-model comparison isn't confounded by inconsistent eval code. |
| `README.md` | This file. |

## Dataset protocol (`configs/dataset_prep.yaml`)

- **Training generator**: StyleGAN. **Cross-generator**: Stable Diffusion.
- StyleGAN pool (real + StyleGAN-fake) → stratified 70/15/15 → train/val/primary-test.
- Real images needed for the Stable Diffusion cross-gen test (so it has both
  classes to score) are reserved **before** the StyleGAN split, from a
  disjoint random subset — this is what guarantees zero leakage rather than
  relying on a check to catch it after the fact.
- Stable Diffusion images never enter training, validation, tuning, threshold
  selection, or architecture/model selection — only the final cross-gen eval.
- **Nothing is hardcoded**: `label_rules` / `generator_rules` in the config
  are keyword matches against the actual downloaded file paths, and every
  count in the verification report and summary table is measured, not
  assumed. **Verify the keywords match your actual folder names after
  downloading** — the exact Kaggle layout wasn't independently confirmed
  when this was written. The script prints any file it can't classify so
  nothing is silently dropped.
- Supplementary/"other" fake images (matching neither keyword set) are
  excluded from both pools by default, per the protocol's instruction not to
  mix in supplementary datasets without explicit configuration.

**Tested**: `prepare_data.py` was run end-to-end against a synthetic dataset
with this same folder shape (`real/<source>/*.jpg`, `fake/stylegan/*.jpg`,
`fake/stable_diffusion/*.jpg`) to confirm the split, leakage detection, and
file materialization all produce consistent, matching counts.

### Leakage checks

Reported (never silently fixed) in `results/vit/dataset_leakage_report.json`:
unique paths per split, duplicate filenames within a split, and content-hash
overlap between every pair of {train, val, test, cross_gen} — this also
directly verifies requirements 4–9 from the protocol (no train/val/test
overlap, and Stable Diffusion absent from train/val/test).

### Dataset summary table

`results/comparison/dataset_summary_table.csv` — `Dataset | Generator | Role | Real | Fake | Usage`,
built from measured counts only.

## Two experiments (fair ablation)

| | Experiment A | Experiment B |
|---|---|---|
| Config | `configs/vit.yaml` | `configs/vit_freq_hybrid.yaml` |
| `run_name` | `vit_b16_baseline_v1` | `vit_b16_freq_hybrid_v1` |
| `model.use_frequency_hybrid` | `false` | `true` |
| Everything else | identical | identical |

Both use the same StyleGAN train/val/test split, the same Stable Diffusion
cross-gen set, the same seed, the same threshold, and the same evaluation
code — so the only thing that can explain a difference in results is the
frequency-hybrid branch itself.

## Metrics (per model, in `metrics.csv` / the harness's `final_comparison_table.csv`)

- Accuracy, precision, recall, F1, ROC-AUC — on both the primary test set and
  the cross-generator test set.
- `accuracy_drop = test_acc - cross_gen_acc`
- `relative_accuracy_drop_pct = accuracy_drop / test_acc * 100`
- Training time, inference time/image, parameter count.

**Per the protocol: the highest primary-test accuracy does not automatically
mean the best model** — the research question is generalization, so
`accuracy_drop` / `relative_accuracy_drop_pct` are the numbers to lead with
in the report's critical analysis, not raw primary accuracy.

## Dependencies

Add to the shared `requirements.txt` if not already there: `torch`,
`torchvision`, `timm`, `scikit-learn`, `pandas`, `numpy`, `pyyaml`, `pillow`,
`matplotlib`, `seaborn`.

## Notebook

`notebooks/04_vit_crossgen_evaluation.ipynb` reads the dataset summary table,
my `metrics.csv`, and the other members' `metrics.csv` files (read-only), and
builds: the dataset summary, the final comparison table, absolute and
relative accuracy-drop charts, ROC overlays, confusion matrix grids,
efficiency-vs-accuracy, and the Experiment A vs. B ablation comparison.

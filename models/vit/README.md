# Member C: Wish Dataset Workflow

Run every command from the repository root. This implementation uses only
`wish096/realvsfake-81k-by-wish`, with Member A's fixed assignments. It never
creates a replacement random split. No measured detector results are included.

## 1. Environment and offline tests

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-member-c.txt
python -m pytest -q
```

Tests use generated images and randomly initialized networks. They do NOT prove
deepfake accuracy. On Colab/Kaggle select a GPU runtime first, retain its matched
CUDA torch/torchvision installation, and install the remaining requirements.
Training records the actual package versions, device and CUDA version for each run.

## 2. Member A's handoff

### Kaggle notebook dataset loading

`notebooks/04_vit_crossgen_evaluation.ipynb` now connects the exact dataset with
`kagglehub.dataset_download("wish096/realvsfake-81k-by-wish")`. Its first cell works
before repository setup, detects direct/nested Real/Fake directories, and prints
the resolved path. Manually attaching the same dataset is also supported through
KaggleHub's resource handling. Enable Internet for initial package/model downloads.
No credentials are embedded. See the [official KaggleHub API](https://github.com/Kaggle/kagglehub#download-dataset).

Set `DATASET_VERSION` to Member A's integer version and `MEMBER_A_CSV` to the shared
manifest path, then rerun the first cell. Latest-version loading is for inspection
only; auditing/training require the pinned version. The manifest describes splits
of this SAME dataset, not an additional image dataset.

If A already has separate CSV files, set `MEMBER_A_SPLIT_CSVS` instead, with keys
`train`, `val`, `test`, `cross_gen` (and optional `excluded`). The notebook combines
them without reassigning rows. Each file needs `filepath,label`; missing source
metadata is inferred from documented filename prefixes, and existing source labels
are validated. Set `MEMBER_A_PATH_PREFIX` only to an exact shared prefix to strip
from old absolute paths. Three-way splits that mix Stable Diffusion into training
or lack cross-generator real controls must be corrected by A, not silently repaired.

Upload/extract this repository to `/kaggle/working/deepfake-detection`, or use
`REPO_OVERRIDE`. The local implementation must be uploaded or pushed before Kaggle
can use it. The notebook generates runtime YAML configs automatically under
`/kaggle/working/member_c` (or `OUTPUT_OVERRIDE`); do not manually edit the tracked
team YAML paths for this notebook workflow. Save the output folder before ending
the session. This connection has local mock coverage, not a live Kaggle/GPU run.

Obtain the images, exact dataset version, fixed splits, preprocessing details,
and any available identity groups. Labels alone are not enough. A shared cloud
dataset mount is sufficient; you do not need another full copy on your Mac.

Use [the data contract](../../data/README.md). A must export one CSV with columns:

```text
filepath,label,source,split,identity
Real/RFF (1).jpg,0,FFHQ,train,
Fake/FSG (1).jpg,1,StyleGAN,train,
```

The above two rows illustrate syntax only, not a valid complete manifest.
The actual manifest must contain both classes in `train`, `val`, `test`, and
`cross_gen`. Paths are relative to the configured dataset root.

```bash
python -m models.vit.prepare_data --manifest /path/to/member_a.csv --data-root /path/to/RealVsFake --dataset-version VERSION_FROM_A --output data/manifests/wish_v1
```

The audit checks decoding, SHA-256, duplicate pixels, source/label conflicts,
generator isolation, class presence and supplied identities. Its outputs are
`manifest.csv`, `audit.json`, and `near_duplicates.csv`. dHash is a candidate
screen, not proof that all near-duplicates or identities were found.

If near-duplicates are reported, A reviews them. Actual overlaps require a
corrected team-wide manifest and a new audit directory. For false positives only,
record a review JSON with `manifest_sha256`, `reviewer`, `rationale`, and
`decision: "false_positives_only"`; set `data.near_duplicate_review` in each training
config and `near_duplicate_review` in the evaluation config. Do not bypass review.

For the terminal workflow, set `data.root` in all training configs and `data_root` in the harness to the
actual mount. Set their manifest paths to the SAME audited CSV. Training verifies
train/validation hashes only; the final evaluator verifies every split.

## 3. Training-only smoke test

```bash
python -m models.vit.train -c configs/vit.yaml --smoke
```

This uses at most 32 existing training and 32 existing validation examples for
one epoch, saves to a separate `_smoke` run, and never predicts either test set.
The first ViT run downloads public pretrained weights. Smoke checkpoints are
rejected by final evaluation. Change run_name before repeating a smoke run.

## 4. Full core experiments

```bash
python -m models.vit.train -c configs/wish/custom_cnn.yaml
python -m models.vit.train -c configs/wish/resnet50.yaml
python -m models.vit.train -c configs/wish/efficientnetv2.yaml
python -m models.vit.train -c configs/vit.yaml
```

The common runner imports each owner's architecture; it does not replace it.
These are the supported Wish training commands, not the legacy standalone
trainers with independently generated splits. A/B can run their models on their
own GPU sessions using the same manifest/configs and return the entire run folder.

All runs use seed 42, 224x224 RGB input, effective batch 32, at most 30 epochs,
and best validation loss with patience 6. CNN warmup/fine-tuning histories stay
in one run. Checkpoint-specific normalization is saved. CUDA uses AMP; CPU/MPS
use float32. Reduce `data.batch_size` to 8/4/2 if memory is limited; accumulation
maintains effective batch size. Do not silently reduce the full training dataset.

```bash
python -m models.vit.train -c configs/vit.yaml --resume results/vit/vit_b16_wish_s42_v1/checkpoints/last.pt
```

Resume requires unchanged config/manifest and the original run directory, including
best_model.pt. It restores optimizer, scheduler, scaler, epoch, RNG and history.
Archive runs to persistent Drive/Kaggle storage between GPU sessions. CPU and
GPU floating-point results need not be bit-identical; record runtime differences.

Every run stores `run_config_used.json`, `environment.json`, `training_history.csv`,
`learning_curves.png`, best weights and resumable last state. No test metrics are
produced by training. Use a NEW run_name for a new experiment.

## 5. Freeze and final evaluation

Edit `configs/crossgen_harness.yaml` to point to all four returned checkpoints.
Checkpoints must identify the shared manifest and validation-only selection.
Legacy plain state dictionaries require an owner-verified `<checkpoint>.json`
sidecar with the same metadata schema; do not invent training provenance.

```bash
python -m models.vit.evaluate_crossgen freeze -c configs/crossgen_harness.yaml
python -m models.vit.evaluate_crossgen run --device cuda
```

Freeze hashes all weights/configs and requires all four models. Evaluation uses
identical ordered test samples for every model. Missing weights, stale hashes or
smoke runs fail instead of creating a misleading partial final comparison.

Final evidence appears in `results/comparison/final/`: predictions, comparison
table, model results JSON, confusion matrices/ROC plots, F1 comparison, and ranked
false-positive/false-negative cases. `status.json` must say `complete`.
Latency is measured after warmup with device synchronization; single-image
forward latency excludes loading, cropping, transfers and preprocessing. Compare
models on the same hardware. Parameter counts include total and trainable values.

```bash
python -m models.vit.evaluate_crossgen plots --predictions results/comparison/final/predictions.csv --output results/comparison/regenerated
```

This regenerates metrics/figures without inference. Never change hyperparameters
after looking at final results and still describe that test as untouched.
Optional repeat seeds or FFT must be declared before the freeze, not selected
because they look better on Stable Diffusion. The FFT configuration is excluded
from the default four-model harness.

## 6. Predict a cropped face

```bash
python -m models.vit.predict --config configs/vit.yaml --checkpoint results/vit/vit_b16_wish_s42_v1/checkpoints/best_model.pt --image /path/to/cropped_face.jpg
```

The CLI and notebook demo use the same probability conversion and preprocessing
as evaluation. Scores are experimental, not calibrated forensic probabilities.
Do not use uncropped group photographs or infer whether a person is trustworthy.

## Integration status

Work is on `feature/kalana`. The requested `origin/dev` merge was not approved;
it has NOT been performed. The main branch's Streamlit demo is not present here,
so GUI integration is pending that merge. `runtime.build_model` supports both
`build_model(cfg)` and `get_model(num_classes=2)`, but different architecture heads
require matching configs/checkpoints and must not be silently interchanged.
The newer dev trainers must have intermediate test evaluation disabled before use
for this protocol. Use the new manifest runner meanwhile.

The only teammate architecture fix in this contribution is constructing pretrained
ResNet with its original classifier size before replacing the head; torchvision
rejects pretrained weights combined with an incompatible initial class count.

## Sources

- [Dataset owner](https://www.kaggle.com/datasets/wish096/realvsfake-81k-by-wish)
- [ViT checkpoint](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k)
- [ViT paper](https://arxiv.org/abs/2010.11929)
- [Training ViT / AugReg](https://arxiv.org/abs/2106.10270)

Model performance, GPU memory requirements and dataset counts must be measured
on the actual handoff. No real images or trained checkpoints are currently included.

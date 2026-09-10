# Data

**Everything under `data/` is gitignored.** Do not commit datasets, extracted
frames, or preprocessed crops. Reproduce them from the instructions below so
every team member shares an identical layout without heavy binaries in history.

## Directory layout (convention — do not change without a team-wide note)

```
data/
├── raw/                  # downloaded sources, untouched
│   ├── real/             #       +-----------+-----------------+
│   └── fake/<generator>/ # e.g.  | StyleGAN2  | ProgressiveGAN  |
├── processed/            # face-cropped + resized, ready for training
└── cross_gen_test/       # held-out generators used by Owner C's eval
    └── <unseen-generator>/
```

Production rule: **do not delete/reorder files between splits while training is
running.** Use `src.data_pipeline.balanced_split(seed=same)` so everyone computes
the same partition.

## Datasets (pick per hardware budget; cite what you use)

| Dataset | Generator(s) | Fakes | Reals | Notes / link |
|---|---|---|---|---|
| **CelebA-HQ / StyleGAN2** | StyleGAN2 | 5 k | 5 k | High-res; good *closed* train set. |
| **FFHQ (real split)** | — | — | 70 k | Clean real faces; pairs well with StyleGAN2 fakes. |
| **DeepFakeFaceForensics (DFFF)** | IDs/GANs/LatentDiffusion | 100 k | 100 k | Large cross-generator diversity — best for the *cross-gen* question. |
| **FaceForensics++ (FF++)** | Deepfakes / FaceSwap / ... | 4 k | 1 k | Classical manipulation; useful as *unseen* test set. |
| **CIFAKE** | GenImage (LDM) | 120 k | 120 k | Easy to download, small images, fine for Custom CNN (Owner A). |

> Recommended pairing for the SE4050 experiment:
> * **Train / val** on StyleGAN2 fakes + FFHQ(real) from **CelebA-HQ**.
> * **In-distribution test** from the same train distribution.
> * **Cross-generator test** (`data/cross_gen_test/`) on **ProGAN**, **StyleGAN3**
>   and **Latent Diffusion (LDM)** fakes that were never seen during training.

## How to download

Prefer scripted downloads. Add one script per dataset in `scripts/download_*`
or a notebook, and record the *exact* URL + checksum here so anyone can re-pull:

- **CelebA-HQ (1024×1024)**: `https://drive.google.com/...` (authorization needed)
  - mirror via `kagglehub`/`huggingface_hub` when available.
- **FFHQ**: click-through licence, then `https://github.com/NVlabs/ffhq-dataset` →
  `ffhq-dataset-v2` images `00000-69999`.
- **DFFF**: `https://huggingface.co/datasets/...` (see dataset card).
- **FF++**: `https://github.com/ondyari/FaceForensics` (c20 / raw).

If a mirror requires a Kaggle login, put the *command* here rather than copying
data, e.g.:

```bash
kaggle datasets download -d $OWNER/$DATASET -p data/raw && unzip -q \
  "$(find data/raw -name '*.zip' | head -1)" -d data/raw
```

If you use an **open-license U-Net/quantized** mirror for CIFAKE, note the
checksum after downloading in this file. Keep the *layer* of the data (raw vs
processed) explicit so no one uploads a 5 GB zip to GitHub by accident.

## Post-download steps

1. Convert everything to RGB and, if needed, split by generator:
   ```python
   from src.data_pipeline import build_dataset_paths
   build_dataset_paths("data/raw")          # ensures processed/, cross_gen_test/
   ```
2. Owner A: add `scripts/make_processed.py` (additive, in your branch) that reads
   `raw/` and writes crops into `processed/`. Do **not** overwrite other members'
   copies of raw data.
3. Sanity check counts and generator labels before agreeing on a frozen test set.

## Citations

When you include one of these datasets in the final report, add its BibTeX here
so the report section writes itself:

```bibtex
@inproceedings{karras2020stylegan,
  title={Analyzing and Improving the Image Quality of StyleGAN},
  author={Karras, Tero and Laine, Samuli and Aittala, Miika and ...},
  booktitle={CVPR}, year={2020}}
```

*(Add entries as you finalize which datasets made the cut.)*
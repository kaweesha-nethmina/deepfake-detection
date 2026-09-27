# RUN.md — Setup, Run & Troubleshooting

Read this **before** running anything. Every error documented here was actually hit
while bringing this project up on a clean machine, so if you follow the steps in
order you should not hit them.

---

## 1. TL;DR (already have `.venv311`?)

```bash
source .venv311/bin/activate
streamlit run demo_app.py
```

If `.venv311` does not exist on your machine, do the full setup in §2.

---

## 2. Prerequisites

| Requirement | Value | Why |
|---|---|---|
| **Python** | **3.11** (3.10 also OK) | `requirements.txt` targets 3.10/3.11. This is the version the project is verified on. |
| RAM | 8 GB+ | ViT-B/16 checkpoint is 343 MB; all four models ≈ 525 MB of weights. |
| Disk | ~5 GB | Dataset is gitignored and must be downloaded separately. |
| GPU | Optional | Runs on CPU. Apple Silicon uses MPS automatically. |

> **Do not use Python 3.12/3.13/3.14 for this project.** Several pinned
> dependencies (notably `facenet-pytorch` and some `torch`/`timm` combinations)
> have no wheels for newer interpreters, and `pip` will either fail or silently
> resolve to a different build. `requirements.txt` explicitly states
> *"Tested target: Python 3.10 / 3.11"*.

Check your version:

```bash
python3.11 --version      # expect: Python 3.11.x
```

---

## 3. Environment setup

### 3.1 Create the virtual environment

```bash
cd /path/to/Project

python3.11 -m venv .venv311
source .venv311/bin/activate          # Windows: .venv311\Scripts\activate
python -m pip install --upgrade pip
```

> **Naming matters.** This project previously had **two** venvs — `.venv`
> (Python 3.14, had Streamlit but *not* torch) and `.venv311` (Python 3.11,
> the complete one). That mismatch is the single most common cause of the
> errors in §7. If you are cloning fresh you only need `.venv311`. If both
> exist on your machine, delete the incomplete one:
>
> ```bash
> rm -rf .venv          # only if it is the Python 3.14 stub
> ```

### 3.2 Install PyTorch first

`requirements.txt` and `requirements-member-c.txt` both say to do this
separately, because the correct build depends on your hardware.

```bash
# Apple Silicon (MPS)
pip install torch torchvision

# NVIDIA CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# NVIDIA CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3.3 Install the remaining dependencies

```bash
pip install -r requirements.txt
```

### 3.4 Install the packages the requirements files omit

**This step is not optional — without it the demo app will not run.**

`demo_app.py` imports `streamlit`, `cv2` and `facenet_pytorch`, but **none of the
three appear in any `requirements*.txt` file**. This is a known gap in the repo
(issue: add them to `requirements.txt`). Until then, install them manually:

```bash
pip install streamlit opencv-python facenet-pytorch
```

| Package | Used by | Consequence if missing |
|---|---|---|
| `streamlit` | `demo_app.py` | App will not start at all. |
| `opencv-python` | `demo_app.py` face detection | Face detection falls back to centre-crop only. |
| `facenet-pytorch` | `demo_app.py` MTCNN detection | Silently degrades to OpenCV Haar cascade. |
| `tqdm` | training loops | Only in `requirements.txt`, not `requirements-member-c.txt`. |

> If you use `requirements-member-c.txt` instead of `requirements.txt`, you must
> also add `opencv-python` and `tqdm` — they are missing from that file too.

---

## 4. Verify the installation

Run this **before** reporting a bug. It exercises every model, checkpoint and
Grad-CAM path end to end and needs no dataset.

```bash
python -c "
import warnings; warnings.filterwarnings('ignore')
import torch, demo_app as d
print('torch', torch.__version__, '| MPS', torch.backends.mps.is_available())
for name, info in d.MODEL_REGISTRY.items():
    m, dev, _ = d.load_model(name)
    out = m(torch.randn(1,3,224,224).to(dev))
    print(f'  OK {name:24} {info[\"gradcam_mode\"]:5} out={tuple(out.shape)}')
"
```

Expected output (device will read `cpu` on non-Apple hardware):

```
torch 2.14.0 | MPS True
  OK Custom CNN             conv  out=(1, 2)
  OK ResNet50               conv  out=(1, 2)
  OK EfficientNetV2         conv  out=(1, 2)
  OK ViT (frequency-hybrid) token out=(1,)
```

If all four print `OK`, your environment is correct. Note ViT outputs `(1,)` not
`(1, 2)` — that is expected, it uses a single logit.

---

## 5. Running the apps

Always confirm the venv is active first (`source .venv311/bin/activate`).

### 5.1 Streamlit demo app (primary)

```bash
streamlit run demo_app.py
```

Opens at <http://localhost:8501>. Upload an image; the app crops the face, runs all
selected models, and renders Grad-CAM overlays.

To pick the port or run without opening a browser:

```bash
streamlit run demo_app.py --server.port 8501
streamlit run demo_app.py --server.headless true
```

### 5.2 Gradio app (optional)

```bash
pip install -r requirements-gradio.txt     # installs gradio + headless opencv
python gradio_app.py
```

> `requirements-gradio.txt` pins `opencv-python-headless`, which **conflicts** with
> the `opencv-python` installed in §3.4. Install one or the other. If Gradio fails
> to start with an OpenCV import error, run
> `pip install --force-reinstall opencv-python` and use the Streamlit app instead.

---

## 6. Receiving `results/` (manual handoff)

> The demo app needs trained checkpoints. **You do not need to train anything** —
> the `results/` folder is sent to you directly.
>
> The dataset is also not in git: `data/raw/`, `data/processed/` and
> `data/manifests/` are gitignored. The demo app does not need the dataset
> either, only the checkpoints.

`results/` is **gitignored** (`.gitignore` rules 28 and 35 ignore
`results/**/*.json` and `*.pt`), so checkpoints and metrics are *not* delivered by
`git pull`. Someone sends the folder out-of-band, and everyone else has to place it
correctly — the paths must match exactly, because the demo app resolves them by
name.

### 6.1 What you should have received

```bash
du -sh results/          # ~515 MB
```

| Path | Size | SHA-256 (first 16) | Needed by |
|---|---:|---|---|
| `results/custom_cnn/best_model.pt` | 4 MB | `f071f317283acf6e` | demo app, evaluation |
| `results/resnet50/best_model.pt` | 91 MB | `c8c29cf87c136631` | demo app, evaluation |
| `results/efficientnetv2/best_model.pt` | 79 MB | `b666ccb0bb05c964` | demo app, evaluation |
| `results/vit/best_model.pt` | 327 MB | `be78ec4969aaf73d` | demo app, evaluation |
| `results/<model>/results.json` | few KB | — | metrics / report tables |

> The ViT checkpoint alone is 327 MB and dominates the transfer. If you only need
> the demo app for three models, send those three.

### 6.2 If you are sending the folder

Zip preserves the directory structure and avoids macOS/Windows dropping
permissions mid-transfer:

```bash
cd /path/to/Project
zip -r results_v1.zip results -x "results/**/*.log" -x "results/**/*.tmp"
# -> results_v1.zip  (~500 MB, expect a few minutes)
shasum -a 256 results_v1.zip          # send this hash along with the file
```

Use the team's approved storage (Drive / OneDrive / a shared course link). Do
**not** commit the zip.

### 6.3 Unpacking it (everyone)

Unzip at the **repository root** — not inside `results/`, and not in a
subfolder. Confirm the layout before doing anything else:

```bash
cd /path/to/Project
unzip /path/to/results_v1.zip

# must print four lines, no errors
ls -la results/*/best_model.pt
```

Expected:

```
results/custom_cnn/best_model.pt
results/resnet50/best_model.pt
results/efficientnetv2/best_model.pt
results/vit/best_model.pt
```

A nested `results/results/vit/...` or a `Project/Project/...` layout is the most
common mistake, and it fails silently — the demo app just reports a missing
checkpoint instead of erroring.

### 6.4 Verify on receipt

```bash
# 1) checksums match what the sender reported
shasum -a 256 results/*/best_model.pt

# 2) every model actually builds and runs (the §4 script)
python -c "
import warnings; warnings.filterwarnings('ignore')
import torch, demo_app as d
for name in d.MODEL_REGISTRY:
    m, dev, _ = d.load_model(name)
    print('OK', name, tuple(m(torch.randn(1,3,224,224).to(dev)).shape))
"
```

If a hash differs, the transfer was truncated or corrupted — re-request it. Do not
run experiments on unverified weights.

### 6.5 Rules for the handoff

1. **Version the bundle** (`results_v1.zip`, `results_v2.zip`, ...). There is no
   commit history for these files, so the version tag is your only record of which
   weights produced which predictions.
2. **Re-send after any retrain.** Training overwrites `best_model.pt` in place, so
   members holding `results_v1.zip` will silently be running *different* weights
   from you. Say which version is current.
3. **Never edit a checkpoint by hand** and never re-save one through
   `torch.save`. It invalidates every recorded hash.
4. **Keep the folder structure.** Do not flatten it, rename the model folders, or
   drop the `results.json` metrics files — members may be reading them.

> **Note — a second copy of the ViT checkpoint is tracked in git** at
> `artifacts/vit_wish_s42_ddp2_v1/best_model.pt` (327 MB), whitelisted by
> `.gitignore:44` and marked for Git LFS in `.gitattributes`. This contradicts
> working rule 3 (*"No data / checkpoints in git"*) and puts 330 MB in `.git/`
> for every clone. Confirm with the team whether this is intentional before
> relying on it as a delivery channel — the `results/` handoff above is the
> safer route.

---

## 7. Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

**Cause — you are in the wrong virtual environment.** This is the most common
issue by far. The traceback shows the interpreter path; check it:

```bash
python -c "import sys; print(sys.executable)"
```

If it prints a path containing `python3.14` or `/.venv/`, you are in the wrong
venv. Fix:

```bash
deactivate
source .venv311/bin/activate
python -c "import torch; print(torch.__version__)"
```

In VS Code: `Cmd+Shift+P` → **Python: Select Interpreter** → `.venv311/bin/python`.
The Streamlit server inherits the editor's selected interpreter, so this matters
even if your terminal is correct.

### `ModuleNotFoundError: No module named 'streamlit'` (or `cv2`, `facenet_pytorch`)

**Cause — the requirements files are incomplete** (see §3.4).

```bash
pip install streamlit opencv-python facenet-pytorch
```

### `ModuleNotFoundError: No module named 'pytest'`

```bash
pip install pytest
```

### `AttributeError: module 'models.custom_cnn.model' has no attribute 'get_model'`

**Already fixed** in `demo_app.py`. The four model modules do not share one
factory convention:

| Module | Factory |
|---|---|
| `models/resnet50/model.py` | `get_model(num_classes)` |
| `models/efficientnetv2/model.py` | `get_model(num_classes)` |
| `models/custom_cnn/model.py` | `build_model(cfg)` |
| `models/vit/model.py` | `build_model(cfg)` |

`load_model()` now dispatches on whichever exists. If you see this error you have
an **old copy of `demo_app.py`** — pull the latest:

```bash
git pull
```

### `IndexError: Dimension out of range` from `F.softmax`

**Already fixed.** The ViT uses `num_classes: 1` and emits a single logit of shape
`(B,)`, so `softmax(dim=1)` is invalid. `run_inference()` now routes single-logit
models through `sigmoid` instead. Same cause — outdated `demo_app.py`.

### `RuntimeError: slow_conv2d_forward_mps: input(device='cpu') and weight(device=mps:0')`

**Cause — a tensor was not moved to the model's device.** Only happens in custom
scripts, not in the app. Fix:

```python
x = x.to(device)     # device from load_model(), or get_device()
```

To force CPU everywhere:

```bash
CUDA_VISIBLE_DEVICES="" python -c "..."
```

### `FileNotFoundError: data/manifests/wish_v1/manifest.csv`

**Cause — the dataset is not in git.** Expected on a fresh clone, and **harmless
for the demo app** — the app only reads checkpoints, never the manifest. You will
only hit this if you try to run training or evaluation.

### `FileNotFoundError` / checkpoint missing in the demo app

`load_model()` returns `(None, None, cfg)` when the checkpoint is absent, and the
app shows a "no trained checkpoint" notice rather than crashing. Unpack the
`results/` folder you were sent (§6.3). All four are expected at:

```
results/custom_cnn/best_model.pt
results/resnet50/best_model.pt
results/efficientnetv2/best_model.pt
results/vit/best_model.pt
```

### Grad-CAM shows nothing for one model

`Grad-CAM unavailable for this run: ...` in the UI means the target layer could
not be resolved. All four models are configured and verified:

| Model | Target | Mode |
|---|---|---|
| Custom CNN | `features.24` (resolved automatically) | conv |
| ResNet50 | `layer4.2.conv3` | conv |
| EfficientNetV2 | `features.7.0` | conv |
| ViT | `backbone.norm` | token |

Custom CNN's registry entry (`features.4.block.0`) does not exist — its stem is a
flat `nn.Sequential` — so `resolve_gradcam_target()` falls back to the last
`Conv2d`. This is intentional and works; the stale registry value is harmless.

### Gradio fails with an OpenCV import error

`requirements-gradio.txt` installs `opencv-python-headless`, which conflicts with
`opencv-python`. See §5.2.

---

## 8. What was changed in `demo_app.py`

For reviewers — these four bugs were all in the app, none in the model code:

1. **Factory dispatch** — `load_model()` called `get_model()` unconditionally;
   `custom_cnn` and `vit` only expose `build_model(cfg)`. Now dispatches on
   `hasattr`.
2. **Checkpoint layout** — ViT checkpoints wrap weights as
   `{model_state, metadata}`; CNN checkpoints are bare `state_dict`. Now unwrapped.
3. **Single-logit softmax** — ViT emits `(B,)`; `softmax(dim=1)` crashed. Now uses
   `sigmoid` for single-logit models.
4. **ViT Grad-CAM** — was hard-disabled. Added `TokenGradCAM`, which averages
   gradients over tokens (not spatial dims), drops the CLS token and reshapes the
   196 patch tokens to 14×14. Note it deliberately skips `ReLU`, since LayerNorm
   output makes the weighted sum uniformly signed and `ReLU` blanks the map.

Also hardened: `pretrained: false` is forced when building, since the checkpoint
overwrites every weight anyway (avoids a needless timm download and allows
offline use).

**If you edit `models/*/model.py`, keep the factory name consistent with the
table above, or `load_model()` will not be able to build your model.**

---

## 9. Quick reference

```bash
# setup (once)
python3.11 -m venv .venv311 && source .venv311/bin/activate
pip install torch torchvision && pip install -r requirements.txt
pip install streamlit opencv-python facenet-pytorch

# unpack the checkpoints you were sent (at the repo root)
unzip /path/to/results_v1.zip
ls results/*/best_model.pt          # expect 4 lines

# run the demo
streamlit run demo_app.py
```

That is the whole workflow — setup, unpack, run. No dataset or training required.

To send `results/` on to someone else:

```bash
zip -r results_v1.zip results -x "results/**/*.log" -x "results/**/*.tmp"
shasum -a 256 results/*/best_model.pt
```

---

## Known gaps in the repo (not fixed here)

These are real issues worth raising before the viva:

- `streamlit`, `facenet-pytorch` missing from all `requirements*.txt`;
  `opencv-python` and `tqdm` also missing from `requirements-member-c.txt`.
- `pytest` is not installed in `.venv311`, so `tests/` has not been run in it.
  Run it with `pip install pytest && python -m pytest -q` if you need it.
- `configs/vit_freq_hybrid.yaml` exists but is not referenced by `demo_app.py`.
- `MODEL_REGISTRY["Custom CNN"]["gradcam_layer"]` is a stale, non-existent path
  (harmless — resolved by fallback, but misleading to read).
- `results/vit/` has no `results.json`, while the other three models do — so the
  ViT's metrics cannot be tabulated the same way. Either an unfinished training
  run or a missing export step.
- A 327 MB ViT checkpoint is tracked in git under `artifacts/`, against working
  rule 3 (see §6.5).

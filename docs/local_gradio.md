# Local ViT face demo

`gradio_app.py` loads the provided trained ViT checkpoint without retraining, downloading Wish, or requiring Kaggle paths. It derives the inference configuration from the checkpoint's metadata and reuses the repository evaluator's model loader, RGB resize, checkpoint normalization and probability conversion.

## Install and run

From the repository root, using Python 3.11 or 3.12:

```sh
python3 -m venv .venv-gradio
source .venv-gradio/bin/activate
python -m pip install -r requirements-gradio.txt
git lfs install
git lfs pull
python gradio_app.py
```

Open http://127.0.0.1:7860. On the developer's Mac the existing environment can be used without reinstalling:

```sh
../wish-vit-local/.venv/bin/python gradio_app.py
```

`--device auto` chooses CUDA, then Apple MPS, then CPU. Use `--device cpu` for an explicit CPU fallback, `--port 7861` if 7860 is occupied, or `--checkpoint /absolute/path/best_model.pt` for another compatible checkpoint. `--check` loads and validates the model without opening a server. Loading uses `weights_only=True`; only the project's ViT module is accepted.

## Inputs and outputs

Use an already-cropped face for the closest match to the training pipeline. Photo mode detects one frontal face and shows the crop; it rejects zero or multiple detected faces. Detection can miss faces or detect non-faces. Manual cropped-face mode does not verify face presence.

The output is Real or Fake, with both model probabilities and threshold 0.5. The optional reference label reports whether the result agrees with the label you supplied. It cannot independently establish ground truth. Arbitrary images, other generators, compression and editing can be outside the model's training distribution. No accuracy or universal detection guarantee follows from this demo.

The server is local-only by default. For intentional sharing, set `WISH_DEMO_USER` and `WISH_DEMO_PASSWORD`, then pass `--share`. Do not expose private images or credentials. The app does not intentionally save an upload dataset, though Gradio uses temporary files while serving uploads. Stop the server when finished.

## Checkpoint provenance

- Imported from the user's downloaded `best_model.pt`.
- Run: `vit_wish_s42_ddp2_v1`; 85,799,425 parameters; plain ViT-B/16, not FFT.
- Metadata records `best_epoch=1`, `world_size=2`, training time 3598.02585191 seconds.
- This file alone does not establish the final epoch count, completion status or final test accuracy. No histories, manifests or measured test results were supplied with this import.
- SHA-256: `be78ec4969aaf73dc82918f6c9aa794e6d196287920d1957c511106f458a452c`.
- Stored through Git LFS at `artifacts/vit_wish_s42_ddp2_v1/best_model.pt`; do not force-add a raw large blob.

The supplied notebook is preserved at `notebooks/07_wish_vit_kaggle_dual_t4_gradio.ipynb`. Its source is unchanged; it is a self-contained training artifact, not a request to execute its training or public-sharing cells locally.

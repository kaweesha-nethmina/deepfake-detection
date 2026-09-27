"""Gradio demo and self test for the Wish ViT-B/16 real vs fake face detector.

Loads the validation selected checkpoint (best_model.pt) with the checkpoint's own
recorded preprocessing, and applies the frozen 0.5 threshold on P(fake) with
real=0, fake=1, exactly like the sealed evaluation. Standalone use:

    python wish_gradio_app.py --run-dir <OUTPUT_ROOT>/runs/<RUN_NAME> --code-dir <OUTPUT_ROOT>/code

Only four files of the run folder are needed: checkpoints/best_model.pt, run_config_used.json,
training_history.csv and (when training finished) training_complete.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class Detector:
    """One loaded checkpoint plus the exact inference contract used in training and evaluation."""

    def __init__(self, run_dir, checkpoint_name="best_model.pt", device=None):
        from models.vit.runtime import load_checkpoint
        self.run_dir = Path(run_dir)
        self.cfg = json.loads((self.run_dir / "run_config_used.json").read_text())
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint = self.run_dir / "checkpoints" / checkpoint_name
        self.model, self.metadata = load_checkpoint(self.cfg, self.checkpoint, self.device)
        if self.metadata.get("smoke"):
            raise ValueError("Refusing a smoke test checkpoint: it was trained on 32 images and means nothing.")
        self.threshold = float(self.metadata["threshold"])
        self.status = self._status()

    def _status(self):
        history_path = self.run_dir / "training_history.csv"
        history = pd.read_csv(history_path) if history_path.is_file() else pd.DataFrame()
        best_epoch = self.metadata.get("best_epoch")
        best = history[history.epoch == best_epoch] if best_epoch is not None and "epoch" in history else pd.DataFrame()
        pick = lambda column: float(best[column].iloc[0]) if column in best and len(best) else None
        return {"run_name": self.metadata.get("run_name"), "best_epoch": best_epoch,
                "epochs_recorded": int(history.epoch.max()) if len(history) else 0,
                "planned_epochs": self.cfg["train"]["epochs"],
                "training_complete": (self.run_dir / "training_complete.json").is_file(),
                "val_accuracy_at_best": pick("val_accuracy"), "val_roc_auc_at_best": pick("val_roc_auc"),
                "threshold": self.threshold, "device": str(self.device)}

    def predict(self, image):
        from models.vit.runtime import predict_image
        probability = predict_image(self.model, image, self.metadata, self.device)
        return {"prediction": "FAKE" if probability >= self.threshold else "REAL",
                "probability_fake": probability, "probability_real": 1.0 - probability,
                "threshold": self.threshold}

    def predict_path(self, path):
        with Image.open(path) as image:
            image.load()
            return self.predict(image)


def select_samples(manifest, data_root, per_source=8, seed=42, sealed_evaluation_complete=False):
    """Real dataset images with known labels, equal numbers per source.

    Before the sealed evaluation, validation images are used: they already guided checkpoint
    selection, so a demo on them cannot leak the sealed test sets (and their accuracy is slightly
    optimistic). Once the sealed evaluation is complete and frozen, never seen primary test images
    and the held out Stable Diffusion fakes are used instead.
    """
    frame = pd.read_csv(manifest)
    if sealed_evaluation_complete:
        plan = [("test", "FFHQ"), ("test", "CelebA"), ("test", "StyleGAN"), ("cross_gen", "StableDiffusion")]
    else:
        plan = [("val", "FFHQ"), ("val", "CelebA"), ("val", "StyleGAN")]
    samples = []
    for split, source in plan:
        pool = frame[(frame.split == split) & (frame.source == source)].sort_values("filepath")
        for row in pool.sample(n=min(per_source, len(pool)), random_state=seed).itertuples():
            samples.append({"path": str(Path(data_root) / row.filepath), "filepath": row.filepath,
                            "split": split, "source": source, "truth": "FAKE" if int(row.label) == 1 else "REAL"})
    if not samples:
        raise ValueError("No samples found in the manifest.")
    return samples


def folder_samples(folder, limit=200):
    """Your own images. The truth comes from a parent folder named real or fake; otherwise it is unknown."""
    folder = Path(folder)
    paths = sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS)
    samples = []
    for path in paths[:limit]:
        parents = {part.lower() for part in path.relative_to(folder).parts[:-1]}
        truth = None if {"real", "fake"} <= parents else "FAKE" if "fake" in parents else "REAL" if "real" in parents else None
        samples.append({"path": str(path), "filepath": str(path.relative_to(folder)), "split": "own",
                        "source": path.parent.name if path.parent != folder else folder.name, "truth": truth})
    return samples


SPLIT_NAMES = {"val": "validation", "test": "sealed test", "cross_gen": "held out Stable Diffusion", "own": "your images"}


def header_markdown(detector):
    s = detector.status
    progress = ("training complete" if s["training_complete"]
                else f"TRAINING INCOMPLETE: {s['epochs_recorded']} of {s['planned_epochs']} epochs recorded")
    quality = ""
    if s["val_accuracy_at_best"] is not None:
        quality = f" Validation accuracy at that epoch: {s['val_accuracy_at_best']:.4f}"
        if s["val_roc_auc_at_best"] is not None:
            quality += f", ROC AUC: {s['val_roc_auc_at_best']:.4f}"
        quality += "."
    return (f"# Real or fake face? ViT-B/16 detector\n"
            f"Run `{s['run_name']}`, best checkpoint from epoch {s['best_epoch']} ({progress}).{quality} "
            f"Verdict is **FAKE** when P(fake) ≥ {s['threshold']:.2f}. Device: {s['device']}.\n\n"
            "Trained on face crops: FFHQ and CelebA (real) against StyleGAN (fake). Stable Diffusion faces "
            "were held out to measure generalization. Photos that are not face crops like these, heavily edited "
            "images, or other generators are outside what this model has evidence for, so treat its score as a "
            "signal, not proof.")


def build_demo(detector, examples=None):
    import gradio as gr

    def classify(image):
        if image is None:
            raise gr.Error("Upload an image first.")
        result = detector.predict(image)
        headline = "FAKE" if result["prediction"] == "FAKE" else "REAL (not fake)"
        verdict = (f"## Verdict: {headline}\n"
                   f"P(fake) = {result['probability_fake']:.4f}, threshold {result['threshold']:.2f}")
        return {"FAKE": result["probability_fake"], "REAL": result["probability_real"]}, verdict, result

    def classify_batch(files):
        if not files:
            raise gr.Error("Upload one or more images.")
        rows = []
        for item in files:
            path = Path(item if isinstance(item, str) else item.name)
            try:
                result = detector.predict_path(path)
                rows.append([path.name, round(result["probability_fake"], 6), result["prediction"]])
            except Exception as error:  # one unreadable file must not hide the others
                rows.append([path.name, None, f"ERROR: {error}"])
        return pd.DataFrame(rows, columns=["file", "p_fake", "verdict"])

    with gr.Blocks(title="Real or fake face? ViT-B/16") as demo:
        gr.Markdown(header_markdown(detector))
        with gr.Tab("Single image"):
            with gr.Row():
                with gr.Column():
                    image = gr.Image(type="pil", image_mode="RGB", label="Face image")
                    check = gr.Button("Check image", variant="primary")
                with gr.Column():
                    verdict = gr.Markdown()
                    label = gr.Label(num_top_classes=2, label="Model probability")
                    details = gr.JSON(label="Raw output")
            if examples:
                gr.Examples(examples=[[sample["path"]] for sample in examples], inputs=image,
                            example_labels=[f"{sample['truth'] or 'unlabelled'} · {sample['source']}" for sample in examples],
                            label="Example images ("
                            + ", ".join(dict.fromkeys(SPLIT_NAMES[sample["split"]] for sample in examples)) + ")",
                            examples_per_page=12)
            check.click(classify, inputs=image, outputs=[label, verdict, details], api_name="predict")
        with gr.Tab("Batch check"):
            files = gr.File(file_count="multiple", file_types=["image"], type="filepath", label="Images")
            run_all = gr.Button("Check all", variant="primary")
            table = gr.Dataframe(headers=["file", "p_fake", "verdict"], label="Results")
            run_all.click(classify_batch, inputs=files, outputs=table, api_name="predict_batch")
    return demo


def self_test(url, detector, samples, tolerance=1e-5):
    """Send every sample through the running Gradio app over HTTP and compare with the model called directly.

    Returns a per image DataFrame and a summary. The pipeline check (upload, Gradio preprocessing,
    model, label) must match exactly; accuracy on these images is measured, never assumed.
    Samples whose truth is None (unlabelled own images) are predicted but not scored.
    """
    from gradio_client import Client, handle_file
    client = Client(url, verbose=False)
    rows = []
    for sample in samples:
        direct = detector.predict_path(sample["path"])
        label, _, details = client.predict(handle_file(sample["path"]), api_name="/predict")
        api_probability = float(details["probability_fake"])
        truth = sample["truth"]
        rows.append({"filepath": sample["filepath"], "path": sample["path"], "split": sample["split"],
                     "source": sample["source"], "truth": truth or "unknown", "prediction": direct["prediction"],
                     "p_fake": direct["probability_fake"],
                     "correct": (direct["prediction"] == truth) if truth else None,
                     "app_prediction": details["prediction"], "app_top_label": label["label"],
                     "app_p_fake": api_probability,
                     "app_matches_model": (abs(api_probability - direct["probability_fake"]) <= tolerance
                                           and details["prediction"] == direct["prediction"] == label["label"])})
    results = pd.DataFrame(rows)
    batch = client.predict([handle_file(sample["path"]) for sample in samples], api_name="/predict_batch")
    batch_frame = pd.DataFrame(batch["data"], columns=batch["headers"])
    batch_ok = (len(batch_frame) == len(results)
                and list(batch_frame.verdict) == list(results.prediction)
                and ((batch_frame.p_fake.astype(float) - results.p_fake).abs() <= tolerance).all())
    scored = results[results.correct.notna()]
    accuracy_of = lambda truth: (float(scored[scored.truth == truth].correct.astype(bool).mean())
                                 if (scored.truth == truth).any() else None)
    summary = {"images": len(results), "labelled_images": len(scored),
               "real_images": int((scored.truth == "REAL").sum()), "fake_images": int((scored.truth == "FAKE").sum()),
               "accuracy_on_real_images": accuracy_of("REAL"), "accuracy_on_fake_images": accuracy_of("FAKE"),
               "accuracy": float(scored.correct.astype(bool).mean()) if len(scored) else None,
               "accuracy_by_source": (scored.groupby("source").correct.apply(lambda c: round(float(c.astype(bool).mean()), 4)).to_dict()
                                      if len(scored) else {}),
               "predicted_fake": int((results.prediction == "FAKE").sum()),
               "predicted_real": int((results.prediction == "REAL").sum()),
               "single_image_endpoint_matches_model": bool(results.app_matches_model.all()),
               "batch_endpoint_matches_model": bool(batch_ok)}
    return results, summary


def plot_self_test(results, output, columns=8, title="Gradio self test (green = correct, red = wrong, grey = no label)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = -(-len(results) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.1 * columns, 2.5 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, row in zip(axes.flat, results.itertuples()):
        with Image.open(row.path) as image:
            axis.imshow(image.convert("RGB"))
        color = "#555555" if row.correct is None or pd.isna(row.correct) else "#1a7f37" if row.correct else "#c62828"
        axis.set_title(f"true {row.truth} ({row.source})\npred {row.prediction} p={row.p_fake:.3f}", fontsize=7, color=color)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--code-dir", required=True, help="folder containing the models package")
    parser.add_argument("--share", action="store_true", help="create a temporary public gradio.live link")
    parser.add_argument("--host", default=None, help="for example 0.0.0.0 to serve on your network or a server")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.code_dir).resolve()))
    build_demo(Detector(args.run_dir)).launch(share=args.share, server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()

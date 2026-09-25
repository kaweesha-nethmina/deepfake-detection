"""Local face-image inference using a saved ViT checkpoint; no training or dataset needed."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from PIL import Image, ImageOps

from models.vit.runtime import device_for, load_checkpoint, predict_image

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "artifacts/vit_wish_s42_ddp2_v1/best_model.pt"
CROP_MODE = "Already-cropped face"
PHOTO_MODE = "Photo: detect one face"


def read_config(checkpoint):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {path}. Run git lfs pull or supply --checkpoint.")
    with path.open("rb") as stream:
        if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("Only the Git LFS pointer is present. Run git lfs pull first.")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    metadata = saved.get("metadata", {})
    if metadata.get("module") != "models.vit.model" or metadata.get("model_name") != "vit_b16":
        raise ValueError("Expected the project's ViT checkpoint, including its inference metadata.")
    if metadata.get("smoke"):
        raise ValueError("Refusing smoke-test weights; supply the actual trained checkpoint.")
    if metadata.get("label_mapping") != {"real": 0, "fake": 1} or metadata.get("threshold") != 0.5:
        raise ValueError("Checkpoint must use real=0, fake=1 and threshold=0.5.")
    return {"model_name": metadata["model_name"], "module": metadata["module"],
            "model": metadata["model_config"], "run_name": metadata.get("run_name", "unknown")}


def prepare_face(image, mode):
    if image is None or not isinstance(image, Image.Image):
        raise ValueError("Upload an image first.")
    if image.width * image.height > 25_000_000:
        raise ValueError("Resize the image to at most 25 megapixels.")
    if min(image.size) < 32:
        raise ValueError("Image too small. Supply a clearer face image.")
    image = ImageOps.exif_transpose(image).convert("RGB")
    if mode == CROP_MODE:
        return image
    if mode != PHOTO_MODE:
        raise ValueError("Unknown image type.")
    import cv2
    import numpy as np
    preview = image.copy()
    preview.thumbnail((1600, 1600))
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("Face detector could not be loaded.")
    boxes = detector.detectMultiScale(cv2.cvtColor(np.asarray(preview), cv2.COLOR_RGB2GRAY),
                                     scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(boxes) == 0:
        raise ValueError("No frontal face detected. Try a clearer photo, or manually crop a face and select cropped-face mode.")
    if len(boxes) != 1:
        raise ValueError("Multiple faces detected. Crop to a single face first.")
    x, y, w, h = [int(v) for v in boxes[0]]
    sx, sy = image.width / preview.width, image.height / preview.height
    return image.crop((max(0, int((x - .15*w)*sx)), max(0, int((y - .15*h)*sy)),
                       min(image.width, int((x + 1.15*w)*sx)), min(image.height, int((y + 1.15*h)*sy))))


class Detector:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device="auto"):
        self.config = read_config(checkpoint)
        self.device = device_for(device)
        self.model, self.metadata = load_checkpoint(self.config, checkpoint, self.device)

    def predict(self, image, mode=CROP_MODE, reference="Unknown"):
        if reference not in ("Unknown", "Real", "Fake"):
            raise ValueError("Unknown reference label.")
        face = prepare_face(image, mode)
        probability = predict_image(self.model, face, self.metadata["preprocessing"], self.device)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError("Invalid model probability; no result reported.")
        label = "Fake" if probability >= .5 else "Real"
        verdict = "Not assessed: reference label unknown"
        if reference != "Unknown":
            verdict = ("Correct" if label == reference else "Incorrect") + " against your supplied label"
        return label, {"Real": 1 - probability, "Fake": probability}, face, verdict


def create_app(detector):
    import gradio as gr

    def predict(image, mode, reference):
        try:
            return detector.predict(image, mode, reference)
        except (ValueError, RuntimeError) as error:
            raise gr.Error(str(error)) from error

    with gr.Blocks(title="Wish ViT Face Detector", analytics_enabled=False) as app:
        gr.Markdown("# Wish ViT Face Detector")
        gr.Markdown("Research prediction, not proof of authenticity. Scores are not calibrated certainty.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", sources=["upload"], label="Face image", height=320)
                mode = gr.Radio([CROP_MODE, PHOTO_MODE], value=CROP_MODE, label="Image type")
                reference = gr.Radio(["Unknown", "Real", "Fake"], value="Unknown", label="Known reference label (optional)")
                submit = gr.Button("Predict", variant="primary")
            with gr.Column():
                label = gr.Textbox(label="Prediction", interactive=False)
                probability = gr.Label(label="Model probabilities", num_top_classes=2)
                crop = gr.Image(label="Analyzed face crop", height=224, interactive=False)
                verdict = gr.Textbox(label="Reference check", interactive=False)
        outputs = [label, probability, crop, verdict]
        submit.click(predict, [image, mode, reference], outputs, api_name="predict", concurrency_limit=1)
        for component in (image, mode, reference):
            component.change(lambda: (None, None, None, None), outputs=outputs, queue=False, api_name=False)
        gr.ClearButton([image, *outputs])
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Explicitly enable a password-protected public tunnel")
    parser.add_argument("--check", action="store_true", help="Validate/load the checkpoint without starting a server")
    args = parser.parse_args()
    auth = None
    if args.share:
        username, password = os.environ.get("WISH_DEMO_USER"), os.environ.get("WISH_DEMO_PASSWORD")
        if not username or not password:
            parser.error("--share requires WISH_DEMO_USER and WISH_DEMO_PASSWORD")
        auth = (username, password)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    detector = Detector(args.checkpoint, args.device)
    print(json.dumps({"checkpoint": str(args.checkpoint), "device": str(detector.device),
        "run_name": detector.metadata.get("run_name"), "best_epoch": detector.metadata.get("best_epoch"),
        "parameters": detector.metadata.get("total_parameters")}, indent=2), flush=True)
    if not args.check:
        create_app(detector).queue(max_size=8).launch(server_name="127.0.0.1", server_port=args.port,
            share=args.share, auth=auth, max_file_size="20mb", show_error=False)


if __name__ == "__main__":
    main()

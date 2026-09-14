"""
Interactive demo app for the Cross-Generator Deepfake Face Detection project.

Run it from the project root with:
    streamlit run demo_app.py

What it does
------------
- Lets you upload a face image (jpg/png).
- Runs it through every model that has a trained checkpoint in results/<model>/best_model.pt.
- Shows REAL vs FAKE prediction + confidence for each model side-by-side.
- Shows a Grad-CAM heatmap for the CNN-based models (Custom CNN, ResNet50, EfficientNetV2)
  so you can point to *why* the model thinks a region looks fake in the viva/video.

This is meant purely as a demo/visualisation layer on top of the checkpoints your
team already trains with models/<name>/train.py — it does not retrain anything.
"""

import io
import os

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from src.data_pipeline import IMAGENET_MEAN, IMAGENET_STD, build_transforms
from src.utils import get_device, load_config

MODEL_REGISTRY = {
    "Custom CNN": {
        "config": "configs/custom_cnn.yaml",
        "import_path": "models.custom_cnn.model",
        "gradcam_layer": "features.4.block.0",  # last conv layer before final pool
    },
    "ResNet50": {
        "config": "configs/resnet50.yaml",
        "import_path": "models.resnet50.model",
        "gradcam_layer": "layer4.2.conv3",
    },
    "EfficientNetV2": {
        "config": "configs/efficientnetv2.yaml",
        "import_path": "models.efficientnetv2.model",
        "gradcam_layer": "features.7.0",
    },
    "ViT (frequency-hybrid)": {
        "config": "configs/vit.yaml",
        "import_path": "models.vit.model",
        "gradcam_layer": None,  # attention-based; Grad-CAM skipped, prediction only
    },
}

CLASS_NAMES = ["REAL", "FAKE"]


@st.cache_resource(show_spinner=False)
def load_model(display_name: str):
    """Loads one model + its checkpoint, if the checkpoint file exists."""
    import importlib

    info = MODEL_REGISTRY[display_name]
    cfg = load_config(info["config"])
    ckpt_path = cfg["output"]["checkpoint_path"]
    if not os.path.exists(ckpt_path):
        return None, None, cfg

    module = importlib.import_module(info["import_path"])
    if display_name == "ViT (frequency-hybrid)":
        use_freq = cfg.get("model", {}).get("use_frequency_branch", True)
        backbone = cfg.get("model", {}).get("vit_backbone", "vit_base_patch16_224")
        model = module.get_model(backbone_name=backbone, use_frequency_branch=use_freq)
    else:
        model = module.get_model(num_classes=2)

    device = get_device()
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device, cfg


def preprocess(image: Image.Image, image_size: int = 224):
    transform = build_transforms(image_size=image_size, train=False)
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    return tensor


def denormalize_for_display(tensor: torch.Tensor) -> np.ndarray:
    """tensor: (3, H, W) normalized -> (H, W, 3) uint8 for overlaying a heatmap."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


class GradCAM:
    """Minimal Grad-CAM implementation using forward/backward hooks — no extra
    dependency beyond torch, so it works with whatever torch version the team
    already has installed for training."""

    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        self.model = model
        self.activations = None
        self.gradients = None
        layer = dict(model.named_modules())[target_layer_name]
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(base_img: np.ndarray, cam: np.ndarray) -> Image.Image:
    """Blends a grayscale Grad-CAM map onto the original image without extra
    plotting dependencies (pure numpy/PIL, so no matplotlib/opencv required)."""
    heat = (cam * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat).convert("L")
    # Simple red-channel heat overlay
    heat_rgb = np.zeros((*heat.shape, 3), dtype=np.uint8)
    heat_rgb[..., 0] = heat  # red channel = intensity
    heat_pil = Image.fromarray(heat_rgb)
    base_pil = Image.fromarray(base_img)
    blended = Image.blend(base_pil, heat_pil, alpha=0.45)
    return blended


def run_inference(display_name: str, image: Image.Image):
    """Runs one model on the uploaded image and renders its result block."""
    st.subheader(display_name)
    model, device, cfg = load_model(display_name)

    if model is None:
        st.warning(
            f"No checkpoint found at "
            f"`{load_config(MODEL_REGISTRY[display_name]['config'])['output']['checkpoint_path']}`. "
            "Train this model first (see models/<name>/train.py)."
        )
        return

    image_size = cfg["data"]["image_size"]
    input_tensor = preprocess(image, image_size=image_size).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    if pred_label == "FAKE":
        st.error(f"**{pred_label}** ({confidence * 100:.1f}% confidence)")
    else:
        st.success(f"**{pred_label}** ({confidence * 100:.1f}% confidence)")

    st.progress(confidence)
    st.caption(f"REAL: {probs[0] * 100:.1f}%  |  FAKE: {probs[1] * 100:.1f}%")

    gradcam_layer = MODEL_REGISTRY[display_name]["gradcam_layer"]
    if gradcam_layer is not None:
        try:
            cam_input = input_tensor.clone().requires_grad_(True)
            cam = GradCAM(model, gradcam_layer)
            heatmap = cam(cam_input, class_idx=pred_idx)
            base_img = denormalize_for_display(input_tensor.squeeze(0))
            overlay = overlay_heatmap(base_img, heatmap)
            st.image(overlay, caption="Grad-CAM: where the model looked", width=280)
        except Exception as e:  # keep the demo robust for the live viva
            st.caption(f"(Grad-CAM unavailable for this run: {e})")
    else:
        st.caption("Grad-CAM not shown for ViT (attention-based, not a conv map).")


def checkpoint_exists(display_name: str) -> bool:
    cfg = load_config(MODEL_REGISTRY[display_name]["config"])
    return os.path.exists(cfg["output"]["checkpoint_path"])


def main():
    st.set_page_config(page_title="Deepfake Face Detector Demo", layout="wide")
    st.title("Cross-Generator Deepfake Face Detection — Live Demo")
    st.caption(
        "Upload a face image, choose which model(s) to run it through, then click Run Detection. "
        "Grad-CAM highlights (red) show which regions most influenced the CNN-based models' decisions."
    )

    uploaded_file = st.file_uploader("Step 1 — Upload a face image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Upload a JPG/PNG face image to continue.")
        return

    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    st.image(image, caption="Uploaded image", width=280)

    st.markdown("### Step 2 — Select which model(s) to run")

    model_names = list(MODEL_REGISTRY.keys())
    availability = {name: checkpoint_exists(name) for name in model_names}

    select_all = st.checkbox(
        f"Select All ({len(model_names)} models)",
        value=False,
        help="Runs every model that has a trained checkpoint.",
    )

    selected = []
    cols = st.columns(len(model_names))
    for i, (col, name) in enumerate(zip(cols, model_names), start=1):
        with col:
            label = f"{i}. {name}"
            if not availability[name]:
                label += "  \n*(no checkpoint yet)*"
            checked = st.checkbox(
                label,
                value=select_all,
                key=f"select_{name}",
                disabled=not availability[name],
            )
            if checked and availability[name]:
                selected.append(name)

    st.markdown("### Step 3 — Run")
    run_clicked = st.button("▶ Run Detection", type="primary", disabled=len(selected) == 0)

    if not selected:
        st.info("Tick one or more models above (or Select All), then click Run Detection.")
        return

    if not run_clicked:
        st.caption(f"Ready to run: {', '.join(selected)}. Click **Run Detection** above.")
        return

    st.markdown("---")
    st.markdown("### Results")
    result_cols = st.columns(len(selected))
    for col, display_name in zip(result_cols, selected):
        with col:
            run_inference(display_name, image)


if __name__ == "__main__":
    main()

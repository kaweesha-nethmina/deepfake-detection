"""
Interactive demo app for the Cross-Generator Deepfake Face Detection project.

Run:
    python -m streamlit run demo_app.py

Pipeline:
    Uploaded image
        ↓
    MTCNN face detection
        ↓
    Expanded + square face crop
        ↓
    Model preprocessing
        ↓
    ResNet50 / EfficientNetV2 / Custom CNN / ViT
        ↓
    Prediction + Grad-CAM

The important point is that the detected face is expanded and converted
to a square crop before being passed to the models. This avoids excessive
cropping/stretching and keeps the inference preprocessing consistent.
"""

import io
import os
import importlib

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from src.data_pipeline import IMAGENET_MEAN, IMAGENET_STD, build_transforms
from src.utils import get_device, load_config


# ============================================================
# MODEL REGISTRY
# ============================================================

MODEL_REGISTRY = {
    "Custom CNN": {
        "config": "configs/custom_cnn.yaml",
        "import_path": "models.custom_cnn.model",
        "gradcam_layer": "features.24",
        "default_checkpoint": "results/custom_cnn/best_model.pt",
    },

    "ResNet50": {
        "config": "configs/resnet50.yaml",
        "import_path": "models.resnet50.model",
        "gradcam_layer": "layer4.2.conv3",
        "default_checkpoint": "results/resnet50/best_model.pt",
    },

    "EfficientNetV2": {
        "config": "configs/efficientnetv2.yaml",
        "import_path": "models.efficientnetv2.model",
        "gradcam_layer": "features.7.0",
        "default_checkpoint": "results/efficientnetv2/best_model.pt",
    },

    "ViT (frequency-hybrid)": {
        "config": "configs/vit.yaml",
        "import_path": "models.vit.model",
        "gradcam_layer": None,
        "default_checkpoint": "results/vit/best_model.pt",
    },
}


CLASS_NAMES = ["REAL", "FAKE"]


# ============================================================
# FACE DETECTION SETTINGS
# ============================================================

# Amount of additional area around the detected face.
#
# 0.35 means:
#   35% extra width on each side
#   35% extra height on each side
#
# This prevents the crop from being too tight around the face.
FACE_MARGIN = 0.35

# Minimum MTCNN probability required to accept a detection.
#
# If the detector is uncertain, OpenCV will be attempted.
MTCNN_MIN_PROBABILITY = 0.80

# If MTCNN fails, OpenCV is used.
OPENCV_MIN_FACE_SIZE = 60


# ============================================================
# CONFIGURATION HELPERS
# ============================================================

def safe_load_config(config_path: str) -> dict:
    """
    Safely load a YAML configuration.

    Returns {} if the file is missing or cannot be loaded.
    """
    try:
        cfg = load_config(config_path)

        if isinstance(cfg, dict):
            return cfg

        return {}

    except Exception:
        return {}


def resolve_checkpoint_path(display_name: str, cfg: dict) -> str:
    """
    Resolve the model checkpoint path.

    Priority:
        1. cfg['output']['checkpoint_path']
        2. cfg['checkpoint_path']
        3. MODEL_REGISTRY default path
    """

    output_section = cfg.get("output") if isinstance(cfg, dict) else None

    if (
        isinstance(output_section, dict)
        and output_section.get("checkpoint_path")
    ):
        return output_section["checkpoint_path"]

    if isinstance(cfg, dict) and cfg.get("checkpoint_path"):
        return cfg["checkpoint_path"]

    return MODEL_REGISTRY[display_name]["default_checkpoint"]


def resolve_image_size(cfg: dict, default: int = 224) -> int:
    """
    Read model image size from configuration.
    """

    data_section = cfg.get("data") if isinstance(cfg, dict) else None

    if (
        isinstance(data_section, dict)
        and data_section.get("image_size")
    ):
        return int(data_section["image_size"])

    return default


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(display_name: str):
    """
    Load a trained model and checkpoint.
    """

    info = MODEL_REGISTRY[display_name]

    cfg = safe_load_config(info["config"])

    ckpt_path = resolve_checkpoint_path(display_name, cfg)

    if not os.path.exists(ckpt_path):
        return None, None, cfg

    module = importlib.import_module(info["import_path"])

    # --------------------------------------------------------
    # ViT
    # --------------------------------------------------------

    if display_name == "ViT (frequency-hybrid)":

        model_cfg = cfg.get("model", {})

        use_freq = model_cfg.get(
            "use_frequency_branch",
            True,
        )

        backbone = model_cfg.get(
            "vit_backbone",
            "vit_base_patch16_224",
        )

        model = module.get_model(
            backbone_name=backbone,
            use_frequency_branch=use_freq,
        )

    # --------------------------------------------------------
    # CNN MODELS
    # --------------------------------------------------------

    else:

        model = module.get_model(
            num_classes=2
        )

    device = get_device()

    state_dict = torch.load(
        ckpt_path,
        map_location=device,
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, device, cfg


# ============================================================
# FACE CROPPING
# ============================================================

def detect_and_crop_face(
    image: Image.Image,
    margin: float = FACE_MARGIN,
):
    """
    Detect the largest/highest-confidence face.

    Detection order:

        1. MTCNN
        2. OpenCV Haar Cascade
        3. Center-square fallback

    The detected face is expanded and converted into a square crop.

    Returns:
        cropped_image, method, confidence
    """

    # --------------------------------------------------------
    # MTCNN
    # --------------------------------------------------------

    try:

        detector = _get_mtcnn()

        boxes, probabilities = detector.detect(image)

        if boxes is not None and len(boxes) > 0:

            # ------------------------------------------------
            # Select best face
            # ------------------------------------------------

            if probabilities is not None:

                valid = [
                    i
                    for i, p in enumerate(probabilities)
                    if p is not None
                    and float(p) >= MTCNN_MIN_PROBABILITY
                ]

                if valid:

                    # Highest confidence face
                    best_idx = max(
                        valid,
                        key=lambda i: float(probabilities[i]),
                    )

                else:

                    # If none reaches threshold, use largest face
                    best_idx = max(
                        range(len(boxes)),
                        key=lambda i:
                        (boxes[i][2] - boxes[i][0])
                        * (boxes[i][3] - boxes[i][1]),
                    )

            else:

                best_idx = max(
                    range(len(boxes)),
                    key=lambda i:
                    (boxes[i][2] - boxes[i][0])
                    * (boxes[i][3] - boxes[i][1]),
                )

            box = boxes[best_idx]

            probability = (
                float(probabilities[best_idx])
                if probabilities is not None
                else None
            )

            cropped = _crop_face_square(
                image,
                box,
                margin,
            )

            if cropped is not None:

                return (
                    cropped,
                    "mtcnn",
                    probability,
                )

    except Exception:
        pass

    # --------------------------------------------------------
    # OpenCV fallback
    # --------------------------------------------------------

    try:

        cropped = _detect_face_opencv(
            image,
            margin,
        )

        if cropped is not None:

            return (
                cropped,
                "opencv",
                None,
            )

    except Exception:
        pass

    # --------------------------------------------------------
    # Center-square fallback
    # --------------------------------------------------------

    cropped = _center_square_crop(image)

    return (
        cropped,
        "center_crop",
        None,
    )


def _crop_face_square(
    image: Image.Image,
    box,
    margin: float = FACE_MARGIN,
):
    """
    Expand the detected face and create a square crop.

    This is intentionally different from simply cropping the
    rectangular MTCNN bounding box.

    The square crop prevents the face from being stretched
    differently along width and height.
    """

    x1, y1, x2, y2 = map(float, box)

    # --------------------------------------------------------
    # Validate bounding box
    # --------------------------------------------------------

    if x2 <= x1 or y2 <= y1:
        return None

    face_w = x2 - x1
    face_h = y2 - y1

    # Face center
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # --------------------------------------------------------
    # Expand the face
    # --------------------------------------------------------

    expanded_w = face_w * (1.0 + 2.0 * margin)
    expanded_h = face_h * (1.0 + 2.0 * margin)

    # Make square
    side = max(
        expanded_w,
        expanded_h,
    )

    # --------------------------------------------------------
    # Initial square coordinates
    # --------------------------------------------------------

    left = cx - side / 2.0
    top = cy - side / 2.0

    right = left + side
    bottom = top + side

    img_w = image.width
    img_h = image.height

    # --------------------------------------------------------
    # Shift square inside image boundaries
    # --------------------------------------------------------

    if left < 0:
        right -= left
        left = 0

    if top < 0:
        bottom -= top
        top = 0

    if right > img_w:
        left -= right - img_w
        right = img_w

    if bottom > img_h:
        top -= bottom - img_h
        bottom = img_h

    # --------------------------------------------------------
    # Final boundary clamp
    # --------------------------------------------------------

    left = max(0, left)
    top = max(0, top)
    right = min(img_w, right)
    bottom = min(img_h, bottom)

    # --------------------------------------------------------
    # Make coordinates integer
    # --------------------------------------------------------

    left = int(round(left))
    top = int(round(top))
    right = int(round(right))
    bottom = int(round(bottom))

    if right <= left or bottom <= top:
        return None

    return image.crop(
        (
            left,
            top,
            right,
            bottom,
        )
    )


def _center_square_crop(image: Image.Image):
    """
    Fallback center-square crop.
    """

    w, h = image.size

    side = min(w, h)

    left = (w - side) // 2
    top = (h - side) // 2

    return image.crop(
        (
            left,
            top,
            left + side,
            top + side,
        )
    )


# ============================================================
# OPENCV FACE DETECTOR
# ============================================================

def _detect_face_opencv(
    image: Image.Image,
    margin: float,
):
    """
    OpenCV Haar cascade fallback.
    """

    import cv2

    cascade = _get_haar_cascade()

    if cascade.empty():
        return None

    image_array = np.array(
        image.convert("RGB")
    )

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY,
    )

    # Improve detection under slightly different lighting.
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(
            OPENCV_MIN_FACE_SIZE,
            OPENCV_MIN_FACE_SIZE,
        ),
    )

    if len(faces) == 0:
        return None

    # Largest face
    x, y, w, h = max(
        faces,
        key=lambda f: f[2] * f[3],
    )

    box = (
        x,
        y,
        x + w,
        y + h,
    )

    return _crop_face_square(
        image,
        box,
        margin,
    )


@st.cache_resource(show_spinner=False)
def _get_haar_cascade():
    """
    Load OpenCV Haar cascade once.
    """

    import cv2

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    return cv2.CascadeClassifier(
        cascade_path
    )


# ============================================================
# MTCNN
# ============================================================

@st.cache_resource(show_spinner=False)
def _get_mtcnn():
    """
    Create MTCNN once.

    CPU is used intentionally because MTCNN is only being used
    for preprocessing/detection.
    """

    from facenet_pytorch import MTCNN

    return MTCNN(
        image_size=160,
        margin=0,
        min_face_size=40,
        thresholds=[
            0.6,
            0.7,
            0.7,
        ],
        factor=0.709,
        post_process=False,
        select_largest=False,
        keep_all=True,
        device="cpu",
    )


# ============================================================
# MODEL PREPROCESSING
# ============================================================

def preprocess(
    image: Image.Image,
    image_size: int = 224,
):
    """
    Apply exactly the inference transform used by the
    project's data pipeline.
    """

    transform = build_transforms(
        image_size=image_size,
        train=False,
    )

    tensor = transform(
        image.convert("RGB")
    ).unsqueeze(0)

    return tensor


# ============================================================
# DISPLAY HELPERS
# ============================================================

def denormalize_for_display(
    tensor: torch.Tensor,
) -> np.ndarray:
    """
    Convert ImageNet-normalized tensor back into RGB uint8.
    """

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=tensor.dtype,
        device=tensor.device,
    ).view(3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD,
        dtype=tensor.dtype,
        device=tensor.device,
    ).view(3, 1, 1)

    img = tensor.cpu() * std.cpu() + mean.cpu()

    img = img.clamp(
        0,
        1,
    )

    img = img.permute(
        1,
        2,
        0,
    ).numpy()

    return (
        img * 255
    ).astype(np.uint8)


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:
    """
    Minimal Grad-CAM implementation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer_name: str,
    ):

        self.model = model
        self.activations = None
        self.gradients = None

        modules = dict(
            model.named_modules()
        )

        if target_layer_name not in modules:
            raise ValueError(
                f"Grad-CAM layer not found: "
                f"{target_layer_name}"
            )

        layer = modules[
            target_layer_name
        ]

        layer.register_forward_hook(
            self._save_activation
        )

        layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(
        self,
        module,
        inp,
        out,
    ):

        self.activations = out.detach()

    def _save_gradient(
        self,
        module,
        grad_in,
        grad_out,
    ):

        self.gradients = (
            grad_out[0].detach()
        )

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: int,
    ) -> np.ndarray:

        self.model.zero_grad(
            set_to_none=True
        )

        output = self.model(
            input_tensor
        )

        score = output[
            0,
            class_idx,
        ]

        score.backward()

        if self.gradients is None:
            raise RuntimeError(
                "Gradients were not captured."
            )

        if self.activations is None:
            raise RuntimeError(
                "Activations were not captured."
            )

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights
            * self.activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = (
            cam
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = np.zeros_like(
                cam
            )

        return cam


def overlay_heatmap(
    base_img: np.ndarray,
    cam: np.ndarray,
) -> Image.Image:
    """
    Red Grad-CAM overlay.
    """

    heat = (
        cam * 255
    ).astype(np.uint8)

    heat_img = Image.fromarray(
        heat
    ).convert("L")

    heat_rgb = np.zeros(
        (
            heat.shape[0],
            heat.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    heat_rgb[..., 0] = heat

    heat_pil = Image.fromarray(
        heat_rgb
    )

    base_pil = Image.fromarray(
        base_img
    )

    return Image.blend(
        base_pil,
        heat_pil,
        alpha=0.45,
    )


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    display_name: str,
    image: Image.Image,
):
    """
    Run a selected model.
    """

    st.subheader(
        display_name
    )

    model, device, cfg = load_model(
        display_name
    )

    # --------------------------------------------------------
    # Missing checkpoint
    # --------------------------------------------------------

    if model is None:

        ckpt_path = (
            resolve_checkpoint_path(
                display_name,
                cfg,
            )
        )

        st.warning(
            f"No checkpoint found at "
            f"`{ckpt_path}`. "
            "Train this model first."
        )

        return

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    image_size = resolve_image_size(
        cfg
    )

    input_tensor = preprocess(
        image,
        image_size=image_size,
    ).to(device)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            input_tensor
        )

        probs = (
            F.softmax(
                logits,
                dim=1,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    pred_idx = int(
        probs.argmax()
    )

    pred_label = (
        CLASS_NAMES[pred_idx]
    )

    confidence = float(
        probs[pred_idx]
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if pred_label == "FAKE":

        st.error(
            f"**{pred_label}** "
            f"({confidence * 100:.1f}% confidence)"
        )

    else:

        st.success(
            f"**{pred_label}** "
            f"({confidence * 100:.1f}% confidence)"
        )

    st.progress(
        confidence
    )

    st.caption(
        f"REAL: {probs[0] * 100:.1f}%  |  "
        f"FAKE: {probs[1] * 100:.1f}%"
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    gradcam_layer = (
        MODEL_REGISTRY[
            display_name
        ]["gradcam_layer"]
    )

    if gradcam_layer is None:

        st.caption(
            "Grad-CAM not shown for ViT "
            "(attention-based model)."
        )

        return

    try:

        cam_input = (
            input_tensor.clone()
            .detach()
            .requires_grad_(True)
        )

        cam = GradCAM(
            model,
            gradcam_layer,
        )

        heatmap = cam(
            cam_input,
            class_idx=pred_idx,
        )

        base_img = (
            denormalize_for_display(
                input_tensor.squeeze(0)
            )
        )

        overlay = overlay_heatmap(
            base_img,
            heatmap,
        )

        st.image(
            overlay,
            caption="Grad-CAM: where the model looked",
            width=280,
        )

    except Exception as e:

        st.caption(
            f"Grad-CAM unavailable for this run: {e}"
        )


# ============================================================
# CHECKPOINT CHECK
# ============================================================

def checkpoint_exists(
    display_name: str,
) -> bool:

    cfg = safe_load_config(
        MODEL_REGISTRY[
            display_name
        ]["config"]
    )

    ckpt_path = (
        resolve_checkpoint_path(
            display_name,
            cfg,
        )
    )

    return os.path.exists(
        ckpt_path
    )


# ============================================================
# MAIN STREAMLIT APP
# ============================================================

def main():

    st.set_page_config(
        page_title="Deepfake Face Detector Demo",
        layout="wide",
    )

    st.title(
        "Cross-Generator Deepfake Face Detection — Live Demo"
    )

    st.caption(
        "Upload a face image, choose which model(s) "
        "to run it through, then click Run Detection. "
        "Grad-CAM highlights show regions that influenced "
        "CNN-based model decisions."
    )

    # ========================================================
    # STEP 1
    # ========================================================

    uploaded_file = st.file_uploader(
        "Step 1 — Upload a face image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
            "bmp",
        ],
    )

    if uploaded_file is None:

        st.info(
            "Upload a JPG/PNG face image to continue."
        )

        return

    image = Image.open(
        io.BytesIO(
            uploaded_file.read()
        )
    ).convert("RGB")

    # ========================================================
    # FACE DETECTION
    # ========================================================

    with st.spinner(
        "Detecting face..."
    ):

        (
            face_crop,
            method,
            detector_probability,
        ) = detect_and_crop_face(
            image
        )

    # ========================================================
    # DISPLAY ORIGINAL + MODEL INPUT
    # ========================================================

    upload_col, crop_col = st.columns(
        2
    )

    with upload_col:

        st.image(
            image,
            caption="Uploaded image",
            width=280,
        )

    with crop_col:

        st.image(
            face_crop,
            caption="What the models actually see",
            width=280,
        )

        if method == "mtcnn":

            if detector_probability is not None:

                st.success(
                    "Face auto-detected and cropped "
                    f"(MTCNN, confidence "
                    f"{detector_probability * 100:.1f}%)."
                )

            else:

                st.success(
                    "Face auto-detected and cropped "
                    "(facenet-pytorch MTCNN)."
                )

        elif method == "opencv":

            st.info(
                "Face auto-detected and cropped "
                "(OpenCV Haar cascade fallback)."
            )

        else:

            st.warning(
                "No face detector successfully detected "
                "a face — using a center-square crop. "
                "Predictions may be less reliable."
            )

    # ========================================================
    # MODEL INPUT
    # ========================================================

    image = face_crop

    # ========================================================
    # STEP 2
    # ========================================================

    st.markdown(
        "### Step 2 — Select which model(s) to run"
    )

    model_names = list(
        MODEL_REGISTRY.keys()
    )

    availability = {
        name: checkpoint_exists(name)
        for name in model_names
    }

    select_all = st.checkbox(
        f"Select All ({len(model_names)} models)",
        value=False,
        help=(
            "Runs every model that has "
            "a trained checkpoint."
        ),
    )

    selected = []

    cols = st.columns(
        len(model_names)
    )

    for i, (
        col,
        name,
    ) in enumerate(
        zip(
            cols,
            model_names,
        ),
        start=1,
    ):

        with col:

            label = f"{i}. {name}"

            if not availability[name]:

                label += (
                    "\n*(no checkpoint yet)*"
                )

            checked = st.checkbox(
                label,
                value=select_all,
                key=f"select_{name}",
                disabled=not availability[name],
            )

            if (
                checked
                and availability[name]
            ):

                selected.append(
                    name
                )

    # ========================================================
    # STEP 3
    # ========================================================

    st.markdown(
        "### Step 3 — Run"
    )

    run_clicked = st.button(
        "▶ Run Detection",
        type="primary",
        disabled=len(selected) == 0,
    )

    if not selected:

        st.info(
            "Tick one or more models above "
            "(or Select All), then click Run Detection."
        )

        return

    if not run_clicked:

        st.caption(
            f"Ready to run: "
            f"{', '.join(selected)}. "
            "Click Run Detection above."
        )

        return

    # ========================================================
    # RESULTS
    # ========================================================

    st.markdown(
        "---"
    )

    st.markdown(
        "### Results"
    )

    result_cols = st.columns(
        len(selected)
    )

    for col, display_name in zip(
        result_cols,
        selected,
    ):

        with col:

            run_inference(
                display_name,
                image,
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
"""Shared image-preprocessing helpers (owner A).

Additive-only API: if you extend this file, add NEW functions and keep the
existing signatures intact. Everyone's `train.py` and notebooks import from here.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2

    _HAVE_CV2 = True
except ImportError:  # opencv not installed (e.g. fresh venv on some platform)
    _HAVE_CV2 = False

try:
    import torch

    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False


# Precomputed ImageNet stats (RGB) used for zero-centering.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image_rgb(path: str) -> np.ndarray:
    """Load an image as an RGB uint8 array (H, W, 3) using OpenCV.

    Raises:
        FileNotFoundError if the image cannot be read.
    """
    if not _HAVE_CV2:
        raise ImportError("opencv-python is required for load_image_rgb().")
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def normalize_rgb(
    img: np.ndarray,
    mode: str = "imagenet",
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
) -> np.ndarray:
    """Normalise a float [0,1] RGB image (H,W,3).

    modes:
        * ``imagenet``: (img - mean) / std  using ImageNet stats,
        * ``unit``    : scale to [-1, 1],
        * ``none``    : leave untouched.
    """
    if mode == "unit":
        return (img - 0.5) * 2.0
    if mode == "imagenet":
        return (img - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    return img


def random_affine_augment(
    img: np.ndarray,
    flip_p: float = 0.5,
    rot_deg: float = 10.0,
    seed: int | None = None,
) -> np.ndarray:
    """Lightweight affine augmentation (no albumentations dependency).

    Returns a *view or copy* — never mutate in place upstream.
    """
    if not _HAVE_CV2:
        raise ImportError("opencv-python is required for random_affine_augment().")
    rng = np.random.default_rng(seed)
    out = img
    if rng.random() < flip_p:
        out = cv2.flip(out, 1)  # horizontal flip
    angle = rng.uniform(-rot_deg, rot_deg)
    if abs(angle) > 1e-3:
        h, w = out.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, m, (w, h), flags=cv2.INTER_LINEAR)
    return np.asarray(out, dtype=img.dtype)


def to_tensor(img: np.ndarray, dtype: str = "float32") -> "torch.Tensor":
    """Convert HWC numpy image to CHW torch tensor on CPU.

    Args:
        img: (H, W, 3) uint8 [0,255] or float [0,1] array.
        dtype: target element dtype ('float32' | 'uint8' | ...).
    """
    if not _HAVE_TORCH:
        raise ImportError("torch is required for to_tensor().")
    t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
    return getattr(torch, dtype)(t)
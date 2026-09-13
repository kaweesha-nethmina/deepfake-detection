"""
Self-contained utilities for Model 4 (ViT / frequency-hybrid).
Owner: Member C

Why this file exists
---------------------
The shared `src/` module (owned collectively, Member A set it up) is supposed
to provide `set_seed`, `load_config`, `build_dataset_paths`, etc. But my part
needs to run and be gradeable on its own, independent of whether `src/` is
finished, broken, or has a slightly different function signature than I
expect. So every helper here:

    1. First tries to import the "real" shared version from `src/`.
    2. Falls back to a local, fully working implementation if that import
       fails for any reason (missing module, missing function, wrong
       signature raising TypeError on call).

This means `models/vit/train.py` and `models/vit/evaluate_crossgen.py` are
runnable today, standalone, and will automatically start using the shared
`src/` versions once the team finishes them — no code change needed later.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from PIL import Image

# torch is optional at import time: prepare_data.py (dataset prep / leakage
# checks) only needs load_config/build_dataset_paths from this file and has
# no reason to require a working torch install. train.py / evaluate_crossgen.py
# DO need torch and will fail loudly and clearly if it isn't available.
try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # broken/incomplete torch installs can raise OSError, not just ImportError
    torch = None

    class Dataset:  # minimal stand-in so this module still imports without torch
        pass


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def _local_set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_seed(seed: int) -> None:
    try:
        from src import set_seed as shared_set_seed  # type: ignore

        shared_set_seed(seed)
    except Exception:
        _local_set_seed(seed)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def _local_load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(path: str) -> dict:
    try:
        from src import load_config as shared_load_config  # type: ignore

        return shared_load_config(path)
    except Exception:
        return _local_load_config(path)


# ---------------------------------------------------------------------------
# Dataset path resolution
# ---------------------------------------------------------------------------
def _local_build_dataset_paths(data_cfg: dict) -> dict:
    keys = ["train_dir", "val_dir", "test_dir", "cross_gen_test_dir"]
    paths = {}
    for k in keys:
        if k not in data_cfg:
            continue
        p = Path(data_cfg[k])
        if not p.exists():
            print(f"[warn] {k} -> '{p}' does not exist yet. "
                  f"Populate it via data/README.md's download scripts before training.")
        paths[k] = p
    return paths


def build_dataset_paths(data_cfg: dict) -> dict:
    try:
        from src import build_dataset_paths as shared_build_paths  # type: ignore

        return shared_build_paths(data_cfg)
    except Exception:
        return _local_build_dataset_paths(data_cfg)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class DeepfakeImageDataset(Dataset):
    """
    Expects a directory laid out as:

        <root>/real/*.jpg
        <root>/fake/*.jpg

    (matches the folder structure specified in the project guide's
    preprocessing pipeline: data/<split>/real, data/<split>/fake).

    label: 0 = real, 1 = fake
    """

    def __init__(self, root, image_size: int = 224, augment: bool = False):
        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment
        self.samples = []

        for label, cls_name in enumerate(["real", "fake"]):
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in IMG_EXTENSIONS:
                    self.samples.append((p, label))

        if len(self.samples) == 0:
            print(f"[warn] No images found under {self.root} "
                  f"(expected {self.root}/real/ and {self.root}/fake/). "
                  f"Dataset length will be 0 until data is downloaded.")

        self.transform = get_transforms(image_size, train=augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label


def get_transforms(image_size: int, train: bool):
    """
    Matches the preprocessing pipeline in the project guide:
    resize -> (train-only) flip/rotation/color-jitter/JPEG-recompression/blur
    -> ImageNet-style normalization (matches what pretrained ViT/CNN
    backbones expect).
    """
    import torchvision.transforms as T

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
            T.RandomApply([JPEGReCompress(quality_range=(40, 90))], p=0.3),
            T.ToTensor(),
            normalize,
        ])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        normalize,
    ])


class JPEGReCompress:
    """
    Simulates a social-media re-upload: re-encodes the PIL image through a
    random-quality JPEG round trip. Called out explicitly in the project
    guide's augmentation step as strengthening robustness / motivating the
    "practical limitations" discussion.
    """

    def __init__(self, quality_range=(40, 90)):
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        import io

        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

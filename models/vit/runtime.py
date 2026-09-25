"""Shared inference contract for training, evaluation and the Member C demo."""
from __future__ import annotations

import copy
import importlib
import inspect
import io
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T

from models.vit.manifest import image_path


def load_config(path):
    with open(path) as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping.")
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_for(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def probabilities(logits):
    if logits.ndim == 1:
        return logits.sigmoid()
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0].sigmoid()
    if logits.ndim == 2 and logits.shape[1] == 2:
        return logits.softmax(dim=1)[:, 1]
    raise ValueError(f"Expected [B], [B,1] or [B,2] logits, got {tuple(logits.shape)}")


def binary_loss(logits, labels):
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
        return torch.nn.functional.binary_cross_entropy_with_logits(logits.reshape(-1), labels.float())
    if logits.ndim == 2 and logits.shape[1] == 2:
        return torch.nn.functional.cross_entropy(logits, labels.long())
    raise ValueError("Unsupported classifier output.")


def build_model(cfg, pretrained=None):
    cfg = copy.deepcopy(cfg)
    module = importlib.import_module(cfg["module"])
    if pretrained is not None:
        cfg.setdefault("model", {})["pretrained"] = pretrained
        if not pretrained:
            cfg["model"]["weights"] = False
    if hasattr(module, "build_model"):
        return module.build_model(cfg)
    if hasattr(module, "get_model"):
        kwargs = {"num_classes": cfg.get("model", {}).get("num_classes", 2)}
        if "pretrained" in inspect.signature(module.get_model).parameters:
            kwargs["pretrained"] = cfg.get("model", {}).get("pretrained", True)
        return module.get_model(**kwargs)
    raise ValueError(f"{cfg['module']} exposes neither build_model nor get_model.")


def preprocessing_for(model, cfg):
    backbone = getattr(model, "backbone", model)
    if hasattr(backbone, "pretrained_cfg"):
        from timm.data import resolve_model_data_config
        data = resolve_model_data_config(backbone)
        mean, std = data["mean"], data["std"]
        interpolation = data.get("interpolation", "bicubic")
    else:
        data = cfg.get("preprocessing", {})
        mean = data.get("mean", [0.485, 0.456, 0.406])
        std = data.get("std", [0.229, 0.224, 0.225])
        interpolation = data.get("interpolation", "bilinear")
    return {"image_size": 224, "mean": list(mean), "std": list(std),
            "interpolation": interpolation, "color": "RGB", "resize": "square",
            "crop": "already_cropped_wish_faces"}


class JPEGRecompress:
    def __call__(self, image):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(70, 100))
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")


def transform_for(metadata, train=False):
    interpolation = T.InterpolationMode(metadata["interpolation"])
    ops = [T.Resize((metadata["image_size"], metadata["image_size"]), interpolation=interpolation)]
    if train:
        ops += [T.RandomHorizontalFlip(), T.RandomRotation(10),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.RandomApply([JPEGRecompress()], p=0.3),
                T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.2)]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(metadata["mean"], metadata["std"])])


class ManifestDataset(Dataset):
    def __init__(self, rows, root, split, metadata, train=False, limit=None):
        self.rows = [row for row in rows if row["split"] == split]
        if limit:
            self.rows = [row for label in (0, 1)
                         for row in [r for r in self.rows if r["label"] == label][:max(1, limit // 2)]]
        if not self.rows:
            raise ValueError(f"Empty dataset: {split}")
        self.root = root
        self.transform = transform_for(metadata, train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(image_path(self.root, row["filepath"])) as image:
            tensor = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
        return tensor, int(row["label"]), index


def load_checkpoint(cfg, path, device):
    model = build_model(cfg, pretrained=False).to(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state, strict=True)
    metadata = checkpoint.get("metadata")
    if metadata is None:
        sidecar = Path(str(path) + ".json")
        if not sidecar.exists():
            raise ValueError("Legacy checkpoint requires a verified <checkpoint>.json metadata sidecar.")
        metadata = json.loads(sidecar.read_text())
    if metadata.get("model_config") != cfg["model"] or metadata.get("module") != cfg["module"]:
        raise ValueError("Checkpoint metadata does not match the selected model configuration.")
    if metadata.get("label_mapping") != {"real": 0, "fake": 1}:
        raise ValueError("Checkpoint must declare real=0 and fake=1.")
    return model.eval(), metadata


@torch.inference_mode()
def predict_image(model, image, metadata, device):
    tensor = transform_for(metadata)(ImageOps.exif_transpose(image).convert("RGB"))
    return float(probabilities(model(tensor.unsqueeze(0).to(device)))[0].cpu())

"""Exact 224x224 decode cache: pay image decoding and resizing once per session.

The first training and validation transform is a deterministic Resize to 224x224.
Storing its uint8 output and applying the remaining (random) transforms to it gives
pixel-identical tensors, so the cache changes speed only, never the experiment.
It lives in local scratch space (never /kaggle/working, so it is not published)
and is rebuilt in each Kaggle session. Training falls back to disk reads when the
cache is absent, stale or does not match the model's preprocessing.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms as T

from models.vit.audit_progress import bounded_map, stage
from models.vit.manifest import image_path, load_audited_manifest

FORMAT = "wish_presize_v1"
ENVIRONMENT_VARIABLE = "WISH_PRESIZE_CACHE"


def resize_op(preprocessing):
    size = preprocessing["image_size"]
    return T.Resize((size, size), interpolation=T.InterpolationMode(preprocessing["interpolation"]))


def signature(manifest_sha256, filepaths, preprocessing):
    return {"format": FORMAT, "manifest_sha256": manifest_sha256,
            "filepaths_sha256": hashlib.sha256("\n".join(filepaths).encode()).hexdigest(),
            "image_size": preprocessing["image_size"], "interpolation": preprocessing["interpolation"],
            "count": len(filepaths)}


def build(manifest, root, directory, preprocessing, splits=("train", "val"), workers=4, headroom_bytes=2 << 30):
    """Create (or verify and reuse) the cache. Returns its directory, or None when there is no room."""
    rows, audit = load_audited_manifest(manifest, root, verify_splits=())
    filepaths = sorted(row["filepath"] for row in rows if row["split"] in splits)
    expected = signature(audit["manifest_sha256"], filepaths, preprocessing)
    directory = Path(directory)
    size = preprocessing["image_size"]
    shape = (len(filepaths), size, size, 3)
    data = directory / "images.u8"
    meta = directory / "cache.json"
    if meta.is_file() and data.is_file() and json.loads(meta.read_text()) == expected \
            and data.stat().st_size == int(np.prod(shape)):
        print(f"Reusing presize cache: {len(filepaths):,} images in {directory}", flush=True)
        return directory
    needed = int(np.prod(shape))
    directory.mkdir(parents=True, exist_ok=True)
    meta.unlink(missing_ok=True)
    data.unlink(missing_ok=True)
    free = shutil.disk_usage(directory).free
    if free < needed + headroom_bytes:
        print(f"Presize cache skipped: needs {needed / 2**30:.1f} GiB, only {free / 2**30:.1f} GiB free "
              f"in {directory}. Training reads images from disk instead.", flush=True)
        return None
    array = np.memmap(data, dtype=np.uint8, mode="w+", shape=shape)
    resize = resize_op(preprocessing)

    def store(item):
        index, relative = item
        with Image.open(image_path(root, relative)) as image:
            array[index] = np.asarray(resize(ImageOps.exif_transpose(image).convert("RGB")), dtype=np.uint8)
        return None

    print(f"Building exact {size}x{size} presize cache for {len(filepaths):,} train/val images "
          f"({needed / 2**30:.1f} GiB in {directory}).", flush=True)
    with stage("Decode and resize once", total=len(filepaths)) as progress:
        bounded_map(store, list(enumerate(filepaths)), workers, progress)
    array.flush()
    del array
    (directory / "index.json").write_text(json.dumps(filepaths))
    meta.write_text(json.dumps(expected, indent=2))  # written last: marks the cache complete
    return directory


class PresizeCache:
    """Read-only view shared by DataLoader workers; the memmap opens lazily inside each worker."""

    def __init__(self, directory, preprocessing, manifest_sha256):
        directory = Path(directory)
        meta = json.loads((directory / "cache.json").read_text())
        filepaths = json.loads((directory / "index.json").read_text())
        if meta != signature(manifest_sha256, filepaths, preprocessing):
            raise ValueError("Presize cache does not match this manifest or preprocessing.")
        size = preprocessing["image_size"]
        self.path = directory / "images.u8"
        self.shape = (len(filepaths), size, size, 3)
        self.index = {relative: position for position, relative in enumerate(filepaths)}
        self._array = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_array"] = None  # never pickle gigabytes into worker processes
        return state

    def get(self, relative):
        position = self.index.get(relative)
        if position is None:
            return None
        if self._array is None:
            self._array = np.memmap(self.path, dtype=np.uint8, mode="r", shape=self.shape)
        return Image.fromarray(np.array(self._array[position]))


def open_cache(preprocessing, manifest_sha256):
    directory = os.environ.get(ENVIRONMENT_VARIABLE)
    if not directory:
        return None
    try:
        return PresizeCache(directory, preprocessing, manifest_sha256)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Presize cache ignored ({error}); reading images from disk.", flush=True)
        return None

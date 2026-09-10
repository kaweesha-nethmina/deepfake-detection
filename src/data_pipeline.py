"""Shared data-pipeline helpers (owner A).

Responsibilities: path layout, face cropping, and deterministic splits.

Additive-only API. The train/val/test *splitting convention* below must stay
stable:

    data/raw/
        real/           ...0001.jpg        # identity-preserving images
        fake/<generator>/...0001.jpg       # generator-labelled fakes

Cross-generator test sets live in ``data/cross_gen_test/<generator>/``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .utils import resolve_path

try:
    import cv2

    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False

_FAKE = 1
_REAL = 0


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def build_dataset_paths(
    raw_dir: str | Path,
    expected_splits: tuple[str, ...] = ("real", "fake"),
) -> dict[str, Path]:
    """Return a dict of Paths for raw/processed/cross-gen dirs.

    Creates the folders if missing (idempotent). Additive-friendly — new keys
    can be appended without breaking old callers.
    """
    raw_dir = resolve_path(raw_dir)
    processed_dir = raw_dir.with_name("processed")
    cross_gen_dir = raw_dir.with_name("cross_gen_test") if raw_dir.name == "raw" else raw_dir.parent / "cross_gen_test"

    for d in (raw_dir, processed_dir, cross_gen_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "raw": raw_dir,
        "processed": processed_dir,
        "cross_gen_test": cross_gen_dir,
        "expected_splits": sorted(expected_splits),
    }


def face_crop_and_align(
    img: np.ndarray,
    detector: str = "opencv_dnn",
    expand_ratio: float = 0.2,
    min_size: int = 64,
) -> np.ndarray:
    """Detect the largest face and return a squared crop.

    Args:
        img: RGB uint8 (H, W, 3).
        detector: ``opencv_dnn`` (default; ships with opencv-python) — swap to a
            better detector (RetinaFace / MTCNN) by adding a new function, do
            not rewrite this signature.
        expand_ratio: fraction of bbox width/height to add as context padding.
        min_size: if the detected bbox is smaller than this, ignore detection
            and return the image resized instead.

    Returns:
        Cropped (and possibly padded) square RGB image. If no face is found the
        original frame is returned *unaltered* so the pipeline never drops data.
    """
    if not _HAVE_CV2:
        raise ImportError("opencv-python is required for face_crop_and_align().")

    rgb = img
    h, w = rgb.shape[:2]
    face = _detect_largest_face(rgb)
    if face is None:
        return rgb

    x, y, bw, bh = face
    if max(bw, bh) < min_size:
        return rgb

    ew = float(bw) * (1 + expand_ratio)
    eh = float(bh) * (1 + expand_ratio)
    cx, cy = x + bw / 2, y + bh / 2
    x0 = int(max(0, cx - ew / 2))
    y0 = int(max(0, cy - eh / 2))
    x1 = int(min(w, cx + ew / 2))
    y1 = int(min(h, cy + eh / 2))
    return rgb[y0:y1, x0:x1]


def _detect_largest_face(rgb: np.ndarray):
    """OpenCV DNN face detector (ResNet + SSD). Returns (x, y, w, h) or None."""
    if not _HAVE_CV2:
        return None
    try:
        proto = cv2.data.haarcascades
        _ = proto  # keep import consistent for non-cv2 builds
    except AttributeError:
        pass
    # OpenCV ships a bundled SSD face detector model.
    proto_path = Path(cv2.__file__).parent / "data" / "dnn" / "deploy.prototxt"
    model_path = Path(cv2.__file__).parent / "data" / "dnn" / "res10_300x300_ssd_iter_140000.caffemodel"
    if not (proto_path.exists() and model_path.exists()):
        # Fall back to Haar cascade bundled with opencv.
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        (x, y, w, h), *_ = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return int(x), int(y), int(w), int(h)

    net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
    blob = cv2.dnn.blobFromImage(cv2.resize(rgb, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    best = None
    best_conf = 0.0
    hh, ww = rgb.shape[:2]
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < 0.6 or conf <= best_conf:
            continue
        best_conf = conf
        box = detections[0, 0, i, 3:7] * np.array([ww, hh, ww, hh])
        x, y, x2, y2 = box.astype(int)
        best = (x, y, x2 - x, y2 - y)
    return best


def balanced_split(
    y: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic, stratified split of row indices.

    Args:
        y: integer class labels aligned to the rows (0/fake, 1/real).
        train_ratio, val_ratio: remaining goes to test.

    Returns:
        (train_idx, val_idx, test_idx) integer arrays. This function is the
        single source of truth for splits, so every member reproduces the same
        partition from the same seed.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    if stratify:
        idx = _stratified_permute(y, rng)
    else:
        rng.shuffle(idx)

    n_tr = int(round(n * train_ratio))
    n_va = int(round(n * val_ratio))
    train_idx = idx[:n_tr]
    val_idx = idx[n_tr : n_tr + n_va]
    test_idx = idx[n_tr + n_va :]
    return train_idx, val_idx, test_idx


def _stratified_permute(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    classes = sorted(set(int(v) for v in y.tolist()))
    per_class: list[np.ndarray] = []
    for c in classes:
        members = np.flatnonzero(y == c)
        rng.shuffle(members)
        per_class.append(members)
    # interleave classes so every slice keeps approximately the class balance
    out: list[int] = []
    ptr = {c: 0 for c in classes}
    counts = {c: len(per_class[c]) for c in classes}
    while True:
        progressed = False
        for c in classes:
            if ptr[c] < counts[c]:
                out.append(int(per_class[c][ptr[c]]))
                ptr[c] += 1
                progressed = True
        if not progressed:
            break
    return np.asarray(out, dtype=np.int64)


def content_hash(path: Path, algo: str = "md5", chunk: int = 1 << 16) -> str:
    """Hash a file's content (not just the name) — used to dedupe/cache splits."""
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
"""Shared utilities: seeding, config loading, device selection, logging.

Public API is additive-only. If you MUST change a signature, flag it in the
PR and BOTH the docstring and the README note the deprecation.

Usage::

    from src import set_seed, load_config, get_device, get_logger
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Seed python, numpy and torch for reproducibility.

    Args:
        seed: master seed.
        deterministic: also call ``torch.backends.cudnn.deterministic`` and
            disable benchmark mode (slightly slower, fully reproducible).

    Returns:
        The effective seed (same as input).
    """
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch not installed on this machine yet
        pass
    return seed


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: str | os.PathLike) -> dict:
    """Load a ``configs/<model>.yaml`` file into a plain dict.

    Backward-compatible notes:
        * all keys are read verbatim; new keys are additive,
        * paths inside the file are treated as relative to the *repo root*
          unless an absolute path is given (see ``resolve_path``).

    Args:
        path: path to a yaml config file.

    Returns:
        A dict mirroring the yaml structure.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        raise ValueError(f"Config file is empty: {path}")
    cfg["_config_path"] = str(path.resolve())
    return cfg


def repo_root() -> Path:
    """Absolute path to the repository root (one level above ``src/``)."""
    return Path(__file__).resolve().parent.parent


def resolve_path(p: str | os.PathLike) -> Path:
    """Resolve a path from the config against the repo root.

    Absolute paths are returned untouched; relative ones are anchored at the
    repo root so that a members' local machine always resolves consistently.
    """
    path = Path(p)
    if path.is_absolute():
        return path
    return repo_root() / path


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def get_device(preference: str = "auto") -> str:
    """Choose the best available torch device.

    Args:
        preference: ``auto`` | ``cpu`` | ``cuda`` | ``mps``.

    Returns:
        A torch device string (``cuda``, ``mps`` or ``cpu``).
    """
    import torch

    if preference == "cpu":
        return "cpu"
    if preference == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preference in ("mps", "auto"):
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    if preference == "auto" and torch.cuda.is_available():
        return "cuda"
    if preference in ("cuda", "mps") and preference != "cpu":
        # requested but unavailable -> fall back with a warning
        logging.getLogger(__name__).warning(
            "Requested device %r unavailable; falling back to cpu.", preference
        )
    return "cpu"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def get_logger(name: str = "deepfake", level: int = logging.INFO) -> logging.Logger:
    """Return a configured module-level logger.

    Additive-safe: does not reconfigure already-initialized loggers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
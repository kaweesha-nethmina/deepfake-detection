"""Shared module for the deepfake cross-generator detection project.

`src` is the ONLY place shared logic lives. Everyone imports from here
instead of copy-pasting. See rules in the root README:

  * changes must be additive and backward-compatible,
  * a breaking change must be flagged in a PR, never silently merged,
  * keep the public API below stable.

Member A owns this folder's EDA/data parts; B/C may propose additive helpers.
"""

from .utils import set_seed, load_config, get_device, get_logger
from .metrics import (
    evaluate_binary,
    accuracy,
    precision_recall,
    f1_score_binary,
    roc_auc,
    confusion_matrix_df,
)
from .preprocessing import (
    load_image_rgb,
    normalize_rgb,
    random_affine_augment,
    to_tensor,
)
from .data_pipeline import (
    build_dataset_paths,
    face_crop_and_align,
    balanced_split,
)

__all__ = [
    # utils
    "set_seed",
    "load_config",
    "get_device",
    "get_logger",
    # metrics
    "evaluate_binary",
    "accuracy",
    "precision_recall",
    "f1_score_binary",
    "roc_auc",
    "confusion_matrix_df",
    # preprocessing
    "load_image_rgb",
    "normalize_rgb",
    "random_affine_augment",
    "to_tensor",
    # data_pipeline
    "build_dataset_paths",
    "face_crop_and_align",
    "balanced_split",
]

__version__ = "0.1.0"
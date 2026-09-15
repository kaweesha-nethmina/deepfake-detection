"""
Shared utility functions used by every model's train.py.
Rule for the team: only ADD new functions here, never change the
signature of an existing one that someone else already depends on.
"""
import os
import random
import yaml
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Make results reproducible across runs and across team members."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load a model's YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Total trainable parameter count, used for the efficiency/complexity
    comparison table in the report."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class EarlyStopping:
    """Stops training when validation loss hasn't improved for `patience` epochs
    and remembers the best model's state_dict."""

    def __init__(self, patience: int = 6, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state_dict = None
        self.should_stop = False

    def step(self, val_loss: float, model: torch.nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

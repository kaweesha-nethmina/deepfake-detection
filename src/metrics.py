"""
Shared evaluation utilities so every model is scored the exact same way.
Rule for the team: only ADD new metric functions here, don't change
what an existing one returns.
"""
import time
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)


@torch.no_grad()
def get_predictions(model, dataloader, device):
    """Runs the model over a dataloader and returns true labels,
    predicted labels (threshold 0.5), and predicted probabilities."""
    model.eval()
    all_labels, all_probs = [], []
    start = time.time()
    n_images = 0
    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]  # probability of "fake"
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
        n_images += images.size(0)
    elapsed = time.time() - start
    ms_per_image = (elapsed / max(n_images, 1)) * 1000

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob, ms_per_image


def compute_all_metrics(y_true, y_pred, y_prob) -> dict:
    """Returns the full metric set required by the assignment's
    classification evaluation criteria."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def get_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return fpr, tpr


def evaluate_model(model, dataloader, device) -> dict:
    """Convenience wrapper: run predictions + compute metrics + timing
    in one call. Used identically by every model's train.py so results
    are directly comparable."""
    y_true, y_pred, y_prob, ms_per_image = get_predictions(model, dataloader, device)
    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    metrics["ms_per_image"] = ms_per_image
    metrics["y_true"] = y_true.tolist()
    metrics["y_prob"] = y_prob.tolist()
    return metrics

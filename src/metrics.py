"""Shared binary-classification evaluation metrics.

API is additive-only. These are thin, dependency-light wrappers so every model
script reports results identically (henece comparable across generators/models).

Usage::

    from src import evaluate_binary
    report = evaluate_binary(y_true, y_score)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class BinaryReport:
    """Container summarising a single binary-evaluation run."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    n: int

    def to_dict(self) -> dict:
        return asdict(self)


def _copy(y) -> np.ndarray:
    return np.asarray(y, dtype=np.float64)


def accuracy(y_true, y_pred) -> float:
    y_true = _copy(y_true)
    y_pred = _copy(y_pred) >= 0.5
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def precision_recall(y_true, y_pred) -> tuple[float, float]:
    y_true = _copy(y_true)
    y_pred = _copy(y_pred) >= 0.5
    tp = float((y_pred & (y_true == 1)).sum())
    fp = float((y_pred & (y_true == 0)).sum())
    fn = float(((~y_pred) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return precision, recall


def f1_score_binary(y_true, y_pred) -> float:
    precision, recall = precision_recall(y_true, y_pred)
    if (precision + recall) == 0:
        return float("nan")
    return float(2 * precision * recall / (precision + recall))


def roc_auc(y_true, y_score) -> float:
    """ROC-AUC from scores, using the Mann-Whitney U estimator (base numpy)."""
    y_true = _copy(y_true)
    y_score = _copy(y_score)
    order = np.argsort(y_score, kind="mergesort")
    sorted_y = y_true[order]
    n_pos = int(sorted_y.sum())
    n_neg = len(sorted_y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, len(sorted_y) + 1)
    ties = np.flatnonzero(np.diff(y_score[order]) == 0)
    if len(ties):
        # average ranks inside each tie group
        group = np.split(ranks, ties + 1)
        group = [g for g in group if len(g)]
        for i, g in enumerate(group):
            ranks[g[0] : g[-1] + 1] = g.mean()  # noqa
    auc = (ranks[sorted_y == 1].sum() - n_pos * (n_pos + 1) / 2) / (
        n_pos * n_neg
    )
    return float(auc)


def confusion_matrix_df(y_true, y_pred, labels=("fake", "real")) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame (base numpy, no sklearn needed)."""
    y_true = _copy(y_true).astype(int)
    y_pred = (_copy(y_pred) >= 0.5).astype(int)
    n = len(y_true)
    mat = np.zeros((2, 2), dtype=int)
    for i in range(n):
        mat[y_true[i], y_pred[i]] += 1
    # rows = true, cols = predicted; labels fixed to 0/"fake",1/"real"
    return pd.DataFrame(
        mat,
        index=pd.Index(labels, name="true"),
        columns=pd.Index(labels, name="pred"),
    )


def evaluate_binary(y_true, y_score) -> BinaryReport:
    """Full report in one call; the standard contract used by all model scripts."""
    y_true = _copy(y_true).astype(int)
    y_score = _copy(y_score)
    pred = y_score >= 0.5
    precision, recall = precision_recall(y_true, pred)
    return BinaryReport(
        accuracy=accuracy(y_true, pred),
        precision=precision,
        recall=recall,
        f1=f1_score_binary(y_true, pred),
        roc_auc=roc_auc(y_true, y_score),
        n=len(y_true),
    )
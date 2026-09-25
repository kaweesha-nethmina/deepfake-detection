"""Metrics and figures regenerated from exported probabilities, never constants."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)


def compute_metrics(labels, scores, threshold=0.5):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("Labels and probabilities must be nonempty, aligned vectors.")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Invalid labels or nonfinite probabilities.")
    if ((scores < 0) | (scores > 1)).any() or not 0 < threshold < 1:
        raise ValueError("Invalid probability or threshold.")
    pred = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred, average="binary", zero_division=0)
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    return {"accuracy": float(accuracy_score(labels, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
            "precision": float(precision), "recall": float(recall), "f1_score": float(f1),
            "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
            "confusion_matrix": cm.tolist(), "n": len(labels), "threshold": threshold,
            "false_positive_rate": float(cm[0, 1] / cm[0].sum()) if cm[0].sum() else None}


def make_figures(predictions, output, threshold=0.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(predictions)
    summaries = []
    for (name, split), group in frame.groupby(["model", "split"], sort=True):
        metrics = compute_metrics(group.label, group.probability, threshold)
        summaries.append({"model": name, "split": split,
                          **{k: v for k, v in metrics.items() if k != "confusion_matrix"}})
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        cm = np.asarray(metrics["confusion_matrix"])
        axes[0].imshow(cm, cmap="Blues")
        for y in (0, 1):
            for x in (0, 1):
                axes[0].text(x, y, str(cm[y, x]), ha="center", va="center",
                             color="white" if cm[y, x] > cm.max() / 2 else "black")
        axes[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Real", "Fake"],
                    yticklabels=["Real", "Fake"], xlabel="Predicted", ylabel="True")
        if metrics["roc_auc"] is not None:
            fpr, tpr, _ = roc_curve(group.label, group.probability)
            axes[1].plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
            axes[1].legend()
        axes[1].plot([0, 1], [0, 1], "k--")
        axes[1].set(xlabel="False positive rate", ylabel="True positive rate")
        fig.suptitle(f"{name}: {split}")
        fig.tight_layout()
        fig.savefig(output / f"{name}_{split}.png", dpi=180)
        plt.close(fig)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "metrics_from_predictions.csv", index=False)
    errors = frame[(frame.probability >= threshold).astype(int) != frame.label].copy()
    errors["error_type"] = np.where(errors.label == 0, "false_positive", "false_negative")
    errors["wrong_confidence"] = np.where(errors.label == 0, errors.probability, 1 - errors.probability)
    errors.sort_values("wrong_confidence", ascending=False).to_csv(output / "failure_cases.csv", index=False)
    pivot = summary.pivot(index="model", columns="split", values="f1_score")
    ax = pivot.plot.bar(rot=15, ylim=(0, 1), ylabel="F1", figsize=(9, 5))
    ax.figure.tight_layout()
    ax.figure.savefig(output / "generalization_f1.png", dpi=180)
    plt.close(ax.figure)
    return summary


def learning_curves(history, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for part in ("train", "val"):
        axes[0].plot(frame.epoch, frame[f"{part}_loss"], label=part)
        axes[1].plot(frame.epoch, frame[f"{part}_accuracy"], label=part)
    axes[0].set(xlabel="Epoch", ylabel="Loss")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)

"""ViT quality evidence from recorded history and frozen per-image predictions."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             cohen_kappa_score, confusion_matrix, log_loss,
                             matthews_corrcoef, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)


def calibration_bins(labels, scores, bins=10):
    y, p = np.asarray(labels), np.asarray(scores, dtype=float)
    indices = np.minimum((p * bins).astype(int), bins - 1)
    return pd.DataFrame([{"bin": i, "count": int((indices == i).sum()),
                          "mean_probability": float(p[indices == i].mean()),
                          "observed_fake_fraction": float(y[indices == i].mean())}
                         for i in range(bins) if (indices == i).any()])


def compute_metrics(labels, scores, threshold=0.5):
    y, p = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Nonempty aligned label/probability vectors required")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Labels must be 0/1; probabilities finite and in [0,1]")
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be in (0,1)")
    predicted = (p >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, predicted, average="binary", zero_division=0)
    cm = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())
    both = len(np.unique(y)) == 2
    specificity = tn / (tn + fp) if tn + fp else None
    calibration = calibration_bins(y, p)
    ece = float((calibration["count"] * (calibration.mean_probability - calibration.observed_fake_fraction).abs()).sum() / len(y))
    return {"accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float((specificity + recall) / 2) if both else None,
            "precision": float(precision), "recall": float(recall), "f1_score": float(f1),
            "roc_auc": float(roc_auc_score(y, p)) if both else None,
            "average_precision": float(average_precision_score(y, p)) if both else None,
            "specificity": specificity, "false_positive_rate": fp / (tn + fp) if tn + fp else None,
            "false_negative_rate": fn / (tp + fn) if tp + fn else None,
            "mcc": float(matthews_corrcoef(y, predicted)) if both else None,
            "cohen_kappa": float(cohen_kappa_score(y, predicted)) if both else None,
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier_score": float(np.mean((p - y) ** 2)), "ece_10_equal_width_bins": ece,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "confusion_matrix": cm.tolist(),
            "n": len(y), "threshold": threshold}


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    pyplot().close(fig)


def learning_curves(history, output):
    frame, output = pd.DataFrame(history), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt = pyplot()
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for axis, metric, title in zip(axes.flat, ("loss", "accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "lr"),
                                    ("BCE loss", "Accuracy", "Fake precision", "Fake recall", "Fake F1", "ROC-AUC", "Average precision (PR)", "Learning rate")):
        plotted = False
        if metric == "lr" and "lr" in frame:
            axis.plot(frame.epoch, frame.lr, color="#333333", marker=".")
            plotted = True
        elif metric != "lr":
            for split, color in (("train", "#007f86"), ("val", "#c33d5b")):
                key = f"{split}_{metric}"
                if key in frame:
                    axis.plot(frame.epoch, frame[key], label=split, color=color, marker=".")
                    plotted = True
            if plotted:
                axis.legend()
        if not plotted:
            axis.text(0.5, 0.5, "Not recorded in this run", ha="center", transform=axis.transAxes)
        axis.set(title=title, xlabel="Epoch")
        if metric not in ("loss", "lr"):
            axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.2)
    fig.suptitle("Training and validation only; training metrics use augmented batches", fontsize=13)
    save(fig, output)
    if "elapsed_seconds" in frame:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(frame.epoch, frame.elapsed_seconds / 3600)
        axes[0].set(title="Cumulative training + validation time", xlabel="Epoch", ylabel="Hours")
        axes[1].bar(frame.epoch, frame.elapsed_seconds.diff().fillna(frame.elapsed_seconds) / 60)
        axes[1].set(title="Epoch duration", xlabel="Epoch", ylabel="Minutes")
        save(fig, output.with_name("epoch_timing.png"))


def validate_predictions(frame):
    required = {"model", "run_name", "filepath", "label", "probability", "predicted_label", "split", "generator"}
    if not required.issubset(frame):
        raise ValueError(f"Predictions missing columns: {sorted(required - set(frame))}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Predictions contain missing required values")
    if frame[["model", "run_name"]].drop_duplicates().shape[0] != 1:
        raise ValueError("This report requires exactly one model/run")
    if set(frame.split) != {"test", "cross_gen"}:
        raise ValueError("Report requires primary and cross-generator test predictions")
    if frame.filepath.duplicated().any():
        raise ValueError("Repeated filepath / test split overlap")
    if not np.array_equal(frame.predicted_label, (frame.probability >= 0.5).astype(int)):
        raise ValueError("Saved predicted labels do not match the frozen 0.5 threshold")
    for _, group in frame.groupby("split"):
        compute_metrics(group.label, group.probability)
        if set(group.label) != {0, 1}:
            raise ValueError("Each test domain must contain real and fake controls")


def make_figures(predictions, output, threshold=0.5):
    if threshold != 0.5:
        raise ValueError("Reporting cannot retune the frozen threshold")
    output = Path(output)
    frame = pd.read_csv(predictions)
    validate_predictions(frame)
    output.mkdir(parents=True, exist_ok=True)
    plt = pyplot()
    parts = [(split, frame[frame.split == split]) for split in ("test", "cross_gen")]
    results = {split: compute_metrics(group.label, group.probability) for split, group in parts}
    rows = [{"split": split, **{k: v for k, v in result.items() if k != "confusion_matrix"}}
            for split, result in results.items()]
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "metrics_from_predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    reports = {split: classification_report(group.label, group.predicted_label, labels=[0, 1],
                                             target_names=["Real", "Fake"], output_dict=True, zero_division=0)
               for split, group in parts}
    (output / "classification_reports.json").write_text(json.dumps(reports, indent=2))
    pd.DataFrame([{"split": split, "class": cls, **reports[split][cls]}
                  for split, _ in parts for cls in ("Real", "Fake")]).to_csv(output / "per_class_metrics.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for col, (split, _) in enumerate(parts):
        cm = np.array(results[split]["confusion_matrix"])
        for row in (0, 1):
            values = cm if row == 0 else cm / cm.sum(axis=1, keepdims=True)
            axis = axes[row, col]
            axis.imshow(values, cmap="Blues", vmin=0, vmax=values.max() if row == 0 else 1)
            for y in (0, 1):
                for x in (0, 1):
                    text = str(cm[y, x]) if row == 0 else f"{values[y, x]:.1%}"
                    axis.text(x, y, text, ha="center", va="center", color="white" if values[y, x] > values.max() / 2 else "black")
            axis.set(title=f"{split}: {'counts' if row == 0 else 'row-normalized'}", xlabel="Predicted", ylabel="True",
                     xticks=[0, 1], yticks=[0, 1], xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    save(fig, output / "confusion_counts_and_rates.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (split, group), color in zip(parts, ("#007f86", "#c33d5b")):
        fpr, tpr, _ = roc_curve(group.label, group.probability)
        precision, recall, _ = precision_recall_curve(group.label, group.probability)
        axes[0].plot(fpr, tpr, color=color, label=f"{split}: AUC {results[split]['roc_auc']:.3f}")
        axes[1].plot(recall, precision, color=color, label=f"{split}: AP {results[split]['average_precision']:.3f}")
        axes[1].axhline(group.label.mean(), color=color, linestyle=":", alpha=0.5)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set(title="ROC", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Precision-recall (dotted = fake prevalence)", xlabel="Recall", ylabel="Precision")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.legend()
    save(fig, output / "roc_precision_recall.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = []
    for axis, (split, group) in zip(axes, parts):
        table = calibration_bins(group.label, group.probability)
        table.insert(0, "split", split)
        bins.append(table)
        axis.plot([0, 1], [0, 1], "k--", alpha=0.4)
        axis.plot(table.mean_probability, table.observed_fake_fraction, "o-", color="#007f86")
        axis.set(title=f"{split}: reliability (10 bins)", xlabel="Mean predicted fake probability",
                 ylabel="Observed fake fraction", xlim=(0, 1), ylim=(0, 1))
    pd.concat(bins).to_csv(output / "calibration_bins.csv", index=False)
    save(fig, output / "calibration.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, (split, group) in zip(axes, parts):
        for label, name, color in ((0, "Real", "#007f86"), (1, "Fake", "#c33d5b")):
            axis.hist(group[group.label == label].probability, bins=np.linspace(0, 1, 26), alpha=0.5, label=name, color=color)
        axis.axvline(0.5, color="black", linestyle="--")
        axis.set(title=split, xlabel="Predicted fake probability", ylabel="Image count")
        axis.legend()
    save(fig, output / "probability_distributions.png")

    names = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "average_precision"]
    ax = summary.set_index("split")[names].T.plot.bar(figsize=(11, 5), rot=15, ylim=(0, 1.05), color=["#007f86", "#c33d5b"])
    ax.set(ylabel="Score", title="Absolute performance by test domain; higher is better")
    save(ax.figure, output / "domain_comparison.png")
    gaps = {"accuracy_drop_pp": 100 * (results["test"]["accuracy"] - results["cross_gen"]["accuracy"]),
            "f1_drop": results["test"]["f1_score"] - results["cross_gen"]["f1_score"]}
    pd.DataFrame([gaps]).to_csv(output / "generalization_gaps.csv", index=False)
    source_rows = []
    for (split, source), group in frame.groupby(["split", "generator"]):
        source_rows.append({"split": split, "source": source, "n": len(group),
                            "accuracy": float((group.label == group.predicted_label).mean()),
                            "fake_prediction_rate": float(group.predicted_label.mean()),
                            "real_count": int((group.label == 0).sum()), "fake_count": int((group.label == 1).sum())})
    pd.DataFrame(source_rows).to_csv(output / "source_breakdown.csv", index=False)
    errors = frame[frame.label != frame.predicted_label].copy()
    errors["error_type"] = np.where(errors.label == 0, "false_positive", "false_negative")
    errors["wrong_confidence"] = np.where(errors.label == 0, errors.probability, 1 - errors.probability)
    errors.sort_values("wrong_confidence", ascending=False).to_csv(output / "failure_cases.csv", index=False)
    checksum = hashlib.sha256(Path(predictions).read_bytes()).hexdigest()
    (output / "report_provenance.json").write_text(json.dumps({"predictions_sha256": checksum, "threshold": 0.5,
        "positive_class": "fake=1", "scope": "independent ViT, two test domains", "no_new_inference": True,
        "notes": ["Average precision is step-weighted AP, not trapezoidal PR area.",
                  "ECE uses ten equal-width fake-probability bins and is sample/bin dependent.",
                  "Source-only accuracy is not a balanced binary generalization score."]}, indent=2))
    return summary


def failure_gallery(errors_path, root, output):
    from PIL import Image, ImageOps
    errors = pd.read_csv(errors_path)
    root = Path(root).resolve()
    selected = pd.concat([errors[(errors.split == split) & (errors.error_type == kind)].head(2)
                          for split in ("test", "cross_gen") for kind in ("false_positive", "false_negative")])
    if selected.empty:
        return
    fig, axes = pyplot().subplots(2, 4, figsize=(13, 7))
    for axis in axes.flat:
        axis.axis("off")
    for axis, (_, row) in zip(axes.flat, selected.iterrows()):
        path = (root / row.filepath).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Image path escapes the dataset root")
        if path.is_file():
            with Image.open(path) as image:
                axis.imshow(ImageOps.exif_transpose(image).convert("RGB"))
        else:
            axis.text(0.5, 0.5, "Image not attached", ha="center", transform=axis.transAxes)
        axis.set_title(f"{row.split}: {row.error_type}\np(fake)={row.probability:.3f}", fontsize=9)
    save(fig, Path(output) / "high_confidence_mistakes.png")


def render_dashboard(run_dir, evaluation_dir, output, data_root=None):
    run_dir, evaluation_dir, output = Path(run_dir), Path(evaluation_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history = run_dir / "training_history.csv"
    if history.exists():
        recorded = pd.read_csv(history)
        learning_curves(recorded, output / "training_dashboard.png")
        metadata_path = run_dir / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        info = {"completed_epochs": int(recorded.epoch.iloc[-1]),
                "training_time_seconds": float(recorded.elapsed_seconds.iloc[-1]) if "elapsed_seconds" in recorded else None,
                "total_parameters": metadata.get("total_parameters"),
                "trainable_parameters": metadata.get("trainable_parameters"),
                "status": "complete" if (run_dir / "training_complete.json").exists() else "incomplete_or_paused"}
        if info["trainable_parameters"] is None and "trainable_parameters" in recorded:
            info["trainable_parameters"] = int(recorded.trainable_parameters.iloc[-1])
        pd.DataFrame([info]).to_csv(output / "training_efficiency.csv", index=False)
        if info["total_parameters"] is not None and info["trainable_parameters"] is not None:
            fig, axis = pyplot().subplots(figsize=(7, 4))
            axis.bar(["Total", "Trainable"], [info["total_parameters"] / 1e6, info["trainable_parameters"] / 1e6], color=["#007f86", "#c33d5b"])
            axis.set(ylabel="Million parameters", title="Model size (not a detection-quality score)")
            save(fig, output / "parameter_counts.png")
    status = evaluation_dir / "status.json"
    if not status.exists() or json.loads(status.read_text()).get("status") != "complete":
        print("Training curves available if recorded. Final metrics await a completed sealed evaluation.")
        return None
    summary = make_figures(evaluation_dir / "predictions.csv", output)
    result = json.loads((evaluation_dir / "results.json").read_text())
    efficiency = []
    for split, timing in result["timing"].items():
        efficiency.append({"split": split, **timing,
                           **{key: result[key] for key in ("total_parameters", "trainable_parameters", "training_time_seconds")}})
    pd.DataFrame(efficiency).to_csv(output / "efficiency.csv", index=False)
    table = pd.DataFrame(efficiency).set_index("split")
    ax = table[["single_image_ms", "batch_ms_per_image"]].plot.bar(rot=0, figsize=(9, 4))
    ax.set(ylabel="Milliseconds per image", title="Synchronized model-forward latency (no IO/preprocessing)")
    save(ax.figure, output / "inference_latency.png")
    if data_root is not None:
        failure_gallery(output / "failure_cases.csv", data_root, output)
    print("Quality dashboard:", output)
    return summary

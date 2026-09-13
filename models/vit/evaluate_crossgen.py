"""
Cross-generator evaluation harness — Owner: Member C.

The project guide's "fair comparison rule" requires all 4 models to be scored
with the *same evaluation code*, on the *same* cross-generator test set. This
script is that shared evaluation code. It:

    1. Loads each model's checkpoint via its own `model.py::build_model(cfg)`
       (every model folder is expected to expose that one function — read-only,
       I never edit another member's model.py).
    2. Runs the identical evaluation loop (same metrics, same threshold, same
       dataset class) against the primary test set and the cross-generator
       test set for every model.
    3. Writes the final merged comparison table + ROC/accuracy-drop plots to
       results/comparison/ — the one results folder I own.

Usage (run after every model has at least one checkpoint in results/<name>/<run>/checkpoints/):

    python models/vit/evaluate_crossgen.py -c configs/crossgen_harness.yaml

If a model's checkpoint or config isn't there yet, it's skipped with a warning
so the harness still produces a partial comparison for whichever models ARE ready.
"""

import argparse
import importlib
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.vit.common import load_config, build_dataset_paths, DeepfakeImageDataset  # noqa: E402


def load_model_builder(module_path: str):
    """module_path e.g. 'models.custom_cnn.model' — must expose build_model(cfg)."""
    module = importlib.import_module(module_path)
    if not hasattr(module, "build_model"):
        raise AttributeError(f"{module_path} has no build_model(cfg) function")
    return module.build_model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())
    return np.array(all_probs), np.array(all_labels)


def compute_metrics(probs, labels, threshold=0.5):
    if len(labels) == 0:
        return {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan}
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = np.nan
    return {"accuracy": accuracy_score(labels, preds), "precision": precision,
            "recall": recall, "f1": f1, "roc_auc": auc}


def relative_drop_pct(primary_acc, cross_acc):
    if primary_acc in (0, None) or (isinstance(primary_acc, float) and np.isnan(primary_acc)):
        return np.nan
    return 100 * (primary_acc - cross_acc) / primary_acc


def evaluate_one_model(name: str, entry: dict, device, comparison_dir: Path) -> dict | None:
    model_cfg_path = entry["config"]
    checkpoint_path = Path(entry["checkpoint"])
    module_path = entry["module"]

    if not Path(model_cfg_path).exists():
        warnings.warn(f"[{name}] config not found at {model_cfg_path} — skipping.")
        return None
    if not checkpoint_path.exists():
        warnings.warn(f"[{name}] checkpoint not found at {checkpoint_path} — skipping (train it first).")
        return None

    model_cfg = load_config(model_cfg_path)
    try:
        build_model = load_model_builder(module_path)
        model = build_model(model_cfg).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:  # a teammate's model.py signature may still be in flux
        warnings.warn(f"[{name}] failed to load model/checkpoint ({e}) — skipping.")
        return None

    paths = build_dataset_paths(model_cfg["data"])
    image_size = model_cfg["data"]["image_size"]
    bs = model_cfg["data"].get("batch_size", 32)
    nw = model_cfg["data"].get("num_workers", 2)

    test_ds = DeepfakeImageDataset(paths["test_dir"], image_size=image_size, augment=False)
    xgen_ds = DeepfakeImageDataset(paths["cross_gen_test_dir"], image_size=image_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)
    xgen_loader = DataLoader(xgen_ds, batch_size=bs, shuffle=False, num_workers=nw)

    test_probs, test_labels = evaluate(model, test_loader, device)
    xgen_probs, xgen_labels = evaluate(model, xgen_loader, device)

    test_m = compute_metrics(test_probs, test_labels)
    xgen_m = compute_metrics(xgen_probs, xgen_labels)

    # save per-model ROC/CM into the shared comparison folder, namespaced by model
    out_dir = comparison_dir / "harness_outputs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(test_labels):
        fpr, tpr, _ = roc_curve(test_labels, test_probs)
        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(out_dir / "roc_primary_test.csv", index=False)
        cm = confusion_matrix(test_labels, (test_probs >= 0.5).astype(int), labels=[0, 1])
        np.savetxt(out_dir / "confusion_matrix_primary_test.csv", cm, delimiter=",", fmt="%d")
    if len(xgen_labels):
        fpr, tpr, _ = roc_curve(xgen_labels, xgen_probs)
        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(out_dir / "roc_cross_gen.csv", index=False)
        cm = confusion_matrix(xgen_labels, (xgen_probs >= 0.5).astype(int), labels=[0, 1])
        np.savetxt(out_dir / "confusion_matrix_cross_gen.csv", cm, delimiter=",", fmt="%d")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accuracy_drop = test_m["accuracy"] - xgen_m["accuracy"] if not np.isnan(test_m["accuracy"]) else np.nan

    return {
        "model": name,
        "test_acc": test_m["accuracy"], "test_precision": test_m["precision"],
        "test_recall": test_m["recall"], "test_f1": test_m["f1"], "test_auc": test_m["roc_auc"],
        "cross_gen_acc": xgen_m["accuracy"], "cross_gen_precision": xgen_m["precision"],
        "cross_gen_recall": xgen_m["recall"], "cross_gen_f1": xgen_m["f1"], "cross_gen_auc": xgen_m["roc_auc"],
        "accuracy_drop": accuracy_drop,
        "relative_accuracy_drop_pct": relative_drop_pct(test_m["accuracy"], xgen_m["accuracy"]),
        "num_params": n_params,
    }


def make_plots(df: pd.DataFrame, comparison_dir: Path):
    if df.empty:
        print("Nothing to plot — no models evaluated successfully yet.")
        return

    plot_df = df.sort_values("accuracy_drop")
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(plot_df["model"], plot_df["accuracy_drop"], color=sns.color_palette("crest", len(plot_df)))
    ax.set_ylabel("Accuracy drop (primary test -> cross-generator test)")
    ax.set_title("Generalization gap across all 4 architectures")
    ax.bar_label(bars, fmt="%.3f")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(comparison_dir / "harness_accuracy_drop.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    for _, row in df.iterrows():
        roc_path = comparison_dir / "harness_outputs" / row["model"] / "roc_cross_gen.csv"
        if roc_path.exists():
            roc_df = pd.read_csv(roc_path)
            ax.plot(roc_df["fpr"], roc_df["tpr"], label=row["model"])
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — Cross-generator test set, all models")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(comparison_dir / "harness_roc_cross_gen.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {comparison_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="configs/crossgen_harness.yaml",
        help="Harness config listing each model's module/config/checkpoint (see that file's comments).",
    )
    args = parser.parse_args()

    harness_cfg = load_config(args.config)
    comparison_dir = Path(harness_cfg.get("comparison_dir", "results/comparison"))
    comparison_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for name, entry in harness_cfg["models"].items():
        print(f"\n--- Evaluating {name} ---")
        result = evaluate_one_model(name, entry, device, comparison_dir)
        if result is not None:
            rows.append(result)
            print(f"  test_acc={result['test_acc']:.4f}  cross_gen_acc={result['cross_gen_acc']:.4f}  "
                  f"accuracy_drop={result['accuracy_drop']:.4f}")

    df = pd.DataFrame(rows)
    out_path = comparison_dir / "final_comparison_table.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFinal comparison table -> {out_path}")

    make_plots(df, comparison_dir)


if __name__ == "__main__":
    main()

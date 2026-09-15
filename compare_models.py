"""
Run AFTER all 4 models have been trained (each produces a results.json
in its own results/<model>/ folder). This script does NOT retrain
anything - it only reads each model's saved results.json, builds the
final comparison table and plots used in the report's "Results and
Model Comparison" and "Critical Analysis" sections.

Run from the project root with:
    python compare_models.py
"""
import json
import os
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import ensure_dir

RESULT_FILES = {
    "Custom CNN": "results/custom_cnn/results.json",
    "ResNet50": "results/resnet50/results.json",
    "EfficientNetV2": "results/efficientnetv2/results.json",
    "ViT (+freq branch)": "results/vit/results.json",
}

COMPARISON_DIR = "results/comparison"


def load_results():
    loaded = {}
    for label, path in RESULT_FILES.items():
        if not os.path.exists(path):
            print(f"WARNING: {path} not found - skipping {label}. "
                  f"Train that model first.")
            continue
        with open(path, "r") as f:
            loaded[label] = json.load(f)
    return loaded


def build_comparison_table(all_results: dict) -> pd.DataFrame:
    rows = []
    for label, res in all_results.items():
        test = res["test_metrics"]
        cross = res["cross_gen_metrics"]
        rows.append({
            "Model": label,
            "Test Accuracy": round(test["accuracy"], 4),
            "Test F1": round(test["f1_score"], 4),
            "Test ROC-AUC": round(test["roc_auc"], 4),
            "Cross-Gen Accuracy": round(cross["accuracy"], 4),
            "Cross-Gen F1": round(cross["f1_score"], 4),
            "Cross-Gen ROC-AUC": round(cross["roc_auc"], 4),
            "Generalization Gap (Acc)": round(res["generalization_gap_accuracy"], 4),
            "Train Time (s)": round(res["total_train_time_seconds"], 1),
            "Inference (ms/image)": round(test["ms_per_image"], 2),
        })
    return pd.DataFrame(rows)


def plot_accuracy_comparison(df: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(df))
    width = 0.35
    ax.bar([i - width / 2 for i in x], df["Test Accuracy"], width, label="In-distribution test")
    ax.bar([i + width / 2 for i in x], df["Cross-Gen Accuracy"], width, label="Cross-generator test")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Model"], rotation=15)
    ax.set_ylabel("Accuracy")
    ax.set_title("In-Distribution vs. Cross-Generator Accuracy by Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_generalization_gap(df: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["Model"], df["Generalization Gap (Acc)"], color="#c0392b")
    ax.set_ylabel("Accuracy Drop (Test -> Cross-Generator)")
    ax.set_title("Generalization Gap by Model (Lower Is Better)")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_efficiency_tradeoff(df: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["Inference (ms/image)"], df["Test Accuracy"], s=80)
    for _, row in df.iterrows():
        ax.annotate(row["Model"], (row["Inference (ms/image)"], row["Test Accuracy"]),
                    textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("Inference Time (ms/image)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy vs. Inference Cost Trade-off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    ensure_dir(COMPARISON_DIR)
    all_results = load_results()

    if not all_results:
        print("No results found yet. Train at least one model before running this script.")
        return

    df = build_comparison_table(all_results)
    csv_path = os.path.join(COMPARISON_DIR, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved comparison table to {csv_path}")
    print(df.to_string(index=False))

    plot_accuracy_comparison(df, os.path.join(COMPARISON_DIR, "accuracy_comparison.png"))
    plot_generalization_gap(df, os.path.join(COMPARISON_DIR, "generalization_gap.png"))
    plot_efficiency_tradeoff(df, os.path.join(COMPARISON_DIR, "efficiency_tradeoff.png"))
    print(f"Saved comparison plots to {COMPARISON_DIR}/")


if __name__ == "__main__":
    main()

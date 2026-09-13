"""
Training entry point for Model 4 (ViT / frequency-hybrid).
Owner: Member C

Usage:
    python models/vit/train.py -c configs/vit.yaml

This script is self-contained: it uses models/vit/common.py, which transparently
uses the shared src/ module if it's ready, and falls back to a local working
implementation otherwise (see common.py's docstring). Nothing here depends on
another member's folder being finished.

Follows the strict experimental protocol: run models/vit/prepare_data.py first
so that data/processed/{train,val,test} contain ONLY StyleGAN(+real) images
and data/cross_gen_test contains ONLY Stable-Diffusion(+held-out real) images.
This script trains and selects the model using train/val only, evaluates the
primary (StyleGAN) test set exactly once, and — without any further model
change — evaluates the cross-generator (Stable Diffusion) set. Label
convention throughout: REAL=0, FAKE=1 (FAKE is the positive class).

Writes:
    results/vit/<run_name>/checkpoints/best_model.pt
    results/vit/<run_name>/training_history.csv
    results/vit/<run_name>/metrics.csv
    results/vit/<run_name>/roc_primary_test.csv, roc_cross_gen.csv
    results/vit/<run_name>/confusion_matrix_primary_test.csv, confusion_matrix_cross_gen.csv
    results/vit/<run_name>/run_config_used.json
    results/comparison/crossgen_<run_name>.csv   (shared folder — I'm the only writer)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `import src` / `import models`
from models.vit.common import set_seed, load_config, build_dataset_paths, DeepfakeImageDataset  # noqa: E402
from models.vit.model import build_model  # noqa: E402


def linear_warmup_decay(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    remaining = max(0, total_steps - step)
    remaining_total = max(1, total_steps - warmup_steps)
    return remaining / remaining_total


def run_epoch(model, loader, criterion, optimizer, scheduler, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []

    if len(loader.dataset) == 0:
        return 0.0, np.array([]), np.array([])

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).float()

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item() * images.size(0)
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_probs), np.array(all_labels)


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    if len(labels) == 0:
        return {"accuracy": float("nan"), "precision": float("nan"),
                "recall": float("nan"), "f1": float("nan"), "roc_auc": float("nan")}
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")  # only one class present in this split
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
    }


def save_curves(probs, labels, out_dir: Path, tag: str, save_roc: bool, save_cm: bool):
    if len(labels) == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_roc:
        fpr, tpr, _ = roc_curve(labels, probs)
        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(out_dir / f"roc_{tag}.csv", index=False)
    if save_cm:
        cm = confusion_matrix(labels, (probs >= 0.5).astype(int), labels=[0, 1])
        np.savetxt(out_dir / f"confusion_matrix_{tag}.csv", cm, delimiter=",", fmt="%d")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to configs/vit.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_name = cfg["run_name"]
    results_dir = Path(cfg["output"]["results_dir"]) / run_name
    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"].format(run_name=run_name))
    comparison_dir = Path(cfg["output"]["comparison_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # ---- data -----------------------------------------------------------
    paths = build_dataset_paths(cfg["data"])
    image_size = cfg["data"]["image_size"]

    train_ds = DeepfakeImageDataset(paths["train_dir"], image_size=image_size, augment=True)
    val_ds = DeepfakeImageDataset(paths["val_dir"], image_size=image_size, augment=False)
    test_ds = DeepfakeImageDataset(paths["test_dir"], image_size=image_size, augment=False)
    cross_gen_ds = DeepfakeImageDataset(paths["cross_gen_test_dir"], image_size=image_size, augment=False)

    bs = cfg["data"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)
    cross_gen_loader = DataLoader(cross_gen_ds, batch_size=bs, shuffle=False, num_workers=nw)

    if len(train_ds) == 0:
        print("[warn] Training set is empty — populate data/ per data/README.md, "
              "then re-run. Continuing to build/save an untrained checkpoint so the "
              "pipeline shape can still be verified end to end.")

    # ---- model / optim / scheduler --------------------------------------
    model = build_model(cfg).to(device)
    criterion = nn.BCEWithLogitsLoss()
    t_cfg = cfg["train"]
    optimizer = AdamW(model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"])

    steps_per_epoch = max(1, len(train_loader))
    total_steps = t_cfg["epochs"] * steps_per_epoch
    warmup_steps = int(total_steps * t_cfg["warmup_steps_pct"])
    scheduler = LambdaLR(
        optimizer, lr_lambda=lambda step: linear_warmup_decay(step, total_steps, warmup_steps)
    )

    # ---- training loop with early stopping -------------------------------
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    train_start = time.time()
    for epoch in range(1, t_cfg["epochs"] + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, scheduler, device, train=True)
        val_loss, val_probs, val_labels = run_epoch(
            model, val_loader, criterion, optimizer, scheduler, device, train=False
        )
        val_metrics = compute_metrics(val_probs, val_labels, cfg["evaluation"]["threshold"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics})
        print(f"[epoch {epoch:02d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} val_auc={val_metrics['roc_auc']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= t_cfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch} (no val_loss improvement).")
                break

    train_time_sec = time.time() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(results_dir / "training_history.csv", index=False)

    # ---- final evaluation: primary (in-distribution) test set, touched once ----
    _, test_probs, test_labels = run_epoch(model, test_loader, criterion, optimizer, scheduler, device, train=False)
    test_metrics = compute_metrics(test_probs, test_labels, cfg["evaluation"]["threshold"])
    save_curves(test_probs, test_labels, results_dir, "primary_test",
                cfg["evaluation"]["save_roc_curve"], cfg["evaluation"]["save_confusion_matrix"])

    # ---- cross-generator evaluation: unseen generator family --------------
    inf_start = time.time()
    _, xgen_probs, xgen_labels = run_epoch(model, cross_gen_loader, criterion, optimizer, scheduler, device, train=False)
    inference_time_per_image = (time.time() - inf_start) / max(1, len(cross_gen_ds))
    xgen_metrics = compute_metrics(xgen_probs, xgen_labels, cfg["evaluation"]["threshold"])
    save_curves(xgen_probs, xgen_labels, results_dir, "cross_gen",
                cfg["evaluation"]["save_roc_curve"], cfg["evaluation"]["save_confusion_matrix"])

    accuracy_drop = test_metrics["accuracy"] - xgen_metrics["accuracy"]
    relative_accuracy_drop_pct = (
        100 * accuracy_drop / test_metrics["accuracy"]
        if test_metrics["accuracy"] not in (0, float("nan")) and not np.isnan(test_metrics["accuracy"])
        else float("nan")
    )
    n_params = count_params(model)

    metrics_row = {
        "model": "vit_frequency_hybrid" if cfg["model"]["use_frequency_hybrid"] else "vit_b16",
        "run_name": run_name,
        "val_acc": history[-1]["accuracy"] if history else float("nan"),
        "val_auc": history[-1]["roc_auc"] if history else float("nan"),
        "test_acc": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_auc": test_metrics["roc_auc"],
        "cross_gen_acc": xgen_metrics["accuracy"],
        "cross_gen_precision": xgen_metrics["precision"],
        "cross_gen_recall": xgen_metrics["recall"],
        "cross_gen_f1": xgen_metrics["f1"],
        "cross_gen_auc": xgen_metrics["roc_auc"],
        "accuracy_drop": accuracy_drop,
        "relative_accuracy_drop_pct": relative_accuracy_drop_pct,
        "train_time_sec": train_time_sec,
        "inference_time_sec_per_image": inference_time_per_image,
        "num_params": n_params,
    }
    pd.DataFrame([metrics_row]).to_csv(results_dir / "metrics.csv", index=False)

    comparison_path = comparison_dir / f"crossgen_{run_name}.csv"
    pd.DataFrame([metrics_row]).to_csv(comparison_path, index=False)

    with open(results_dir / "run_config_used.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print("\n=== Run complete ===")
    print(f"Run name: {run_name}")
    print(f"Primary test accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"Cross-generator accuracy:   {xgen_metrics['accuracy']:.4f}")
    print(f"Accuracy drop (gen. gap):   {accuracy_drop:.4f}")
    print(f"Params: {n_params:,} | Train time: {train_time_sec:.1f}s | "
          f"Inference: {inference_time_per_image * 1000:.2f} ms/image")
    print(f"Metrics saved to:     {results_dir / 'metrics.csv'}")
    print(f"Comparison row saved: {comparison_path}")


if __name__ == "__main__":
    main()

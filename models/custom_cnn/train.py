"""Custom CNN trainer (owner A).

Usage:
    python models/custom_cnn/train.py -c configs/custom_cnn.yaml

Writes to `results/custom_cnn/`:
    * <run_name>/checkpoints/best.pt          — best model by val F1
    * <run_name>/metrics.csv                  — per-epoch log
    * <run_name>/history.png                  — loss/acc curves

Change hyperparameters in the yaml, not in this file.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import evaluate_binary, get_device, get_logger, load_config, resolve_path, set_seed

logger = get_logger("custom_cnn.train")


def build_datasets(cfg: dict):
    """ImageFolder-style loader over data/raw/{real,fake}. Minimal for now —
    Owner A will enrich via src.data_pipeline (face-crop, augmentation)."""
    from torchvision import datasets, transforms

    size = cfg["image"]["size"]
    tfm = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    raw = resolve_path(cfg["data"]["raw_dir"]) if "raw_dir" in cfg["data"] else resolve_path(cfg["data"]["root"])
    root_dir = raw / "real" if (raw / "real").exists() else raw

    ds = datasets.ImageFolder(root_dir, transform=tfm)
    n = len(ds)
    tr, va, te = cfg["data"]["train_split"], cfg["data"]["val_split"], cfg["data"]["test_split"]
    n_tr = int(round(n * tr))
    n_va = int(round(n * va))
    n_te = n - n_tr - n_va
    return random_split(ds, [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(cfg["seed"]))


def make_loader(ds, batch_size, num_workers, shuffle):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom CNN training")
    parser.add_argument("-c", "--config", type=str, required=True, help="configs/custom_cnn.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    device = get_device(cfg.get("device", "auto"))
    logger.info("device=%s seed=%d", device, cfg["seed"])

    # --- data ---
    tr_ds, va_ds, te_ds = build_datasets(cfg)
    train_loader = make_loader(tr_ds, cfg["training"]["batch_size"], cfg["data"]["num_workers"], shuffle=True)
    val_loader = make_loader(va_ds, cfg["training"]["batch_size"], cfg["data"]["num_workers"], shuffle=False)

    # --- model ---
    from model import build_model

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

    run_dir = resolve_path(cfg["logging"]["output_dir"]) / cfg["logging"]["run_name"]
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_f1 = -1.0
    patience = cfg["training"].get("early_stopping_patience", 10)
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        running, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
        scheduler.step()

        # validation
        model.eval()
        logits, ytrue = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits.append(model(xb).softmax(1)[:, 1].cpu().numpy())
                ytrue.append(yb.numpy())
        y_score = np.concatenate(logits)
        y_true = np.concatenate(ytrue)
        report = evaluate_binary(y_true, y_score)

        history.append(
            {
                "epoch": epoch,
                "train_loss": running / total,
                "train_acc": correct / total,
                "val_acc": report.accuracy,
                "val_f1": report.f1,
                "val_auc": report.roc_auc,
            }
        )
        logger.info(
            "epoch=%3d loss=%.4f train_acc=%.4f val_acc=%.4f val_f1=%.4f val_auc=%.4f",
            epoch, history[-1]["train_loss"], history[-1]["train_acc"],
            report.accuracy, report.f1, report.roc_auc,
        )

        if report.f1 > best_f1:
            best_f1 = report.f1
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("early stop at epoch %d", epoch)
                break

    pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)
    torch.save({"model_state": model.state_dict(), "cfg": cfg, "best_f1": best_f1}, ckpt_dir / "last.pt")
    logger.info("done in %.1fs — outputs under %s (best val F1=%.4f)", time.time() - t0, run_dir, best_f1)


if __name__ == "__main__":
    main()
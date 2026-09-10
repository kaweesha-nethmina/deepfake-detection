"""ViT trainer + cross-generator evaluation (owner C).

Usage:
    python models/vit/train.py -c configs/vit.yaml

Writes to `results/vit/`:
    * <run_name>/checkpoints/best.pt
    * <run_name>/metrics.csv
And (cross-generator comparison) to `results/comparison/`:
    * crossgen_<run_name>.csv

Note: the cross-generator eval expects `data/cross_gen_test/<generator>/` to
exist (see data/README.md).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import evaluate_binary, get_device, get_logger, load_config, resolve_path, set_seed

logger = get_logger("vit.train")


def build_datasets(cfg: dict):
    from torchvision import datasets, transforms

    size = cfg["image"]["size"]
    tfm = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    raw = resolve_path(cfg["data"]["root"])
    root_dir = raw / "real" if (raw / "real").exists() else raw
    ds = datasets.ImageFolder(root_dir, transform=tfm)
    n = len(ds)
    tr, va, te = cfg["data"]["train_split"], cfg["data"]["val_split"], cfg["data"]["test_split"]
    n_tr = int(round(n * tr))
    n_va = int(round(n * va))
    return random_split(ds, [n_tr, n_va, n - n_tr - n_va], generator=torch.Generator().manual_seed(cfg["seed"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="ViT training")
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    device = get_device(cfg.get("device", "auto"))
    logger.info("device=%s seed=%d", device, cfg["seed"])

    from model import build_model

    train_ds, val_ds, _ = build_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
                              num_workers=cfg["data"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                            num_workers=cfg["data"]["num_workers"])

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                 weight_decay=cfg["training"]["weight_decay"])
    warmup = cfg["training"].get("warmup_epochs", 0)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / max(1, cfg["training"]["epochs"] - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    run_dir = resolve_path(cfg["logging"]["output_dir"]) / cfg["logging"]["run_name"]
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    history, best_f1, no_improve, patience, t0 = [], -1.0, 0, cfg["training"].get("early_stopping_patience", 8), time.time()
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

        model.eval()
        logits, ytrue = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits.append(model(xb.to(device)).softmax(1)[:, 1].cpu().numpy())
                ytrue.append(yb.numpy())
        report = evaluate_binary(np.concatenate(ytrue), np.concatenate(logits))
        history.append({"epoch": epoch, "train_loss": running / total, "train_acc": correct / total,
                        "val_acc": report.accuracy, "val_f1": report.f1, "val_auc": report.roc_auc})
        logger.info("epoch=%3d loss=%.4f train_acc=%.4f val_acc=%.4f val_f1=%.4f val_auc=%.4f",
                    epoch, history[-1]["train_loss"], history[-1]["train_acc"],
                    report.accuracy, report.f1, report.roc_auc)
        if report.f1 > best_f1:
            best_f1 = report.f1
            torch.save(model.state_dict(), run_dir / "checkpoints/best.pt")
            no_improve = 0
        elif (no_improve := no_improve + 1) >= patience:
            logger.info("early stop at epoch %d", epoch)
            break

    pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)
    torch.save({"model_state": model.state_dict(), "cfg": cfg, "best_f1": best_f1}, run_dir / "checkpoints/last.pt")

    run_crossgen(cfg, model, device)
    logger.info("done in %.1fs — outputs under %s (best val F1=%.4f)", time.time() - t0, run_dir, best_f1)


def run_crossgen(cfg: dict, model: nn.Module, device: str) -> None:
    """Evaluate the trained head on unseen generators (owner C's core deliverable)."""
    from torchvision import datasets, transforms

    gen_dir = resolve_path(cfg["data"]["cross_gen_test_dir"])
    if not gen_dir.exists() or not any(gen_dir.iterdir()):
        logger.warning("cross_gen_test dir missing/empty — skipping crossgen eval. See data/README.md.")
        return

    size = cfg["image"]["size"]
    tfm = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ckpt = torch.load(resolve_path(cfg["logging"]["output_dir"]) / cfg["logging"]["run_name"] / "checkpoints/best.pt",
                      map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rows = []
    for gen in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
        ds = datasets.ImageFolder(str(gen), transform=tfm)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], num_workers=cfg["data"]["num_workers"])
        logits, ytrue = [], []
        with torch.no_grad():
            for xb, yb in loader:
                logits.append(model(xb.to(device)).softmax(1)[:, 1].cpu().numpy())
                ytrue.append(yb.numpy())
        rep = evaluate_binary(np.concatenate(ytrue), np.concatenate(logits))
        rows.append({"generator": gen.name, "n": rep.n, "acc": rep.accuracy, "f1": rep.f1, "auc": rep.roc_auc})
        logger.info("crossgen [%s] n=%d acc=%.4f f1=%.4f auc=%.4f", gen.name, rep.n, rep.accuracy, rep.f1, rep.roc_auc)

    if rows:
        out_dir = resolve_path(cfg["crossgen"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / f"crossgen_{cfg['logging']['run_name']}.csv", index=False)


if __name__ == "__main__":
    main()
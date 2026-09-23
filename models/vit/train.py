"""Manifest-based team training. This command never evaluates either test set."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.vit.manifest import load_audited_manifest
from models.vit.reporting import compute_metrics, learning_curves
from models.vit.runtime import (ManifestDataset, binary_loss, build_model, device_for,
                                load_config, preprocessing_for, probabilities, set_seed)


def phase_parameters(model, cfg, warmup):
    """Keep teammates' heads/architectures; only apply their staged unfreezing."""
    for parameter in model.parameters():
        parameter.requires_grad = True
    if not cfg["train"].get("warmup_epochs", 0):
        return
    for parameter in model.parameters():
        parameter.requires_grad = False
    backbone = getattr(model, "backbone", model)
    head = getattr(model, "head", None)
    if head is None:
        head = getattr(backbone, "fc", getattr(backbone, "classifier", None))
    if head is None:
        raise ValueError("Cannot identify classifier for staged training.")
    for parameter in head.parameters():
        parameter.requires_grad = True
    if not warmup:
        if hasattr(backbone, "layer4"):
            blocks = [backbone.layer3, backbone.layer4]
        elif hasattr(backbone, "blocks"):
            blocks = list(backbone.blocks.children())[-2:]
        elif hasattr(backbone, "features"):
            blocks = list(backbone.features.children())[-2:]
        else:
            raise ValueError("Cannot identify backbone blocks for fine-tuning.")
        for block in blocks:
            for parameter in block.parameters():
                parameter.requires_grad = True


def run_epoch(model, loader, device, optimizer=None, scheduler=None, scaler=None,
              accumulation=1, grad_clip=1.0, amp=False):
    training = optimizer is not None
    model.train(training)
    if training:
        # Frozen BatchNorm must not silently update running statistics.
        for module in model.modules():
            parameters = list(module.parameters(recurse=False))
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and parameters and not any(p.requires_grad for p in parameters):
                module.eval()
        optimizer.zero_grad(set_to_none=True)
    total, count, scores, labels_out = 0.0, 0, [], []
    with torch.set_grad_enabled(training):
        for step, (images, labels, _) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = binary_loss(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss; refusing to save a misleading result.")
            if training:
                # Weight a partial final accumulation window by sample count.
                start = (step // accumulation) * accumulation * loader.batch_size
                window_samples = min(accumulation * loader.batch_size, len(loader.dataset) - start)
                scaled_loss = loss * len(labels) / window_samples
                scaler.scale(scaled_loss).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None and scaler.get_scale() >= old_scale:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(labels)
            count += len(labels)
            scores.extend(probabilities(logits).detach().cpu().tolist())
            labels_out.extend(labels.cpu().tolist())
    if not count:
        raise ValueError("Empty loader.")
    return total / count, compute_metrics(labels_out, scores)


def rng_state(generator):
    np_state = np.random.get_state()
    state = {"python": random.getstate(), "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
             "torch": torch.get_rng_state(), "loader": generator.get_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, generator):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype="uint32"), *n[2:]))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def train(cfg, resume=None):
    cfg = copy.deepcopy(cfg)
    t = cfg["train"]
    if not 1 <= t["epochs"] <= 30 or t["early_stopping_patience"] != 6:
        raise ValueError("Team protocol: 1..30 epochs and patience=6.")
    batch, effective = cfg["data"]["batch_size"], t.get("effective_batch_size", 32)
    if batch < 1 or effective < batch or effective % batch:
        raise ValueError("Effective batch size must be a positive multiple of microbatch size.")
    rows, audit = load_audited_manifest(cfg["data"]["manifest"], cfg["data"]["root"],
                                       cfg["data"].get("near_duplicate_review"), ("train", "val"))
    out = Path(cfg["output"]["results_dir"]) / cfg["run_name"]
    if out.exists() and not resume:
        raise FileExistsError(f"Use a new run_name or --resume: {out}")
    device = device_for(cfg.get("device", "auto"))
    set_seed(cfg["seed"])
    model = build_model(cfg, pretrained=False if resume else None).to(device)
    preproc = preprocessing_for(model, cfg)
    generator = torch.Generator().manual_seed(cfg["seed"])
    loaders = {part: DataLoader(ManifestDataset(rows, cfg["data"]["root"], part, preproc,
                       train=part == "train", limit=32 if cfg.get("smoke") else None),
                       batch_size=batch, shuffle=part == "train", generator=generator,
                       num_workers=cfg["data"].get("num_workers", 0), drop_last=False)
               for part in ("train", "val")}
    amp = bool(t.get("mixed_precision", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, best_loss, stale, elapsed, start_epoch = [], float("inf"), 0, 0.0, 1
    checkpoint = None
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint["config"] != cfg or checkpoint["metadata"]["manifest_sha256"] != audit["manifest_sha256"]:
            raise ValueError("Resume config/manifest mismatch.")
        model.load_state_dict(checkpoint["model_state"])
        history, best_loss, stale = checkpoint["history"], checkpoint["best_loss"], checkpoint["stale"]
        elapsed, start_epoch = checkpoint["elapsed"], checkpoint["epoch"] + 1
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"], generator)
        if not (out / "checkpoints/best_model.pt").exists():
            raise FileNotFoundError("Resume needs the original best checkpoint as well as last.pt.")
        if checkpoint["phase"] == "finetune" and stale >= t["early_stopping_patience"]:
            print("This run already reached early stopping; no further updates are allowed.")
            return out
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    (out / "run_config_used.json").write_text(json.dumps(cfg, indent=2))
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "timm", "numpy", "pandas", "scikit-learn", "Pillow", "PyYAML")}
    environment = {"python": platform.python_version(), "packages": versions, "device": str(device),
                   "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine()}
    try:
        environment["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        environment["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        environment["git_commit"] = None
    (out / "environment.json").write_text(json.dumps(environment, indent=2))
    metadata = {"model_name": cfg["model_name"], "run_name": cfg["run_name"],
                "manifest_sha256": audit["manifest_sha256"], "dataset_version": audit["dataset_version"],
                "preprocessing": preproc, "label_mapping": {"real": 0, "fake": 1},
                "threshold": 0.5, "selection": "validation_loss", "smoke": cfg.get("smoke", False),
                "model_config": cfg["model"], "module": cfg["module"], "seed": cfg["seed"]}
    current_phase, optimizer, scheduler = None, None, None
    for epoch in range(start_epoch, t["epochs"] + 1):
        warmup = epoch <= t.get("warmup_epochs", 0)
        phase = "warmup" if warmup else "finetune"
        if phase != current_phase:
            phase_parameters(model, cfg, warmup)
            lr = t.get("warmup_lr", t["lr"]) if warmup else t["lr"]
            cls = torch.optim.Adam if t.get("optimizer") == "adam" else torch.optim.AdamW
            optimizer = cls(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                            weight_decay=t["weight_decay"])
            total_steps = max(1, t["epochs"] * math.ceil(len(loaders["train"]) / (effective // batch)))
            warm_steps = int(total_steps * t.get("warmup_steps_pct", 0))
            def schedule(step):
                if step < warm_steps:
                    return (step + 1) / max(1, warm_steps)
                return max(0.0, (total_steps - step) / max(1, total_steps - warm_steps))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
            if checkpoint and checkpoint["phase"] == phase:
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
            elif current_phase is not None or checkpoint:
                stale = 0
            current_phase = phase
        tick = time.perf_counter()
        train_loss, train_metrics = run_epoch(model, loaders["train"], device, optimizer, scheduler,
                                              scaler, effective // batch, t.get("grad_clip_norm", 1.0), amp)
        val_loss, val_metrics = run_epoch(model, loaders["val"], device, amp=amp)
        elapsed += time.perf_counter() - tick
        history.append({"epoch": epoch, "phase": phase, "train_loss": train_loss, "val_loss": val_loss,
                        "train_accuracy": train_metrics["accuracy"], "val_accuracy": val_metrics["accuracy"],
                        "train_f1": train_metrics["f1_score"], "val_f1": val_metrics["f1_score"],
                        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                        "lr": optimizer.param_groups[0]["lr"], "elapsed_seconds": elapsed})
        metadata.update(training_time_seconds=elapsed,
                        total_parameters=sum(p.numel() for p in model.parameters()),
                        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            torch.save({"model_state": model.state_dict(), "metadata": dict(metadata, best_epoch=epoch)},
                       ckpt_dir / "best_model.pt")
        else:
            stale += 1
        state = {"model_state": model.state_dict(), "metadata": metadata, "config": cfg,
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": epoch, "phase": phase, "history": history,
                 "best_loss": best_loss, "stale": stale, "elapsed": elapsed, "rng": rng_state(generator)}
        temporary = ckpt_dir / "last.tmp"
        torch.save(state, temporary)
        temporary.replace(ckpt_dir / "last.pt")
        pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
        print(f"epoch={epoch} phase={phase} train_loss={train_loss:.4f} val_loss={val_loss:.4f}", flush=True)
        if not warmup and stale >= t["early_stopping_patience"]:
            break
    if not history:
        raise ValueError("No training history; check resume epoch.")
    learning_curves(history, out / "learning_curves.png")
    # Total cost includes epochs after the selected checkpoint, not just its prefix.
    best = torch.load(ckpt_dir / "best_model.pt", map_location="cpu", weights_only=True)
    best["metadata"]["training_time_seconds"] = elapsed
    torch.save(best, ckpt_dir / "best_model.pt")
    (out / "training_complete.json").write_text(json.dumps({"status": "smoke_only" if cfg.get("smoke") else "trained",
                                                           "epochs": len(history), "seconds": elapsed}, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="configs/vit.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--manifest")
    parser.add_argument("--device")
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for key, value in (("root", args.data_root), ("manifest", args.manifest)):
        if value:
            cfg["data"][key] = value
    if args.device:
        cfg["device"] = args.device
    if args.smoke:
        cfg.update(smoke=True, run_name=cfg["run_name"] + "_smoke")
        cfg["train"].update(epochs=1, warmup_epochs=0)
        cfg["data"].update(batch_size=2, num_workers=0)
    print(f"Training artifacts: {train(cfg, args.resume)}. Test sets have not been evaluated.")


if __name__ == "__main__":
    main()

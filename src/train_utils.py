"""
Generic training loop shared by every model, so training behaves
identically across all 4 experiments (fair comparison requirement).
Rule for the team: only ADD to this file, don't change existing behavior
without telling everyone first.
"""
import json
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils import EarlyStopping, ensure_dir
from src.metrics import evaluate_model


def run_one_epoch(model, dataloader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(dataloader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train_model(model, loaders, optimizer, device, epochs: int,
                 early_stopping_patience: int, results_dir: str,
                 checkpoint_path: str, model_name: str):
    """Trains `model`, tracks history, applies early stopping, saves the
    best checkpoint, then evaluates on both the primary test set and the
    cross-generator held-out set. Returns the full results dictionary
    that gets written to results/<model>/results.json."""
    ensure_dir(results_dir)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=early_stopping_patience)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    import time
    wall_start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_one_epoch(
            model, loaders["train"], criterion, optimizer, device, train=True)
        val_loss, val_acc = run_one_epoch(
            model, loaders["val"], criterion, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[{model_name}] Epoch {epoch}/{epochs} - "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if early_stopper.step(val_loss, model):
            print(f"[{model_name}] Early stopping triggered at epoch {epoch}.")
            break

    total_train_time = time.time() - wall_start

    # Restore best weights before final evaluation
    if early_stopper.best_state_dict is not None:
        model.load_state_dict(early_stopper.best_state_dict)

    torch.save(model.state_dict(), checkpoint_path)

    test_metrics = evaluate_model(model, loaders["test"], device)
    cross_gen_metrics = evaluate_model(model, loaders["cross_gen"], device)

    results = {
        "model_name": model_name,
        "history": history,
        "total_train_time_seconds": total_train_time,
        "test_metrics": test_metrics,
        "cross_gen_metrics": cross_gen_metrics,
        "generalization_gap_accuracy": test_metrics["accuracy"] - cross_gen_metrics["accuracy"],
    }

    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"[{model_name}] Test accuracy: {test_metrics['accuracy']:.4f} | "
          f"Cross-generator accuracy: {cross_gen_metrics['accuracy']:.4f} | "
          f"Generalization gap: {results['generalization_gap_accuracy']:.4f}")

    return results

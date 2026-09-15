"""
Run from the project root with:
    python -m models.resnet50.train
Only this file and model.py in this folder should be edited by the
member responsible for ResNet50. Do not edit other models' folders.
"""
import torch

from src.utils import set_seed, load_config, get_device, count_parameters, ensure_dir
from src.data_pipeline import build_dataloaders
from src.train_utils import train_model
from models.resnet50.model import get_model, freeze_backbone, unfreeze_last_blocks


def main():
    cfg = load_config("configs/resnet50.yaml")
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Using device: {device}")

    ensure_dir(cfg["output"]["results_dir"])

    loaders = build_dataloaders(cfg["data"])

    model = get_model(num_classes=2).to(device)

    # ---- Phase 1: warmup (base frozen, only new head trains) ----
    freeze_backbone(model)
    print(f"[Phase 1] Trainable parameters: {count_parameters(model):,}")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["warmup_lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    train_model(
        model=model,
        loaders=loaders,
        optimizer=optimizer,
        device=device,
        epochs=cfg["train"]["warmup_epochs"],
        early_stopping_patience=cfg["train"]["early_stopping_patience"],
        results_dir=cfg["output"]["results_dir"],
        checkpoint_path=cfg["output"]["checkpoint_path"],
        model_name="resnet50_phase1_warmup",
    )

    # ---- Phase 2: fine-tune last blocks at a low learning rate ----
    unfreeze_last_blocks(model, n_blocks=cfg["train"]["unfreeze_last_n_blocks"])
    print(f"[Phase 2] Trainable parameters: {count_parameters(model):,}")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["finetune_lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    train_model(
        model=model,
        loaders=loaders,
        optimizer=optimizer,
        device=device,
        epochs=cfg["train"]["finetune_epochs"],
        early_stopping_patience=cfg["train"]["early_stopping_patience"],
        results_dir=cfg["output"]["results_dir"],
        checkpoint_path=cfg["output"]["checkpoint_path"],
        model_name="resnet50",
    )


if __name__ == "__main__":
    main()

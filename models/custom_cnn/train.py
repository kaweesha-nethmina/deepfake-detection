"""
Custom CNN trainer.

Owner: Member A

This trainer uses the shared project pipeline so that the Custom CNN
follows the same data loading, training, validation, and evaluation
framework as the other team models.

Usage:
    python models/custom_cnn/train.py -c configs/custom_cnn.yaml
"""

from __future__ import annotations

import argparse

import torch

from src.data_pipeline import build_dataloaders
from src.train_utils import train_model
from src.utils import (
    count_parameters,
    ensure_dir,
    get_device,
    load_config,
    set_seed,
)

from models.custom_cnn.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Custom CNN model."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the Custom CNN YAML configuration.",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1. Load configuration
    # ---------------------------------------------------------
    cfg = load_config(args.config)

    # ---------------------------------------------------------
    # 2. Reproducibility
    # ---------------------------------------------------------
    set_seed(cfg["seed"])

    # ---------------------------------------------------------
    # 3. Device
    # ---------------------------------------------------------
    device = get_device()

    print("=" * 70)
    print("CUSTOM CNN TRAINING")
    print("=" * 70)
    print(f"Device       : {device}")
    print(f"Seed         : {cfg['seed']}")
    print(f"Model        : {cfg['model_name']}")
    print()

    # ---------------------------------------------------------
    # 4. Create output directory
    # ---------------------------------------------------------
    results_dir = cfg["output"]["results_dir"]
    checkpoint_path = cfg["output"]["checkpoint_path"]

    ensure_dir(results_dir)

    # ---------------------------------------------------------
    # 5. Build data loaders
    # ---------------------------------------------------------
    print("Building data loaders...")

    loaders = build_dataloaders(
        cfg["data"]
    )

    print(
        f"Train batches       : {len(loaders['train'])}"
    )
    print(
        f"Validation batches  : {len(loaders['val'])}"
    )
    print(
        f"Primary test batches: {len(loaders['test'])}"
    )
    print(
        f"Cross-gen batches   : {len(loaders['cross_gen'])}"
    )
    print()

    # ---------------------------------------------------------
    # 6. Build Custom CNN
    # ---------------------------------------------------------
    model = build_model(cfg)

    model = model.to(device)

    print(
        f"Trainable parameters: {count_parameters(model):,}"
    )
    print()

    # ---------------------------------------------------------
    # 7. Optimizer
    # ---------------------------------------------------------
    learning_rate = cfg["train"]["learning_rate"]
    weight_decay = cfg["train"]["weight_decay"]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    print("Optimizer           : Adam")
    print(f"Learning rate       : {learning_rate}")
    print(f"Weight decay        : {weight_decay}")
    print(
        f"Max epochs          : {cfg['train']['epochs']}"
    )
    print(
        f"Early stopping      : {cfg['train']['early_stopping_patience']}"
    )
    print()

    # ---------------------------------------------------------
    # 8. Train + validate + test + cross-generator evaluation
    # ---------------------------------------------------------
    results = train_model(
        model=model,
        loaders=loaders,
        optimizer=optimizer,
        device=device,
        epochs=cfg["train"]["epochs"],
        early_stopping_patience=cfg["train"][
            "early_stopping_patience"
        ],
        results_dir=results_dir,
        checkpoint_path=checkpoint_path,
        model_name=cfg["model_name"],
    )

    # ---------------------------------------------------------
    # 9. Final summary
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("CUSTOM CNN EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"Primary Test Accuracy      : "
        f"{results['test_metrics']['accuracy']:.4f}"
    )

    print(
        f"Cross-Generator Accuracy   : "
        f"{results['cross_gen_metrics']['accuracy']:.4f}"
    )

    print(
        f"Generalization Gap         : "
        f"{results['generalization_gap_accuracy']:.4f}"
    )

    print()
    print(f"Results directory : {results_dir}")
    print(f"Checkpoint        : {checkpoint_path}")


if __name__ == "__main__":
    main()
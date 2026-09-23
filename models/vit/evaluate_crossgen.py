"""Freeze selected checkpoints, then run a strict four-model final evaluation."""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.vit.manifest import load_audited_manifest, sha256_file
from models.vit.reporting import compute_metrics, make_figures
from models.vit.runtime import (ManifestDataset, device_for, load_checkpoint, load_config,
                                probabilities)

REQUIRED = {"custom_cnn", "resnet50", "efficientnetv2", "vit_b16"}


def validate_metadata(metadata, cfg, checksum):
    if metadata.get("manifest_sha256") != checksum:
        raise ValueError("Checkpoint was not trained on the frozen shared manifest.")
    if metadata.get("smoke", True) or metadata.get("selection") != "validation_loss":
        raise ValueError("Final evaluation requires a non-smoke, validation-selected checkpoint.")
    if metadata.get("label_mapping") != {"real": 0, "fake": 1} or metadata.get("threshold") != 0.5:
        raise ValueError("Checkpoint label mapping / threshold conflicts with team protocol.")
    if metadata.get("model_config") != cfg["model"] or metadata.get("module") != cfg["module"]:
        raise ValueError("Checkpoint architecture/config mismatch.")
    if metadata.get("preprocessing", {}).get("image_size") != 224:
        raise ValueError("Final comparison requires 224x224 preprocessing.")
    for key in ("training_time_seconds", "total_parameters", "trainable_parameters", "run_name"):
        if key not in metadata:
            raise ValueError(f"Checkpoint metadata missing {key}; obtain it from the model owner.")


def freeze(config, destination):
    cfg = load_config(config)
    if not REQUIRED.issubset(cfg["models"]):
        raise ValueError("Final comparison requires all four core models.")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Do not overwrite a frozen experiment.")
    _, audit = load_audited_manifest(cfg["manifest"], cfg["data_root"],
                                     cfg.get("near_duplicate_review"), verify_splits=())
    entries = {}
    for name, entry in cfg["models"].items():
        if not Path(entry["checkpoint"]).is_file():
            raise FileNotFoundError(f"{name}: missing checkpoint {entry['checkpoint']}")
        model_cfg = load_config(entry["config"])
        bundle = torch.load(entry["checkpoint"], map_location="cpu", weights_only=True)
        metadata = bundle.get("metadata")
        if metadata is None:
            metadata = json.loads(Path(entry["checkpoint"] + ".json").read_text())
        validate_metadata(metadata, model_cfg, audit["manifest_sha256"])
        if model_cfg["model_name"] != name:
            raise ValueError(f"Model name mismatch for {name}")
        entries[name] = {"config": model_cfg, "checkpoint": entry["checkpoint"],
                         "checkpoint_sha256": sha256_file(entry["checkpoint"]), "metadata": metadata}
        del bundle
    protocol = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "manifest": cfg["manifest"], "manifest_sha256": audit["manifest_sha256"],
                "data_root": cfg["data_root"], "near_duplicate_review": cfg.get("near_duplicate_review"),
                "threshold": 0.5, "models": entries}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def predict_loader(model, loader, device, name, run_name):
    model.eval()
    predictions, seconds = [], 0.0
    for images, labels, indices in loader:
        images = images.to(device)
        synchronize(device)
        start = time.perf_counter()
        scores = probabilities(model(images))
        synchronize(device)
        seconds += time.perf_counter() - start
        for label, score, index in zip(labels.tolist(), scores.cpu().tolist(), indices.tolist()):
            row = loader.dataset.rows[index]
            predictions.append({"model": name, "run_name": run_name, "image_id": row["sha256"],
                                "filepath": row["filepath"], "label": label, "probability": score,
                                "predicted_label": int(score >= 0.5), "split": row["split"],
                                "generator": row["source"]})
    return predictions, seconds * 1000 / len(predictions)


@torch.inference_mode()
def single_image_latency(model, dataset, device, repeats=30):
    image = dataset[0][0].unsqueeze(0).to(device)
    for _ in range(5):
        model(image)
    synchronize(device)
    tick = time.perf_counter()
    for _ in range(repeats):
        model(image)
    synchronize(device)
    return 1000 * (time.perf_counter() - tick) / repeats


def evaluate(frozen, output, device="auto", batch_size=32):
    protocol = json.loads(Path(frozen).read_text())
    if not REQUIRED.issubset(protocol["models"]) or protocol.get("threshold") != 0.5:
        raise ValueError("Invalid frozen four-model protocol.")
    rows, audit = load_audited_manifest(protocol["manifest"], protocol["data_root"],
                                       protocol.get("near_duplicate_review"))
    if protocol["manifest_sha256"] != audit["manifest_sha256"]:
        raise ValueError("Manifest changed after experiment freeze.")
    for name, entry in protocol["models"].items():
        if sha256_file(entry["checkpoint"]) != entry["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed after freeze: {name}")
        validate_metadata(entry["metadata"], entry["config"], audit["manifest_sha256"])
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new evaluation directory; never overwrite final evidence.")
    output.mkdir(parents=True)
    status = output / "status.json"
    status.write_text(json.dumps({"status": "running", "frozen_sha256": sha256_file(frozen)}))
    device = device_for(device)
    all_predictions, comparison, expected_ids = [], [], {}
    try:
        for name, entry in protocol["models"].items():
            print(f"Evaluating frozen {name}", flush=True)
            model, metadata = load_checkpoint(entry["config"], entry["checkpoint"], device)
            if metadata != entry["metadata"]:
                raise ValueError("Checkpoint metadata changed since freeze.")
            metrics, times = {}, {}
            model_predictions = []
            for split in ("test", "cross_gen"):
                dataset = ManifestDataset(rows, protocol["data_root"], split, metadata["preprocessing"])
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                predictions, batch_latency = predict_loader(model, loader, device, name, metadata["run_name"])
                ids = [p["image_id"] for p in predictions]
                if split in expected_ids and expected_ids[split] != ids:
                    raise ValueError("Models did not evaluate identical ordered sample lists.")
                expected_ids[split] = ids
                metrics[split] = compute_metrics([p["label"] for p in predictions],
                                                 [p["probability"] for p in predictions])
                times[split] = {"batch_ms_per_image": batch_latency,
                                "single_image_ms": single_image_latency(model, dataset, device)}
                all_predictions.extend(predictions)
                model_predictions.extend(predictions)
            model_dir = output / name
            model_dir.mkdir()
            pd.DataFrame(model_predictions).to_csv(model_dir / "predictions.csv", index=False)
            result = {"model_name": name, "test_metrics": metrics["test"],
                      "cross_gen_metrics": metrics["cross_gen"], "timing": times,
                      "total_train_time_seconds": metadata["training_time_seconds"],
                      "total_parameters": metadata["total_parameters"],
                      "trainable_parameters": metadata["trainable_parameters"],
                      "generalization_gap_accuracy": metrics["test"]["accuracy"] - metrics["cross_gen"]["accuracy"],
                      "generalization_gap_f1": metrics["test"]["f1_score"] - metrics["cross_gen"]["f1_score"]}
            (model_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
            comparison.append({"model": name, "primary_accuracy": metrics["test"]["accuracy"],
                               "cross_accuracy": metrics["cross_gen"]["accuracy"],
                               "primary_f1": metrics["test"]["f1_score"],
                               "cross_f1": metrics["cross_gen"]["f1_score"],
                               "cross_auc": metrics["cross_gen"]["roc_auc"],
                               "accuracy_drop_pp": 100 * result["generalization_gap_accuracy"],
                               "f1_drop": result["generalization_gap_f1"],
                               "parameters": metadata["total_parameters"],
                               "training_seconds": metadata["training_time_seconds"],
                               "latency_ms": times["test"]["single_image_ms"]})
            del model
            gc.collect()
        predictions_path = output / "predictions.csv"
        pd.DataFrame(all_predictions).to_csv(predictions_path, index=False)
        pd.DataFrame(comparison).to_csv(output / "model_comparison.csv", index=False)
        make_figures(predictions_path, output / "figures")
        status.write_text(json.dumps({"status": "complete", "models": list(protocol["models"]),
                                     "frozen_sha256": sha256_file(frozen), "device": str(device)}, indent=2))
    except Exception as error:
        status.write_text(json.dumps({"status": "failed_not_final", "error": str(error)}, indent=2))
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("-c", "--config", default="configs/crossgen_harness.yaml")
    freeze_parser.add_argument("--output", default="results/comparison/frozen.json")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--frozen", default="results/comparison/frozen.json")
    run_parser.add_argument("--output", default="results/comparison/final")
    run_parser.add_argument("--device", default="auto")
    run_parser.add_argument("--batch-size", type=int, default=32)
    plots_parser = sub.add_parser("plots")
    plots_parser.add_argument("--predictions", required=True)
    plots_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.config, args.output)
        print(f"Protocol frozen: {args.output}. No test predictions were computed.")
    elif args.command == "run":
        print(evaluate(args.frozen, args.output, args.device, args.batch_size))
    else:
        make_figures(args.predictions, args.output)


if __name__ == "__main__":
    main()

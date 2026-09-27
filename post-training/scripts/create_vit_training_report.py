"""Create evidence-scoped training JSON and a model-card report from an exported run."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

import torch


METRICS = {"loss": "bce_loss", "accuracy": "accuracy", "precision": "precision",
           "recall": "recall", "f1": "f1_score", "roc_auc": "roc_auc",
           "average_precision": "average_precision", "balanced_accuracy": "balanced_accuracy",
           "specificity": "specificity", "mcc": "mcc"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)


def read_history(path):
    with path.open(newline="") as stream:
        rows = [{key: value if key == "phase" else float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]
    if not rows:
        raise ValueError("Empty training history")
    for index, row in enumerate(rows, 1):
        if row["epoch"] != index or not all(math.isfinite(v) for v in row.values() if isinstance(v, float)):
            raise ValueError("Nonfinite values or noncontiguous history")
        for key in ("epoch", "world_size", "trainable_parameters"):
            row[key] = int(row[key])
        if row["elapsed_seconds"] <= (rows[index-2]["elapsed_seconds"] if index > 1 else 0):
            raise ValueError("Training timer is not increasing")
        for key in ("train_accuracy", "val_accuracy", "train_precision", "val_precision", "train_recall",
                    "val_recall", "train_f1", "val_f1", "train_roc_auc", "val_roc_auc", "val_specificity"):
            if not 0 <= row[key] <= 1:
                raise ValueError(f"Metric outside [0,1]: {key}")
    return rows


def metrics(row, part):
    result = {name: row[f"{part}_{key}"] for key, name in METRICS.items()}
    result.update(false_positive_rate=1 - result["specificity"], false_negative_rate=1 - result["recall"])
    return result


def build_result(source):
    runs = sorted((source / "runs").glob("*/training_history.csv"))
    if len(runs) != 1:
        raise ValueError("Expected exactly one exported training run")
    run = runs[0].parent
    cfg = json.loads((run / "run_config_used.json").read_text())
    completion = json.loads((run / "training_complete.json").read_text())
    demo = json.loads((source / "gradio_app/self_test_summary.json").read_text())
    rows = read_history(runs[0])
    checkpoint = run / "checkpoints/best_model.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    observed_parameters = sum(t.numel() for t in saved["model_state"].values())
    best = rows[int(metadata["best_epoch"]) - 1]
    last = rows[-1]
    files = {p.relative_to(source / "code").as_posix(): p.read_text()
             for p in (source / "code").rglob("*.py")}
    runtime_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    checks = {
        "completion_matches_history_length": completion["epochs"] == len(rows),
        "completion_status_trained": completion["status"] == "trained",
        "completion_time_matches_history": close(completion["seconds"], last["elapsed_seconds"]),
        "checkpoint_time_matches_completion": close(metadata["training_time_seconds"], completion["seconds"]),
        "best_epoch_matches_minimum_validation_loss": best["epoch"] == min(rows, key=lambda r: r["val_loss"])["epoch"],
        "runtime_code_hash_matches_configuration": runtime_hash == cfg["runtime_sha256"],
        "model_config_matches_checkpoint": cfg["model"] == metadata["model_config"],
        "run_identifiers_match": cfg["run_name"] == metadata["run_name"] == demo["run"]["run_name"] == run.name,
        "label_mapping_matches": metadata["label_mapping"] == {"real": 0, "fake": 1},
        "threshold_matches": metadata["threshold"] == demo["run"]["threshold"] == .5,
        "parameter_counts_match": observed_parameters == metadata["total_parameters"] == last["trainable_parameters"],
        "world_sizes_match": cfg["distributed"]["world_size"] == metadata["world_size"] == 2 and all(r["world_size"] == 2 for r in rows),
        "demo_best_accuracy_matches_history": close(demo["run"]["val_accuracy_at_best"], best["val_accuracy"]),
        "demo_best_auc_matches_history": close(demo["run"]["val_roc_auc_at_best"], best["val_roc_auc"]),
    }
    if not all(checks.values()):
        raise ValueError(f"Inconsistent evidence: {[k for k, v in checks.items() if not v]}")
    stale_epochs = len(rows) - best["epoch"]
    early_stop = stale_epochs >= cfg["train"]["early_stopping_patience"] and len(rows) < cfg["train"]["epochs"]
    durations = [r["elapsed_seconds"] - (rows[i-1]["elapsed_seconds"] if i else 0) for i, r in enumerate(rows)]
    evidence_files = [runs[0], run / "training_complete.json", run / "run_config_used.json", checkpoint,
                      source / "gradio_app/self_test_summary.json", *sorted((source / "code").rglob("*.py"))]
    result = {
        "schema_version": "1.0",
        "report_type": "training_validation_model_card",
        "format_note": "Project-specific evidence schema; not a claimed proprietary OpenCodeGen schema.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run.name,
        "scope": "Recorded training and validation evidence, not final held-out test evaluation",
        "metric_units": "Fractions in [0,1] unless named seconds, bytes, images_per_second or percentage_points; BCE/MCC are not percentages",
        "metric_definition": {"positive_class": "fake", "positive_label": 1, "threshold": .5,
                              "precision_recall_f1_averaging": "binary", "average_precision": "AP, not trapezoidal PR-AUC"},
        "model": {"architecture": "ViT-B/16", "backbone": cfg["model"]["backbone"],
                  "pretraining_recipe": "ImageNet-21k pretrained, then ImageNet-1k fine-tuned, then Wish fine-tuned",
                  "frequency_branch": False, "parameters": metadata["total_parameters"],
                  "trainable_parameters": metadata["trainable_parameters"], "preprocessing": metadata["preprocessing"]},
        "data": {"dataset": "wish096/realvsfake-81k-by-wish", "manifest_sha256_recorded": metadata["manifest_sha256"],
                 "dataset_version_recorded": metadata["dataset_version"], "manifest_supplied": False,
                 "audit_supplied": False, "split_counts": None, "identity_independence_verified": False,
                 "protocol_from_code": {"train_validation_primary": ["FFHQ", "CelebA", "StyleGAN"],
                                        "cross_generator": ["separate real controls", "StableDiffusion"],
                                        "excluded_source": "AiGenImage"},
                 "limitation": "Protocol inspected in supplied code; split membership/leakage cannot be independently checked without manifest and audit."},
        "training": {"status": "completed", "epochs_completed": len(rows), "maximum_epochs": cfg["train"]["epochs"],
                     "best_epoch": best["epoch"], "selection_criterion": metadata["selection"],
                     "early_stopping_patience": cfg["train"]["early_stopping_patience"],
                     "stop_reason": "early_stopping_inferred_from_history_and_training_code" if early_stop else "not_established",
                     "epochs_without_improvement_after_best": stale_epochs,
                     "seed": cfg["seed"], "optimizer": cfg["train"]["optimizer"], "learning_rate_target": cfg["train"]["lr"],
                     "weight_decay": cfg["train"]["weight_decay"], "warmup_fraction": cfg["train"]["warmup_steps_pct"],
                     "scheduler": "linear_warmup_then_decay", "gradient_clip_norm": cfg["train"]["grad_clip_norm"],
                     "mixed_precision": cfg["train"]["mixed_precision"], "world_size": cfg["distributed"]["world_size"],
                     "microbatch_per_gpu": cfg["data"]["batch_size"], "effective_batch_size": cfg["train"]["effective_batch_size"],
                     "accumulation_steps": cfg["train"]["effective_batch_size"] // (cfg["data"]["batch_size"] * cfg["distributed"]["world_size"])},
        "metrics": {"validation_at_selected_checkpoint": metrics(best, "val"),
                    "training_online_during_selected_epoch": metrics(best, "train"),
                    "validation_at_last_epoch": metrics(last, "val"),
                    "training_online_during_last_epoch": metrics(last, "train")},
        "metrics_provenance": "Copied from recorded epoch history; not independently recomputed from per-image predictions. Training metrics are online augmented-batch metrics, not fixed-checkpoint train-set evaluation.",
        "efficiency": {"recorded_training_validation_seconds": completion["seconds"],
                       "recorded_time_scope": "Sum of epoch train/validation calls; excludes dataset audit, cache build, setup, most checkpoint/plot writing and demo.",
                       "epoch_seconds": durations, "mean_epoch_seconds": statistics.mean(durations),
                       "recorded_train_images_per_second_mean": statistics.mean(r["train_images_per_second"] for r in rows),
                       "recorded_train_images_per_second_range": [min(r["train_images_per_second"] for r in rows), max(r["train_images_per_second"] for r in rows)],
                       "throughput_caveat": "Trainer-reported throughput; not a controlled synchronized benchmark or inference latency.",
                       "inference_ms_per_image": None, "peak_gpu_memory_bytes": None, "single_gpu_speedup": None,
                       "gpu_model_verified": None},
        "checkpoint": {"file": "checkpoints/best_model.pt", "sha256": sha256(checkpoint), "size_bytes": checkpoint.stat().st_size,
                       "metadata": metadata},
        "gradio_self_test": {"status": "reported_pass", "evidence": demo,
                             "scope": "36 selected validation examples; not an independent test set",
                             "per_image_results_supplied": False,
                             "limitation": "Summary reports 100% accuracy and endpoint agreement; sample identities and individual outputs cannot be verified from this export."},
        "held_out_evaluation": {"status": "not_completed_according_to_supplied_self_test_summary",
                                "primary_test": None, "cross_generator_test": None,
                                "accuracy_drop_percentage_points": None, "f1_drop": None},
        "unavailable_evidence": ["data_audit/manifest.csv", "data_audit/audit.json", "per-image validation predictions",
                                 "primary and cross-generator prediction CSVs", "confusion matrix counts", "ROC/PR curve points",
                                 "calibration metrics", "run environment/device log", "last.pt resume state",
                                 "Gradio per-image self-test CSV", "other models' matching-split results"],
        "integrity_checks": checks,
        "source_files": [{"path": p.relative_to(source).as_posix(), "sha256": sha256(p)} for p in evidence_files],
        "history": rows,
        "conclusion": "Strong recorded within-dataset validation discrimination. Best checkpoint is epoch 1 after a seven-epoch run. Unseen-generator robustness and comparative superiority remain unestablished."
    }
    return result, run


def plots(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = result["history"]
    epochs = [r["epoch"] for r in rows]
    target = output / "figures"
    target.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), layout="constrained")
    for ax, key, label in zip(axes.flat, ("loss", "accuracy", "precision", "recall", "f1", "roc_auc"),
                              ("BCE loss", "Accuracy", "Fake precision", "Fake recall", "Fake F1", "ROC-AUC")):
        for part, color, display in (("train", "#137c66", "Training (online)"), ("val", "#b13e65", "Validation")):
            ax.plot(epochs, [r[f"{part}_{key}"] for r in rows], marker="o", markersize=3, label=display, color=color)
        ax.axvline(result["training"]["best_epoch"], color="#555555", linestyle=":", linewidth=1)
        ax.set(title=label, xlabel="Epoch", xticks=epochs)
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Training and validation only | selected checkpoint: epoch 1\nMetric axes are auto-scaled; no held-out test results", fontsize=12)
    fig.savefig(target / "training_validation.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    axes[0].plot(epochs, [r["val_loss"] for r in rows], marker="o", color="#b13e65")
    axes[0].scatter([1], [rows[0]["val_loss"]], s=80, facecolors="none", edgecolors="black", label="Selected")
    axes[0].legend(fontsize=8)
    axes[0].set(title="Validation BCE loss", ylabel="Loss")
    axes[1].plot(epochs, [100*r["val_accuracy"] for r in rows], marker="o", color="#137c66")
    axes[1].set(title="Validation accuracy (zoomed)", ylabel="Percent")
    axes[2].plot(epochs, [r["lr"] for r in rows], marker="o", color="#675397")
    axes[2].set(title="Recorded end-of-epoch LR", ylabel="Learning rate")
    for ax in axes:
        ax.set(xlabel="Epoch", xticks=epochs)
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.grid(alpha=.2)
    fig.savefig(target / "selection_and_schedule.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    axes[0].bar(epochs, [v/60 for v in result["efficiency"]["epoch_seconds"]], color="#137c66")
    axes[0].set(title="Recorded train + validation duration", ylabel="Minutes", xlabel="Epoch", xticks=epochs)
    axes[1].plot(epochs, [r["train_images_per_second"] for r in rows], marker="o", color="#675397")
    axes[1].set(title="Trainer-reported throughput", ylabel="Images / second", xlabel="Epoch", xticks=epochs)
    axes[1].set_ylim(0, max(r["train_images_per_second"] for r in rows)*1.1)
    fig.savefig(target / "efficiency.png", dpi=160)
    plt.close(fig)


def report(result):
    best = result["metrics"]["validation_at_selected_checkpoint"]
    last = result["metrics"]["validation_at_last_epoch"]
    total = result["efficiency"]["recorded_training_validation_seconds"]
    metrics_table = "\n".join(f"| {name} | {best[key]:.8f} | {last[key]:.8f} |" for key, name in
        (("bce_loss", "BCE loss"), ("accuracy", "Accuracy"), ("precision", "Fake precision"), ("recall", "Fake recall"),
         ("f1_score", "Fake F1"), ("roc_auc", "ROC-AUC"), ("average_precision", "Average precision"),
         ("balanced_accuracy", "Balanced accuracy"), ("specificity", "Real specificity"), ("mcc", "MCC")))
    epoch_table = "\n".join(f"| {r['epoch']} | {r['train_loss']:.6f} | {r['val_loss']:.6f} | {100*r['val_accuracy']:.4f}% | {r['val_f1']:.6f} |"
                            for r in result["history"])
    return f'''# ViT-B/16 Training Evidence Report

Run: `{result['run_name']}`  
Scope: recorded training, validation and Gradio validation-subset checks.  
Format: model-card-style experiment report. No proprietary OpenCodeGen format is claimed.

## 1. Executive Summary

The exported run completed **7 epochs** of a maximum 30. The checkpoint selected by minimum validation loss is **epoch 1**, not epoch 7. Its recorded validation accuracy is **{100*best['accuracy']:.4f}%**, fake-class F1 is **{best['f1_score']:.6f}**, and ROC-AUC is **{best['roc_auc']:.8f}**. The epoch 1 checkpoint and this report refer to the same SHA-256-verified weight file already stored in the project repository.

Training plus validation calls account for **{total:.2f} seconds**, approximately **59 minutes 58 seconds**. Six consecutive later epochs did not improve validation loss. Together with the supplied stopping code and patience 6, this explains the stop at epoch 7. The completion file records status `trained` but does not itself name a stop reason; early stopping is inferred from the corroborating evidence.

The Gradio summary reports correct labels for **36/36 validation examples**, plus agreement between direct inference and both endpoints. **No final primary-test or cross-generator results are included.** Strong validation performance is not proof of unseen-generator robustness or superiority over the team's CNNs.

## 2. Model and Intended Use

Plain ViT-B/16 with 16x16 patches, a single-logit binary head, dropout 0.1 and **85,799,425 trainable parameters**. The named timm checkpoint is `vit_base_patch16_224.augreg_in21k_ft_in1k`: ImageNet-21k pretraining followed by ImageNet-1k fine-tuning before this Wish run. The optional FFT branch is disabled.

Inference uses RGB input, square 224x224 bicubic resizing, and mean/std **[0.5, 0.5, 0.5]**. Real is 0; fake is 1. A fake probability of at least 0.5 predicts fake. These are model scores, not calibrated assurances of authenticity. Intended inputs are face crops resembling the training data; arbitrary photos, face swaps and other generators are outside the demonstrated evidence.

## 3. Data and Provenance

Dataset: `wish096/realvsfake-81k-by-wish`. The supplied code defines train/validation/primary testing using FFHQ and CelebA real faces plus StyleGAN fakes. Stable Diffusion and separate real controls are reserved for cross-generator testing; AiGenImage is excluded. These are **code-level protocol statements**, not independently verified split membership in this report.

The export does not contain the manifest, audit, image inventory or dataset images. Exact split sizes, source proportions, duplicate exclusions, identity overlap and leakage status therefore cannot be independently verified. A recorded checksum identifies an expected manifest; it does not substitute for inspecting that manifest.

Manifest SHA-256 recorded in checkpoint:  
`{result['data']['manifest_sha256_recorded']}`

Dataset snapshot recorded in checkpoint:  
`{result['data']['dataset_version_recorded']}`

## 4. Training Protocol

| Setting | Recorded value |
| --- | --- |
| Seed | 42 |
| Optimizer / loss | AdamW / BCEWithLogitsLoss |
| Target learning rate / weight decay | 0.00003 / 0.01 |
| Schedule | 10% linear warmup, then linear decay |
| Gradient clipping | 1.0 |
| Distributed workers | 2 CUDA ranks |
| Per-GPU microbatch / effective batch | 16 / 32 |
| Accumulation steps | 1 |
| Mixed precision | Enabled |
| Maximum epochs / patience | 30 / 6 |
| Selection criterion | Lowest validation loss |

The notebook is intended for T4 x2, but the export lacks an environment/device log, so the exact GPU SKU and dependency versions used during training are not independently verified. The supplied training implementation uses a padding DistributedSampler for training and removes repeated sample IDs from reported metrics. Depending on sample-count divisibility, padding can still repeat training examples in gradient updates. The absent manifest prevents checking whether this occurred. Validation uses disjoint, unpadded shards.

## 5. Recorded Results

All metrics below are **validation** metrics. Classification scores use a 0-1 scale, MCC uses -1 to 1, and BCE loss is nonnegative without an upper bound. They are read from the epoch history, not recomputed from missing per-image predictions. Precision, recall and F1 treat fake as the positive class. AP means average precision, not trapezoidal PR-AUC.

| Metric | Selected checkpoint: epoch 1 | Last recorded epoch: 7 |
| --- | ---: | ---: |
{metrics_table}

Epoch 2 has the highest recorded validation accuracy, but its validation loss is worse than epoch 1. It is therefore **not** the checkpoint selected by the declared protocol. Changing selection criteria after observing results would describe a different experiment.

| Epoch | Training BCE | Validation BCE | Validation accuracy | Validation F1 |
| --- | ---: | ---: | ---: | ---: |
{epoch_table}

![Training and validation history](figures/training_validation.png)

Training metrics summarize augmented batches while model weights are changing; they are not a fixed-checkpoint evaluation of the entire training set. Metric axes in this figure are auto-scaled and do not all start at zero.

## 6. Convergence and Checkpoint Selection

Training loss falls from 0.138305 to 0.023548, while validation loss is best at the first epoch and remains worse in all six later epochs. The final validation loss is approximately {last['bce_loss']/best['bce_loss']:.2f} times the best value. This supports retaining the early checkpoint rather than assuming the latest weights are better.

The target learning rate is approached around epoch 3 during warmup. Subsequent worsening validation loss may reflect increased confidence on mistakes, optimization sensitivity or overfitting to the training domain. Aggregate curves cannot establish which explanation is correct, and they do not prove leakage. Strong pretraining and training-only augmentation/dropout can also help explain validation performance exceeding the online training metrics.

![Selection and schedule](figures/selection_and_schedule.png)

Further tuning, if justified, must use validation only and retain a fresh final test protocol. Do not extend training merely to reach 30 epochs or select the most flattering metric after the fact.

## 7. Efficiency

Recorded train/validation time: **{total:.2f} seconds**. Mean recorded epoch duration: **{result['efficiency']['mean_epoch_seconds']/60:.2f} minutes**. Mean reported training throughput: **{result['efficiency']['recorded_train_images_per_second_mean']:.2f} images/second** across the run's distributed training process.

![Recorded efficiency](figures/efficiency.png)

These timings exclude dataset audit/cache preparation, setup and most output writing. The trainer's throughput is not a controlled synchronized inference benchmark. There is no matched single-GPU baseline, so neither a twofold speedup nor a total end-to-end runtime improvement can be claimed. Inference latency and peak GPU memory are unreported.

## 8. Gradio Integration Evidence

The supplied summary reports **24 real and 12 fake validation images**, all predicted correctly. Sources are CelebA, FFHQ and StyleGAN. Both single-image and batch endpoints reportedly match direct model inference. The summary records Gradio 5.50.0 for this self-test.

This is a small selected subset already involved in model selection. **100% on these 36 examples is not final detector accuracy.** The export omits the per-image self-test CSV and example identities, so the summary cannot be independently regenerated. No Stable Diffusion examples were part of this reported self-test. No confusion matrix or uncertainty interval for the full validation/test population is invented from this summary.

## 9. Evaluation Gaps and Limitations

| Evidence | Availability |
| --- | --- |
| Training history and best weights | Supplied and internally reconciled |
| Primary held-out test | Not completed according to supplied self-test summary; no results supplied |
| Cross-generator test | Not completed according to supplied self-test summary; no results supplied |
| Accuracy/F1 generalization drops | Not calculable |
| Per-image probabilities and confusion counts | Not supplied |
| ROC/PR curve points and calibration | Not supplied; cannot reconstruct from scalar AUC/AP |
| Shared-split four-model comparison | Not supplied |
| Repeated-seed uncertainty | One seed only |
| Dataset/identity leakage verification | Manifest, audit and identity evidence absent |
| Exact hardware and training package versions | Environment log absent |

Source/compression shortcuts, unequal pretraining, unknown identity overlap and changes in image generator can all limit external validity. The defensible conclusion is excellent **recorded within-dataset validation** performance, not a universal deepfake detector.

## 10. Reproducibility and Next Evidence

The checkpoint SHA-256 is `{result['checkpoint']['sha256']}`. The supplied Python runtime reproduces configuration hash `{result['training_configuration']['runtime_sha256']}`. The JSON includes source-file hashes, the complete seven-epoch history, metric definitions and consistency checks. No training, test inference or checkpoint modification was performed to create this report.

Retrieve the exact manifest/audit and complete run environment. Freeze the selected checkpoint and threshold, then perform the agreed primary and cross-generator evaluations without tuning on either test. Export per-image predictions, confusion matrices, ROC/PR curves, calibration diagnostics, false positives/negatives and synchronized inference timing. Compare A/B models only after verifying compatible training and held-out assignments. Keep missing results explicitly unavailable until these measurements exist.

Evidence inputs: `training_history.csv`, `training_complete.json`, `run_config_used.json`, `best_model.pt`, `gradio_app/self_test_summary.json`, and supplied runtime source. Original lightweight inputs are preserved in `evidence/`; the large checkpoint is identified by its hash rather than duplicated.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result, run = build_result(args.source)
    result["training_configuration"] = json.loads((run / "run_config_used.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = args.output / "evidence"
    evidence.mkdir(exist_ok=True)
    for path in (run / "training_history.csv", run / "training_complete.json", run / "run_config_used.json",
                 args.source / "gradio_app/self_test_summary.json", run / "learning_curves.png"):
        shutil.copy2(path, evidence / path.name)
    plots(result, args.output)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    markdown = report(result)
    (args.output / "training_report.md").write_text(markdown)
    from markdown_it import MarkdownIt
    body = MarkdownIt("commonmark").enable("table").render(markdown)
    (args.output / "training_report.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>ViT Training Evidence Report</title>
<style>body{font:16px/1.65 system-ui,sans-serif;color:#20252a;background:#fff;max-width:1050px;margin:40px auto;padding:0 24px}
h1{font-size:30px}h2{font-size:23px;border-top:1px solid #d8dfdd;padding-top:22px;margin-top:38px;color:#145b4b}
table{border-collapse:collapse;width:100%;display:block;overflow-x:auto;font-size:14px}th,td{padding:9px 12px;border-bottom:1px solid #d8dfdd;text-align:left}th{background:#edf5f1}
code{font-size:13px;overflow-wrap:anywhere}img{max-width:100%;height:auto}p{overflow-wrap:anywhere}
@media print{body{margin:0;max-width:none;font-size:11px}h2{break-after:avoid}img,tr{break-inside:avoid}}</style><body>''' + body + "</body></html>\n")
    loaded = json.loads((args.output / "result.json").read_text())
    assert loaded["training"]["epochs_completed"] == 7 and loaded["training"]["best_epoch"] == 1
    assert loaded["held_out_evaluation"]["cross_generator_test"] is None
    assert loaded["gradio_self_test"]["evidence"]["splits_used"] == ["val"]
    assert all(loaded["integrity_checks"].values())
    print(json.dumps({"output": str(args.output), "epochs": 7, "selected_epoch": 1,
                      "validation_accuracy": loaded["metrics"]["validation_at_selected_checkpoint"]["accuracy"],
                      "checks_passed": len(loaded["integrity_checks"])}))


if __name__ == "__main__":
    main()

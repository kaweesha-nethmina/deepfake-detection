import copy
import csv
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from models.vit import manifest
from models.vit.evaluate_crossgen import evaluate, freeze, predict_loader, validate_metadata
from models.vit.reporting import compute_metrics, make_figures
from models.vit.runtime import (ManifestDataset, binary_loss, build_model, load_checkpoint,
                                load_config, predict_image, preprocessing_for, probabilities,
                                transform_for)
from models.vit.train import phase_parameters, run_epoch, train

torch.set_num_threads(1)


@pytest.fixture
def wish(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    rows = []
    rng = np.random.default_rng(10)
    for i, split in enumerate(manifest.SPLITS):
        for label in (0, 1):
            for j in (0, 1):
                prefix = "RFF" if label == 0 else ("FSD" if split == "cross_gen" else "FSG")
                name = f"{prefix} ({i * 2 + j + 1}).jpg"
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(root / name)
                rows.append({"filepath": name, "label": label,
                             "source": manifest.SOURCES[prefix][1], "split": split})
    source = tmp_path / "member_a.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    audit = tmp_path / "audit"
    manifest.audit_manifest(source, root, audit, "fixture-only", near_distance=0)
    return root, audit / "manifest.csv", rows, source


@pytest.fixture
def tiny(monkeypatch):
    module = types.ModuleType("fixture_model")
    module.build_model = lambda cfg: torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                         torch.nn.Flatten(), torch.nn.Linear(3, cfg["model"].get("num_classes", 1)))
    monkeypatch.setitem(sys.modules, "fixture_model", module)
    return module


def configuration(tmp_path, wish, name="vit_b16"):
    root, csv_path, _, _ = wish
    return {"model_name": name, "module": "fixture_model", "model": {"num_classes": 1},
            "run_name": name + "_fixture", "seed": 42, "device": "cpu",
            "data": {"root": str(root), "manifest": str(csv_path), "batch_size": 2, "num_workers": 0},
            "train": {"epochs": 2, "early_stopping_patience": 6, "lr": 0.001,
                      "weight_decay": 0.01, "effective_batch_size": 4},
            "output": {"results_dir": str(tmp_path / "runs")}}


@pytest.mark.parametrize("name,label,source", [("RFF (1).jpg", 0, "FFHQ"),
                         ("RCA (1).jpg", 0, "CelebA"), ("FSG (1).jpg", 1, "StyleGAN"),
                         ("FSD (1).jpg", 1, "StableDiffusion"), ("AI (1).jpg", 1, "AiGenImage")])
def test_source_names_not_parent_substrings(name, label, source):
    assert manifest.source_from_name("RealVsFake/" + name) == (label, source)


def test_manifest_preserves_assignments(wish):
    root, path, original, _ = wish
    rows, audit = manifest.load_audited_manifest(path, root)
    assert [(r["filepath"], r["split"]) for r in rows] == [(r["filepath"], r["split"]) for r in original]
    assert audit["dataset"] == manifest.DATASET


@pytest.mark.parametrize("problem", ["overlap", "diffusion", "label", "unknown", "identity", "aigen"])
def test_bad_assignments_fail(wish, problem):
    rows = copy.deepcopy(wish[2])
    if problem == "overlap":
        rows[-1]["filepath"] = rows[0]["filepath"]
    elif problem == "diffusion":
        rows[-1]["split"] = "train"
    elif problem == "label":
        rows[0]["label"] = 1
    elif problem == "unknown":
        rows[0]["filepath"] = "unknown.jpg"
    elif problem == "identity":
        rows[0]["identity"] = rows[4]["identity"] = "same-person"
    else:
        rows[2].update(filepath="AI (1).jpg", source="AiGenImage")
    with pytest.raises(ValueError):
        manifest.validate_rows(rows)


@pytest.mark.parametrize("relative", ["../escape.jpg", "/tmp/absolute.jpg"])
def test_paths_cannot_escape(tmp_path, relative):
    with pytest.raises(ValueError):
        manifest.image_path(tmp_path, relative)


def test_changed_content_fails_audit(wish):
    root, path, rows, _ = wish
    (root / rows[0]["filepath"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        manifest.load_audited_manifest(path, root)


def test_duplicate_pixels_fail(wish, tmp_path):
    root, _, rows, source = wish
    (root / rows[4]["filepath"]).write_bytes((root / rows[0]["filepath"]).read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        manifest.audit_manifest(source, root, tmp_path / "duplicate", "fixture")


def test_missing_image_fails(wish, tmp_path):
    root, _, rows, source = wish
    (root / rows[0]["filepath"]).unlink()
    with pytest.raises(FileNotFoundError):
        manifest.audit_manifest(source, root, tmp_path / "missing", "fixture")


def test_near_duplicates_require_explicit_review(wish):
    root, path, _, _ = wish
    report = json.loads(path.with_name("audit.json").read_text())
    report["near_duplicate_pairs"] = 1
    path.with_name("audit.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Review"):
        manifest.load_audited_manifest(path, root)


def test_output_conventions_agree():
    logits = torch.tensor([-3.0, 0.0, 2.0])
    scores = probabilities(logits)
    assert torch.allclose(scores, probabilities(logits[:, None]))
    assert torch.allclose(scores, probabilities(torch.stack([torch.zeros_like(logits), logits], dim=1)))
    with pytest.raises(ValueError):
        probabilities(torch.zeros(2, 3))


def test_metrics_ties_and_no_positive_predictions():
    metrics = compute_metrics([0, 1, 0, 1], [0.2, 0.2, 0.2, 0.2])
    assert metrics["roc_auc"] == 0.5
    assert metrics["f1_score"] == 0
    assert metrics["confusion_matrix"] == [[2, 0], [2, 0]]
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [float("nan"), 1])


def test_train_resume_is_reproducible(tmp_path, wish, tiny, monkeypatch):
    import models.vit.train as training
    cfg = configuration(tmp_path, wish)
    reference = train(cfg)
    expected = torch.load(reference / "checkpoints/last.pt", weights_only=True)
    cfg["run_name"] += "_interrupted"
    original = training.run_epoch
    calls = 0
    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated session interruption")
        return original(*args, **kwargs)
    monkeypatch.setattr(training, "run_epoch", interrupt)
    with pytest.raises(RuntimeError, match="simulated"):
        train(cfg)
    monkeypatch.setattr(training, "run_epoch", original)
    last = Path(cfg["output"]["results_dir"]) / cfg["run_name"] / "checkpoints/last.pt"
    resumed = train(cfg, str(last))
    actual = torch.load(resumed / "checkpoints/last.pt", weights_only=True)
    for key in expected["model_state"]:
        assert torch.equal(expected["model_state"][key], actual["model_state"][key])
    assert len(actual["history"]) == 2
    assert not list(resumed.glob("*test*"))


def test_demo_and_evaluator_agree_without_mutation(tmp_path, wish, tiny):
    cfg = configuration(tmp_path, wish)
    out = train(cfg)
    model, metadata = load_checkpoint(cfg, out / "checkpoints/best_model.pt", torch.device("cpu"))
    rows, _ = manifest.load_audited_manifest(wish[1], wish[0])
    ds = ManifestDataset(rows, wish[0], "test", metadata["preprocessing"])
    before = {k: v.clone() for k, v in model.state_dict().items()}
    predictions, _ = predict_loader(model, DataLoader(ds, batch_size=2), torch.device("cpu"), "tiny", "fixture")
    with Image.open(wish[0] / ds.rows[0]["filepath"]) as image:
        score = predict_image(model, image, metadata["preprocessing"], torch.device("cpu"))
    assert score == pytest.approx(predictions[0]["probability"], abs=1e-7)
    assert torch.equal(ds[0][0], ds[0][0])
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_four_model_workflow_and_regeneration(tmp_path, wish, tiny):
    harness = {"manifest": str(wish[1]), "data_root": str(wish[0]), "models": {}}
    for name in ("custom_cnn", "resnet50", "efficientnetv2", "vit_b16"):
        cfg = configuration(tmp_path, wish, name)
        out = train(cfg)
        config_path = tmp_path / (name + ".yaml")
        config_path.write_text(yaml.safe_dump(cfg))
        harness["models"][name] = {"config": str(config_path), "checkpoint": str(out / "checkpoints/best_model.pt")}
    harness_path = tmp_path / "harness.yaml"
    harness_path.write_text(yaml.safe_dump(harness))
    frozen = tmp_path / "frozen.json"
    freeze(harness_path, frozen)
    output = evaluate(frozen, tmp_path / "final", "cpu", 2)
    assert json.loads((output / "status.json").read_text())["status"] == "complete"
    frame = pd.read_csv(output / "predictions.csv")
    assert len(frame) == 4 * 8
    regenerated = make_figures(output / "predictions.csv", tmp_path / "regenerated")
    for name in harness["models"]:
        results = json.loads((output / name / "results.json").read_text())
        row = regenerated[(regenerated.model == name) & (regenerated.split == "test")].iloc[0]
        assert row.f1_score == results["test_metrics"]["f1_score"]
    with pytest.raises(FileExistsError):
        evaluate(frozen, output, "cpu", 2)
    checkpoint = Path(harness["models"]["vit_b16"]["checkpoint"])
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed after freeze"):
        evaluate(frozen, tmp_path / "tampered", "cpu", 2)
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="missing checkpoint"):
        freeze(harness_path, tmp_path / "missing_weights.json")
    harness["models"].pop("vit_b16")
    harness_path.write_text(yaml.safe_dump(harness))
    with pytest.raises(ValueError, match="four"):
        freeze(harness_path, tmp_path / "missing.json")


def test_smoke_checkpoint_cannot_be_final(tmp_path, wish, tiny):
    cfg = configuration(tmp_path, wish)
    cfg["smoke"] = True
    out = train(cfg)
    saved = torch.load(out / "checkpoints/best_model.pt", weights_only=True)
    with pytest.raises(ValueError, match="non-smoke"):
        validate_metadata(saved["metadata"], cfg, saved["metadata"]["manifest_sha256"])


def test_resume_rejects_config_change(tmp_path, wish, tiny):
    cfg = configuration(tmp_path, wish)
    out = train(cfg)
    cfg["train"]["lr"] = 0.1
    with pytest.raises(ValueError, match="mismatch"):
        train(cfg, out / "checkpoints/last.pt")


def test_partial_accumulation_matches_full_batch():
    torch.manual_seed(9)
    model = torch.nn.Linear(3, 1)
    twin = copy.deepcopy(model)
    ds = TensorDataset(torch.randn(7, 3), torch.tensor([0, 1, 0, 1, 0, 1, 0]), torch.arange(7))
    for net, batch, accumulation in ((model, 2, 2), (twin, 4, 1)):
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        run_epoch(net, DataLoader(ds, batch_size=batch), torch.device("cpu"),
                  optimizer=optimizer, scaler=scaler, accumulation=accumulation, grad_clip=100)
    for a, b in zip(model.parameters(), twin.parameters()):
        assert torch.allclose(a, b, atol=1e-7)


def test_training_does_not_open_held_out_images(tmp_path, wish, tiny):
    cfg = configuration(tmp_path, wish)
    for row in wish[2]:
        if row["split"] in ("test", "cross_gen"):
            (wish[0] / row["filepath"]).unlink()
    # Hash/schema audit exists; no final images should be opened by this command.
    assert (train(cfg) / "checkpoints/best_model.pt").exists()


def test_factory_supports_teammates_get_model(monkeypatch):
    module = types.ModuleType("fixture_get_model")
    module.get_model = lambda num_classes: torch.nn.Linear(3, num_classes)
    monkeypatch.setitem(sys.modules, "fixture_get_model", module)
    model = build_model({"module": "fixture_get_model", "model": {"num_classes": 2}}, pretrained=False)
    assert model(torch.zeros(1, 3)).shape == (1, 2)


def test_fft_half_input_is_finite_at_224():
    from models.vit.model import FrequencyBranch
    result = FrequencyBranch.to_fft_magnitude(torch.zeros(1, 3, 224, 224, dtype=torch.float16))
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()


def test_notebook_schema_and_cells_compile():
    import nbformat
    path = Path("notebooks/04_vit_crossgen_evaluation.ipynb")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(path), "exec")


@pytest.mark.architecture
@pytest.mark.parametrize("path", ["configs/vit.yaml", "configs/vit_freq_hybrid.yaml",
                                 "configs/wish/custom_cnn.yaml", "configs/wish/resnet50.yaml",
                                 "configs/wish/efficientnetv2.yaml"])
def test_real_architecture_offline_forward(path):
    cfg = load_config(path)
    model = build_model(cfg, pretrained=False).eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 224, 224))
    assert probabilities(logits).shape == (1,)
    assert torch.isfinite(logits).all()
    assert preprocessing_for(model, cfg)["image_size"] == 224
    if cfg["train"].get("warmup_epochs"):
        phase_parameters(model, cfg, True)
        warm = sum(p.numel() for p in model.parameters() if p.requires_grad)
        phase_parameters(model, cfg, False)
        assert warm < sum(p.numel() for p in model.parameters() if p.requires_grad)

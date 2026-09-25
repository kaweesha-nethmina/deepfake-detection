"""Exercise notebook dataset/setup cells without network or GPU downloads."""
import ast
import csv
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from models.vit.manifest import combine_split_csvs, read_manifest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/04_vit_crossgen_evaluation.ipynb"


def source(cell_id):
    nb = json.loads(NOTEBOOK.read_text())
    return "".join(next(cell for cell in nb["cells"] if cell["id"] == cell_id)["source"])


def execute_dataset(monkeypatch, directory, version=None):
    calls = []
    module = types.ModuleType("kagglehub")
    def download(handle):
        calls.append(handle)
        return str(directory)
    module.dataset_download = download
    monkeypatch.setitem(sys.modules, "kagglehub", module)
    tree = ast.parse(source("dataset"))
    if version is not None:
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DATASET_VERSION" for t in node.targets):
                node.value = ast.Constant(value=version)
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), str(NOTEBOOK), "exec"), namespace)
    return calls, namespace


@pytest.mark.parametrize("nested", ["", "RealVsFake", "RealVsFake/RealVsFake"])
def test_exact_dataset_and_nested_root(monkeypatch, tmp_path, nested):
    root = tmp_path / nested
    (root / "Real").mkdir(parents=True)
    (root / "Fake").mkdir()
    calls, ns = execute_dataset(monkeypatch, tmp_path)
    assert calls == ["wish096/realvsfake-81k-by-wish"]
    assert ns["DATA_ROOT"] == root.resolve()


def test_pinned_version_and_ambiguous_roots(monkeypatch, tmp_path):
    (tmp_path / "Real").mkdir()
    (tmp_path / "Fake").mkdir()
    calls, ns = execute_dataset(monkeypatch, tmp_path, version=3)
    assert calls == ["wish096/realvsfake-81k-by-wish/versions/3"]
    for version in (True, 0, -1, "latest", 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            ns["wish_handle"](version)
    other = tmp_path / "ambiguous"
    for name in ("one", "two"):
        (other / name / "Real").mkdir(parents=True)
        (other / name / "Fake").mkdir()
    with pytest.raises(RuntimeError, match="Expected one"):
        ns["find_wish_root"](other)


def test_version_and_manifest_are_required(monkeypatch, tmp_path):
    (tmp_path / "Real").mkdir()
    (tmp_path / "Fake").mkdir()
    _, ns = execute_dataset(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="exact DATASET_VERSION"):
        exec(source("audit"), ns)
    ns.update(DATASET_VERSION=3, REQUESTED_HANDLE="wish096/realvsfake-81k-by-wish/versions/3")
    with pytest.raises(FileNotFoundError, match="shared CSV"):
        exec(source("audit"), ns)


def test_runtime_configs_use_loaded_data_without_editing_team_files(monkeypatch, tmp_path):
    repo = NOTEBOOK.parents[1]
    monkeypatch.chdir(repo)
    original = {p: p.read_bytes() for p in [repo / "configs/vit.yaml", *list((repo / "configs/wish").glob("*.yaml"))]}
    ns = {"Path": Path, "OUTPUT_ROOT": tmp_path, "DATA_ROOT": tmp_path / "attached/RealVsFake",
          "AUDIT": tmp_path / "data_audit/wish_v3", "review": None}
    exec(source("runtime-configs"), ns)
    harness = yaml.safe_load(ns["HARNESS_CONFIG"].read_text())
    assert len(harness["models"]) == 4
    for name, path in ns["CONFIGS"].items():
        cfg = yaml.safe_load(path.read_text())
        assert cfg["data"]["root"] == str(ns["DATA_ROOT"])
        assert cfg["data"]["manifest"] == str(ns["AUDIT"] / "manifest.csv")
        assert harness["models"][name]["config"] == str(path)
        assert str(tmp_path) in cfg["output"]["results_dir"]
    assert all(p.read_bytes() == value for p, value in original.items())
    # An unchanged rerun is safe, but a changed dataset cannot overwrite run configs.
    exec(source("runtime-configs"), ns)
    ns["DATA_ROOT"] = tmp_path / "different_data"
    with pytest.raises(ValueError, match="Existing run configuration differs"):
        exec(source("runtime-configs"), ns)


def split_csvs(tmp_path, prefix=""):
    files = {}
    for index, split in enumerate(("train", "val", "test", "cross_gen"), start=1):
        path = tmp_path / f"{split}_labels.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filepath", "label"])
            writer.writeheader()
            writer.writerow({"filepath": f"{prefix}Real/RFF ({index}).jpg", "label": 0})
            generator = "FSD" if split == "cross_gen" else "FSG"
            writer.writerow({"filepath": f"{prefix}Fake/{generator} ({index}).jpg", "label": 1})
        files[split] = path
    return files


def test_existing_csv_assignments_preserved_and_sources_inferred(tmp_path):
    files = split_csvs(tmp_path)
    path = combine_split_csvs(files, tmp_path / "combined.csv")
    rows = read_manifest(path)
    assert len(rows) == 8
    assert [r["split"] for r in rows] == [s for s in files for _ in range(2)]
    assert [r["source"] for r in rows] == ["FFHQ", "StyleGAN"] * 3 + ["FFHQ", "StableDiffusion"]
    before = path.read_bytes()
    combine_split_csvs(files, path)
    assert path.read_bytes() == before


def test_explicit_kaggle_prefix_mapping(tmp_path):
    prefix = "/kaggle/input/realvsfake-81k-by-wish/RealVsFake/RealVsFake/"
    files = split_csvs(tmp_path, prefix)
    with pytest.raises(ValueError, match="paths must be relative"):
        combine_split_csvs(files, tmp_path / "bad.csv")
    path = combine_split_csvs(files, tmp_path / "combined.csv", prefix)
    assert read_manifest(path)[0]["filepath"] == "Real/RFF (1).jpg"


def test_three_way_csvs_do_not_silently_create_cross_generator_split(tmp_path):
    files = split_csvs(tmp_path)
    files.pop("cross_gen")
    with pytest.raises(ValueError, match="cross_gen requires both classes"):
        combine_split_csvs(files, tmp_path / "combined.csv")


def test_csv_generator_leakage_is_not_silently_reassigned(tmp_path):
    files = split_csvs(tmp_path)
    files["train"].write_text(files["train"].read_text().replace("FSG (1)", "FSD (1)"))
    with pytest.raises(ValueError, match="StableDiffusion leakage"):
        combine_split_csvs(files, tmp_path / "combined.csv")

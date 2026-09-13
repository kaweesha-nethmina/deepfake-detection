"""
Dataset preparation / verification / leakage-check — Owner: Member C.

Implements the strict experimental protocol:

    StyleGAN  (real + StyleGAN-fake)  -> stratified 70/15/15 -> train/val/primary-test
    Stable Diffusion (fake) + held-out real -> cross-generator test ONLY

Rules enforced (see the protocol doc this was written against):
  - Never hardcode dataset counts — everything is computed from the files
    actually present under `raw_dir`.
  - Stable Diffusion images never enter train/val/primary-test.
  - Real images used for the cross-generator test are held out BEFORE the
    StyleGAN train/val/test split, so there is zero overlap.
  - REAL = 0, FAKE = 1 everywhere (matches common.py's DeepfakeImageDataset).
  - Report duplicates/leakage rather than silently fixing them.

Usage:
    python models/vit/prepare_data.py -c configs/dataset_prep.yaml

Outputs:
    data/processed/{train,val,test}/{real,fake}/         (symlinks or copies)
    data/cross_gen_test/{real,fake}/                      (symlinks or copies)
    results/vit/dataset_verification_report.json
    results/vit/dataset_leakage_report.json
    results/comparison/dataset_summary_table.csv          (shared folder, Member C owns it)
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.vit.common import load_config  # noqa: E402

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ---------------------------------------------------------------------------
# Step 1 — classify every file under raw_dir (label + generator), report unmatched
# ---------------------------------------------------------------------------
def classify_path(rel_path_lower: str, rules: list, key: str):
    for rule in rules:
        if rule["keyword"] in rel_path_lower:
            return rule[key]
    return None


def scan_raw_dataset(raw_dir: Path, label_rules: list, generator_rules: list):
    records = []
    unmatched = []
    for path in raw_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        rel = str(path.relative_to(raw_dir)).lower()
        label = classify_path(rel, label_rules, "label")
        generator = classify_path(rel, generator_rules, "generator") if label == "fake" else "real_source"

        if label is None:
            unmatched.append(str(path))
            continue

        records.append({"path": path, "label": label, "generator": generator or "other"})

    return records, unmatched


# ---------------------------------------------------------------------------
# Step 2 — verification report (actual measured counts, never hardcoded)
# ---------------------------------------------------------------------------
def build_verification_report(records: list, unmatched: list) -> dict:
    labels = Counter(r["label"] for r in records)
    generators = Counter(r["generator"] for r in records)
    return {
        "total_images_classified": len(records),
        "total_images_unclassified": len(unmatched),
        "real_images": labels.get("real", 0),
        "fake_images": labels.get("fake", 0),
        "class_distribution_pct": {
            k: round(100 * v / max(1, len(records)), 2) for k, v in labels.items()
        },
        "generator_distribution": dict(generators),
        "stylegan_fake_images": generators.get("stylegan", 0),
        "stable_diffusion_fake_images": generators.get("stable_diffusion", 0),
        "other_unassigned_fake_images": generators.get("other", 0),
        "unmatched_file_examples": unmatched[:20],
    }


# ---------------------------------------------------------------------------
# Step 3 — allocate real images: cross-gen holdout vs. StyleGAN experiment pool
# ---------------------------------------------------------------------------
def allocate_real_images(real_records: list, n_needed_for_crossgen: int, rng: random.Random):
    """Reserve `n_needed_for_crossgen` real images (seeded, disjoint) for the
    Stable Diffusion cross-generator test, so it has both classes to score
    ROC-AUC/precision/recall on. The rest go to the StyleGAN experiment pool.
    Reservation happens BEFORE the StyleGAN split -> zero leakage by construction.
    """
    shuffled = real_records.copy()
    rng.shuffle(shuffled)
    n_reserve = min(n_needed_for_crossgen, len(shuffled))
    crossgen_real = shuffled[:n_reserve]
    stylegan_pool_real = shuffled[n_reserve:]
    return stylegan_pool_real, crossgen_real


# ---------------------------------------------------------------------------
# Step 4 — stratified 70/15/15 split of the StyleGAN experiment pool
# ---------------------------------------------------------------------------
def stratified_split(records: list, split_cfg: dict, rng: random.Random):
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    splits = {"train": [], "val": [], "test": []}
    for label, items in by_label.items():
        items = items.copy()
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * split_cfg["train"])
        n_val = int(n * split_cfg["val"])
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train:n_train + n_val])
        splits["test"].extend(items[n_train + n_val:])
    return splits


# ---------------------------------------------------------------------------
# Step 5 — leakage checks (paths, filenames, content hashes, set overlap)
# ---------------------------------------------------------------------------
def file_hash(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def leakage_checks(splits: dict, crossgen_records: list) -> dict:
    report = {"issues": [], "checked": []}

    def paths_of(records):
        return [str(r["path"]) for r in records]

    def hashes_of(records):
        return {file_hash(r["path"]): str(r["path"]) for r in records}

    all_groups = {**{k: splits[k] for k in ("train", "val", "test")}, "cross_gen": crossgen_records}

    # 1/2/3 — unique paths, duplicate filenames within each split
    for name, recs in all_groups.items():
        paths = paths_of(recs)
        report["checked"].append(f"unique_paths::{name}")
        if len(paths) != len(set(paths)):
            report["issues"].append(f"Duplicate file paths found within '{name}'.")
        filenames = [Path(p).name for p in paths]
        dup_names = [n for n, c in Counter(filenames).items() if c > 1]
        if dup_names:
            report["issues"].append(
                f"'{name}' has {len(dup_names)} duplicate filenames (different folders, "
                f"same name) — inspect these: {dup_names[:10]}"
            )

    # 4/5/6 — no image occurs in more than one of train/val/primary-test (by content hash)
    group_names = ["train", "val", "test"]
    group_hashes = {n: hashes_of(all_groups[n]) for n in group_names}
    for i, a in enumerate(group_names):
        for b in group_names[i + 1:]:
            overlap = set(group_hashes[a]) & set(group_hashes[b])
            report["checked"].append(f"hash_overlap::{a}_vs_{b}")
            if overlap:
                report["issues"].append(
                    f"LEAKAGE: {len(overlap)} identical images appear in both '{a}' and '{b}'. "
                    f"Example: {group_hashes[a][next(iter(overlap))]}"
                )

    # 7/8/9 — Stable Diffusion images not present in train/val/primary-test
    crossgen_hashes = hashes_of(crossgen_records)
    for n in group_names:
        overlap = set(group_hashes[n]) & set(crossgen_hashes)
        report["checked"].append(f"hash_overlap::cross_gen_vs_{n}")
        if overlap:
            report["issues"].append(
                f"LEAKAGE: {len(overlap)} images in the cross-generator set also appear in '{n}'."
            )

    report["passed"] = len(report["issues"]) == 0
    return report


# ---------------------------------------------------------------------------
# Step 6 — materialize splits on disk (symlink or copy) in common.py's expected layout
# ---------------------------------------------------------------------------
def unique_dest_name(src: Path, taken: set) -> str:
    """
    Source subfolders can reuse filenames (e.g. ffhq/0000.jpg and celeba/0000.jpg,
    or two generator dumps both starting numbering at 0000). Prefixing with the
    immediate parent folder name keeps names traceable and avoids silently
    dropping files on a collision; a numeric suffix is the final fallback.
    """
    candidate = f"{src.parent.name}__{src.name}"
    if candidate not in taken:
        return candidate
    stem, suffix = Path(candidate).stem, Path(candidate).suffix
    i = 1
    while f"{stem}__{i}{suffix}" in taken:
        i += 1
    return f"{stem}__{i}{suffix}"


def materialize(records: list, dest_root: Path, link_mode: str):
    import shutil

    for label in ("real", "fake"):
        (dest_root / label).mkdir(parents=True, exist_ok=True)

    taken_by_label = {
        "real": {p.name for p in (dest_root / "real").iterdir()} if (dest_root / "real").exists() else set(),
        "fake": {p.name for p in (dest_root / "fake").iterdir()} if (dest_root / "fake").exists() else set(),
    }

    n_written = 0
    for r in records:
        src = r["path"]
        taken = taken_by_label[r["label"]]
        dest_name = unique_dest_name(src, taken)
        dest = dest_root / r["label"] / dest_name
        taken.add(dest_name)

        if dest.exists():
            n_written += 1
            continue  # already materialized from a previous run

        if link_mode == "symlink":
            try:
                dest.symlink_to(src.resolve())
                n_written += 1
                continue
            except (OSError, NotImplementedError):
                pass  # fall back to copy (e.g. filesystems without symlink support)
        shutil.copy2(src, dest)
        n_written += 1

    if n_written != len(records):
        print(f"[warn] materialize: expected {len(records)} files under {dest_root}, wrote {n_written}.")
    return n_written


# ---------------------------------------------------------------------------
# Step 7 — dataset summary table (measured counts only, never invented)
# ---------------------------------------------------------------------------
def build_summary_table(splits: dict, crossgen_records: list) -> pd.DataFrame:
    def counts(records):
        c = Counter(r["label"] for r in records)
        return c.get("real", 0), c.get("fake", 0)

    rows = []
    role_map = {"train": "Training", "val": "Validation", "test": "Final ID test"}
    for split_name in ("train", "val", "test"):
        real, fake = counts(splits[split_name])
        rows.append({
            "Dataset": "StyleGAN", "Generator": "StyleGAN",
            "Role": split_name.capitalize() if split_name != "test" else "Primary Test",
            "Real": real, "Fake": fake, "Usage": role_map[split_name],
        })

    real, fake = counts(crossgen_records)
    rows.append({
        "Dataset": "Stable Diffusion", "Generator": "Diffusion", "Role": "Cross-Generator",
        "Real": real, "Fake": fake, "Usage": "Final OOD test",
    })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = random.Random(cfg["seed"])

    raw_dir = Path(cfg["raw_dir"])
    if not raw_dir.exists():
        print(f"[error] raw_dir '{raw_dir}' does not exist. Extract the Kaggle download there first.")
        sys.exit(1)

    print(f"Scanning {raw_dir} ...")
    records, unmatched = scan_raw_dataset(raw_dir, cfg["label_rules"], cfg["generator_rules"])
    if unmatched:
        print(f"[warn] {len(unmatched)} files could not be classified as real/fake — "
              f"see the verification report's 'unmatched_file_examples' and adjust "
              f"label_rules/generator_rules in configs/dataset_prep.yaml.")

    report = build_verification_report(records, unmatched)
    print(json.dumps(report, indent=2))

    out_paths = cfg["output"]
    Path(out_paths["verification_report"]).parent.mkdir(parents=True, exist_ok=True)
    with open(out_paths["verification_report"], "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Split records by generator
    real_records = [r for r in records if r["label"] == "real"]
    stylegan_fake = [r for r in records if r["label"] == "fake" and r["generator"] == "stylegan"]
    sd_fake = [r for r in records if r["label"] == "fake" and r["generator"] == "stable_diffusion"]
    other_fake = [r for r in records if r["label"] == "fake" and r["generator"] == "other"]

    if other_fake:
        print(f"[note] {len(other_fake)} fake images matched neither 'stylegan' nor "
              f"'stable_diffusion' keywords — excluded from both experiment pools "
              f"(main experiment stays exactly StyleGAN -> Stable Diffusion). "
              f"See dataset_verification_report.json if you want to use them separately.")

    if not stylegan_fake or not sd_fake:
        print("[error] Need at least one StyleGAN-fake and one Stable-Diffusion-fake image "
              "to build the protocol's split. Check your generator_rules keywords against "
              "the actual downloaded folder names.")
        sys.exit(1)

    # Reserve real images for cross-gen BEFORE splitting the StyleGAN pool (no leakage by construction)
    stylegan_pool_real, crossgen_real = allocate_real_images(real_records, len(sd_fake), rng)

    stylegan_pool = stylegan_pool_real + stylegan_fake
    crossgen_records = crossgen_real + sd_fake

    print(f"\nStyleGAN experiment pool: {len(stylegan_pool_real)} real + {len(stylegan_fake)} fake "
          f"= {len(stylegan_pool)} images")
    print(f"Cross-generator (Stable Diffusion) pool: {len(crossgen_real)} real + {len(sd_fake)} fake "
          f"= {len(crossgen_records)} images")

    splits = stratified_split(stylegan_pool, cfg["split"], rng)
    for name in ("train", "val", "test"):
        c = Counter(r["label"] for r in splits[name])
        print(f"  {name}: real={c.get('real', 0)} fake={c.get('fake', 0)} total={len(splits[name])}")

    # Leakage checks — report only, never silently fix
    print("\nRunning leakage checks (this hashes every file; may take a while for large datasets)...")
    leakage_report = leakage_checks(splits, crossgen_records)
    Path(out_paths["leakage_report"]).parent.mkdir(parents=True, exist_ok=True)
    with open(out_paths["leakage_report"], "w") as f:
        json.dump(leakage_report, f, indent=2)
    if leakage_report["passed"]:
        print("Leakage checks passed: no overlap detected.")
    else:
        print("!!! LEAKAGE CHECK FOUND ISSUES !!!")
        for issue in leakage_report["issues"]:
            print(f"  - {issue}")
        print(f"Full report: {out_paths['leakage_report']}")

    # Materialize on disk in the layout common.py's DeepfakeImageDataset expects
    processed_dir = Path(cfg["processed_dir"])
    cross_gen_dir = Path(cfg["cross_gen_dir"])
    link_mode = cfg.get("link_mode", "symlink")

    print(f"\nMaterializing splits under {processed_dir} and {cross_gen_dir} (mode={link_mode}) ...")
    materialize(splits["train"], processed_dir / "train", link_mode)
    materialize(splits["val"], processed_dir / "val", link_mode)
    materialize(splits["test"], processed_dir / "test", link_mode)
    materialize(crossgen_records, cross_gen_dir, link_mode)

    # Dataset summary table -> shared results/comparison/ (Member C owns this folder)
    summary_df = build_summary_table(splits, crossgen_records)
    Path(out_paths["summary_table"]).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_paths["summary_table"], index=False)
    print(f"\nDataset summary table:\n{summary_df.to_string(index=False)}")
    print(f"\nSaved -> {out_paths['summary_table']}")
    print(f"Verification report -> {out_paths['verification_report']}")
    print(f"Leakage report -> {out_paths['leakage_report']}")


if __name__ == "__main__":
    main()

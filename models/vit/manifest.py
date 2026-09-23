"""Validate Member A's Wish assignments without generating replacement splits."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

DATASET = "wish096/realvsfake-81k-by-wish"
SPLITS = ("train", "val", "test", "cross_gen")
SOURCES = {"RFF": (0, "FFHQ"), "RCA": (0, "CelebA"),
           "FSG": (1, "StyleGAN"), "FSD": (1, "StableDiffusion"),
           "AI": (1, "AiGenImage")}
FIELDS = ("filepath", "label", "source", "split", "identity", "sha256", "pixel_sha256", "dhash")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_from_name(path):
    match = re.fullmatch(r"(RFF|RCA|FSG|FSD|AI)\s*\(\d+\)\.(?:jpg|jpeg|png|webp)",
                         Path(path).name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unrecognized Wish filename: {path}; retain original filenames.")
    return SOURCES[match.group(1).upper()]


def image_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"filepath must be relative to data root: {relative}")
    resolved = (Path(root).resolve() / path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Image escapes data root: {relative}")
    return resolved


def read_manifest(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"filepath", "label", "source", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty.")
    return rows


def combine_split_csvs(split_files, output, strip_prefix=""):
    """Convert A's existing split files to the shared schema; never reassign a row."""
    if not split_files or not set(split_files).issubset(set(SPLITS) | {"excluded"}):
        raise ValueError("CSV keys must be train, val, test, cross_gen (and optional excluded).")
    rows = []
    prefix = strip_prefix.replace("\\", "/")
    for split, path in split_files.items():
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {"filepath", "label"}.issubset(reader.fieldnames or []):
                raise ValueError(f"{path} requires filepath and label columns.")
            for row in reader:
                if row.get("split") and row["split"] != split:
                    raise ValueError(f"CSV filename assignment conflicts with row split: {path}")
                relative = row["filepath"].replace("\\", "/")
                if prefix:
                    if not relative.startswith(prefix):
                        raise ValueError(f"Explicit strip prefix does not match: {relative}")
                    relative = relative[len(prefix):]
                # This changes only path notation; source/label are cross-checked.
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise ValueError("CSV paths must be relative; set the exact MEMBER_A_PATH_PREFIX if needed.")
                _, known_source = source_from_name(relative)
                rows.append({"filepath": relative, "label": row["label"],
                             "source": row.get("source") or known_source,
                             "split": split, "identity": row.get("identity", "")})
    # Missing cross-generator controls or SD in train/val must be corrected by A.
    rows = validate_rows(rows)
    output = Path(output)
    comparable = lambda entries: [{k: str(row.get(k, "")) for k in FIELDS[:5]} for row in entries]
    if output.exists():
        if comparable(read_manifest(output)) != comparable(rows):
            raise ValueError("Combined manifest already exists with different assignments; choose a new output directory.")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS[:5])
        writer.writeheader()
        writer.writerows(rows)
    return output


def validate_rows(rows, require_all=True):
    seen, identities, counts = set(), {}, Counter()
    result = []
    for original in rows:
        row = dict(original)
        row["filepath"] = row["filepath"].replace("\\", "/")
        if row["filepath"] in seen:
            raise ValueError(f"Repeated filepath / split overlap: {row['filepath']}")
        seen.add(row["filepath"])
        label, source = source_from_name(row["filepath"])
        if str(row["label"]) != str(label) or row["source"] != source:
            raise ValueError(f"Filename/label/source conflict: {row['filepath']}")
        split = row["split"]
        if source == "AiGenImage":
            if split != "excluded":
                raise ValueError("AiGenImage must be explicitly assigned to excluded by Member A.")
        elif split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        elif source == "StableDiffusion" and split != "cross_gen":
            raise ValueError("StableDiffusion leakage: only cross_gen is permitted.")
        elif source == "StyleGAN" and split == "cross_gen":
            raise ValueError("StyleGAN cannot appear in the held-out cross_gen set.")
        identity = row.get("identity", "").strip()
        if identity and split != "excluded":
            key = (source, identity)
            if key in identities and identities[key] != split:
                raise ValueError(f"Identity overlaps splits: {key}")
            identities[key] = split
        row["label"] = label
        row["identity"] = identity
        counts[(split, label)] += 1
        result.append(row)
    if require_all:
        for split in SPLITS:
            for label in (0, 1):
                if not counts[(split, label)]:
                    raise ValueError(f"{split} requires both classes; missing label {label}.")
    return result


def audit_manifest(manifest, root, output, dataset_version, near_distance=4):
    if not str(dataset_version).strip():
        raise ValueError("Record the downloaded Kaggle dataset version.")
    if not 0 <= near_distance <= 8:
        raise ValueError("near_distance must be between 0 and 8.")
    rows = validate_rows(read_manifest(manifest))
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {output}")
    file_seen, pixel_seen, buckets, near = {}, {}, {}, []
    counts, dimensions = Counter(), Counter()
    # A d-bit neighbor must share one of d+1 hash pieces. This avoids a
    # quadratic scan while retaining every dHash candidate within distance d.
    pieces = near_distance + 1
    for row in rows:
        path = image_path(root, row["filepath"])
        digest = sha256_file(path)
        with Image.open(path) as image:
            image.load()
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            dimensions[f"{rgb.width}x{rgb.height}"] += 1
            pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            small = list(rgb.convert("L").resize((9, 8)).tobytes())
            value = sum(int(small[y * 9 + x] > small[y * 9 + x + 1]) << (y * 8 + x)
                        for y in range(8) for x in range(8))
        row.update(sha256=digest, pixel_sha256=pixel_hash, dhash=f"{value:016x}")
        counts[f"{row['split']}:{row['source']}:{row['label']}"] += 1
        if row["split"] == "excluded":
            continue
        for index, fingerprint in ((file_seen, digest), (pixel_seen, pixel_hash)):
            if fingerprint in index:
                raise ValueError(f"Duplicate image content: {row['filepath']} and {index[fingerprint]}")
            index[fingerprint] = row["filepath"]
        candidates = {}
        keys = []
        for piece in range(pieces):
            start, end = 64 * piece // pieces, 64 * (piece + 1) // pieces
            key = (piece, (value >> start) & ((1 << (end - start)) - 1))
            keys.append(key)
            for previous in buckets.get(key, []):
                candidates[previous[0]] = previous
        for previous_path, previous_split, previous_hash in candidates.values():
            distance = (value ^ previous_hash).bit_count()
            if previous_split != row["split"] and distance <= near_distance:
                near.append({"filepath_a": previous_path, "filepath_b": row["filepath"],
                             "distance": distance})
        for key in keys:
            buckets.setdefault(key, []).append((row["filepath"], row["split"], value))
    output.mkdir(parents=True)
    with open(output / "manifest.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in FIELDS} for row in rows)
    with open(output / "near_duplicates.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("filepath_a", "filepath_b", "distance"))
        writer.writeheader()
        writer.writerows(near)
    report = {"dataset": DATASET, "dataset_version": str(dataset_version),
              "manifest_sha256": sha256_file(output / "manifest.csv"),
              "input_manifest_sha256": sha256_file(manifest), "counts": dict(counts),
              "dimensions": dict(dimensions), "near_duplicate_pairs": len(near),
              "near_distance": near_distance,
              "identity_metadata_available": any(row["identity"] for row in rows),
              "status": "needs_near_duplicate_review" if near else "passed"}
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_audited_manifest(manifest, root, review=None, verify_splits=SPLITS):
    manifest = Path(manifest)
    rows = validate_rows(read_manifest(manifest))
    audit = json.loads(manifest.with_name("audit.json").read_text())
    if audit["dataset"] != DATASET or audit["manifest_sha256"] != sha256_file(manifest):
        raise ValueError("Audit does not match this Wish manifest. Re-audit through Member A.")
    if audit["near_duplicate_pairs"]:
        if not review:
            raise ValueError("Review near_duplicates.csv and provide a signed review JSON before training.")
        decision = json.loads(Path(review).read_text())
        if (decision.get("manifest_sha256") != audit["manifest_sha256"]
                or decision.get("decision") != "false_positives_only"
                or not decision.get("reviewer") or not decision.get("rationale")):
            raise ValueError("Review must identify reviewer, rationale and audited manifest.")
    for row in rows:
        if row["split"] in verify_splits:
            if row.get("sha256") != sha256_file(image_path(root, row["filepath"])):
                raise ValueError(f"Image changed since audit: {row['filepath']}")
    return rows, audit

"""
Prepares the RealVsFake_162k_by_Wish dataset (CSV-labeled) into the
train/val/test/cross_gen_test structure the pipeline expects.

Assumes you copied the downloaded 'archive' folder into data/raw/, so:
    data/raw/archive/train_labels.csv
    data/raw/archive/val_labels.csv
    data/raw/archive/test_labels.csv
    data/raw/archive/RealVsFake/RealVsFake/Real/*.jpg
    data/raw/archive/RealVsFake/RealVsFake/Fake/*.jpg

The CSV 'filepath' column looks like:
    /kaggle/input/realvsfake-81k-by-wish/RealVsFake/RealVsFake/Real/RCA (1).jpg
We only need the part from 'RealVsFake/RealVsFake/' onward, joined with
your local archive folder.

Cross-generator logic:
    All rows whose `source` column matches --cross_gen_source (default
    'StableDiffusion') are pulled OUT of the primary train/val/test sets
    entirely and placed in data/cross_gen_test/fake/, so the model never
    sees a diffusion-generated image during training or tuning.
    An equal-sized sample of real images originally in the test split is
    moved to data/cross_gen_test/real/ (and removed from processed/test/)
    to keep that folder balanced.

Usage:
    python scripts/prepare_from_csv.py
"""
import argparse
import random
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def resolve_local_path(csv_filepath: str, local_archive_root: Path) -> Path:
    marker = "RealVsFake/RealVsFake/"
    idx = csv_filepath.replace("\\", "/").find(marker)
    if idx == -1:
        raise ValueError(f"Could not find '{marker}' in path: {csv_filepath}")
    relative = csv_filepath.replace("\\", "/")[idx + len(marker):]
    return local_archive_root / "RealVsFake" / "RealVsFake" / relative


def copy_row(row, local_archive_root: Path, dest_folder: Path):
    src = resolve_local_path(row["filepath"], local_archive_root)
    if not src.exists():
        return False
    dest_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_folder / src.name)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive_dir", default="data/raw/archive",
                         help="Path to the extracted Kaggle 'archive' folder")
    parser.add_argument("--out_processed", default="data/processed")
    parser.add_argument("--out_cross_gen", default="data/cross_gen_test")
    parser.add_argument("--cross_gen_source", default="StableDiffusion",
                         help="Value in the 'source' column to hold out entirely "
                              "for cross-generator testing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    out_processed = Path(args.out_processed)
    out_cross_gen = Path(args.out_cross_gen)

    splits = {
        "train": pd.read_csv(archive_dir / "train_labels.csv"),
        "val": pd.read_csv(archive_dir / "val_labels.csv"),
        "test": pd.read_csv(archive_dir / "test_labels.csv"),
    }
    for split_name, df in splits.items():
        df["split"] = split_name

    full_df = pd.concat(splits.values(), ignore_index=True)
    print("Row counts by label and source:")
    print(full_df.groupby(["label", "source"]).size())

    is_cross_gen = full_df["source"] == args.cross_gen_source
    cross_gen_fake_df = full_df[is_cross_gen]
    primary_df = full_df[~is_cross_gen]

    print(f"\n{len(cross_gen_fake_df)} images from source "
          f"'{args.cross_gen_source}' will be held out for cross-generator testing.")

    # Pull a matching number of REAL images out of the primary test split to
    # pair with the held-out fakes, so cross_gen_test is balanced.
    n_needed = len(cross_gen_fake_df)
    real_test_mask = (primary_df["label"] == 0) & (primary_df["split"] == "test")
    real_test_indices = primary_df[real_test_mask].index.tolist()
    random.Random(args.seed).shuffle(real_test_indices)
    cross_gen_real_indices = real_test_indices[:n_needed]

    cross_gen_real_df = primary_df.loc[cross_gen_real_indices]
    primary_df = primary_df.drop(index=cross_gen_real_indices)

    # ---- Copy primary train/val/test ----
    label_to_name = {0: "real", 1: "fake"}
    copied, missing = 0, 0
    for _, row in tqdm(primary_df.iterrows(), total=len(primary_df), desc="Copying primary set"):
        dest = out_processed / row["split"] / label_to_name[row["label"]]
        ok = copy_row(row, archive_dir, dest)
        copied += int(ok)
        missing += int(not ok)

    # ---- Copy cross-generator held-out set ----
    for _, row in tqdm(cross_gen_fake_df.iterrows(), total=len(cross_gen_fake_df),
                        desc="Copying cross-gen fake"):
        dest = out_cross_gen / "fake"
        ok = copy_row(row, archive_dir, dest)
        copied += int(ok)
        missing += int(not ok)

    for _, row in tqdm(cross_gen_real_df.iterrows(), total=len(cross_gen_real_df),
                        desc="Copying cross-gen real"):
        dest = out_cross_gen / "real"
        ok = copy_row(row, archive_dir, dest)
        copied += int(ok)
        missing += int(not ok)

    print(f"\nDone. Copied {copied} images. {missing} files were listed in the CSV "
          f"but not found on disk (safe to ignore if a small number).")
    print(f"Primary dataset: {out_processed}")
    print(f"Cross-generator held-out dataset: {out_cross_gen}")


if __name__ == "__main__":
    main()

"""
Prepare the RealVsFake_162k_by_Wish dataset for the final Custom CNN
experiment.

This script reproduces the dataset split used in the Kaggle experiment.

Dataset sources:
    Real:
        - CelebA
        - FFHQ

    Fake:
        - StyleGAn              -> primary train/val/test
        - StableDiffusion       -> unseen cross-generator test
        - AiGenImage            -> excluded from the primary experiment

Final dataset structure:

    data/processed/
        train/
            real/
            fake/
        val/
            real/
            fake/
        test/
            real/
            fake/

    data/cross_gen_test/
        real/
        fake/

Experimental split:

    TRAIN:
        CelebA real
        FFHQ real
        StyleGAn fake

    VALIDATION:
        CelebA real
        FFHQ real
        StyleGAn fake

    PRIMARY TEST:
        CelebA real
        FFHQ real
        StyleGAn fake

    CROSS-GENERATOR TEST:
        CelebA real from original test split
        FFHQ real from original test split
        StableDiffusion fake from original test split

    AiGenImage:
        Excluded from the primary experiment.

This prevents StableDiffusion images from being seen during training
or validation and evaluates generalization to an unseen generator family.
"""

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------
# PATH RESOLUTION
# ---------------------------------------------------------------------

def resolve_local_path(csv_filepath: str, local_archive_root: Path) -> Path:
    """
    Convert the original Kaggle filepath into the local extracted path.

    Example CSV path:
        /kaggle/input/realvsfake-81k-by-wish/
        RealVsFake/RealVsFake/Real/example.jpg

    Local path:
        data/raw/archive/RealVsFake/RealVsFake/Real/example.jpg
    """

    marker = "RealVsFake/RealVsFake/"

    normalized = str(csv_filepath).replace("\\", "/")

    idx = normalized.find(marker)

    if idx == -1:
        raise ValueError(
            f"Could not find '{marker}' in filepath:\n{csv_filepath}"
        )

    relative = normalized[idx + len(marker):]

    return local_archive_root / "RealVsFake" / "RealVsFake" / relative


# ---------------------------------------------------------------------
# SAFE FILE COPYING
# ---------------------------------------------------------------------

def make_destination_name(csv_filepath: str) -> str:
    """
    Create a collision-safe filename.

    The original dataset contains files from different folders.
    Using only src.name could overwrite files with the same filename.

    We therefore append a short hash based on the original filepath.
    """

    original_name = Path(str(csv_filepath)).name
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix

    digest = hashlib.md5(
        str(csv_filepath).encode("utf-8")
    ).hexdigest()[:10]

    return f"{stem}__{digest}{suffix}"


def copy_row(
    row,
    local_archive_root: Path,
    destination_folder: Path,
) -> bool:

    src = resolve_local_path(
        row["filepath"],
        local_archive_root,
    )

    if not src.exists():
        return False

    destination_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_name = make_destination_name(
        row["filepath"]
    )

    destination = destination_folder / destination_name

    shutil.copy2(src, destination)

    return True


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Prepare RealVsFake dataset for Custom CNN experiment."
    )

    parser.add_argument(
        "--archive_dir",
        default="data/raw/archive",
        help="Extracted Kaggle archive directory.",
    )

    parser.add_argument(
        "--out_processed",
        default="data/processed",
        help="Primary train/val/test output directory.",
    )

    parser.add_argument(
        "--out_cross_gen",
        default="data/cross_gen_test",
        help="Cross-generator test output directory.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed kept for reproducibility.",
    )

    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    out_processed = Path(args.out_processed)
    out_cross_gen = Path(args.out_cross_gen)

    # -----------------------------------------------------------------
    # VERIFY INPUT FILES
    # -----------------------------------------------------------------

    required_files = [
        archive_dir / "train_labels.csv",
        archive_dir / "val_labels.csv",
        archive_dir / "test_labels.csv",
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    print("=" * 70)
    print("REALVSFAKE DATASET PREPARATION")
    print("=" * 70)

    print(f"Archive directory : {archive_dir}")
    print(f"Processed output  : {out_processed}")
    print(f"Cross-gen output  : {out_cross_gen}")
    print()

    # -----------------------------------------------------------------
    # LOAD CSV FILES
    # -----------------------------------------------------------------

    splits = {
        "train": pd.read_csv(
            archive_dir / "train_labels.csv"
        ),
        "val": pd.read_csv(
            archive_dir / "val_labels.csv"
        ),
        "test": pd.read_csv(
            archive_dir / "test_labels.csv"
        ),
    }

    for split_name, dataframe in splits.items():

        dataframe["split"] = split_name

    full_df = pd.concat(
        splits.values(),
        ignore_index=True,
    )

    # -----------------------------------------------------------------
    # DATASET SUMMARY
    # -----------------------------------------------------------------

    print("Original dataset distribution:")
    print()

    print(
        full_df
        .groupby(["split", "source", "label"])
        .size()
        .to_string()
    )

    print()
    print("-" * 70)

    # -----------------------------------------------------------------
    # DEFINE FINAL EXPERIMENTAL DATASETS
    # -----------------------------------------------------------------

    # Primary experiment sources.
    primary_sources = {
        "CelebA",
        "FFHQ",
        "StyleGAn",
    }

    # -----------------------------------------------------------------
    # PRIMARY TRAIN
    # -----------------------------------------------------------------

    train_df = full_df[
        (full_df["split"] == "train")
        & (full_df["source"].isin(primary_sources))
    ].copy()

    # -----------------------------------------------------------------
    # PRIMARY VALIDATION
    # -----------------------------------------------------------------

    val_df = full_df[
        (full_df["split"] == "val")
        & (full_df["source"].isin(primary_sources))
    ].copy()

    # -----------------------------------------------------------------
    # PRIMARY TEST
    # -----------------------------------------------------------------

    primary_test_df = full_df[
        (full_df["split"] == "test")
        & (full_df["source"].isin(primary_sources))
    ].copy()

    # -----------------------------------------------------------------
    # CROSS-GENERATOR TEST
    #
    # IMPORTANT:
    # Only StableDiffusion images from the ORIGINAL TEST SPLIT
    # are used.
    # -----------------------------------------------------------------

    cross_gen_fake_df = full_df[
        (full_df["split"] == "test")
        & (full_df["source"] == "StableDiffusion")
        & (full_df["label"] == 1)
    ].copy()

    # All REAL images from the original test split.
    #
    # This gives:
    #   CelebA = 6075
    #   FFHQ   = 6075
    #   Total  = 12150
    #
    # This exactly matches the Kaggle cross-generator evaluation.
    cross_gen_real_df = full_df[
        (full_df["split"] == "test")
        & (full_df["label"] == 0)
        & (full_df["source"].isin({"CelebA", "FFHQ"}))
    ].copy()

    # -----------------------------------------------------------------
    # EXCLUDED DATA
    # -----------------------------------------------------------------

    excluded_df = full_df[
        ~full_df.index.isin(
            set(train_df.index)
            | set(val_df.index)
            | set(primary_test_df.index)
            | set(cross_gen_fake_df.index)
            | set(cross_gen_real_df.index)
        )
    ].copy()

    # -----------------------------------------------------------------
    # PRINT FINAL SPLIT COUNTS
    # -----------------------------------------------------------------

    print("FINAL EXPERIMENTAL SPLIT")
    print()

    print(
        f"Train:              {len(train_df):,}"
    )

    print(
        f"Validation:         {len(val_df):,}"
    )

    print(
        f"Primary test:       {len(primary_test_df):,}"
    )

    print(
        f"Cross-gen fake:     {len(cross_gen_fake_df):,}"
    )

    print(
        f"Cross-gen real:     {len(cross_gen_real_df):,}"
    )

    print(
        f"Excluded:           {len(excluded_df):,}"
    )

    print()

    # -----------------------------------------------------------------
    # SAFETY ASSERTIONS
    # -----------------------------------------------------------------

    assert len(train_df) == 105700, (
        f"Unexpected train size: {len(train_df)}"
    )

    assert len(val_df) == 22650, (
        f"Unexpected validation size: {len(val_df)}"
    )

    assert len(primary_test_df) == 22650, (
        f"Unexpected primary test size: {len(primary_test_df)}"
    )

    assert len(cross_gen_fake_df) == 1495, (
        f"Unexpected cross-gen fake size: {len(cross_gen_fake_df)}"
    )

    assert len(cross_gen_real_df) == 12150, (
        f"Unexpected cross-gen real size: {len(cross_gen_real_df)}"
    )

    # StableDiffusion must NOT appear in training or validation.
    assert not (
        train_df["source"]
        .eq("StableDiffusion")
        .any()
    )

    assert not (
        val_df["source"]
        .eq("StableDiffusion")
        .any()
    )

    # StableDiffusion in cross-gen must come only from test.
    assert set(cross_gen_fake_df["split"]) == {"test"}

    # AiGenImage must not enter the primary experiment.
    assert not (
        train_df["source"]
        .eq("AiGenImage")
        .any()
    )

    assert not (
        val_df["source"]
        .eq("AiGenImage")
        .any()
    )

    assert not (
        primary_test_df["source"]
        .eq("AiGenImage")
        .any()
    )

    print("Safety checks: PASSED")
    print()

    # -----------------------------------------------------------------
    # CLEAN OLD OUTPUTS
    # -----------------------------------------------------------------

    print("Cleaning previous prepared dataset...")

    if out_processed.exists():
        shutil.rmtree(out_processed)

    if out_cross_gen.exists():
        shutil.rmtree(out_cross_gen)

    # Re-create root folders.
    out_processed.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_cross_gen.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # COPY PRIMARY DATASET
    # -----------------------------------------------------------------

    label_to_name = {
        0: "real",
        1: "fake",
    }

    copied = 0
    missing = 0

    print("Copying training images...")

    for _, row in tqdm(
        train_df.iterrows(),
        total=len(train_df),
        desc="Train",
    ):

        destination = (
            out_processed
            / "train"
            / label_to_name[int(row["label"])]
        )

        if copy_row(
            row,
            archive_dir,
            destination,
        ):
            copied += 1
        else:
            missing += 1

    print("Copying validation images...")

    for _, row in tqdm(
        val_df.iterrows(),
        total=len(val_df),
        desc="Validation",
    ):

        destination = (
            out_processed
            / "val"
            / label_to_name[int(row["label"])]
        )

        if copy_row(
            row,
            archive_dir,
            destination,
        ):
            copied += 1
        else:
            missing += 1

    print("Copying primary test images...")

    for _, row in tqdm(
        primary_test_df.iterrows(),
        total=len(primary_test_df),
        desc="Primary Test",
    ):

        destination = (
            out_processed
            / "test"
            / label_to_name[int(row["label"])]
        )

        if copy_row(
            row,
            archive_dir,
            destination,
        ):
            copied += 1
        else:
            missing += 1

    # -----------------------------------------------------------------
    # COPY CROSS-GENERATOR TEST
    # -----------------------------------------------------------------

    print("Copying cross-generator fake images...")

    for _, row in tqdm(
        cross_gen_fake_df.iterrows(),
        total=len(cross_gen_fake_df),
        desc="Cross-gen Fake",
    ):

        destination = (
            out_cross_gen
            / "fake"
        )

        if copy_row(
            row,
            archive_dir,
            destination,
        ):
            copied += 1
        else:
            missing += 1

    print("Copying cross-generator real images...")

    for _, row in tqdm(
        cross_gen_real_df.iterrows(),
        total=len(cross_gen_real_df),
        desc="Cross-gen Real",
    ):

        destination = (
            out_cross_gen
            / "real"
        )

        if copy_row(
            row,
            archive_dir,
            destination,
        ):
            copied += 1
        else:
            missing += 1

    # -----------------------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)

    print(f"Copied files : {copied:,}")
    print(f"Missing files: {missing:,}")

    print()
    print("Primary dataset:")
    print(out_processed)

    print()
    print("Cross-generator dataset:")
    print(out_cross_gen)

    print()
    print("Final structure:")
    print()
    print("data/processed/")
    print("  train/")
    print("    real/")
    print("    fake/")
    print("  val/")
    print("    real/")
    print("    fake/")
    print("  test/")
    print("    real/")
    print("    fake/")
    print()
    print("data/cross_gen_test/")
    print("  real/")
    print("  fake/")
    print()

    if missing > 0:
        print(
            "WARNING: Some CSV-listed files were not found on disk."
        )
    else:
        print(
            "All CSV-listed experiment files were copied successfully."
        )


if __name__ == "__main__":
    main()
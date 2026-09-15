"""
Splits a raw downloaded dataset (one folder of real images, one folder of
fake images) into the train/val/test structure the pipeline expects.

Usage example, for the PRIMARY dataset (e.g. the StyleGAN subset):
    python scripts/prepare_dataset.py \
        --real_dir data/raw/stylegan/real \
        --fake_dir data/raw/stylegan/fake \
        --out_dir data/processed \
        --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15

Usage example, for the CROSS-GENERATOR held-out set (e.g. Stable Diffusion):
this one is NOT split - all of it goes straight into cross_gen_test/,
since it must never be touched during training or tuning.
    python scripts/prepare_dataset.py \
        --real_dir data/raw/stable_diffusion/real \
        --fake_dir data/raw/stable_diffusion/fake \
        --out_dir data/cross_gen_test \
        --no_split
"""
import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path):
    return [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]


def copy_files(files, dest_folder: Path):
    dest_folder.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dest_folder / f.name)


def split_and_copy(files, out_dir: Path, class_name: str,
                    train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    random.Random(seed).shuffle(files)
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    copy_files(train_files, out_dir / "train" / class_name)
    copy_files(val_files, out_dir / "val" / class_name)
    copy_files(test_files, out_dir / "test" / class_name)

    print(f"{class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")


def copy_no_split(files, out_dir: Path, class_name: str):
    copy_files(files, out_dir / class_name)
    print(f"{class_name}: {len(files)} images copied (no split)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", required=True, help="Folder containing only real images")
    parser.add_argument("--fake_dir", required=True, help="Folder containing only fake images")
    parser.add_argument("--out_dir", required=True, help="Where to write the organized dataset")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_split", action="store_true",
                         help="Use for the cross-generator held-out set: copies everything "
                              "into <out_dir>/real and <out_dir>/fake with no train/val/test split.")
    args = parser.parse_args()

    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir)
    out_dir = Path(args.out_dir)

    real_files = list_images(real_dir)
    fake_files = list_images(fake_dir)
    print(f"Found {len(real_files)} real images and {len(fake_files)} fake images.")

    if args.no_split:
        copy_no_split(real_files, out_dir, "real")
        copy_no_split(fake_files, out_dir, "fake")
    else:
        ratios_sum = args.train_ratio + args.val_ratio + args.test_ratio
        assert abs(ratios_sum - 1.0) < 1e-6, "train/val/test ratios must sum to 1.0"
        split_and_copy(real_files, out_dir, "real", args.train_ratio, args.val_ratio,
                        args.test_ratio, args.seed)
        split_and_copy(fake_files, out_dir, "fake", args.train_ratio, args.val_ratio,
                        args.test_ratio, args.seed)

    print(f"Done. Dataset organized under: {out_dir}")


if __name__ == "__main__":
    main()

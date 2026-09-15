"""
Shared data loading pipeline.

Expected folder layout (created by you after downloading the dataset(s)):

    data/processed/train/real/*.jpg
    data/processed/train/fake/*.jpg
    data/processed/val/real/*.jpg
    data/processed/val/fake/*.jpg
    data/processed/test/real/*.jpg
    data/processed/test/fake/*.jpg
    data/cross_gen_test/real/*.jpg     <- different generator family, held out entirely
    data/cross_gen_test/fake/*.jpg

Label convention (fixed for the whole team, do not change):
    real -> 0
    fake -> 1

Rule for the team: only ADD to this file, don't change existing
function signatures without telling everyone.
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, train: bool = False):
    """Training transforms include augmentation (flip, rotation, jitter,
    blur/compression-like noise) to simulate social-media re-uploads.
    Val/test/cross-gen transforms are deterministic (no augmentation)."""
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_dataloaders(data_cfg: dict):
    """Builds train/val/test/cross_gen dataloaders from a config's `data` section.
    Uses torchvision.datasets.ImageFolder, which assigns labels alphabetically
    from subfolder names -> 'fake'=0, 'real'=1 by default. We explicitly
    remap to our fixed convention (real=0, fake=1) below so every model
    and every metric uses the same label meaning."""
    image_size = data_cfg["image_size"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 4)

    train_tf = build_transforms(image_size, train=True)
    eval_tf = build_transforms(image_size, train=False)

    train_ds = datasets.ImageFolder(data_cfg["train_dir"], transform=train_tf)
    val_ds = datasets.ImageFolder(data_cfg["val_dir"], transform=eval_tf)
    test_ds = datasets.ImageFolder(data_cfg["test_dir"], transform=eval_tf)
    cross_gen_ds = datasets.ImageFolder(data_cfg["cross_gen_dir"], transform=eval_tf)

    _remap_labels(train_ds)
    _remap_labels(val_ds)
    _remap_labels(test_ds)
    _remap_labels(cross_gen_ds)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        "cross_gen": DataLoader(cross_gen_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True),
    }
    return loaders


def _remap_labels(dataset: datasets.ImageFolder):
    """Force the label convention real=0, fake=1 regardless of alphabetical
    folder order, and keep dataset.classes consistent with it."""
    class_to_idx = {"real": 0, "fake": 1}
    old_to_new = {}
    for class_name, old_idx in dataset.class_to_idx.items():
        if class_name not in class_to_idx:
            raise ValueError(
                f"Expected subfolders named 'real' and 'fake', found '{class_name}'. "
                f"Rename your dataset folders to match this convention."
            )
        old_to_new[old_idx] = class_to_idx[class_name]

    dataset.samples = [(path, old_to_new[old_idx]) for path, old_idx in dataset.samples]
    dataset.targets = [old_to_new[old_idx] for old_idx in dataset.targets]
    dataset.class_to_idx = class_to_idx
    dataset.classes = ["real", "fake"]

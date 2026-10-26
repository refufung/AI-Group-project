"""Dataset and transforms: shared resize/normalize; train-only mild augmentation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


class DogBreedDataset(Dataset):
    def __init__(self, csv_path: Path | str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        with Image.open(row["path"]) as im:
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label"])
        return image, label


def compute_train_mean_std(train_csv: Path, image_size: int = 224, max_samples: int = 512) -> tuple[list[float], list[float]]:
    """Estimate per-channel mean/std on a subsample of the training set (after resize, before normalize)."""
    to_tensor = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    ds = DogBreedDataset(train_csv, transform=to_tensor)
    n = min(len(ds), max_samples)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)
    count = 0
    for images, _ in loader:
        b = images.size(0)
        images = images.view(b, 3, -1)
        sum_ += images.sum(dim=(0, 2))
        sum_sq += (images ** 2).sum(dim=(0, 2))
        count += b * images.size(2)
        if count // (image_size * image_size) >= n:
            break
    mean = (sum_ / count).tolist()
    var = sum_sq / count - torch.tensor(mean) ** 2
    std = torch.sqrt(var.clamp(min=1e-8)).tolist()
    return mean, std


def build_transforms(
    mean: list[float],
    std: list[float],
    train: bool,
) -> transforms.Compose:
    normalize = transforms.Normalize(mean=mean, std=std)
    if train:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.05,
                    hue=0.0,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def load_or_compute_norm_stats(train_csv: Path, meta_path: Path) -> tuple[list[float], list[float]]:
    meta: dict = {}
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
    if "mean" in meta and "std" in meta:
        return meta["mean"], meta["std"]
    mean, std = compute_train_mean_std(train_csv)
    meta["mean"] = mean
    meta["std"] = std
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Computed train mean={mean}, std={std}; saved to {meta_path}")
    return mean, std


def make_loaders(
    batch_size: int = 32,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, list[float], list[float]]:
    train_csv = OUTPUT_DIR / "train.csv"
    val_csv = OUTPUT_DIR / "val.csv"
    test_csv = OUTPUT_DIR / "test.csv"
    meta_path = OUTPUT_DIR / "split_meta.json"
    mean, std = load_or_compute_norm_stats(train_csv, meta_path)

    train_ds = DogBreedDataset(train_csv, transform=build_transforms(mean, std, train=True))
    val_ds = DogBreedDataset(val_csv, transform=build_transforms(mean, std, train=False))
    test_ds = DogBreedDataset(test_csv, transform=build_transforms(mean, std, train=False))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    return train_loader, val_loader, test_loader, mean, std

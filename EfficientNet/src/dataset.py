"""Dataset and ImageNet-style transforms for EfficientNet-B0."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import autoaugment

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
        return image, int(row["label"])


def build_transforms(
    train: bool,
    use_randaugment: bool = True,
    randaugment_magnitude: int = 5,
) -> transforms.Compose:
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if train:
        ops: list = [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        if use_randaugment:
            ops.append(
                autoaugment.RandAugment(num_ops=2, magnitude=int(randaugment_magnitude))
            )
        else:
            ops.append(
                transforms.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.05, hue=0.0
                )
            )
        ops.extend([transforms.ToTensor(), normalize])
        return transforms.Compose(ops)
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def make_loaders(
    batch_size: int = 32,
    num_workers: int = 0,
    use_randaugment: bool = True,
    randaugment_magnitude: int = 5,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_csv = OUTPUT_DIR / "train.csv"
    val_csv = OUTPUT_DIR / "val.csv"
    test_csv = OUTPUT_DIR / "test.csv"

    meta_path = OUTPUT_DIR / "split_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}
    meta["mean"] = IMAGENET_MEAN
    meta["std"] = IMAGENET_STD
    meta["normalize"] = "ImageNet"
    meta["train_augment"] = (
        f"RandAugment(m={randaugment_magnitude})" if use_randaugment else "ColorJitter"
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    train_ds = DogBreedDataset(
        train_csv,
        transform=build_transforms(
            train=True,
            use_randaugment=use_randaugment,
            randaugment_magnitude=randaugment_magnitude,
        ),
    )
    val_ds = DogBreedDataset(val_csv, transform=build_transforms(train=False))
    test_ds = DogBreedDataset(test_csv, transform=build_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

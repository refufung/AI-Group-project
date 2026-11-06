"""Dataset and transforms for the from-scratch track.

Train (strong aug): RandomResizedCrop(0.6-1.0) -> HFlip -> RandAugment(2,9)
  -> ToTensor -> Normalize -> RandomErasing(p=0.25, random fill)
Eval: Resize(224) -> ToTensor -> Normalize
Eval-TTA view: Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
IMAGE_SIZE = 224


class DogBreedDataset(Dataset):
    def __init__(self, csv_path: Path | str, transform=None):
        self.df = pd.read_csv(csv_path)
        exists = self.df["path"].map(lambda p: Path(p).is_file())
        n_missing = int((~exists).sum())
        if n_missing:
            print(f"Warning: dropping {n_missing} missing paths from {csv_path}")
            self.df = self.df.loc[exists].reset_index(drop=True)
        if len(self.df) == 0:
            raise FileNotFoundError(f"No readable images in {csv_path}")
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


def load_norm_stats(meta_path: Path | None = None) -> tuple[list[float], list[float]]:
    meta_path = meta_path or (OUTPUT_DIR / "split_meta.json")
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    return meta["mean"], meta["std"]


def build_train_transform(mean: list[float], std: list[float], image_size: int = IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
        ]
    )


def build_eval_transform(mean: list[float], std: list[float], image_size: int = IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_eval_tta_transform(mean: list[float], std: list[float], image_size: int = IMAGE_SIZE):
    """Second TTA view: mild zoom-in via 256 resize + center crop."""
    return transforms.Compose(
        [
            transforms.Resize((int(image_size * 8 / 7),) * 2),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def make_loaders(
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    image_size: int = IMAGE_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader, list[float], list[float]]:
    mean, std = load_norm_stats()
    train_ds = DogBreedDataset(OUTPUT_DIR / "train.csv", transform=build_train_transform(mean, std, image_size))
    val_ds = DogBreedDataset(OUTPUT_DIR / "val.csv", transform=build_eval_transform(mean, std, image_size))
    test_ds = DogBreedDataset(OUTPUT_DIR / "test.csv", transform=build_eval_transform(mean, std, image_size))

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, mean, std

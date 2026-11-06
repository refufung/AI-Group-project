"""Re-split Tsinghua low-resolution 4-breed dataset at 1000 unique images per class.

Split: 70/15/15 -> 700 train / 150 val / 150 test per class (2800/600/600 total), seed 42.

Duplicate guard: the on-disk low-resolution folder contains byte-identical macOS
Finder copies named " <n>.jpg" from a double zip extraction (verified by md5).
Those are excluded here; without this guard a re-split would leak train<->test
duplicates.

Writes: outputs/train.csv, outputs/val.csv, outputs/test.csv, outputs/split_meta.json
(existing files are moved to outputs/backup_prev/ first, never silently overwritten).
"""

from __future__ import annotations

import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SEED = 42
PER_CLASS = 1000
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
IMAGE_SIZE = 224

TRACK_DIR = Path(__file__).resolve().parents[1]  # resBuildScratch/
PROJECT_ROOT = TRACK_DIR.parent  # "AI Group project "
PAPER_DIR = PROJECT_ROOT / "OneDrive_2026-10-24" / "0. Original Tsinghua Dataset and Paper"
LOW_RES = PAPER_DIR / "low-resolution"
OUTPUT_DIR = TRACK_DIR / "outputs"

BREEDS = {
    "1121-n000002-French_bulldog": ("French_bulldog", 0),
    "2909-n000116-Cardigan": ("Cardigan", 1),
    "3580-n000122-Labrador_retriever": ("Labrador_retriever", 2),
    "5355-n000126-golden_retriever": ("golden_retriever", 3),
}
CLASS_NAMES = [name for name, _ in sorted(BREEDS.values(), key=lambda t: t[1])]

# macOS duplicate copies end in " 2.jpg", " 3.jpg", ... -> stem ends with " <digits>"
DUP_RE = re.compile(r"\s\d+$")
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def collect_valid_images() -> dict[str, list[Path]]:
    """Collect unique, readable RGB images per class, skipping Finder duplicate copies."""
    by_class: dict[str, list[Path]] = defaultdict(list)
    for folder, (class_name, _) in BREEDS.items():
        breed_dir = LOW_RES / folder
        if not breed_dir.is_dir():
            raise FileNotFoundError(breed_dir)
        paths = sorted(
            p
            for p in breed_dir.iterdir()
            if p.suffix.lower() in IMG_EXTS and p.is_file()
        )
        skipped_dup = 0
        skipped_bad = 0
        for path in paths:
            if DUP_RE.search(path.stem):
                skipped_dup += 1
                continue
            try:
                with Image.open(path) as im:
                    im.convert("RGB")
                by_class[class_name].append(path.resolve())
            except Exception as exc:
                skipped_bad += 1
                print(f"  skip unreadable {path.name}: {exc}")
        print(
            f"{class_name}: {len(by_class[class_name])} unique usable "
            f"(skipped {skipped_dup} duplicate copies, {skipped_bad} unreadable)"
        )
    return dict(by_class)


def balance_and_split(
    by_class: dict[str, list[Path]], seed: int = SEED
) -> tuple[dict[str, list[dict]], dict]:
    rng = random.Random(seed)
    for class_name, paths in by_class.items():
        if len(paths) < PER_CLASS:
            raise ValueError(f"{class_name} has only {len(paths)} unique images (< {PER_CLASS})")

    rows_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    class_counts: dict = {"train": {}, "val": {}, "test": {}, "per_class_cap": PER_CLASS}

    n_train = int(round(PER_CLASS * TRAIN_RATIO))  # 700
    n_val = int(round(PER_CLASS * VAL_RATIO))  # 150

    for _, (class_name, label) in BREEDS.items():
        paths = list(by_class[class_name])
        rng.shuffle(paths)
        paths = paths[:PER_CLASS]
        splits = {
            "train": paths[:n_train],
            "val": paths[n_train : n_train + n_val],
            "test": paths[n_train + n_val :],
        }
        for split_name, split_paths in splits.items():
            class_counts[split_name][class_name] = len(split_paths)
            for p in split_paths:
                rows_by_split[split_name].append(
                    {"path": str(p), "label": label, "class_name": class_name}
                )

    for split_name in rows_by_split:
        rng.shuffle(rows_by_split[split_name])
    return rows_by_split, class_counts


class _CsvDataset(Dataset):
    def __init__(self, csv_path: Path, transform=None):
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


def compute_train_mean_std(csv_path: Path, max_samples: int = 512) -> tuple[list[float], list[float]]:
    """Per-channel mean/std on a subsample of the train split (after resize, before normalize)."""
    tf = transforms.Compose(
        [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
    )
    ds = _CsvDataset(csv_path, transform=tf)
    n = min(len(ds), max_samples)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)
    count = 0
    for images, _ in loader:
        b = images.size(0)
        flat = images.view(b, 3, -1)
        sum_ += flat.sum(dim=(0, 2))
        sum_sq += (flat**2).sum(dim=(0, 2))
        count += b * flat.size(2)
        if count // (IMAGE_SIZE * IMAGE_SIZE) >= n:
            break
    mean = (sum_ / count).tolist()
    var = sum_sq / count - torch.tensor(mean) ** 2
    std = torch.sqrt(var.clamp(min=1e-8)).tolist()
    return mean, std


def backup_existing() -> None:
    existing = [OUTPUT_DIR / f"{s}.csv" for s in ("train", "val", "test")]
    existing.append(OUTPUT_DIR / "split_meta.json")
    existing = [p for p in existing if p.is_file()]
    if not existing:
        return
    backup_dir = OUTPUT_DIR / "backup_prev"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for p in existing:
        shutil.move(str(p), backup_dir / p.name)
    print(f"Moved {len(existing)} existing file(s) to {backup_dir}")


def main() -> None:
    print(f"LOW_RES    = {LOW_RES}  exists={LOW_RES.is_dir()}")
    print(f"OUTPUT_DIR = {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    backup_existing()

    by_class = collect_valid_images()
    rows_by_split, class_counts = balance_and_split(by_class)

    for name, rows in rows_by_split.items():
        csv_path = OUTPUT_DIR / f"{name}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Wrote {csv_path} ({len(rows)} rows)")

    mean, std = compute_train_mean_std(OUTPUT_DIR / "train.csv")
    print(f"train mean={mean}")
    print(f"train std ={std}")

    meta = {
        "seed": SEED,
        "per_class": PER_CLASS,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "breeds": {
            folder: {"class_name": name, "label": label}
            for folder, (name, label) in BREEDS.items()
        },
        "class_counts": class_counts,
        "mean": mean,
        "std": std,
        "note": (
            "1000 unique images per class; macOS Finder duplicate copies "
            "(stem ends with ' <n>') excluded to prevent train/val/test leakage. "
            "Split is by image; augmentation is applied only in the training "
            "DataLoader and never written to disk."
        ),
    }
    meta_path = OUTPUT_DIR / "split_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")

    # --- verification ---
    print("\n=== verification ===")
    print(json.dumps(class_counts, indent=2))
    seen: set[str] = set()
    ok = True
    for name in ("train", "val", "test"):
        df = pd.read_csv(OUTPUT_DIR / f"{name}.csv")
        dup_rows = df["path"].str.contains(r"\s\d+\.(?:jpg|jpeg|png)$", case=False, regex=True).sum()
        overlap = seen.intersection(df["path"])
        print(
            f"{name}: {len(df)} rows, labels={sorted(df['label'].unique().tolist())}, "
            f"duplicate-name rows={dup_rows}, overlap-with-previous-splits={len(overlap)}"
        )
        if dup_rows or overlap or len(df) != df["path"].nunique():
            ok = False
        seen.update(df["path"])
    print("OK" if ok else "VERIFICATION FAILED")


if __name__ == "__main__":
    main()

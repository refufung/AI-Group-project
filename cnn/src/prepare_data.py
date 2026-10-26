"""Extract (if needed), validate, balance, and split the four-breed dog dataset."""

from __future__ import annotations

import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Folder name -> class index / display name
BREEDS = {
    "1121-n000002-French_bulldog": ("French_bulldog", 0),
    "2909-n000116-Cardigan": ("Cardigan", 1),
    "3580-n000122-Labrador_retriever": ("Labrador_retriever", 2),
    "5355-n000126-golden_retriever": ("golden_retriever", 3),
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = PROJECT_ROOT / "OneDrive_2026-10-24" / "0. Original Tsinghua Dataset and Paper"
LOW_RES = PAPER_DIR / "low-resolution"
ZIP_PATH = PAPER_DIR / "low-resolution.zip"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def ensure_breed_folders() -> None:
    """Unzip missing breed folders from the local archive."""
    missing = [folder for folder in BREEDS if not (LOW_RES / folder).is_dir()]
    if not missing:
        print("All breed folders present.")
        return
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(f"Missing breed folders {missing} and zip not found: {ZIP_PATH}")
    print(f"Extracting {missing} from {ZIP_PATH.name} ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = [
            m
            for m in zf.namelist()
            if any(m.startswith(f"low-resolution/{folder}/") for folder in missing)
        ]
        zf.extractall(PAPER_DIR, members=members)
    print("Extraction done.")


def collect_valid_images() -> dict[str, list[Path]]:
    """Open every image as RGB; drop unreadable files. Key = class_name."""
    by_class: dict[str, list[Path]] = defaultdict(list)
    for folder, (class_name, _) in BREEDS.items():
        breed_dir = LOW_RES / folder
        if not breed_dir.is_dir():
            raise FileNotFoundError(breed_dir)
        paths = sorted(
            p
            for p in breed_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file()
        )
        valid = 0
        for path in paths:
            try:
                with Image.open(path) as im:
                    im.convert("RGB")
                by_class[class_name].append(path.resolve())
                valid += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {path.name}: {exc}")
        print(f"{class_name}: {valid}/{len(paths)} readable")
    return dict(by_class)


def balance_and_split(
    by_class: dict[str, list[Path]], seed: int = SEED
) -> tuple[list[dict], list[dict], list[dict], dict]:
    rng = random.Random(seed)
    min_count = min(len(v) for v in by_class.values())
    print(f"Balancing all classes to {min_count} images (seed={seed})")

    rows_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    class_counts = {"train": {}, "val": {}, "test": {}, "balanced_per_class": min_count}

    for folder, (class_name, label) in BREEDS.items():
        del folder  # folder key unused after lookup
        paths = list(by_class[class_name])
        rng.shuffle(paths)
        paths = paths[:min_count]
        n_train = int(round(min_count * TRAIN_RATIO))
        n_val = int(round(min_count * VAL_RATIO))
        # Remainder goes to test so totals match min_count
        splits = {
            "train": paths[:n_train],
            "val": paths[n_train : n_train + n_val],
            "test": paths[n_train + n_val :],
        }
        for split_name, split_paths in splits.items():
            class_counts[split_name][class_name] = len(split_paths)
            for p in split_paths:
                rows_by_split[split_name].append(
                    {
                        "path": str(p),
                        "label": label,
                        "class_name": class_name,
                    }
                )

    for split_name in rows_by_split:
        rng.shuffle(rows_by_split[split_name])

    meta = {
        "seed": seed,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "breeds": {folder: {"class_name": name, "label": label} for folder, (name, label) in BREEDS.items()},
        "class_counts": class_counts,
        "note": (
            "No leaf-group IDs in Tsinghua low-resolution set; each JPEG is one "
            "original image. Split is by image; augmentation is applied only in "
            "the training DataLoader and never written to disk."
        ),
    }
    return rows_by_split["train"], rows_by_split["val"], rows_by_split["test"], meta


def write_outputs(
    train_rows: list[dict],
    val_rows: list[dict],
    test_rows: list[dict],
    meta: dict,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        path = OUTPUT_DIR / f"{name}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Wrote {path} ({len(rows)} rows)")
    meta_path = OUTPUT_DIR / "split_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")
    print("Class counts:")
    print(json.dumps(meta["class_counts"], indent=2))


def main() -> None:
    ensure_breed_folders()
    by_class = collect_valid_images()
    train_rows, val_rows, test_rows, meta = balance_and_split(by_class)
    write_outputs(train_rows, val_rows, test_rows, meta)


if __name__ == "__main__":
    main()

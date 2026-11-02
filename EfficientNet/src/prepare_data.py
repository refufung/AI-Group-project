"""Copy shared train/val/test CSVs from the Custom CNN track for a fair comparison."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

EFF_ROOT = Path(__file__).resolve().parents[1]
CNN_OUTPUTS = EFF_ROOT.parent / "cnn" / "outputs"
OUTPUT_DIR = EFF_ROOT / "outputs"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CNN_OUTPUTS.is_dir():
        raise FileNotFoundError(f"Missing CNN outputs at {CNN_OUTPUTS}")

    for name in ("train.csv", "val.csv", "test.csv"):
        src = CNN_OUTPUTS / name
        dst = OUTPUT_DIR / name
        shutil.copy2(src, dst)
        print(f"Copied {src} → {dst}")

    cnn_meta_path = CNN_OUTPUTS / "split_meta.json"
    cnn_meta = json.loads(cnn_meta_path.read_text(encoding="utf-8")) if cnn_meta_path.is_file() else {}
    meta = {
        "seed": cnn_meta.get("seed", 42),
        "train_ratio": cnn_meta.get("train_ratio", 0.7),
        "val_ratio": cnn_meta.get("val_ratio", 0.15),
        "test_ratio": cnn_meta.get("test_ratio", 0.15),
        "breeds": cnn_meta.get("breeds"),
        "class_counts": cnn_meta.get("class_counts"),
        "note": (
            "Reused cnn/outputs train/val/test CSVs (seed 42) for fair EfficientNet comparison. "
            + str(cnn_meta.get("note", ""))
        ),
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "normalize": "ImageNet",
        "source_csvs": str(CNN_OUTPUTS),
    }
    out_meta = OUTPUT_DIR / "split_meta.json"
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_meta}")


if __name__ == "__main__":
    main()

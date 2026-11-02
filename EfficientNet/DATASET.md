# Four-breed dog dataset (EfficientNet-B0 track)

**Primary entry:** `python src/train.py`, or notebook [`efficientnet_b0.ipynb`](efficientnet_b0.ipynb).

**Continuous search until target:**  
`PYTHONPATH=src python -u src/continuous_improve.py --target 0.85 --max-rounds 40 --resume`

This track **reuses the same train/val/test CSVs** as the Custom CNN track for a fair comparison (seed 42).

Source: Tsinghua Dogs low-resolution set  
Path: `OneDrive_2026-10-24/0. Original Tsinghua Dataset and Paper/low-resolution`

## Classes

| Label | Class name           | Source folder                         |
|------:|----------------------|---------------------------------------|
| 0     | French_bulldog       | `1121-n000002-French_bulldog`         |
| 1     | Cardigan             | `2909-n000116-Cardigan`               |
| 2     | Labrador_retriever   | `3580-n000122-Labrador_retriever`     |
| 3     | golden_retriever     | `5355-n000126-golden_retriever`       |

## Split (shared with `cnn/`)

- **Seed:** `42`
- **Per class:** 785 train / 168 val / 168 test
- **Totals:** 3,140 / 672 / 672
- Manifests: `EfficientNet/outputs/{train,val,test}.csv` (copied from `cnn/outputs/`)

## Preprocessing (EfficientNet / ImageNet)

- RGB, resize/crop to **224×224**
- Normalize with **ImageNet** mean/std (not the custom CNN train-set stats)
- Train-only: `RandomResizedCrop(0.75–1.0)`, horizontal flip, light ColorJitter
- Val/test: deterministic resize + ImageNet normalize

## Model

**Primary:** custom from-scratch **EfficientNet** (`src/model.py`) only — no torchvision / ImageNet weights.

**Training:** Stage A train-all → Stage B fine-tune. AdamW, RandAugment, MixUp/CutMix, EMA. Search can vary `width_mult`, `dropout`, `depth_drop`.

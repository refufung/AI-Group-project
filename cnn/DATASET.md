# Four-breed dog dataset (Custom CNN track)

**Primary entry:** run [`cnn/custom_cnn.ipynb`](custom_cnn.ipynb) with the `Python (cnn/.venv)` kernel (Run All).  
The notebook is self-contained (split → preprocess → SmallCNN → val tuning → one-shot test). Optional reference scripts remain under `cnn/src/`.

Source: Tsinghua Dogs low-resolution set  
Path: `OneDrive_2026-10-24/0. Original Tsinghua Dataset and Paper/low-resolution`

## Classes

| Label | Class name           | Source folder                         |
|------:|----------------------|---------------------------------------|
| 0     | French_bulldog       | `1121-n000002-French_bulldog`         |
| 1     | Cardigan             | `2909-n000116-Cardigan`               |
| 2     | Labrador_retriever   | `3580-n000122-Labrador_retriever`     |
| 3     | golden_retriever     | `5355-n000126-golden_retriever`       |

## Balancing

French Bulldog is the smallest readable class (1,121 images). Every class was randomly subsampled to **1,121** images so class counts stay equal (4,484 total).

## Split

- **Seed:** `42`
- **Ratios:** 70% train / 15% validation / 15% test (stratified by class)
- **Per class:** 785 train / 168 val / 168 test
- **Totals:** 3,140 / 672 / 672

There are no leaf-group or dog-identity IDs in this release: each JPEG is one original image. The split is by image. Random augmentation is applied only inside the training `DataLoader` and is never written to disk, so no augmented view of a training image can appear in validation or test.

CSV manifests (shared with ResNet / VLM tracks):

- `cnn/outputs/train.csv`
- `cnn/outputs/val.csv`
- `cnn/outputs/test.csv`
- `cnn/outputs/split_meta.json` (seed, counts, train mean/std)

Columns: `path`, `label`, `class_name`.

## Preprocessing

Shared for all splits:

- RGB conversion and label check from CSV
- Resize to 224×224 (train uses resize-256 + random crop 224)
- Scale to [0, 1], then per-channel normalize with **training-set** mean/std (stored in `split_meta.json`)

Training-only (symptom / breed-cue preserving):

- Random horizontal flip
- Rotation ±15°
- Color jitter: brightness/contrast 0.15, saturation 0.05

Validation and test use deterministic resize + the same normalization; no random transforms.

## Usage for teammates

```python
import pandas as pd
train = pd.read_csv("cnn/outputs/train.csv")
# Use the same mean/std from split_meta.json for ResNet / VLM preprocessing consistency.
```

Validation is for hyperparameter tuning and checkpoint selection only. The test set is for a single final evaluation after that choice.

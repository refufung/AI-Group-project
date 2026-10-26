"""Train SmallCNN with validation tuning; evaluate once on the held-out test set."""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from dataset import make_loaders
from model import SmallCNN

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
CLASS_NAMES = ["French_bulldog", "Cardigan", "Labrador_retriever", "golden_retriever"]
SEED = 42
MAX_EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 32

# Validation-only hyperparameter grid
LR_GRID = [1e-3, 3e-4]
DROPOUT_GRID = [0.3, 0.5]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / max(n, 1)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))
    return avg_loss, acc, macro_f1, cm, all_labels, all_preds


def train_one_run(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    dropout: float,
) -> dict:
    model = SmallCNN(num_classes=len(CLASS_NAMES), dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience_left = PATIENCE

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        print(
            f"  epoch {epoch:02d}/{MAX_EPOCHS}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stop at epoch {epoch} (best val macro-F1={best_f1:.4f} @ {best_epoch})")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "best_val_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "lr": lr,
        "dropout": dropout,
    }


def plot_curves(history: dict, out_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train acc")
    axes[1].plot(epochs, history["val_acc"], label="val acc")
    axes[1].plot(epochs, history["val_macro_f1"], label="val macro-F1")
    axes[1].set_title("Accuracy / Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, mean, std = make_loaders(batch_size=BATCH_SIZE)
    print(f"Norm mean={mean}, std={std}")

    tuning_results = []
    best_run = None

    for lr in LR_GRID:
        for dropout in DROPOUT_GRID:
            print(f"\n=== Tuning lr={lr}, dropout={dropout} ===")
            set_seed(SEED)
            run = train_one_run(train_loader, val_loader, device, lr=lr, dropout=dropout)
            summary = {
                "lr": lr,
                "dropout": dropout,
                "best_val_macro_f1": run["best_val_macro_f1"],
                "best_epoch": run["best_epoch"],
            }
            tuning_results.append(summary)
            print(f"  best val macro-F1={run['best_val_macro_f1']:.4f} @ epoch {run['best_epoch']}")
            if best_run is None or run["best_val_macro_f1"] > best_run["best_val_macro_f1"]:
                best_run = run

    assert best_run is not None
    print(
        f"\nSelected config: lr={best_run['lr']}, dropout={best_run['dropout']}, "
        f"val_macro_f1={best_run['best_val_macro_f1']:.4f}"
    )

    # Save best checkpoint
    ckpt_path = OUTPUT_DIR / "best_small_cnn.pt"
    torch.save(
        {
            "state_dict": best_run["model"].state_dict(),
            "lr": best_run["lr"],
            "dropout": best_run["dropout"],
            "best_val_macro_f1": best_run["best_val_macro_f1"],
            "best_epoch": best_run["best_epoch"],
            "class_names": CLASS_NAMES,
            "seed": SEED,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint {ckpt_path}")

    plot_curves(best_run["history"], OUTPUT_DIR / "training_curves.png")

    # Single final test evaluation
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_f1, cm, _, _ = evaluate(
        best_run["model"], test_loader, device, criterion
    )
    plot_confusion(cm, OUTPUT_DIR / "confusion_matrix.png")

    # Class counts from split meta
    meta_path = OUTPUT_DIR / "split_meta.json"
    with meta_path.open(encoding="utf-8") as f:
        split_meta = json.load(f)

    metrics = {
        "seed": SEED,
        "device": str(device),
        "selected_lr": best_run["lr"],
        "selected_dropout": best_run["dropout"],
        "best_val_macro_f1": best_run["best_val_macro_f1"],
        "best_epoch": best_run["best_epoch"],
        "tuning_results": tuning_results,
        "test": {
            "loss": test_loss,
            "accuracy": test_acc,
            "macro_f1": test_f1,
            "confusion_matrix": cm.tolist(),
            "class_names": CLASS_NAMES,
        },
        "class_counts": split_meta.get("class_counts"),
        "mean": mean,
        "std": std,
    }
    metrics_path = OUTPUT_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Final test (one-shot) ===")
    print(f"accuracy={test_acc:.4f}  macro_f1={test_f1:.4f}  loss={test_loss:.4f}")
    print("confusion matrix:")
    print(cm)
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()

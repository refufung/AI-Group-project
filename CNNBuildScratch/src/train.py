"""Train CNNSE from scratch on the 1000/class split; TTA + ensemble one-shot test.

Recipe:
  AdamW (wd on conv/linear weights only) + per-step linear warmup -> cosine to 0
  label smoothing 0.1, MixUp(0.2)/CutMix(1.0) per batch with prob 0.5,
  grad clip 1.0, EMA(0.999), best checkpoint selected by EMA val macro-F1.

Usage:
  python train.py --smoke                      # 2-min end-to-end sanity check
  python train.py --epochs 60 --seeds 42       # sanity run
  python train.py --epochs 150 --seeds 42 1337 2025   # full run
  python train.py --eval-only --seeds 42 1337 2025    # TTA + ensemble test from checkpoints
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

from dataset import (
    OUTPUT_DIR,
    DogBreedDataset,
    build_eval_tta_transform,
    make_loaders,
)
from model import CNNSE, count_params

CLASS_NAMES = ["French_bulldog", "Cardigan", "Labrador_retriever", "golden_retriever"]


def set_seed(seed: int) -> None:
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


# ---------------- MixUp / CutMix ----------------

def rand_bbox(size: torch.Size, lam: float) -> tuple[int, int, int, int]:
    h, w = size[2], size[3]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(w * cut_rat), int(h * cut_rat)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1, x2 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)
    y1, y2 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)
    return x1, y1, x2, y2


def maybe_mix(
    images: torch.Tensor, targets: torch.Tensor, mixup_alpha: float, cutmix_alpha: float, prob: float
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, float] | None]:
    if random.random() > prob:
        return images, targets, None
    perm = torch.randperm(images.size(0), device=images.device)
    if random.random() < 0.5:  # CutMix
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        images[:, :, y1:y2, x1:x2] = images[perm][:, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
    else:  # MixUp
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        images = lam * images + (1.0 - lam) * images[perm]
    return images, targets, (perm, lam)


def soft_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float,
    mix: tuple[torch.Tensor, float] | None = None,
) -> torch.Tensor:
    """Cross-entropy with label smoothing; linearly combined for mixed batches."""
    logp = F.log_softmax(logits, dim=1)

    def ce(t: torch.Tensor) -> torch.Tensor:
        nll = -logp.gather(1, t.unsqueeze(1)).squeeze(1)
        return (1.0 - smoothing) * nll + smoothing * (-logp.mean(dim=1))

    loss = ce(targets)
    if mix is not None:
        perm, lam = mix
        loss = lam * loss + (1.0 - lam) * ce(targets[perm])
    return loss.mean()


# ---------------- EMA ----------------

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


# ---------------- eval ----------------

@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, smoothing: float = 0.0
) -> tuple[float, float, float, np.ndarray, list[int], list[int]]:
    model.eval()
    total_loss, n = 0.0, 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = soft_ce(logits, labels, smoothing)
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))
    return total_loss / max(n, 1), acc, f1, cm, all_labels, all_preds


@torch.no_grad()
def predict_probs(
    model: nn.Module, loader: DataLoader, device: torch.device, tta_loader: DataLoader | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Mean softmax over views: [x, hflip(x)] for the plain loader (+ TTA zoom loader if given)."""
    model.eval()
    loaders = [loader] + ([tta_loader] if tta_loader is not None else [])
    view_probs: list[np.ndarray] = []
    labels_all: list[int] = []
    for view_idx, current in enumerate(loaders):
        batch_probs: list[np.ndarray] = []
        for images, labels in current:
            images = images.to(device)
            p = F.softmax(model(images), dim=1) + F.softmax(model(torch.flip(images, dims=[3])), dim=1)
            batch_probs.append((p / 2.0).cpu().numpy())
            if view_idx == 0:
                labels_all.extend(labels.tolist())
        view_probs.append(np.concatenate(batch_probs, axis=0))
    views = np.stack(view_probs, axis=0)  # (n_views, n, C)
    return views.mean(axis=0), np.array(labels_all)


# ---------------- train ----------------

def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim > 1 else no_decay).append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def lr_at(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def train_one_seed(args, seed: int, device: torch.device) -> dict:
    set_seed(seed)
    train_loader, val_loader, _, mean, std = make_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers, seed=seed
    )
    if args.smoke:
        args.max_batches = 6

    model = CNNSE(num_classes=len(CLASS_NAMES), dropout=args.dropout).to(device)
    if seed == args.seeds[0]:
        print(f"CNNSE params: {count_params(model) / 1e6:.2f}M")
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.wd)
    ema = ModelEMA(model, decay=args.ema_decay)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    global_step = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
    best_f1, best_epoch, best_state = -1.0, 0, None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for i, (images, labels) in enumerate(train_loader):
            if args.smoke and i >= args.max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            images, labels, mix = maybe_mix(images, labels, args.mixup_alpha, args.cutmix_alpha, args.mix_prob)

            lr = lr_at(global_step, total_steps, warmup_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = soft_ce(logits, labels, args.label_smoothing, mix)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            global_step += 1

            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        val_loss, val_acc, val_f1, _, _, _ = evaluate(ema.module, val_loader, device)
        history["train_loss"].append(running_loss / max(total, 1))
        history["train_acc"].append(correct / max(total, 1))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        marker = ""
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in ema.module.state_dict().items()}
            marker = " *"
        print(
            f"[seed {seed}] epoch {epoch:03d}/{args.epochs} lr={lr:.2e} "
            f"train_loss={history['train_loss'][-1]:.4f} train_acc={history['train_acc'][-1]:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
            f"({time.time() - t0:.0f}s){marker}",
            flush=True,
        )
        if args.smoke and epoch >= 1:
            break

    assert best_state is not None
    ckpt = {
        "arch": "CNN_se",
        "state_dict": best_state,
        "seed": seed,
        "best_val_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "class_names": CLASS_NAMES,
        "mean": mean,
        "std": std,
        "config": vars(args),
    }
    ckpt_path = OUTPUT_DIR / ("smoke_ckpt.pt" if args.smoke else f"best_CNN_se_seed{seed}.pt")
    torch.save(ckpt, ckpt_path)
    print(f"[seed {seed}] saved {ckpt_path} (best val macro-F1={best_f1:.4f} @ epoch {best_epoch})")

    return {
        "seed": seed,
        "history": history,
        "best_val_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "ckpt_path": str(ckpt_path),
        "train_seconds": time.time() - t0,
    }


# ---------------- eval helpers ----------------

def load_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CNNSE(num_classes=len(CLASS_NAMES))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def make_tta_loader(split: str, mean, std, batch_size: int) -> DataLoader:
    ds = DogBreedDataset(OUTPUT_DIR / f"{split}.csv", transform=build_eval_tta_transform(mean, std))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def plot_curves(histories: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for h in histories:
        hist, seed = h["history"], h["seed"]
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], alpha=0.6, label=f"train s{seed}")
        axes[0].plot(epochs, hist["val_loss"], ls="--", alpha=0.8, label=f"val s{seed}")
        axes[1].plot(epochs, hist["val_acc"], alpha=0.8, label=f"val acc s{seed}")
        axes[1].plot(epochs, hist["val_macro_f1"], ls="--", alpha=0.8, label=f"val F1 s{seed}")
    axes[0].set_title("Loss")
    axes[1].set_title("Val accuracy / macro-F1")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_final_eval(args, device: torch.device) -> dict:
    """TTA per seed on val + test; ensemble = mean over seeds. Test evaluated once."""
    _, _, test_loader, mean, std = make_loaders(batch_size=args.batch_size, num_workers=0)
    tta_test = make_tta_loader("test", mean, std, args.batch_size)

    results = {"seeds": [], "test_ensemble": None}
    test_probs: list[np.ndarray] = []
    test_labels = None
    for seed in args.seeds:
        ckpt_path = OUTPUT_DIR / f"best_CNN_se_seed{seed}.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)
        model = load_model(ckpt_path, device)
        probs, labels = predict_probs(model, test_loader, device, tta_loader=tta_test)
        test_probs.append(probs)
        test_labels = labels
        preds = probs.argmax(1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        results["seeds"].append({"seed": seed, "test_tta_acc": acc, "test_tta_macro_f1": f1})
        print(f"[eval] seed {seed} single+TTA test acc={acc:.4f} f1={f1:.4f}")

    ens_probs = np.mean(test_probs, axis=0)
    ens_preds = ens_probs.argmax(1)
    acc = accuracy_score(test_labels, ens_preds)
    f1 = f1_score(test_labels, ens_preds, average="macro", zero_division=0)
    cm = confusion_matrix(test_labels, ens_preds, labels=list(range(len(CLASS_NAMES))))
    results["test_ensemble"] = {
        "accuracy": acc,
        "macro_f1": f1,
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "n_seeds": len(args.seeds),
        "tta": "mean softmax over {224, hflip, 256-center-crop, hflip} x seeds",
    }
    print(f"[eval] ENSEMBLE+TTA test acc={acc:.4f} f1={f1:.4f}")
    plot_confusion(cm, OUTPUT_DIR / "confusion_matrix.png", "Ensemble+TTA test confusion matrix")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=5e-2)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mix-prob", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 1
        args.warmup_epochs = 0

    device = get_device()
    print(f"device={device} epochs={args.epochs} seeds={args.seeds} bs={args.batch_size} "
          f"lr={args.lr} wd={args.wd} warmup={args.warmup_epochs} mix_prob={args.mix_prob}")

    runs = []
    if not args.eval_only:
        for seed in args.seeds:
            runs.append(train_one_seed(args, seed, device))
        if not args.smoke:
            plot_curves(runs, OUTPUT_DIR / "training_curves.png")

    if args.smoke:
        print("SMOKE OK")
        return

    eval_results = run_final_eval(args, device)
    metrics = {
        "config": {k: v for k, v in vars(args).items()},
        "device": str(device),
        "class_counts": json.loads((OUTPUT_DIR / "split_meta.json").read_text())["class_counts"],
        "runs": [
            {k: r[k] for k in ("seed", "best_val_macro_f1", "best_epoch", "train_seconds")}
            for r in runs
        ],
        **eval_results,
    }
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {OUTPUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()

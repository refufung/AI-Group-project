"""From-scratch EfficientNet: Stage A train-all, Stage B fine-tune.

Supports full training and shorter search budgets via ``TrainConfig``.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from dataset import IMAGENET_MEAN, IMAGENET_STD, make_loaders
from model import build_efficientnet_b0, set_backbone_trainable

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
CLASS_NAMES = ["French_bulldog", "Cardigan", "Labrador_retriever", "golden_retriever"]
NUM_CLASSES = 4


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 32
    num_classes: int = 4
    image_size: int = 224
    # Defaults from continuous_improve winner (final full retrain ~78.6%)
    width_mult: float = 0.65
    dropout: float = 0.1
    depth_drop: int = 0
    depth_mult: float = 1.0
    se_ratio: float = 0.25
    drop_path_rate: float = 0.0
    focal_gamma: float = 0.0  # 0 = soft CE; 1.5 = soft focal
    label_smoothing: float = 0.05
    weight_decay: float = 0.01
    ema_decay: float = 0.995
    ema_start_epoch: int = 5
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.5
    randaugment: bool = True
    randaugment_magnitude: int = 5
    stage_a_max_epochs: int = 100
    stage_a_patience: int = 15
    stage_a_lr: float = 1e-3
    stage_b_max_epochs: int = 40
    stage_b_patience: int = 10
    stage_b_lr: float = 1e-4
    save_plots: bool = True
    write_metrics: bool = True
    ckpt_name: str = "best_efficientnet_b0.pt"


def search_config(**overrides: Any) -> TrainConfig:
    """Medium Stage A/B budgets for continuous search rounds."""
    base = TrainConfig(
        stage_a_max_epochs=50,
        stage_a_patience=8,
        stage_b_max_epochs=20,
        stage_b_patience=6,
        save_plots=False,
        write_metrics=False,
        # Never overwrite the official champion checkpoint during short search.
        ckpt_name="search_candidate.pt",
    )
    return replace(base, **overrides) if overrides else base


def full_config(**overrides: Any) -> TrainConfig:
    return replace(TrainConfig(), **overrides) if overrides else TrainConfig()


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


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if not v.dtype.is_floating_point:
                continue
            v.copy_(v * d + msd[k].detach() * (1.0 - d))


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def soft_focal_loss(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    gamma: float = 1.5,
) -> torch.Tensor:
    """Soft-label focal loss; gamma=0 reduces to soft CE."""
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    if gamma <= 0:
        return -(soft_targets * log_probs).sum(dim=1).mean()
    return -(soft_targets * (1.0 - probs).pow(gamma) * log_probs).sum(dim=1).mean()


def train_loss_fn(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    focal_gamma: float,
) -> torch.Tensor:
    if focal_gamma and focal_gamma > 0:
        return soft_focal_loss(logits, soft_targets, gamma=focal_gamma)
    return soft_cross_entropy(logits, soft_targets)


def one_hot(labels: torch.Tensor, num_classes: int, label_smoothing: float) -> torch.Tensor:
    n = labels.size(0)
    off = label_smoothing / num_classes
    on = 1.0 - label_smoothing + off
    y = torch.full((n, num_classes), off, device=labels.device, dtype=torch.float32)
    y.scatter_(1, labels.unsqueeze(1), on)
    return y


def rand_bbox(size: torch.Size, lam: float) -> tuple[int, int, int, int]:
    _, _, h, w = size
    cut_rat = float(np.sqrt(1.0 - lam))
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = int(np.random.randint(w))
    cy = int(np.random.randint(h))
    x1 = int(np.clip(cx - cut_w // 2, 0, w))
    y1 = int(np.clip(cy - cut_h // 2, 0, h))
    x2 = int(np.clip(cx + cut_w // 2, 0, w))
    y2 = int(np.clip(cy + cut_h // 2, 0, h))
    return x1, y1, x2, y2


def apply_mixup_cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    cfg: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    soft = one_hot(labels, cfg.num_classes, cfg.label_smoothing)
    if random.random() > cfg.mix_prob:
        return images, soft

    use_cutmix = random.random() < 0.5
    if use_cutmix:
        lam = float(np.random.beta(cfg.cutmix_alpha, cfg.cutmix_alpha))
        idx = torch.randperm(images.size(0), device=images.device)
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
        lam_adj = 1.0 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
        soft2 = one_hot(labels[idx], cfg.num_classes, cfg.label_smoothing)
        soft = soft * lam_adj + soft2 * (1.0 - lam_adj)
        return mixed, soft

    lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
    idx = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[idx]
    soft2 = one_hot(labels[idx], cfg.num_classes, cfg.label_smoothing)
    soft = soft * lam + soft2 * (1.0 - lam)
    return mixed, soft


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
    return avg_loss, acc, macro_f1, cm


def train_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion_eval: nn.Module,
    cfg: TrainConfig,
    max_epochs: int,
    patience: int | None,
    scheduler: CosineAnnealingLR | None,
    history: dict,
    stage_name: str,
    ema: ModelEMA | None = None,
) -> tuple[float, int, dict]:
    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    patience_left = patience if patience is not None else 10**9

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            images, soft_targets = apply_mixup_cutmix(images, labels, cfg)
            optimizer.zero_grad()
            logits = model(images)
            loss = train_loss_fn(logits, soft_targets, cfg.focal_gamma)
            loss.backward()
            optimizer.step()
            if ema is not None and epoch >= cfg.ema_start_epoch:
                ema.update(model)
            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc, val_f1, _ = evaluate(model, val_loader, device, criterion_eval)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["stage"].append(stage_name)

        lr_msg = ""
        if scheduler is not None:
            lr_msg = f" lr={scheduler.get_last_lr()[0]:.6f}"
        elif optimizer.param_groups:
            lr_msg = f" lr={optimizer.param_groups[0]['lr']:.6f}"

        print(
            f"  [{stage_name}] epoch {epoch:02d}/{max_epochs}{lr_msg}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience if patience is not None else 10**9
        else:
            patience_left -= 1
            if patience is not None and patience_left <= 0:
                print(
                    f"  early stop at {stage_name} epoch {epoch} "
                    f"(best val macro-F1={best_f1:.4f} @ {best_epoch})"
                )
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    if ema is not None:
        _, _, ema_f1, _ = evaluate(ema.ema, val_loader, device, criterion_eval)
        if ema_f1 > best_f1:
            print(f"  using EMA weights (val macro-F1 {ema_f1:.4f} > {best_f1:.4f})")
            best_f1 = ema_f1
            best_state = {k: v.detach().cpu().clone() for k, v in ema.ema.state_dict().items()}
            model.load_state_dict(best_state)
        else:
            ema.ema.load_state_dict(best_state)
    return best_f1, best_epoch, best_state


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
    ax.set_title("Test Confusion Matrix (EfficientNet from-scratch)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_and_eval(cfg: TrainConfig, output_dir: Path | None = None) -> dict[str, Any]:
    """Train Stage A→B, one-shot test. Returns metrics dict (includes state_dict)."""
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")
    print(
        f"Config: width={cfg.width_mult} drop={cfg.dropout} depth_drop={cfg.depth_drop} "
        f"depth_mult={cfg.depth_mult} se={cfg.se_ratio} drop_path={cfg.drop_path_rate} "
        f"focal_g={cfg.focal_gamma} img={cfg.image_size} "
        f"lrA={cfg.stage_a_lr} wd={cfg.weight_decay} mix={cfg.mix_prob} "
        f"ra_m={cfg.randaugment_magnitude} bs={cfg.batch_size} seed={cfg.seed}"
    )

    train_loader, val_loader, test_loader = make_loaders(
        batch_size=cfg.batch_size,
        use_randaugment=cfg.randaugment,
        randaugment_magnitude=cfg.randaugment_magnitude,
        image_size=cfg.image_size,
    )

    model = build_efficientnet_b0(
        num_classes=cfg.num_classes,
        pretrained=False,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
        depth_drop=cfg.depth_drop,
        depth_mult=cfg.depth_mult,
        se_ratio=cfg.se_ratio,
        drop_path_rate=cfg.drop_path_rate,
    ).to(device)
    set_backbone_trainable(model, True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    ema = ModelEMA(model, decay=cfg.ema_decay)
    criterion_eval = nn.CrossEntropyLoss()
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "stage": [],
    }

    print("\n=== Stage A: train all layers from scratch ===")
    opt_a = torch.optim.AdamW(
        model.parameters(), lr=cfg.stage_a_lr, weight_decay=cfg.weight_decay
    )
    t0 = max(10, cfg.stage_a_max_epochs // 4)
    sched_a = CosineAnnealingWarmRestarts(opt_a, T_0=t0, T_mult=2)
    best_a, epoch_a, state_a = train_epochs(
        model,
        train_loader,
        val_loader,
        device,
        opt_a,
        criterion_eval,
        cfg,
        max_epochs=cfg.stage_a_max_epochs,
        patience=cfg.stage_a_patience,
        scheduler=sched_a,
        history=history,
        stage_name="A",
        ema=ema,
    )
    print(f"Stage A best val macro-F1={best_a:.4f} @ epoch {epoch_a}")
    model.load_state_dict(state_a)
    ema.ema.load_state_dict(state_a)

    print("\n=== Stage B: fine-tune all layers (lower LR) ===")
    set_backbone_trainable(model, True)
    opt_b = torch.optim.AdamW(
        model.parameters(), lr=cfg.stage_b_lr, weight_decay=cfg.weight_decay
    )
    sched_b = CosineAnnealingLR(opt_b, T_max=cfg.stage_b_max_epochs)
    best_b, epoch_b, best_state = train_epochs(
        model,
        train_loader,
        val_loader,
        device,
        opt_b,
        criterion_eval,
        cfg,
        max_epochs=cfg.stage_b_max_epochs,
        patience=cfg.stage_b_patience,
        scheduler=sched_b,
        history=history,
        stage_name="B",
        ema=ema,
    )
    print(f"Stage B best val macro-F1={best_b:.4f} @ epoch {epoch_b}")
    model.load_state_dict(best_state)

    test_loss, test_acc, test_f1, cm = evaluate(model, test_loader, device, criterion_eval)
    print(f"TEST accuracy={test_acc:.4f} macro_f1={test_f1:.4f}")

    ckpt = {
        "state_dict": best_state,
        "class_names": CLASS_NAMES,
        "seed": cfg.seed,
        "architecture": "efficientnet_b0_from_scratch",
        "width_mult": cfg.width_mult,
        "dropout": cfg.dropout,
        "depth_drop": cfg.depth_drop,
        "depth_mult": cfg.depth_mult,
        "se_ratio": cfg.se_ratio,
        "drop_path_rate": cfg.drop_path_rate,
        "image_size": cfg.image_size,
        "focal_gamma": cfg.focal_gamma,
        "pretrained": None,
        "config": asdict(cfg),
        "label_smoothing": cfg.label_smoothing,
        "weight_decay": cfg.weight_decay,
        "ema_decay": cfg.ema_decay,
        "best_val_macro_f1": best_b,
        "best_stage_b_epoch": epoch_b,
        "stage_a_best_val_macro_f1": best_a,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    }
    ckpt_path = out / cfg.ckpt_name
    torch.save(ckpt, ckpt_path)
    print(f"Saved checkpoint {ckpt_path}")

    if cfg.save_plots:
        plot_curves(history, out / "training_curves.png")
        plot_confusion(cm, out / "confusion_matrix.png")

    meta_path = out / "split_meta.json"
    split_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    metrics: dict[str, Any] = {
        "seed": cfg.seed,
        "device": str(device),
        "architecture": "efficientnet_b0_from_scratch",
        "width_mult": cfg.width_mult,
        "dropout": cfg.dropout,
        "depth_drop": cfg.depth_drop,
        "depth_mult": cfg.depth_mult,
        "se_ratio": cfg.se_ratio,
        "drop_path_rate": cfg.drop_path_rate,
        "image_size": cfg.image_size,
        "focal_gamma": cfg.focal_gamma,
        "params": n_params,
        "pretrained": None,
        "optimizer": "AdamW",
        "config": asdict(cfg),
        "stage_a": {
            "max_epochs": cfg.stage_a_max_epochs,
            "patience": cfg.stage_a_patience,
            "lr": cfg.stage_a_lr,
            "best_val_macro_f1": best_a,
            "best_epoch": epoch_a,
        },
        "stage_b": {
            "max_epochs": cfg.stage_b_max_epochs,
            "patience": cfg.stage_b_patience,
            "lr": cfg.stage_b_lr,
            "best_val_macro_f1": best_b,
            "best_epoch": epoch_b,
        },
        "test": {
            "loss": float(test_loss),
            "accuracy": float(test_acc),
            "macro_f1": float(test_f1),
            "confusion_matrix": cm.tolist(),
            "class_names": CLASS_NAMES,
        },
        "class_counts": split_meta.get("class_counts"),
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "ckpt_path": str(ckpt_path),
    }
    if cfg.write_metrics:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote {out / 'metrics.json'}")

    metrics["state_dict"] = best_state
    return metrics


def main() -> None:
    train_and_eval(full_config())


if __name__ == "__main__":
    main()

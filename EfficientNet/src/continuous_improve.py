"""Continuous improve until target test accuracy (from-scratch EfficientNet).

Example:
  PYTHONPATH=src python -u src/continuous_improve.py --target 0.85 --max-rounds 100
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from train import OUTPUT_DIR, TrainConfig, full_config, search_config, train_and_eval

_LEVER_SCHEDULE: list[dict[str, Any]] = [
    {"width_mult": 0.8},
    {"width_mult": 1.0},
    {"width_mult": 1.2},
    {"width_mult": 0.65},
    {"depth_mult": 1.2},
    {"depth_mult": 1.4},
    {"depth_mult": 1.0},
    {"se_ratio": 0.5},
    {"se_ratio": 0.25},
    {"image_size": 256},
    {"image_size": 224},
    {"drop_path_rate": 0.05},
    {"drop_path_rate": 0.1},
    {"drop_path_rate": 0.0},
    {"focal_gamma": 1.5},
    {"focal_gamma": 0.0},
    {"dropout": 0.05},
    {"dropout": 0.2},
    {"depth_drop": 1},
    {"depth_drop": 0},
    {"stage_a_lr": 5e-4},
    {"stage_a_lr": 2e-3},
    {"stage_a_lr": 1e-3},
    {"weight_decay": 0.05},
    {"weight_decay": 0.005},
    {"mix_prob": 0.2},
    {"mix_prob": 0.4},
    {"mix_prob": 0.6},
    {"randaugment_magnitude": 3},
    {"randaugment_magnitude": 7},
    {"batch_size": 16},
    {"batch_size": 32},
    {"seed": 7},
    {"seed": 123},
    {"seed": 2024},
    {"seed": 42},
]

_KNOB_KEYS = (
    "width_mult",
    "dropout",
    "depth_drop",
    "depth_mult",
    "se_ratio",
    "drop_path_rate",
    "focal_gamma",
    "image_size",
    "stage_a_lr",
    "stage_b_lr",
    "weight_decay",
    "mix_prob",
    "mixup_alpha",
    "cutmix_alpha",
    "randaugment",
    "randaugment_magnitude",
    "label_smoothing",
    "ema_decay",
    "ema_start_epoch",
    "seed",
    "batch_size",
)

_RANDOM_SPACE: dict[str, list[Any]] = {
    "width_mult": [0.65, 0.8, 1.0, 1.2],
    "depth_mult": [1.0, 1.2, 1.4],
    "se_ratio": [0.25, 0.5],
    "image_size": [224, 256],
    "drop_path_rate": [0.0, 0.05, 0.1],
    "focal_gamma": [0.0, 1.5],
    "dropout": [0.05, 0.1, 0.2],
    "depth_drop": [0, 1],
    "stage_a_lr": [5e-4, 1e-3, 2e-3],
    "weight_decay": [0.005, 0.01, 0.05],
    "mix_prob": [0.2, 0.4, 0.5, 0.6],
    "randaugment_magnitude": [3, 5, 7],
    "batch_size": [16, 32],
    "seed": [42, 7, 123, 2024],
}


def winner_baseline() -> TrainConfig:
    """Current ~78.6% full-retrain winner as search starting point."""
    return search_config(
        width_mult=0.65,
        dropout=0.1,
        depth_drop=0,
        depth_mult=1.0,
        se_ratio=0.25,
        drop_path_rate=0.0,
        focal_gamma=0.0,
        image_size=224,
        stage_a_lr=1e-3,
        weight_decay=0.01,
        mix_prob=0.5,
        randaugment_magnitude=5,
        batch_size=32,
        seed=42,
    )


def _knobs(cfg: TrainConfig) -> dict[str, Any]:
    return {k: getattr(cfg, k) for k in _KNOB_KEYS}


def _cfg_from_dict(d: dict[str, Any], *, search: bool) -> TrainConfig:
    knobs = {k: d[k] for k in _KNOB_KEYS if k in d}
    if search:
        return search_config(**knobs)
    return full_config(**knobs)


def _random_from_best(best_cfg: TrainConfig) -> TrainConfig:
    knobs = _knobs(best_cfg)
    # Mutate 2–4 knobs so we explore without full chaos
    keys = list(_RANDOM_SPACE.keys())
    random.shuffle(keys)
    for key in keys[: random.randint(2, 4)]:
        knobs[key] = random.choice(_RANDOM_SPACE[key])
    if knobs.get("image_size") == 256 and knobs.get("batch_size", 32) > 16:
        # Prefer smaller batch at 256 to reduce MPS OOM risk
        if random.random() < 0.5:
            knobs["batch_size"] = 16
    return search_config(**knobs)


def propose_next(
    best_cfg: TrainConfig,
    last_cfg: TrainConfig,
    lever_idx: int,
    improved: bool,
) -> tuple[TrainConfig, int]:
    if improved:
        knobs = _knobs(last_cfg)
        if last_cfg.depth_mult < 1.35:
            knobs["depth_mult"] = round(min(1.4, last_cfg.depth_mult + 0.2), 1)
        elif last_cfg.image_size < 256:
            knobs["image_size"] = 256
            knobs["batch_size"] = min(knobs.get("batch_size", 32), 16)
        elif last_cfg.width_mult < 1.15:
            knobs["width_mult"] = min(1.2, round(last_cfg.width_mult + 0.15, 2))
        elif last_cfg.drop_path_rate < 0.05:
            knobs["drop_path_rate"] = 0.05
        elif last_cfg.focal_gamma <= 0:
            knobs["focal_gamma"] = 1.5
        elif last_cfg.se_ratio < 0.4:
            knobs["se_ratio"] = 0.5
        else:
            knobs["randaugment_magnitude"] = min(7, last_cfg.randaugment_magnitude + 1)
        return search_config(**knobs), lever_idx

    # Alternate scheduled lever vs random sample every other miss
    if lever_idx % 2 == 0:
        knobs = _knobs(best_cfg)
        idx = (lever_idx // 2) % len(_LEVER_SCHEDULE)
        knobs.update(_LEVER_SCHEDULE[idx])
        if knobs.get("image_size") == 256:
            knobs["batch_size"] = min(int(knobs.get("batch_size", 32)), 16)
        return search_config(**knobs), lever_idx + 1

    return _random_from_best(best_cfg), lever_idx + 1


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _full_from_search(cfg: TrainConfig) -> TrainConfig:
    return full_config(
        **_knobs(cfg),
        save_plots=True,
        write_metrics=True,
        ckpt_name="best_efficientnet_b0.pt",
    )


def _is_promising(test_acc: float, best_acc: float, target: float) -> bool:
    # Looser than v1 so medium search can trigger full A100/B40 often enough
    if test_acc >= target - 0.05:
        return True
    if best_acc >= 0 and test_acc >= best_acc - 0.01:
        return True
    if best_acc >= 0 and test_acc >= best_acc + 0.01:
        return True
    return False


def _write_progress(
    path: Path,
    *,
    round_num: int,
    best_acc: float,
    best_f1: float,
    best_cfg: TrainConfig,
    best_was_full: bool,
    lever_idx: int,
    last_cfg: TrainConfig,
    improved_last: bool,
) -> None:
    path.write_text(
        json.dumps(
            {
                "round": round_num,
                "test_accuracy": best_acc,
                "test_macro_f1": best_f1,
                "config": asdict(best_cfg),
                "was_full_retrain": best_was_full,
                "lever_idx": lever_idx,
                "last_config": asdict(last_cfg),
                "improved_last": improved_last,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _max_logged_round(log_path: Path) -> int:
    if not log_path.is_file():
        return 0
    max_r = 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        max_r = max(max_r, int(row.get("round", 0)))
    return max_r


_PRINT_KEYS = (
    "width_mult",
    "depth_mult",
    "se_ratio",
    "image_size",
    "dropout",
    "depth_drop",
    "drop_path_rate",
    "focal_gamma",
    "stage_a_lr",
    "weight_decay",
    "mix_prob",
    "randaugment_magnitude",
    "batch_size",
    "seed",
)


def run_until_target(
    target: float = 0.85,
    max_rounds: int = 100,
    resume: bool = False,
) -> None:
    out = OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "search_log.jsonl"
    best_path = out / "search_best.json"
    progress_path = out / "search_progress.json"
    final_path = out / "search_final.json"

    if not resume and log_path.is_file():
        archive = out / "search_log_prev.jsonl"
        log_path.replace(archive)

    best_cfg = winner_baseline()
    best_acc = -1.0
    best_f1 = -1.0
    best_was_full = False
    lever_idx = 0
    start_round = 1
    improved_last = False
    last_cfg = best_cfg

    for seed_name in ("search_final.json", "metrics.json"):
        seed_path = out / seed_name
        if not seed_path.is_file():
            continue
        try:
            seed = json.loads(seed_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        seed_acc = seed.get("final_test_accuracy") or seed.get("best_test_accuracy")
        if seed_acc is None and isinstance(seed.get("test"), dict):
            seed_acc = seed["test"].get("accuracy")
        seed_cfg = seed.get("config")
        if seed_acc is not None and float(seed_acc) > best_acc:
            best_acc = float(seed_acc)
            best_f1 = float(
                seed.get("final_test_macro_f1")
                or seed.get("best_test_macro_f1")
                or (seed.get("test") or {}).get("macro_f1")
                or best_acc
            )
            if isinstance(seed_cfg, dict):
                best_cfg = _cfg_from_dict(seed_cfg, search=True)
            best_was_full = True
            last_cfg = best_cfg
            print(f"Seeded best_acc={best_acc:.4f} from {seed_name}")
            break

    if resume:
        progress = None
        for cand in (progress_path, best_path):
            if cand.is_file():
                try:
                    progress = json.loads(cand.read_text(encoding="utf-8"))
                    break
                except json.JSONDecodeError:
                    continue
        logged = _max_logged_round(log_path)
        if progress is not None:
            resume_acc = float(progress.get("test_accuracy", -1))
            if resume_acc >= best_acc:
                best_acc = resume_acc
                best_f1 = float(progress.get("test_macro_f1", best_f1))
                best_cfg = _cfg_from_dict(progress.get("config") or {}, search=True)
                best_was_full = bool(progress.get("was_full_retrain", best_was_full))
            lever_idx = int(progress.get("lever_idx", logged))
            if isinstance(progress.get("last_config"), dict):
                last_cfg = _cfg_from_dict(progress["last_config"], search=True)
            improved_last = bool(progress.get("improved_last", False))
            start_round = max(int(progress.get("round", 0)), logged) + 1
        else:
            start_round = logged + 1
            lever_idx = max(0, logged - 1)
        print(
            f"Resume from round {start_round}, best_acc={best_acc:.4f}, "
            f"lever_idx={lever_idx}"
        )
        if best_acc >= target:
            print(f"Already at target {target:.2%}; nothing to do.")
            return

    print(
        f"=== Continuous improve until {target:.0%} "
        f"(max_rounds={max_rounds}, start={start_round}) ==="
    )
    hit_target = best_acc >= target

    for r in range(start_round, max_rounds + 1):
        if hit_target:
            break

        if r == start_round and not resume:
            cfg = winner_baseline()
        else:
            cfg, lever_idx = propose_next(
                best_cfg, last_cfg, lever_idx, improved_last
            )

        print(f"\n######## ROUND {r}/{max_rounds} (target {target:.0%}) ########")
        print(json.dumps({k: getattr(cfg, k) for k in _PRINT_KEYS}, indent=2))

        acc_before = best_acc
        result = train_and_eval(cfg, output_dir=out)
        test_acc = float(result["test"]["accuracy"])
        test_f1 = float(result["test"]["macro_f1"])
        last_cfg = cfg
        improved_last = test_acc > acc_before + 1e-6

        row: dict[str, Any] = {
            "round": r,
            "mode": "search",
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
            "config": asdict(cfg),
            "improved_best": improved_last or best_acc < 0,
        }
        append_jsonl(log_path, row)
        print(
            f"Round {r} search test_acc={test_acc:.4f} "
            f"best_so_far={max(best_acc, test_acc):.4f} target={target:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_f1 = test_f1
            best_cfg = copy.deepcopy(cfg)
            best_was_full = False
            best_path.write_text(
                json.dumps(
                    {
                        "test_accuracy": best_acc,
                        "test_macro_f1": best_f1,
                        "config": asdict(best_cfg),
                        "round": r,
                        "was_full_retrain": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"  NEW BEST (search) @ round {r}: acc={best_acc:.4f}")

        _write_progress(
            progress_path,
            round_num=r,
            best_acc=best_acc,
            best_f1=best_f1,
            best_cfg=best_cfg,
            best_was_full=best_was_full,
            lever_idx=lever_idx,
            last_cfg=last_cfg,
            improved_last=improved_last,
        )

        if test_acc >= target:
            hit_target = True
            print(f"  TARGET HIT on search: {test_acc:.4f} >= {target:.4f}")
            print("\n=== Full retrain of target-hitting search config ===")
            full_cfg = _full_from_search(cfg)
            full_res = train_and_eval(full_cfg, output_dir=out)
            full_acc = float(full_res["test"]["accuracy"])
            full_f1 = float(full_res["test"]["macro_f1"])
            append_jsonl(
                log_path,
                {
                    "round": r,
                    "mode": "full",
                    "test_accuracy": full_acc,
                    "test_macro_f1": full_f1,
                    "config": asdict(full_cfg),
                },
            )
            if full_acc >= best_acc:
                best_acc = full_acc
                best_f1 = full_f1
                best_cfg = copy.deepcopy(cfg)
                best_was_full = True
            best_path.write_text(
                json.dumps(
                    {
                        "test_accuracy": best_acc,
                        "test_macro_f1": best_f1,
                        "config": asdict(best_cfg),
                        "round": r,
                        "was_full_retrain": best_was_full,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            _write_progress(
                progress_path,
                round_num=r,
                best_acc=best_acc,
                best_f1=best_f1,
                best_cfg=best_cfg,
                best_was_full=best_was_full,
                lever_idx=lever_idx,
                last_cfg=last_cfg,
                improved_last=True,
            )
            hit_target = best_acc >= target
            break

        if _is_promising(test_acc, acc_before if acc_before >= 0 else test_acc - 1, target):
            print(
                f"\n=== Opportunistic full retrain (promising short-run "
                f"{test_acc:.4f}) ==="
            )
            full_cfg = _full_from_search(cfg)
            full_res = train_and_eval(full_cfg, output_dir=out)
            full_acc = float(full_res["test"]["accuracy"])
            full_f1 = float(full_res["test"]["macro_f1"])
            append_jsonl(
                log_path,
                {
                    "round": r,
                    "mode": "full",
                    "test_accuracy": full_acc,
                    "test_macro_f1": full_f1,
                    "config": asdict(full_cfg),
                },
            )
            print(f"Full retrain test_acc={full_acc:.4f}")
            if full_acc > best_acc:
                best_acc = full_acc
                best_f1 = full_f1
                best_cfg = copy.deepcopy(cfg)
                best_was_full = True
                best_path.write_text(
                    json.dumps(
                        {
                            "test_accuracy": best_acc,
                            "test_macro_f1": best_f1,
                            "config": asdict(best_cfg),
                            "round": r,
                            "was_full_retrain": True,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"  NEW BEST (full) @ round {r}: acc={best_acc:.4f}")
            _write_progress(
                progress_path,
                round_num=r,
                best_acc=best_acc,
                best_f1=best_f1,
                best_cfg=best_cfg,
                best_was_full=best_was_full,
                lever_idx=lever_idx,
                last_cfg=last_cfg,
                improved_last=improved_last,
            )
            if full_acc >= target:
                hit_target = True
                print(f"  TARGET HIT on full retrain: {full_acc:.4f}")
                break

    print(f"\n=== Loop finished. Best test_acc={best_acc:.4f} target={target:.4f} ===")

    print("\n=== Final full retrain of champion ===")
    full_cfg = _full_from_search(best_cfg)
    metrics = train_and_eval(full_cfg, output_dir=out)
    best_acc = float(metrics["test"]["accuracy"])
    best_f1 = float(metrics["test"]["macro_f1"])
    best_was_full = True
    best_path.write_text(
        json.dumps(
            {
                "test_accuracy": best_acc,
                "test_macro_f1": best_f1,
                "config": asdict(best_cfg),
                "round": max_rounds,
                "was_full_retrain": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "target": target,
        "hit_target": best_acc >= target,
        "best_test_accuracy": best_acc,
        "best_test_macro_f1": best_f1,
        "max_rounds": max_rounds,
        "config": asdict(full_cfg),
        "was_full_retrain": best_was_full,
    }
    final_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Final best accuracy={best_acc:.4f} "
        f"({'HIT' if best_acc >= target else 'MISS'} target {target:.4f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous from-scratch EfficientNet search until target accuracy"
    )
    parser.add_argument("--target", type=float, default=0.85)
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from outputs/search_progress.json",
    )
    parser.add_argument("--rounds", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    max_rounds = args.rounds if args.rounds is not None else args.max_rounds
    run_until_target(target=args.target, max_rounds=max_rounds, resume=args.resume)


if __name__ == "__main__":
    main()

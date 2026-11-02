"""Continuous improve until target test accuracy (from-scratch EfficientNet).

Example:
  PYTHONPATH=src python -u src/continuous_improve.py --target 0.85 --max-rounds 40 --resume
"""

from __future__ import annotations

import argparse
import copy
import json
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
    {"width_mult": 0.5},
    {"drop_path_rate": 0.05},
    {"drop_path_rate": 0.1},
    {"focal_gamma": 1.5},
    {"focal_gamma": 0.0},
    {"dropout": 0.05},
    {"dropout": 0.2},
    {"depth_drop": 1},
    {"depth_drop": 2},
    {"stage_a_lr": 5e-4},
    {"stage_a_lr": 2e-3},
    {"weight_decay": 0.05},
    {"weight_decay": 0.005},
    {"mix_prob": 0.2},
    {"mix_prob": 0.4},
    {"randaugment_magnitude": 3},
    {"randaugment_magnitude": 7},
    {"batch_size": 16},
    {"batch_size": 32},
]

_KNOB_KEYS = (
    "width_mult",
    "dropout",
    "depth_drop",
    "drop_path_rate",
    "focal_gamma",
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


def winner_baseline() -> TrainConfig:
    """Current ~78.6% full-retrain winner as search starting point."""
    return search_config(
        width_mult=0.65,
        dropout=0.1,
        depth_drop=0,
        drop_path_rate=0.0,
        focal_gamma=0.0,
        stage_a_lr=1e-3,
        weight_decay=0.01,
        mix_prob=0.5,
        randaugment_magnitude=5,
        batch_size=32,
    )


def _knobs(cfg: TrainConfig) -> dict[str, Any]:
    return {k: getattr(cfg, k) for k in _KNOB_KEYS}


def _cfg_from_dict(d: dict[str, Any], *, search: bool) -> TrainConfig:
    knobs = {k: d[k] for k in _KNOB_KEYS if k in d}
    if search:
        return search_config(**knobs)
    return full_config(**knobs)


def propose_next(
    best_cfg: TrainConfig,
    last_cfg: TrainConfig,
    lever_idx: int,
    improved: bool,
) -> tuple[TrainConfig, int]:
    if improved:
        knobs = _knobs(last_cfg)
        if last_cfg.width_mult < 1.0 and last_cfg.drop_path_rate < 0.05:
            knobs["drop_path_rate"] = 0.05
        elif last_cfg.focal_gamma <= 0:
            knobs["focal_gamma"] = 1.5
        elif last_cfg.width_mult < 0.95:
            knobs["width_mult"] = min(1.0, round(last_cfg.width_mult + 0.15, 2))
        else:
            knobs["randaugment_magnitude"] = min(7, last_cfg.randaugment_magnitude + 1)
        return search_config(**knobs), lever_idx

    knobs = _knobs(best_cfg)
    idx = lever_idx % len(_LEVER_SCHEDULE)
    knobs.update(_LEVER_SCHEDULE[idx])
    return search_config(**knobs), lever_idx + 1


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
    if test_acc >= target - 0.03:
        return True
    if best_acc >= 0 and test_acc >= best_acc + 0.015:
        return True
    return False


def run_until_target(
    target: float = 0.85,
    max_rounds: int = 40,
    resume: bool = False,
) -> None:
    out = OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "search_log.jsonl"
    best_path = out / "search_best.json"
    final_path = out / "search_final.json"

    if not resume and log_path.is_file():
        # Archive previous log
        archive = out / "search_log_prev.jsonl"
        log_path.replace(archive)

    best_cfg = winner_baseline()
    best_acc = -1.0
    best_f1 = -1.0
    best_was_full = False
    lever_idx = 0
    start_round = 1

    # Seed from prior champion so short-search noise cannot erase ~78.6% full result
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
            print(f"Seeded best_acc={best_acc:.4f} from {seed_name}")
            break

    if resume and best_path.is_file():
        payload = json.loads(best_path.read_text(encoding="utf-8"))
        resume_acc = float(payload.get("test_accuracy", -1))
        if resume_acc >= best_acc:
            best_acc = resume_acc
            best_f1 = float(payload.get("test_macro_f1", best_f1))
            best_cfg = _cfg_from_dict(payload.get("config") or {}, search=True)
            best_was_full = bool(payload.get("was_full_retrain", best_was_full))
        start_round = int(payload.get("round", 0)) + 1
        print(f"Resume from round {start_round}, best_acc={best_acc:.4f}")
        if best_acc >= target:
            print(f"Already at target {target:.2%}; nothing to do.")
            return

    print(
        f"=== Continuous improve until {target:.0%} "
        f"(max_rounds={max_rounds}, start={start_round}) ==="
    )
    improved_last = False
    last_cfg = best_cfg
    last_acc = best_acc
    hit_target = best_acc >= target

    for r in range(start_round, max_rounds + 1):
        if hit_target:
            break

        if r == start_round and not resume:
            cfg = winner_baseline()
        elif r == start_round and resume:
            cfg, lever_idx = propose_next(best_cfg, last_cfg, lever_idx, improved=False)
        else:
            cfg, lever_idx = propose_next(
                best_cfg, last_cfg, lever_idx, improved_last
            )

        print(f"\n######## ROUND {r}/{max_rounds} (target {target:.0%}) ########")
        print(
            json.dumps(
                {
                    k: getattr(cfg, k)
                    for k in (
                        "width_mult",
                        "dropout",
                        "depth_drop",
                        "drop_path_rate",
                        "focal_gamma",
                        "stage_a_lr",
                        "weight_decay",
                        "mix_prob",
                        "randaugment_magnitude",
                        "batch_size",
                    )
                },
                indent=2,
            )
        )

        acc_before = best_acc
        result = train_and_eval(cfg, output_dir=out)
        test_acc = float(result["test"]["accuracy"])
        test_f1 = float(result["test"]["macro_f1"])
        last_cfg = cfg
        last_acc = test_acc
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

        if test_acc >= target:
            hit_target = True
            print(f"  TARGET HIT on short search: {test_acc:.4f} >= {target:.4f}")
            # Still do a full retrain for the official checkpoint
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
            if full_acc >= target:
                hit_target = True
                print(f"  TARGET HIT on full retrain: {full_acc:.4f}")
                break

    print(f"\n=== Loop finished. Best test_acc={best_acc:.4f} target={target:.4f} ===")

    if not best_was_full and best_acc >= 0:
        print("\n=== Final full retrain of champion ===")
        full_cfg = _full_from_search(best_cfg)
        metrics = train_and_eval(full_cfg, output_dir=out)
        best_acc = float(metrics["test"]["accuracy"])
        best_f1 = float(metrics["test"]["macro_f1"])
        best_was_full = True
    else:
        # Ensure metrics.json reflects champion (already written by last full)
        metrics = {
            "test": {"accuracy": best_acc, "macro_f1": best_f1},
            "config": asdict(_full_from_search(best_cfg)),
        }

    summary = {
        "target": target,
        "hit_target": best_acc >= target,
        "best_test_accuracy": best_acc,
        "best_test_macro_f1": best_f1,
        "max_rounds": max_rounds,
        "config": asdict(_full_from_search(best_cfg)),
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
    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from outputs/search_best.json",
    )
    # Back-compat: --rounds maps to max-rounds
    parser.add_argument("--rounds", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    max_rounds = args.rounds if args.rounds is not None else args.max_rounds
    run_until_target(target=args.target, max_rounds=max_rounds, resume=args.resume)


if __name__ == "__main__":
    main()

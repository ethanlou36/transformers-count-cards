from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from blackjackgpt.dataset import load_stream, max_shoe_length
from blackjackgpt.expert.policy import bet_units_for_raw_true_count
from blackjackgpt.sidecar import load_sidecar

_BET_ACC_TARGET = 0.95
_DEPTH_BANDS = [[0, 10], [10, 20], [20, 30], [30, 40], [40, 52]]


def bet_vs_count(sidecar, seed: int = 0) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (
        train_test_split,
    )

    category = sidecar.column("token_category")
    is_bet = category == "BET"
    x = sidecar.column("decision_raw_true_count")[is_bet].astype(np.float64)
    bet_units = sidecar.column("decision_bet_units")[is_bet].astype(np.int64)

    ramp = np.array([bet_units_for_raw_true_count(float(v)) for v in x])
    ramp_consistency = float(np.mean(ramp == bet_units))

    n_classes = int(np.unique(bet_units).size)
    if n_classes < 2:
        return {
            "n_bet_positions": int(is_bet.sum()),
            "n_bet_classes": n_classes,
            "logreg_test_accuracy": None,
            "single_class_note": "flat bet: only one bet token, prediction is trivial",
            "ramp_consistency": ramp_consistency,
            "target": _BET_ACC_TARGET,
            "passes_target": None,
        }

    x = x.reshape(-1, 1)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, bet_units, test_size=0.2, random_state=seed, stratify=bet_units
    )
    model = LogisticRegression(max_iter=2000, C=100.0)
    model.fit(x_tr, y_tr)
    acc = float(model.score(x_te, y_te))
    return {
        "n_bet_positions": int(is_bet.sum()),
        "n_bet_classes": n_classes,
        "logreg_test_accuracy": acc,
        "ramp_consistency": ramp_consistency,
        "target": _BET_ACC_TARGET,
        "passes_target": acc > _BET_ACC_TARGET,
    }


def player_ev(dataset_dir: Path) -> dict[str, Any]:
    shoes = pq.read_table(
        dataset_dir / "shoes.parquet",
        columns=["profit_half_units", "initial_units_wagered", "total_units_wagered"],
    )
    profit_half = np.asarray(shoes.column("profit_half_units").to_numpy())
    init_units = np.asarray(shoes.column("initial_units_wagered").to_numpy())
    total_units = np.asarray(shoes.column("total_units_wagered").to_numpy())

    profit_units = profit_half.sum() / 2.0
    ev_on_initial = float(profit_units / init_units.sum())
    ev_on_action = float(profit_units / total_units.sum())
    return {
        "n_shoes": len(profit_half),
        "total_profit_units": float(profit_units),
        "ev_on_initial_wager_pct": ev_on_initial * 100.0,
        "ev_on_total_action_pct": ev_on_action * 100.0,
        "positive": ev_on_initial > 0.0,
    }


def count_depth_decorrelation(sidecar) -> dict[str, Any]:
    rc = sidecar.column("after_running_count").astype(np.float64)
    dealt = sidecar.column("after_cards_dealt").astype(np.float64)
    pearson = float(np.corrcoef(rc, dealt)[0, 1])
    bands = []
    for lo, hi in _DEPTH_BANDS:
        sel = (dealt >= lo) & (dealt < hi)
        if sel.sum() > 1:
            bands.append(
                {
                    "band": [lo, hi],
                    "n": int(sel.sum()),
                    "running_count_variance": float(np.var(rc[sel])),
                    "running_count_std": float(np.std(rc[sel])),
                }
            )
    return {
        "pearson_running_count_vs_cards_dealt": pearson,
        "abs_corr_low": abs(pearson) < 0.5,
        "per_depth_band": bands,
    }


def validate_signal(dataset_dir: Path, seed: int = 0) -> dict[str, Any]:
    sidecar = load_sidecar(dataset_dir)
    _, offsets = load_stream(dataset_dir)

    bet = bet_vs_count(sidecar, seed=seed)
    ev = player_ev(dataset_dir)
    decorr = count_depth_decorrelation(sidecar)
    max_len = max_shoe_length(offsets)

    warnings: list[str] = []
    if bet["passes_target"] is False:
        warnings.append(
            f"bet-vs-count accuracy {bet['logreg_test_accuracy']:.3f} < "
            f"{_BET_ACC_TARGET}: count signal weaker than designed"
        )
    if not ev["positive"]:
        warnings.append(
            f"player EV {ev['ev_on_initial_wager_pct']:.3f}% is not positive"
        )

    if bet.get("n_bet_classes", 2) >= 2 and bet["ramp_consistency"] < 1.0:
        warnings.append(
            f"expert bet disagrees with frozen ramp on "
            f"{(1 - bet['ramp_consistency']) * 100:.2f}% of positions"
        )

    return {
        "dataset_dir": str(dataset_dir),
        "num_shoes": int(len(offsets) - 1),
        "max_shoe_length_tokens": int(max_len),
        "bet_vs_count": bet,
        "player_ev": ev,
        "count_depth_decorrelation": decorr,
        "warnings": warnings,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate data signal pre-training.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = validate_signal(args.dataset_dir, seed=args.seed)
    out = args.output or (args.dataset_dir / "signal_report.json")
    Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    bet = report["bet_vs_count"]
    ev = report["player_ev"]
    acc = bet["logreg_test_accuracy"]
    acc_str = f"{acc:.4f}" if acc is not None else "n/a (single bet class)"
    print(
        f"[signal] bet-vs-count acc={acc_str} "
        f"(ramp_consistency={bet['ramp_consistency']:.4f})"
    )
    print(f"[signal] player EV={ev['ev_on_initial_wager_pct']:.3f}% on initial wager")
    print(
        f"[signal] corr(count, depth)="
        f"{report['count_depth_decorrelation']['pearson_running_count_vs_cards_dealt']:.3f}"
        f" | max shoe len={report['max_shoe_length_tokens']}"
    )
    for w in report["warnings"]:
        print(f"[signal] WARNING: {w}")
    print(f"[signal] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

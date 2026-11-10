from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from blackjackgpt.checkpoint import load_model
from blackjackgpt.model import load_vocabulary
from blackjackgpt.train import resolve_device

_CARD_PREFIXES = ("P:", "D_UP:", "D_REVEAL:", "D_DRAW:")
_HIGH_RANKS = ("T", "A")


def _rank_groups(vocab):
    from blackjackgpt.blackjack.cards import Rank

    return {
        r.value: [vocab.encode(f"{p}{r.value}") for p in _CARD_PREFIXES] for r in Rank
    }


@torch.no_grad()
def _high_card_belief(model, seg: np.ndarray, pred_pos: int, groups, device) -> float:
    idx = torch.from_numpy(seg[None, :].astype(np.int64)).to(device)
    logits = model(idx)["logits"][0, pred_pos]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    rank_mass = {r: float(probs[ids].sum()) for r, ids in groups.items()}
    total = sum(rank_mass.values())
    if total <= 0:
        return float("nan")
    return sum(rank_mass[r] for r in _HIGH_RANKS) / total


def run_twin_next_card(
    paired_dir: Path, checkpoint: Path, dataset_dir: str, device: torch.device
) -> dict:
    vocab, _ = load_vocabulary(dataset_dir)
    groups = _rank_groups(vocab)
    tokens = np.load(paired_dir / "counterfactual_tokens.npy")
    model, _ = load_model(checkpoint, device)

    deltas, gaps, dev_ids = [], [], []
    with (paired_dir / "counterfactual_pairs.jsonl").open() as handle:
        for line in handle:
            pair = json.loads(line)
            belief = {}
            for side in ("high", "low"):
                m = pair[side]
                seg = tokens[m["token_start"] : m["token_end_exclusive"]]
                belief[side] = _high_card_belief(
                    model, seg, m["prediction_position"], groups, device
                )
            deltas.append(belief["high"] - belief["low"])
            gaps.append(pair["high"]["running_count"] - pair["low"]["running_count"])
            dev_ids.append(pair["deviation_id"])

    deltas_arr = np.array(deltas)
    return {
        "checkpoint": str(checkpoint),
        "n_pairs": len(deltas_arr),
        "mean_high_card_belief_delta": float(np.nanmean(deltas_arr)),
        "frac_positive": float(np.mean(deltas_arr > 0)),
        "per_pair_delta": deltas_arr.tolist(),
        "running_count_gap": [int(g) for g in gaps],
        "deviation_ids": dev_ids,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Twin next-card belief delta.")
    parser.add_argument("--paired-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = resolve_device(args.device)
    report = run_twin_next_card(
        args.paired_dir, args.checkpoint, args.dataset_dir, device
    )
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[twin-nc] mean high-card belief delta="
        f"{report['mean_high_card_belief_delta']:+.4f} "
        f"({report['frac_positive'] * 100:.0f}% positive over {report['n_pairs']} pairs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

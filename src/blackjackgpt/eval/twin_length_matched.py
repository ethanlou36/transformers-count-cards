from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from blackjackgpt.checkpoint import load_model, resolve_checkpoint
from blackjackgpt.eval.twin_next_card import run_twin_next_card
from blackjackgpt.model import load_vocabulary
from blackjackgpt.train import resolve_device

_ACTIONS = ("HIT", "STAND", "DOUBLE")


@torch.no_grad()
def _action_lean(model, seg, pred_pos, dev_id, basic_id, device) -> float:
    idx = torch.from_numpy(seg[None, :].astype(np.int64)).to(device)
    logits = model(idx)["logits"][0, pred_pos]
    return float(logits[dev_id].item() - logits[basic_id].item())


def run_twin_length_matched(
    paired_dir: Path, run_dir: Path, dataset_dir: str, device: torch.device
) -> dict:
    vocab, _ = load_vocabulary(dataset_dir)
    action_id = {a: int(vocab.encode(f"ACTION:{a}")) for a in _ACTIONS}
    tokens = np.load(paired_dir / "counterfactual_tokens.npy")
    model, _ = load_model(resolve_checkpoint(run_dir), device)

    pairs = [
        json.loads(line) for line in (paired_dir / "counterfactual_pairs.jsonl").open()
    ]
    len_gaps, pred_gaps = [], []
    matched_correct: list[bool] = []
    n_deviation = 0
    for p in pairs:
        hi, lo = p["high"], p["low"]
        lh = hi["token_end_exclusive"] - hi["token_start"]
        ll = lo["token_end_exclusive"] - lo["token_start"]
        len_gaps.append(lh - ll)
        pred_gaps.append(hi["prediction_position"] - lo["prediction_position"])
        matched = (lh == ll) and (
            hi["prediction_position"] == lo["prediction_position"]
        )
        dev_act, basic_act = hi["expert_action"], lo["expert_action"]
        if dev_act == basic_act:
            continue
        n_deviation += 1
        if not matched:
            continue
        leans = {}
        for side, m in (("high", hi), ("low", lo)):
            seg = tokens[m["token_start"] : m["token_end_exclusive"]]
            leans[side] = _action_lean(
                model,
                seg,
                m["prediction_position"],
                action_id[dev_act],
                action_id[basic_act],
                device,
            )
        matched_correct.append((leans["high"] - leans["low"]) > 0)

    n_matched = sum(1 for g in len_gaps if g == 0)
    mc = np.array(matched_correct)

    belief = run_twin_next_card(
        paired_dir, resolve_checkpoint(run_dir), dataset_dir, device
    )
    n_dir_matched = len(mc)
    dir_acc = float(mc.mean()) if n_dir_matched else float("nan")
    return {
        "run_dir": str(run_dir),
        "n_pairs_total": len(pairs),
        "n_length_matched": int(n_matched),
        "frac_length_matched": float(n_matched / max(1, len(pairs))),
        "length_gap_distribution": {
            int(k): int(v) for k, v in zip(*np.unique(len_gaps, return_counts=True))
        },
        "prediction_position_gap_distribution": {
            int(k): int(v) for k, v in zip(*np.unique(pred_gaps, return_counts=True))
        },
        "n_deviation_pairs_matched": int(n_dir_matched),
        "directional_accuracy_matched": dir_acc,
        "belief_frac_positive_matched": belief["frac_positive"],
        "belief_mean_delta_matched": belief["mean_high_card_belief_delta"],
        "verdict": (
            "length-matched by construction; twin result clean of length cues"
            if n_matched == len(pairs) and dir_acc >= 0.85
            else (
                "data request: too few length-matched pairs; generate "
                "length-matched counterfactuals"
                if n_dir_matched < 30
                else "partially matched; see numbers"
            )
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Length-matched twins control (B3).")
    p.add_argument("--paired-dir", default="runs/paired-lambda1")
    p.add_argument("--run-dir", default="runs/full-latest")
    p.add_argument("--dataset-dir", default="data/merged")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = resolve_device(args.device or "cpu")
    report = run_twin_length_matched(
        Path(args.paired_dir), Path(args.run_dir), args.dataset_dir, device
    )
    out = Path(args.out) if args.out else Path(args.paired_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "twin_length_matched.json").write_text(json.dumps(report, indent=2))
    print(
        f"[b3] {report['n_length_matched']}/{report['n_pairs_total']} length-matched; "
        f"directional accuracy on matched deviation pairs "
        f"({report['n_deviation_pairs_matched']}) = "
        f"{report['directional_accuracy_matched']:.3f}"
    )
    print(f"[b3] verdict: {report['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

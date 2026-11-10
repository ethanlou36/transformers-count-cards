from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from blackjackgpt.checkpoint import load_model, resolve_checkpoint
from blackjackgpt.config import ExperimentConfig, load_config
from blackjackgpt.dataset import load_stream, split_by_shoe
from blackjackgpt.eval.transcript import TranscriptTracker
from blackjackgpt.expert.policy import bet_ramp_entry_for_raw_true_count
from blackjackgpt.model import GPT, load_vocabulary
from blackjackgpt.sidecar import load_sidecar
from blackjackgpt.tokens import TokenClass, TokenTables
from blackjackgpt.train import resolve_device

_BET_UNITS = (1, 2, 4, 6, 8)
_BET_UNIT_INDEX = {u: i for i, u in enumerate(_BET_UNITS)}


@torch.no_grad()
def _shoe_logits(model: GPT, seq: np.ndarray, device: torch.device) -> torch.Tensor:
    idx = torch.from_numpy(seq[None, :].astype(np.int64)).to(device)
    return model(idx)["logits"][0]


@dataclass
class LegalityResult:
    legality_rate: float
    legal_prob_mass: float
    positions: int
    samples: int


def legality_eval(
    model: GPT,
    tokens: np.ndarray,
    offsets: np.ndarray,
    shoe_ids: np.ndarray,
    vocab,
    device: torch.device,
    samples_per_position: int,
    temperature: float,
    seed: int,
) -> LegalityResult:
    tracker = TranscriptTracker(vocab)
    rng = torch.Generator().manual_seed(seed)
    legal_hits = 0
    total_samples = 0
    prob_mass_sum = 0.0
    positions = 0

    for shoe_id in shoe_ids:
        seq = tokens[offsets[shoe_id] : offsets[shoe_id + 1]]
        logits = _shoe_logits(model, seq, device).float().cpu()
        probs = torch.softmax(logits / temperature, dim=-1)
        tracker.reset()
        length = len(seq)
        for t in range(length - 1):
            tracker.advance(int(seq[t]))
            legal_ids = tracker.legal_next()
            if not legal_ids:
                continue
            legal_idx = torch.tensor(sorted(legal_ids), dtype=torch.long)
            dist = probs[t]
            prob_mass_sum += float(dist[legal_idx].sum())
            sampled = torch.multinomial(
                dist, samples_per_position, replacement=True, generator=rng
            )
            legal_set = set(legal_ids)
            legal_hits += sum(int(s) in legal_set for s in sampled.tolist())
            total_samples += samples_per_position
            positions += 1

    return LegalityResult(
        legality_rate=legal_hits / max(1, total_samples),
        legal_prob_mass=prob_mass_sum / max(1, positions),
        positions=positions,
        samples=total_samples,
    )


@dataclass
class BetCalibration:
    accuracy: float
    n: int
    confusion: list[list[int]]
    count_bucket_labels: list[str]
    bet_labels: list[str]


def bet_calibration_eval(
    model: GPT,
    tokens: np.ndarray,
    offsets: np.ndarray,
    shoe_ids: np.ndarray,
    vocab,
    tables: TokenTables,
    sidecar,
    device: torch.device,
) -> BetCalibration:
    round_start_id = vocab.encode("ROUND_START")
    bet_ids = np.array(sorted(tables.ids_in_class(TokenClass.BET)), dtype=np.int64)
    bet_units_of = {vocab.encode(f"BET_{u}"): u for u in _BET_UNITS}

    dec_bet = sidecar.column("decision_bet_units")
    dec_tc = sidecar.column("decision_raw_true_count")

    confusion = np.zeros((len(_BET_UNITS), len(_BET_UNITS)), dtype=np.int64)
    correct = 0
    n = 0
    for shoe_id in shoe_ids:
        start, end = int(offsets[shoe_id]), int(offsets[shoe_id + 1])
        seq = tokens[start:end]
        logits = _shoe_logits(model, seq, device).float().cpu().numpy()
        for p in np.nonzero(seq == round_start_id)[0]:
            bet_offset = start + p + 1

            pred_id = int(bet_ids[np.argmax(logits[p, bet_ids])])
            pred_units = bet_units_of[pred_id]
            expert_units = int(dec_bet[bet_offset])
            count_bucket, _ = bet_ramp_entry_for_raw_true_count(
                float(dec_tc[bet_offset])
            )
            confusion[_BET_UNIT_INDEX[pred_units], count_bucket] += 1
            correct += int(pred_units == expert_units)
            n += 1

    return BetCalibration(
        accuracy=correct / max(1, n),
        n=n,
        confusion=confusion.tolist(),
        count_bucket_labels=["<1", "[1,2)", "[2,3)", "[3,4)", ">=4"],
        bet_labels=[f"BET_{u}" for u in _BET_UNITS],
    )


def run_behavior(exp: ExperimentConfig, run_dir: Path, device: torch.device) -> dict:
    vocab, _ = load_vocabulary(exp.data.dataset_dir)
    tables = TokenTables(vocab)
    tokens, offsets = load_stream(exp.data.dataset_dir)
    split = split_by_shoe(offsets, exp.train.val_frac, exp.train.split_seed)
    shoe_ids = split.val
    if exp.eval.max_shoes:
        shoe_ids = shoe_ids[: exp.eval.max_shoes]

    model, _ = load_model(resolve_checkpoint(run_dir), device)

    legality = legality_eval(
        model,
        tokens,
        offsets,
        shoe_ids,
        vocab,
        device,
        exp.eval.samples_per_position,
        exp.eval.sample_temperature,
        exp.eval.seed,
    )
    sidecar = load_sidecar(exp.data.dataset_dir)
    calibration = bet_calibration_eval(
        model, tokens, offsets, shoe_ids, vocab, tables, sidecar, device
    )

    return {
        "n_shoes": len(shoe_ids),
        "legality": {
            "legality_rate": legality.legality_rate,
            "legal_prob_mass": legality.legal_prob_mass,
            "positions": legality.positions,
            "samples": legality.samples,
            "target": 0.99,
            "passes_target": legality.legality_rate > 0.99,
        },
        "bet_calibration": {
            "accuracy": calibration.accuracy,
            "n": calibration.n,
            "confusion_pred_by_countbucket": calibration.confusion,
            "count_bucket_labels": calibration.count_bucket_labels,
            "bet_labels": calibration.bet_labels,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Behavioral evals for the GPT.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    exp = load_config(args.config)
    if args.seed is not None:
        exp.eval.seed = args.seed
    run_dir = Path(args.run_dir) if args.run_dir else exp.run_dir
    device = resolve_device(args.device or exp.train.device)

    report = run_behavior(exp, run_dir, device)
    out = run_dir / "behavior_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    leg = report["legality"]
    cal = report["bet_calibration"]
    print(
        f"[behavior] legality_rate={leg['legality_rate']:.4f} "
        f"(prob_mass={leg['legal_prob_mass']:.4f}) over {leg['positions']} positions"
    )
    print(f"[behavior] bet accuracy={cal['accuracy']:.4f} over {cal['n']} bets")
    print(f"[behavior] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

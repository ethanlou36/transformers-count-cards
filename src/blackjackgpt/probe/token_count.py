from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from blackjackgpt.checkpoint import load_model, resolve_checkpoint
from blackjackgpt.config import load_config
from blackjackgpt.dataset import load_stream, split_by_shoe
from blackjackgpt.model import GPT, load_vocabulary
from blackjackgpt.probe.activations import build_activation_cache
from blackjackgpt.probe.probes import split_positions_by_shoe
from blackjackgpt.probe.regression import held_out_r2
from blackjackgpt.tokens import TokenTables
from blackjackgpt.train import resolve_device, seed_everything


def running_count_at(
    tokens: np.ndarray,
    offsets: np.ndarray,
    global_offsets: np.ndarray,
    shoe_index: np.ndarray,
    tables: TokenTables,
) -> np.ndarray:
    weight = np.array(tables.hilo_weight, dtype=np.int64)
    running = np.cumsum(weight[tokens])
    out = np.empty(len(global_offsets), dtype=np.float64)
    for shoe_id in np.unique(shoe_index):
        start = int(offsets[shoe_id])
        base = running[start] - weight[tokens[start]]
        sel = shoe_index == shoe_id
        out[sel] = running[global_offsets[sel]] - base
    return out


def probe_running_count(exp, run_dir: Path, device: torch.device) -> dict:
    vocab, _pad_id = load_vocabulary(exp.data.dataset_dir)
    tables = TokenTables(vocab)
    tokens, offsets = load_stream(exp.data.dataset_dir)
    split = split_by_shoe(offsets, exp.train.val_frac, exp.train.split_seed)
    shoe_ids = split.val
    if exp.probe.max_shoes:
        shoe_ids = shoe_ids[: exp.probe.max_shoes]
    alphas = exp.probe.alphas

    trained, gpt_cfg = load_model(resolve_checkpoint(run_dir), device)
    seed_everything(0)
    random_model = GPT(gpt_cfg).to(device).eval()

    def r2_by_layer(model, out_name):
        cache = build_activation_cache(
            model,
            tokens,
            offsets,
            shoe_ids,
            tables,
            run_dir / "probe" / out_name,
            device,
            "card",
        )
        y = running_count_at(
            tokens,
            offsets,
            cache.positions.global_offset,
            cache.positions.shoe_index,
            tables,
        )
        sp = split_positions_by_shoe(
            cache.positions.shoe_index,
            exp.probe.probe_eval_frac,
            exp.probe.probe_split_seed,
        )
        tr, ev = sp.train_mask, sp.eval_mask
        return [
            held_out_r2(cache.layer(i)[tr], y[tr], cache.layer(i)[ev], y[ev], alphas)
            for i in range(cache.n_layer)
        ]

    trained_r2 = r2_by_layer(trained, "token_count_trained")
    random_r2 = r2_by_layer(random_model, "token_count_random")
    return {
        "dataset_dir": exp.data.dataset_dir,
        "n_held_out_shoes": len(shoe_ids),
        "trained_running_count_r2_by_layer": trained_r2,
        "random_init_running_count_r2_by_layer": random_r2,
        "trained_best": float(np.nanmax(trained_r2)),
        "random_best": float(np.nanmax(random_r2)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sidecar-free running-count probe.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    exp = load_config(args.config)
    run_dir = Path(args.run_dir)
    device = resolve_device(args.device or exp.train.device)
    report = probe_running_count(exp, run_dir, device)
    out = run_dir / "probe" / "token_count_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"[token-count] trained best R^2={report['trained_best']:.3f} "
        f"random-init best R^2={report['random_best']:.3f}"
    )
    print(f"[token-count] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

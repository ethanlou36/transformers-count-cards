from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from blackjackgpt.checkpoint import load_model, resolve_checkpoint
from blackjackgpt.config import load_config
from blackjackgpt.dataset import load_stream, split_by_shoe
from blackjackgpt.model import load_vocabulary
from blackjackgpt.probe.activations import build_activation_cache
from blackjackgpt.probe.alignment import load_labels, verify_alignment
from blackjackgpt.probe.probes import combine_report, plot_report, run_probe_class
from blackjackgpt.sidecar import load_sidecar
from blackjackgpt.tokens import TokenTables
from blackjackgpt.train import resolve_device

_KIND_STATE = {"card": "after", "bet": "before"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run count-decodability probes.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    exp = load_config(args.config)
    if args.seed is not None:
        exp.probe.probe_split_seed = args.seed
    run_dir = Path(args.run_dir) if args.run_dir else exp.run_dir
    device = resolve_device(args.device or exp.train.device)

    vocab, _pad_id = load_vocabulary(exp.data.dataset_dir)
    tables = TokenTables(vocab)
    tokens, offsets = load_stream(exp.data.dataset_dir)
    split = split_by_shoe(offsets, exp.train.val_frac, exp.train.split_seed)
    shoe_ids = split.val
    if exp.probe.max_shoes:
        shoe_ids = shoe_ids[: exp.probe.max_shoes]

    model, _ = load_model(resolve_checkpoint(run_dir), device)
    sidecar = load_sidecar(exp.data.dataset_dir)
    probe_dir = run_dir / "probe"

    class_payloads: dict[str, dict] = {}
    for kind, state in _KIND_STATE.items():
        verify_alignment(tokens, offsets, shoe_ids, vocab, tables, sidecar, kind)
        cache = build_activation_cache(
            model, tokens, offsets, shoe_ids, tables, probe_dir / kind, device, kind
        )
        labels = load_labels(cache.positions, sidecar, state)
        print(
            f"[probe] {kind}: alignment verified, "
            f"{len(cache.positions)} positions cached"
        )
        class_payloads[kind] = run_probe_class(
            cache,
            labels,
            alphas=exp.probe.alphas,
            eval_frac=exp.probe.probe_eval_frac,
            split_seed=exp.probe.probe_split_seed,
            depth_bands=exp.probe.depth_bands,
        )

    report = combine_report(
        class_payloads,
        {
            "dataset_dir": exp.data.dataset_dir,
            "n_held_out_shoes": len(shoe_ids),
            "checkpoint": str(resolve_checkpoint(run_dir)),
        },
    )
    report.save_json(probe_dir / "probe_report.json")
    plot_report(report, probe_dir / "probe_r2.png")

    for kind, payload in class_payloads.items():
        best = payload["best_layer"]["running_count"]
        partial = payload["controls"]["running_count_partial_out_cards_dealt"]
        print(
            f"[probe] {kind}: running_count best R^2={best['r2']:.3f} @ layer "
            f"{best['layer']} | partial-out-depth max R^2={np.nanmax(partial):.3f}"
        )
    print(f"[probe] wrote {probe_dir / 'probe_report.json'} and probe_r2.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

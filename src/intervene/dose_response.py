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
from blackjackgpt.probe.alignment import Positions, load_labels, select_positions
from blackjackgpt.probe.probes import split_positions_by_shoe
from blackjackgpt.probe.regression import fit_ridge
from blackjackgpt.sidecar import load_sidecar
from blackjackgpt.tokens import TokenTables

_BET_UNITS = (1, 2, 4, 6, 8)


def _round_start_positions(tokens, offsets, shoe_ids, vocab) -> Positions:
    rs_id = np.array([vocab.encode("ROUND_START")], dtype=np.int64)
    return select_positions(tokens, offsets, shoe_ids, rs_id)


@torch.no_grad()
def _expected_bet_under_patch(
    model: GPT,
    tokens: np.ndarray,
    offsets: np.ndarray,
    shoe_ids: np.ndarray,
    layer_block_idx: int | None,
    delta: torch.Tensor | None,
    bet_ids: np.ndarray,
    bet_units: np.ndarray,
    device: torch.device,
    pad_id: int,
    round_start_id: int,
) -> float:
    handle = None
    if delta is not None and layer_block_idx is not None:
        mask_holder: dict[str, torch.Tensor] = {}

        def hook(_module, _inp, out):
            m = mask_holder["mask"]
            return out + m.unsqueeze(-1) * delta

        handle = model.blocks[layer_block_idx].register_forward_hook(hook)

    exp_bets: list[float] = []
    try:
        for start in range(0, len(shoe_ids), 16):
            batch = shoe_ids[start : start + 16]
            seqs = [tokens[offsets[s] : offsets[s + 1]] for s in batch]
            max_len = max(len(s) for s in seqs)
            idx = np.full((len(batch), max_len), pad_id, dtype=np.int64)
            rs_mask = np.zeros((len(batch), max_len), dtype=bool)
            for i, seq in enumerate(seqs):
                idx[i, : len(seq)] = seq
                rs_mask[i, : len(seq)] = seq == round_start_id
            idx_t = torch.from_numpy(idx).to(device)
            if delta is not None and layer_block_idx is not None:
                mask_holder["mask"] = torch.from_numpy(rs_mask).to(
                    device, dtype=delta.dtype
                )
            logits = model(idx_t)["logits"]
            bet_logits = logits[:, :, bet_ids]
            probs = torch.softmax(bet_logits, dim=-1).cpu().numpy()
            units = np.asarray(bet_units, dtype=np.float64)
            eb = probs @ units
            exp_bets.extend(eb[rs_mask].tolist())
    finally:
        if handle is not None:
            handle.remove()
    return float(np.mean(exp_bets)) if exp_bets else float("nan")


def run_intervention(
    exp,
    run_dir: Path,
    device: torch.device,
    n_shoes: int = 300,
    count_grid: tuple[int, ...] = tuple(range(-5, 6)),
) -> dict:
    vocab, pad_id = load_vocabulary(exp.data.dataset_dir)
    tables = TokenTables(vocab)
    tokens, offsets = load_stream(exp.data.dataset_dir)
    split = split_by_shoe(offsets, exp.train.val_frac, exp.train.split_seed)
    shoe_ids = split.val[:n_shoes]
    sidecar = load_sidecar(exp.data.dataset_dir)
    model, _ = load_model(resolve_checkpoint(run_dir), device)
    alphas = exp.probe.alphas
    n_block = model.config.n_layer

    bet_ids = np.array([vocab.encode(f"BET_{u}") for u in _BET_UNITS], dtype=np.int64)
    bet_units = np.array(_BET_UNITS, dtype=np.float64)
    round_start_id = int(vocab.encode("ROUND_START"))

    positions = _round_start_positions(tokens, offsets, shoe_ids, vocab)
    cache = build_activation_cache(
        model,
        tokens,
        offsets,
        shoe_ids,
        tables,
        run_dir / "probe" / "intervene",
        device,
        "bet",
        positions=positions,
    )
    labels = load_labels(cache.positions, sidecar, "before")
    rc = labels.running_count
    split_p = split_positions_by_shoe(
        cache.positions.shoe_index,
        exp.probe.probe_eval_frac,
        exp.probe.probe_split_seed,
    )
    tr, ev = split_p.train_mask, split_p.eval_mask

    per_layer = [
        fit_ridge(cache.layer(i)[tr], rc[tr], cache.layer(i)[ev], rc[ev], alphas).r2
        for i in range(n_block)
    ]
    best_layer = int(np.nanargmax(per_layer))
    fit = fit_ridge(
        cache.layer(best_layer)[tr], rc[tr], cache.layer(best_layer)[ev], rc[ev], alphas
    )
    w = fit.coef.astype(np.float64)
    w_norm2 = float(w @ w)

    probe_delta_unit = w / w_norm2
    hi = rc >= np.median(rc)
    md = cache.layer(best_layer)[hi].mean(0) - cache.layer(best_layer)[~hi].mean(0)
    md_gain = float(w @ md)
    md_delta_unit = md / md_gain if md_gain != 0 else md * 0.0

    md_step_norm = float(np.linalg.norm(md_delta_unit))
    probe_unit = w / (np.linalg.norm(w) + 1e-12)
    probe_matched_unit = probe_unit * md_step_norm
    rng = np.random.default_rng(0)
    rand = rng.standard_normal(w.shape)
    rand *= md_step_norm / (np.linalg.norm(rand) + 1e-12)

    def sweep(delta_unit: np.ndarray, block_idx: int) -> list[float]:
        out = []
        for c in count_grid:
            delta = torch.from_numpy((c * delta_unit).astype(np.float32)).to(device)
            out.append(
                _expected_bet_under_patch(
                    model,
                    tokens,
                    offsets,
                    shoe_ids,
                    block_idx,
                    delta,
                    bet_ids,
                    bet_units,
                    device,
                    pad_id,
                    round_start_id,
                )
            )
        return out

    baseline = _expected_bet_under_patch(
        model,
        tokens,
        offsets,
        shoe_ids,
        None,
        None,
        bet_ids,
        bet_units,
        device,
        pad_id,
        round_start_id,
    )
    curves = {
        "mean_difference": sweep(md_delta_unit, best_layer),
        "probe_direction": sweep(probe_delta_unit, best_layer),
        "probe_direction_normmatched": sweep(probe_matched_unit, best_layer),
        "random_direction": sweep(rand, best_layer),
        "mean_difference_last_layer": sweep(md_delta_unit, n_block - 1),
    }
    delta_norms = {
        "mean_difference": md_step_norm,
        "probe_direction": float(np.linalg.norm(probe_delta_unit)),
        "probe_direction_normmatched": md_step_norm,
        "random_direction": md_step_norm,
    }

    ref_counts = sorted({int(x) for x in count_grid})
    dec_bet = sidecar.column("decision_bet_units")

    bet_offsets = cache.positions.global_offset + 1
    expert_bet = dec_bet[bet_offsets].astype(np.float64)
    empirical = []
    for c in ref_counts:
        sel = np.round(rc) == c
        empirical.append(float(expert_bet[sel].mean()) if sel.any() else float("nan"))

    def slope(curve: list[float]) -> float:
        x = np.array(count_grid, dtype=np.float64)
        y = np.array(curve, dtype=np.float64)
        ok = np.isfinite(y)
        return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() > 1 else float("nan")

    return {
        "best_layer": best_layer + 1,
        "running_count_r2_by_layer": per_layer,
        "count_grid": list(count_grid),
        "baseline_expected_bet": baseline,
        "curves": curves,
        "slopes": {k: slope(v) for k, v in curves.items()},
        "delta_activation_norms_per_step": delta_norms,
        "empirical_reference": {"counts": ref_counts, "expert_bet": empirical},
        "n_round_start_positions": len(cache.positions),
    }


def plot_intervention(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = report["count_grid"]
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "mean_difference": ("tab:blue", "o", "-"),
        "probe_direction": ("tab:red", "v", "-"),
        "probe_direction_normmatched": ("tab:orange", "^", "-"),
        "random_direction": ("gray", "x", ":"),
        "mean_difference_last_layer": ("tab:green", "s", "--"),
    }
    for name, curve in report["curves"].items():
        color, marker, ls = styles.get(name, ("black", ".", "-"))
        ax.plot(
            g,
            curve,
            marker=marker,
            ls=ls,
            color=color,
            label=f"{name} (slope {report['slopes'][name]:.2f})",
        )
    ref = report["empirical_reference"]
    ax.plot(
        ref["counts"],
        ref["expert_bet"],
        marker="D",
        color="tab:green",
        label="empirical expert bet vs count",
    )
    ax.set_xlabel("implied running-count shift (α, calibrated to count units)")
    ax.set_ylabel("expected bet size (units)")
    ax.set_title(f"Bet dose-response to count patching (layer {report['best_layer']})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Count-patching dose-response.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--n-shoes", type=int, default=300)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    from blackjackgpt.train import resolve_device

    args = build_arg_parser().parse_args(argv)
    exp = load_config(args.config)
    run_dir = Path(args.run_dir)
    device = resolve_device(args.device or exp.train.device)
    report = run_intervention(exp, run_dir, device, n_shoes=args.n_shoes)
    out = run_dir / "probe" / "intervention_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_intervention(report, run_dir / "probe" / "intervention.png")
    print(
        f"[intervene] baseline bet={report['baseline_expected_bet']:.2f} | "
        f"slopes: probe={report['slopes']['probe_direction']:.2f} "
        f"random={report['slopes']['random_direction']:.2f}"
    )
    print(f"[intervene] wrote {out} and intervention.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from blackjackgpt.checkpoint import load_model, resolve_checkpoint
from blackjackgpt.config import load_config
from blackjackgpt.dataset import load_stream, split_by_shoe
from blackjackgpt.model import load_vocabulary
from blackjackgpt.probe.activations import build_activation_cache
from blackjackgpt.probe.alignment import load_labels
from blackjackgpt.probe.counting_systems import COUNTING_SYSTEMS, RANKS
from blackjackgpt.probe.effect_of_removal import eor_entropy_weights, eor_ev_weights
from blackjackgpt.probe.probes import split_positions_by_shoe
from blackjackgpt.probe.regression import cosine, fit_ridge
from blackjackgpt.sidecar import load_sidecar
from blackjackgpt.tokens import TokenTables
from blackjackgpt.train import resolve_device

_BET_UNITS = (1, 2, 4, 6, 8)


def _center(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v - v.mean()


def _ccos(a: np.ndarray, b: np.ndarray) -> float:
    return cosine(_center(a), _center(b))


def _sign_align(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return -v if float(_center(v) @ _center(ref)) < 0 else v


@torch.no_grad()
def _expected_bet(model, idx_t, rs_mask_t, bet_ids, delta, block_idx):
    handle = None
    if delta is not None:

        def hook(_m, _i, out):
            return out + rs_mask_t.unsqueeze(-1) * delta

        handle = model.blocks[block_idx].register_forward_hook(hook)
    try:
        logits = model(idx_t)["logits"][:, :, bet_ids]
        probs = torch.softmax(logits, dim=-1)
        units = torch.tensor(_BET_UNITS, dtype=probs.dtype, device=probs.device)
        eb = (probs * units).sum(-1)
        return eb[rs_mask_t.bool()].mean().item()
    finally:
        if handle is not None:
            handle.remove()


def run_weight_audit(
    exp, run_dir: Path, device: torch.device, n_shoes: int = 800, alpha: float = 2.0
) -> dict:
    vocab, pad_id = load_vocabulary(exp.data.dataset_dir)
    tables = TokenTables(vocab)
    tokens, offsets = load_stream(exp.data.dataset_dir)
    split = split_by_shoe(offsets, exp.train.val_frac, exp.train.split_seed)
    shoe_ids = split.val[:n_shoes]
    sidecar = load_sidecar(exp.data.dataset_dir)
    model, _ = load_model(resolve_checkpoint(run_dir), device)
    alphas = exp.probe.alphas
    round_start_id = int(vocab.encode("ROUND_START"))
    bet_ids = np.array([vocab.encode(f"BET_{u}") for u in _BET_UNITS], dtype=np.int64)

    from blackjackgpt.intervene.dose_response import _round_start_positions

    positions = _round_start_positions(tokens, offsets, shoe_ids, vocab)
    cache = build_activation_cache(
        model,
        tokens,
        offsets,
        shoe_ids,
        tables,
        run_dir / "probe" / "weight_audit",
        device,
        "bet",
        positions=positions,
    )
    labels = load_labels(cache.positions, sidecar, "before")
    comp = labels.remaining_composition
    split_p = split_positions_by_shoe(
        cache.positions.shoe_index,
        exp.probe.probe_eval_frac,
        exp.probe.probe_split_seed,
    )
    tr, ev = split_p.train_mask, split_p.eval_mask

    comp_r2 = [
        fit_ridge(cache.layer(i)[tr], comp[tr], cache.layer(i)[ev], comp[ev], alphas).r2
        for i in range(model.config.n_layer)
    ]
    best_layer = int(np.nanargmax(comp_r2))
    fit = fit_ridge(
        cache.layer(best_layer)[tr],
        comp[tr],
        cache.layer(best_layer)[ev],
        comp[ev],
        alphas,
    )
    W = fit.coef

    U, svals, _ = np.linalg.svd(W, full_matrices=False)
    svd_weights = U[:, 0]

    from sklearn.linear_model import Ridge

    enc = Ridge(alpha=10.0).fit(comp[tr], cache.layer(best_layer)[tr])
    encoding = np.asarray(enc.coef_)
    sens = _bet_sensitivity(
        model,
        tokens,
        offsets,
        shoe_ids,
        best_layer,
        encoding,
        bet_ids,
        device,
        pad_id,
        round_start_id,
        alpha,
    )

    used_weights = -sens

    hilo = COUNTING_SYSTEMS["hi_lo"]
    vectors = {
        "svd_top_mode": _sign_align(svd_weights, hilo),
        "bet_sensitivity_used": _sign_align(used_weights, hilo),
        "hi_lo": hilo,
        "ko": COUNTING_SYSTEMS["ko"],
        "hi_opt_i": COUNTING_SYSTEMS["hi_opt_i"],
        "eor_ev": _sign_align(eor_ev_weights(exp.data.dataset_dir), hilo),
        "eor_entropy": _sign_align(eor_entropy_weights(), hilo),
    }
    names = list(vectors)

    table = {
        a: {b: round(_ccos(vectors[a], vectors[b]), 4) for b in names} for a in names
    }

    svd_mode_cos_hilo = [round(_ccos(U[:, i], hilo), 4) for i in range(U.shape[1])]

    def norm_to_hilo(v):
        cv, ch = _center(v), _center(hilo)
        s = float(cv @ ch) / float(ch @ ch)
        return (cv / s if s else cv).tolist()

    return {
        "run_dir": str(run_dir),
        "best_composition_layer": best_layer + 1,
        "composition_r2_by_layer": comp_r2,
        "svd_singular_values": svals.tolist(),
        "ranks": [r.value for r in RANKS],
        "derived_weights": {
            "svd_top_mode": vectors["svd_top_mode"].tolist(),
            "bet_sensitivity_used": vectors["bet_sensitivity_used"].tolist(),
        },
        "reference_weights": {
            "hi_lo": hilo.tolist(),
            "eor_ev": vectors["eor_ev"].tolist(),
            "eor_entropy": vectors["eor_entropy"].tolist(),
        },
        "weights_centered_scaled_to_hilo": {
            "bet_sensitivity_used": norm_to_hilo(vectors["bet_sensitivity_used"]),
            "eor_ev": norm_to_hilo(vectors["eor_ev"]),
            "hi_lo": _center(hilo).tolist(),
        },
        "svd_mode_cosine_to_hilo": svd_mode_cos_hilo,
        "cosine_table_depth_removed": table,
        "circular_note": (
            "Phase 4 weights were solved from the running-count probe (Hi-Lo "
            "target); their 0.998 cosine to Hi-Lo is circular. The two derivations "
            "here reference no counting system. Cosines are depth-removed "
            "(mean-centered), since a counting weight is defined only modulo an "
            "additive constant (the constant component == depth)."
        ),
    }


@torch.no_grad()
def _bet_sensitivity(
    model,
    tokens,
    offsets,
    shoe_ids,
    block_idx,
    Wp,
    bet_ids,
    device,
    pad_id,
    round_start_id,
    alpha,
) -> np.ndarray:
    sens = np.zeros(Wp.shape[1])
    for r in range(Wp.shape[1]):
        dir_r = torch.from_numpy(Wp[:, r].astype(np.float32)).to(device)
        plus, minus, npos = 0.0, 0.0, 0
        for start in range(0, len(shoe_ids), 16):
            batch = shoe_ids[start : start + 16]
            seqs = [tokens[offsets[s] : offsets[s + 1]] for s in batch]
            mx = max(len(s) for s in seqs)
            idx = np.full((len(batch), mx), pad_id, dtype=np.int64)
            rs = np.zeros((len(batch), mx), dtype=bool)
            for i, seq in enumerate(seqs):
                idx[i, : len(seq)] = seq
                rs[i, : len(seq)] = seq == round_start_id
            idx_t = torch.from_numpy(idx).to(device)
            rs_t = torch.from_numpy(rs).to(device, dtype=torch.float32)
            plus += (
                _expected_bet(model, idx_t, rs_t, bet_ids, alpha * dir_r, block_idx)
                * rs.sum()
            )
            minus += (
                _expected_bet(model, idx_t, rs_t, bet_ids, -alpha * dir_r, block_idx)
                * rs.sum()
            )
            npos += int(rs.sum())
        sens[r] = (plus - minus) / (2 * alpha * max(npos, 1))
    return sens


def plot_weight_audit(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = report["ranks"]
    x = np.arange(len(ranks))
    s = report["weights_centered_scaled_to_hilo"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.3, s["hi_lo"], 0.2, label="Hi-Lo", color="tab:green")
    ax.bar(x - 0.1, s["eor_ev"], 0.2, label="EoR (EV)", color="tab:red")
    ax.bar(
        x + 0.1,
        s["bet_sensitivity_used"],
        0.2,
        label="model betting sensitivity (used)",
        color="tab:blue",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ranks)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("rank")
    ax.set_ylabel("per-rank weight (depth-removed, scaled to Hi-Lo)")
    ax.set_title("Non-circular model weights vs Hi-Lo and Effect-of-Removal")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Non-circular counting-weight audit.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--n-shoes", type=int, default=800)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    exp = load_config(args.config)
    run_dir = Path(args.run_dir)
    device = resolve_device(args.device or exp.train.device)
    report = run_weight_audit(exp, run_dir, device, n_shoes=args.n_shoes)
    out = run_dir / "probe" / "weight_audit_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_weight_audit(report, run_dir / "probe" / "weight_audit.png")
    t = report["cosine_table_depth_removed"]
    print("[audit] depth-removed cosine to Hi-Lo / EoR-EV:")
    for k in ("svd_top_mode", "bet_sensitivity_used"):
        print(f"  {k:22s} hi_lo={t[k]['hi_lo']:+.3f}  eor_ev={t[k]['eor_ev']:+.3f}")
    print(f"[audit] wrote {out} and weight_audit.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

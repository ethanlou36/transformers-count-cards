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
from blackjackgpt.probe.alignment import load_labels, verify_alignment
from blackjackgpt.probe.counting_systems import (
    COUNTING_SYSTEMS,
    RANKS,
    system_running_count,
)
from blackjackgpt.probe.probes import split_positions_by_shoe
from blackjackgpt.probe.regression import cosine, fit_ridge
from blackjackgpt.sidecar import load_sidecar
from blackjackgpt.tokens import TokenTables
from blackjackgpt.train import resolve_device


def run_representation(exp, run_dir: Path, device: torch.device, kind: str) -> dict:
    vocab, _pad_id = load_vocabulary(exp.data.dataset_dir)
    tables = TokenTables(vocab)
    tokens, offsets = load_stream(exp.data.dataset_dir)
    split = split_by_shoe(offsets, exp.train.val_frac, exp.train.split_seed)
    shoe_ids = split.val
    if exp.probe.max_shoes:
        shoe_ids = shoe_ids[: exp.probe.max_shoes]

    sidecar = load_sidecar(exp.data.dataset_dir)
    verify_alignment(tokens, offsets, shoe_ids, vocab, tables, sidecar, kind)
    model, _ = load_model(resolve_checkpoint(run_dir), device)
    state = "after" if kind == "card" else "before"
    cache = build_activation_cache(
        model,
        tokens,
        offsets,
        shoe_ids,
        tables,
        run_dir / "probe" / "representation" / kind,
        device,
        kind,
    )
    labels = load_labels(cache.positions, sidecar, state)
    comp = labels.remaining_composition

    targets: dict[str, np.ndarray] = {
        "running_count": labels.running_count,
        "true_count": labels.raw_true_count,
        "cards_dealt": labels.cards_dealt,
    }
    for name, w in COUNTING_SYSTEMS.items():
        targets[f"count_{name}"] = system_running_count(comp, w)

    alphas = exp.probe.alphas
    split_p = split_positions_by_shoe(
        cache.positions.shoe_index,
        exp.probe.probe_eval_frac,
        exp.probe.probe_split_seed,
    )
    tr, ev = split_p.train_mask, split_p.eval_mask

    per_layer_rc = []
    per_layer_comp = []
    for layer in range(cache.n_layer):
        feats = cache.layer(layer)
        per_layer_rc.append(
            fit_ridge(
                feats[tr],
                targets["running_count"][tr],
                feats[ev],
                targets["running_count"][ev],
                alphas,
            ).r2
        )
        per_layer_comp.append(
            fit_ridge(feats[tr], comp[tr], feats[ev], comp[ev], alphas).r2
        )
    best_layer = int(np.nanargmax(per_layer_rc))
    feats = cache.layer(best_layer)

    scalar_r2 = {
        name: fit_ridge(feats[tr], y[tr], feats[ev], y[ev], alphas).r2
        for name, y in targets.items()
    }

    comp_fit = fit_ridge(feats[tr], comp[tr], feats[ev], comp[ev], alphas)
    per_rank_r2 = {
        RANKS[i].value: fit_ridge(
            feats[tr], comp[tr, i], feats[ev], comp[ev, i], alphas
        ).r2
        for i in range(10)
    }

    W = comp_fit.coef
    rc_fit = fit_ridge(
        feats[tr],
        targets["running_count"][tr],
        feats[ev],
        targets["running_count"][ev],
        alphas,
    )
    w_rc = rc_fit.coef

    svals = np.linalg.svd(W, compute_uv=False)
    sv_norm = svals / svals.sum()
    participation_rank = float((svals.sum() ** 2) / (svals**2).sum())

    h = COUNTING_SYSTEMS["hi_lo"]
    d_hilo = -(h @ W)
    cos_rc_hilo_subspace = cosine(w_rc, d_hilo)

    u, *_ = np.linalg.lstsq(W.T, w_rc, rcond=None)

    learned_weights = -u
    system_cos = {
        name: cosine(learned_weights, w) for name, w in COUNTING_SYSTEMS.items()
    }

    scale = float(np.dot(learned_weights, h) / max(np.dot(h, h), 1e-9))
    learned_weights_scaled = learned_weights / scale if scale != 0 else learned_weights

    return {
        "kind": kind,
        "n_positions": len(cache.positions),
        "best_layer": best_layer + 1,
        "per_layer_running_count_r2": per_layer_rc,
        "per_layer_composition_r2": per_layer_comp,
        "scalar_r2_at_best_layer": scalar_r2,
        "composition_joint_r2": comp_fit.r2,
        "composition_per_rank_r2": per_rank_r2,
        "geometry": {
            "svd_singular_values": svals.tolist(),
            "svd_normalized": sv_norm.tolist(),
            "participation_rank": participation_rank,
            "cos_running_count_vs_hilo_subspace": cos_rc_hilo_subspace,
            "learned_per_rank_weights": learned_weights.tolist(),
            "learned_per_rank_weights_scaled_to_hilo": learned_weights_scaled.tolist(),
            "hi_lo_weights": h.tolist(),
            "cosine_to_systems": system_cos,
            "ranks": [r.value for r in RANKS],
        },
    }


def plot_representation(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    card = report["position_classes"].get("card")
    if card is None:
        card = next(iter(report["position_classes"].values()))
    ranks = card["geometry"]["ranks"]
    x = np.arange(len(ranks))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(
        x, [card["composition_per_rank_r2"][r] for r in ranks], color="tab:blue"
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ranks)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"per-rank composition R² (layer {card['best_layer']})")
    axes[0].set_xlabel("rank")
    axes[0].set_ylabel("R²")

    sv = card["geometry"]["svd_normalized"]
    axes[1].plot(range(1, len(sv) + 1), sv, marker="o", color="tab:purple")
    axes[1].set_title(
        f"composition W SVD spectrum (participation rank "
        f"{card['geometry']['participation_rank']:.2f})"
    )
    axes[1].set_xlabel("singular value index")
    axes[1].set_ylabel("normalized σ")

    lw = card["geometry"]["learned_per_rank_weights_scaled_to_hilo"]
    hl = card["geometry"]["hi_lo_weights"]
    axes[2].bar(x - 0.2, lw, width=0.4, label="learned (scaled)", color="tab:orange")
    axes[2].bar(x + 0.2, hl, width=0.4, label="Hi-Lo", color="tab:green")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(ranks)
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].set_title("model's own counting weights vs Hi-Lo")
    axes[2].set_xlabel("rank")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"Representation analysis ({report.get('label', '')})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe representation analysis.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--kinds", type=str, default="card,bet")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--device", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    exp = load_config(args.config)
    run_dir = Path(args.run_dir)
    device = resolve_device(args.device or exp.train.device)

    report: dict = {"label": args.label, "position_classes": {}}
    for kind in [k.strip() for k in args.kinds.split(",") if k.strip()]:
        report["position_classes"][kind] = run_representation(
            exp, run_dir, device, kind
        )
        g = report["position_classes"][kind]["geometry"]
        print(
            f"[repr] {kind}: best_layer={report['position_classes'][kind]['best_layer']}"
            f" comp_joint_R2={report['position_classes'][kind]['composition_joint_r2']:.3f}"
            f" cos(rc,hilo-subspace)={g['cos_running_count_vs_hilo_subspace']:.3f}"
            f" cos(learned,hilo)={g['cosine_to_systems']['hi_lo']:.3f}"
        )
    out = run_dir / "probe" / "representation_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_representation(report, run_dir / "probe" / "representation.png")
    print(f"[repr] wrote {out} and representation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

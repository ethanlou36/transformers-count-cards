from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from blackjackgpt.checkpoint import load_model, resolve_checkpoint
from blackjackgpt.config import load_config
from blackjackgpt.dataset import load_stream, split_by_shoe
from blackjackgpt.intervene.next_card import (
    _mean_rank_dist_under_patch,
    _rank_token_ids,
)
from blackjackgpt.model import load_vocabulary
from blackjackgpt.probe.activations import build_activation_cache
from blackjackgpt.probe.alignment import load_labels, positions_for_kind
from blackjackgpt.probe.counting_systems import RANKS
from blackjackgpt.probe.probes import split_positions_by_shoe
from blackjackgpt.probe.regression import cosine, fit_ridge
from blackjackgpt.sidecar import load_sidecar
from blackjackgpt.tokens import TokenTables
from blackjackgpt.train import resolve_device


def _analytical_sensitivity(comp: np.ndarray) -> np.ndarray:
    n_ranks = comp.shape[1]
    totals = np.zeros((n_ranks, n_ranks))
    npos = 0
    for c in comp:
        nn = c.sum()
        if nn <= 1:
            continue
        p = c / nn

        totals += (np.eye(n_ranks) - p[None, :]) / nn
        npos += 1
    return totals / max(npos, 1)


def _grad_sensitivity(
    model,
    tokens,
    offsets,
    shoe_ids,
    block,
    encoding,
    rank_groups,
    device,
    pad_id,
    card_ids_set,
) -> np.ndarray:
    enc_t = torch.tensor(encoding, dtype=torch.float32, device=device)
    all_rank_ids = torch.tensor(
        [i for g in rank_groups for i in g], dtype=torch.long, device=device
    )
    group_index = torch.tensor(
        [gi for gi, g in enumerate(rank_groups) for _ in g], device=device
    )
    n_ranks = len(rank_groups)

    def mean_dist(c: torch.Tensor) -> torch.Tensor:
        delta = enc_t @ c
        total = torch.zeros(n_ranks, device=device)
        npos = 0
        for start in range(0, len(shoe_ids), 16):
            batch = shoe_ids[start : start + 16]
            seqs = [tokens[offsets[s] : offsets[s + 1]] for s in batch]
            mx = max(len(s) for s in seqs)
            idx = np.full((len(batch), mx), pad_id, dtype=np.int64)
            pred = np.zeros((len(batch), mx), dtype=bool)
            for i, seq in enumerate(seqs):
                idx[i, : len(seq)] = seq
                nxt = np.isin(seq, list(card_ids_set))
                pred[i, : len(seq) - 1] = nxt[1:]
            idx_t = torch.from_numpy(idx).to(device)
            mask = torch.from_numpy(pred).to(device, dtype=torch.float32)

            def hook(_m, _i, out, _mask=mask, _delta=delta):
                return out + _mask.unsqueeze(-1) * _delta

            h = model.blocks[block].register_forward_hook(hook)
            try:
                logits = model(idx_t)["logits"]
            finally:
                h.remove()
            probs = torch.softmax(logits, dim=-1)
            rank_mass = probs[:, :, all_rank_ids]
            grouped = torch.zeros((*rank_mass.shape[:2], n_ranks), device=device)
            grouped.index_add_(2, group_index, rank_mass)
            grouped = grouped / grouped.sum(-1, keepdim=True).clamp_min(1e-9)
            sel = torch.from_numpy(pred).to(device)
            total = total + grouped[sel].sum(0)
            npos += int(pred.sum())
        return total / max(npos, 1)

    c0 = torch.zeros(n_ranks, device=device, requires_grad=True)
    jac = torch.autograd.functional.jacobian(mean_dist, c0)
    return jac.detach().cpu().numpy().T


def run_next_card_sensitivity(
    exp,
    run_dir: Path,
    device: torch.device,
    n_shoes_fd: int = 600,
    n_shoes_grad: int = 150,
    alpha: float = 1.0,
) -> dict:
    vocab, pad_id = load_vocabulary(exp.data.dataset_dir)
    tables = TokenTables(vocab)
    tokens, offsets = load_stream(exp.data.dataset_dir)
    split = split_by_shoe(offsets, exp.train.val_frac, exp.train.split_seed)
    shoe_ids = split.val[:n_shoes_fd]
    sidecar = load_sidecar(exp.data.dataset_dir)
    model, _ = load_model(resolve_checkpoint(run_dir), device)
    alphas = exp.probe.alphas
    rank_groups = _rank_token_ids(vocab)
    card_ids_set = {int(i) for g in rank_groups for i in g}

    positions = positions_for_kind(tokens, offsets, shoe_ids, tables, "card")
    cache = build_activation_cache(
        model,
        tokens,
        offsets,
        shoe_ids,
        tables,
        run_dir / "probe" / "next_card_sensitivity",
        device,
        "card",
        positions=positions,
    )
    labels = load_labels(cache.positions, sidecar, "after")
    comp = labels.remaining_composition
    sp = split_positions_by_shoe(
        cache.positions.shoe_index,
        exp.probe.probe_eval_frac,
        exp.probe.probe_split_seed,
    )
    tr, ev = sp.train_mask, sp.eval_mask
    comp_r2 = [
        fit_ridge(cache.layer(i)[tr], comp[tr], cache.layer(i)[ev], comp[ev], alphas).r2
        for i in range(model.config.n_layer)
    ]
    best_layer = int(np.nanargmax(comp_r2))

    from sklearn.linear_model import Ridge

    enc = Ridge(alpha=10.0).fit(comp[tr], cache.layer(best_layer)[tr])
    encoding = np.asarray(enc.coef_)

    n_ranks = len(rank_groups)
    fd = np.zeros((n_ranks, n_ranks))
    for r in range(n_ranks):
        dir_r = encoding[:, r].astype(np.float32)
        plus = _mean_rank_dist_under_patch(
            model,
            tokens,
            offsets,
            shoe_ids,
            best_layer,
            torch.tensor(alpha * dir_r, device=device),
            rank_groups,
            device,
            pad_id,
            card_ids_set,
        )
        minus = _mean_rank_dist_under_patch(
            model,
            tokens,
            offsets,
            shoe_ids,
            best_layer,
            torch.tensor(-alpha * dir_r, device=device),
            rank_groups,
            device,
            pad_id,
            card_ids_set,
        )
        fd[r] = (plus - minus) / (2 * alpha)

    grad = _grad_sensitivity(
        model,
        tokens,
        offsets,
        shoe_ids[:n_shoes_grad],
        best_layer,
        encoding,
        rank_groups,
        device,
        pad_id,
        card_ids_set,
    )

    analytic = _analytical_sensitivity(comp)

    def cos_flat(a, b):
        return cosine(a.reshape(-1), b.reshape(-1))

    def pearson(a, b):
        return float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])

    per_rank_cos = [float(cosine(fd[r], analytic[r])) for r in range(n_ranks)]
    ranks = [r.value for r in RANKS]
    return {
        "run_dir": str(run_dir),
        "readout": "next_card_distribution",
        "best_composition_layer": best_layer + 1,
        "composition_r2_by_layer": comp_r2,
        "alpha": alpha,
        "n_shoes_finite_diff": len(shoe_ids),
        "n_shoes_gradient": len(shoe_ids[:n_shoes_grad]),
        "ranks": ranks,
        "sensitivity_finite_diff": fd.tolist(),
        "sensitivity_gradient": grad.tolist(),
        "sensitivity_analytical": analytic.tolist(),
        "diagonal_finite_diff": np.diag(fd).tolist(),
        "diagonal_analytical": np.diag(analytic).tolist(),
        "grad_vs_finite_diff_cosine": cos_flat(grad, fd),
        "grad_vs_finite_diff_pearson": pearson(grad, fd),
        "learned_vs_analytical_cosine": cos_flat(fd, analytic),
        "learned_vs_analytical_pearson": pearson(fd, analytic),
        "diagonal_cosine": float(cosine(np.diag(fd), np.diag(analytic))),
        "per_rank_cosine": per_rank_cos,
    }


def plot_next_card_sensitivity(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = report["ranks"]
    x = np.arange(len(ranks))
    fd = np.array(report["sensitivity_finite_diff"])
    an = np.array(report["sensitivity_analytical"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(
        x - 0.2,
        report["diagonal_analytical"],
        0.4,
        label="analytical (combinatorics)",
        color="tab:gray",
    )
    axes[0].bar(
        x + 0.2,
        report["diagonal_finite_diff"],
        0.4,
        label="model (finite diff)",
        color="tab:blue",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ranks)
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].set_xlabel("rank r")
    axes[0].set_ylabel("dP(next=r) / d(one more rank-r card)")
    axes[0].set_title(
        f"Self-sensitivity per rank (diag cos {report['diagonal_cosine']:.3f})"
    )
    axes[0].legend(fontsize=8)

    axes[1].scatter(an.reshape(-1), fd.reshape(-1), s=18, alpha=0.7, color="tab:blue")
    lim = max(np.abs(an).max(), np.abs(fd).max()) * 1.1
    axes[1].plot([-lim, lim], [-lim, lim], color="tab:gray", ls="--", lw=1)
    axes[1].set_xlabel("analytical dP(next=j)/dc_r")
    axes[1].set_ylabel("model (finite diff)")
    axes[1].set_title(
        f"All rank pairs: cos {report['learned_vs_analytical_cosine']:.3f}, "
        f"r {report['learned_vs_analytical_pearson']:.3f}"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="λ=0 next-card sensitivity vs analytical.")
    p.add_argument("--config", default="configs/train_lambda0.yaml")
    p.add_argument("--run-dir", default="runs/lambda0-v1")
    p.add_argument("--n-shoes-fd", type=int, default=600)
    p.add_argument("--n-shoes-grad", type=int, default=150)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    exp = load_config(args.config)
    run_dir = Path(args.run_dir)
    device = resolve_device(args.device or exp.train.device)
    report = run_next_card_sensitivity(
        exp, run_dir, device, n_shoes_fd=args.n_shoes_fd, n_shoes_grad=args.n_shoes_grad
    )
    out = Path(args.out) if args.out else run_dir / "probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / "next_card_sensitivity.json").write_text(json.dumps(report, indent=2))
    plot_next_card_sensitivity(report, out / "next_card_sensitivity.png")
    print(
        f"[b1] grad-vs-fd cos {report['grad_vs_finite_diff_cosine']:.3f}; "
        f"learned-vs-analytical cos {report['learned_vs_analytical_cosine']:.3f}, "
        f"pearson {report['learned_vs_analytical_pearson']:.3f}; "
        f"diag cos {report['diagonal_cosine']:.3f}"
    )
    print(f"[b1] wrote {out / 'next_card_sensitivity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

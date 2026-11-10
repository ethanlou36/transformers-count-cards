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
from blackjackgpt.probe.alignment import load_labels, positions_for_kind
from blackjackgpt.probe.counting_systems import RANKS
from blackjackgpt.probe.probes import split_positions_by_shoe
from blackjackgpt.probe.regression import fit_ridge
from blackjackgpt.sidecar import load_sidecar
from blackjackgpt.tokens import TokenTables
from blackjackgpt.train import resolve_device

_CARD_PREFIXES = ("P:", "D_UP:", "D_REVEAL:", "D_DRAW:")


def _rank_token_ids(vocab) -> list[list[int]]:
    groups = []
    for r in RANKS:
        ids = [vocab.encode(f"{p}{r.value}") for p in _CARD_PREFIXES]
        groups.append(ids)
    return groups


@torch.no_grad()
def _mean_rank_dist_under_patch(
    model,
    tokens,
    offsets,
    shoe_ids,
    layer_block_idx,
    delta,
    rank_groups,
    device,
    pad_id,
    card_ids_set,
) -> np.ndarray:
    handle = None
    if delta is not None:
        mask_holder: dict[str, torch.Tensor] = {}

        def hook(_m, _i, out):
            return out + mask_holder["mask"].unsqueeze(-1) * delta

        handle = model.blocks[layer_block_idx].register_forward_hook(hook)

    all_rank_ids = torch.tensor(
        [i for g in rank_groups for i in g], dtype=torch.long, device=device
    )
    group_index = torch.tensor(
        [gi for gi, g in enumerate(rank_groups) for _ in g], device=device
    )
    totals = np.zeros(len(rank_groups))
    n = 0
    try:
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
            if delta is not None:
                mask_holder["mask"] = torch.from_numpy(pred).to(
                    device, dtype=delta.dtype
                )
            logits = model(idx_t)["logits"]
            probs = torch.softmax(logits, dim=-1)
            rank_mass = probs[:, :, all_rank_ids]

            grouped = torch.zeros(
                (*rank_mass.shape[:2], len(rank_groups)), device=device
            )
            grouped.index_add_(2, group_index, rank_mass)
            grouped = grouped / grouped.sum(-1, keepdim=True).clamp_min(1e-9)
            sel = torch.from_numpy(pred).to(device)
            totals += grouped[sel].sum(0).cpu().numpy()
            n += int(pred.sum())
    finally:
        if handle is not None:
            handle.remove()
    return totals / max(n, 1)


def run_next_card_intervention(
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
    rank_groups = _rank_token_ids(vocab)
    card_ids_set = {int(i) for g in rank_groups for i in g}

    positions = positions_for_kind(tokens, offsets, shoe_ids, tables, "card")
    cache = build_activation_cache(
        model,
        tokens,
        offsets,
        shoe_ids,
        tables,
        run_dir / "probe" / "next_card",
        device,
        "card",
        positions=positions,
    )
    labels = load_labels(cache.positions, sidecar, "after")
    rc = labels.running_count
    sp = split_positions_by_shoe(
        cache.positions.shoe_index,
        exp.probe.probe_eval_frac,
        exp.probe.probe_split_seed,
    )
    tr, ev = sp.train_mask, sp.eval_mask
    per_layer = [
        fit_ridge(cache.layer(i)[tr], rc[tr], cache.layer(i)[ev], rc[ev], alphas).r2
        for i in range(n_block)
    ]
    best_layer = int(np.nanargmax(per_layer))
    w = fit_ridge(
        cache.layer(best_layer)[tr], rc[tr], cache.layer(best_layer)[ev], rc[ev], alphas
    ).coef.astype(np.float64)
    hi = rc >= np.median(rc)
    md = cache.layer(best_layer)[hi].mean(0) - cache.layer(best_layer)[~hi].mean(0)
    md_unit = md / float(w @ md)
    rng = np.random.default_rng(0)
    rand = rng.standard_normal(w.shape)
    rand *= np.linalg.norm(md_unit) / (np.linalg.norm(rand) + 1e-12)

    t_idx, a_idx = 9, 0

    def sweep(direction, block):
        pt, pa = [], []
        for c in count_grid:
            delta = torch.from_numpy((c * direction).astype(np.float32)).to(device)
            dist = _mean_rank_dist_under_patch(
                model,
                tokens,
                offsets,
                shoe_ids,
                block,
                delta,
                rank_groups,
                device,
                pad_id,
                card_ids_set,
            )
            pt.append(float(dist[t_idx]))
            pa.append(float(dist[a_idx]))
        return pt, pa

    md_t, md_a = sweep(md_unit, best_layer)
    rand_t, rand_a = sweep(rand, best_layer)
    last_t, last_a = sweep(md_unit, n_block - 1)

    emp_t, emp_a = [], []
    for c in count_grid:
        sel = np.round(rc) == c
        if sel.sum() > 5:
            comp = labels.remaining_composition[sel].mean(0)
            comp = comp / comp.sum()
            emp_t.append(float(comp[9]))
            emp_a.append(float(comp[0]))
        else:
            emp_t.append(float("nan"))
            emp_a.append(float("nan"))

    def slope(y):
        x = np.array(count_grid, float)
        yy = np.array(y, float)
        ok = np.isfinite(yy)
        return float(np.polyfit(x[ok], yy[ok], 1)[0]) if ok.sum() > 1 else float("nan")

    return {
        "best_layer": best_layer + 1,
        "running_count_r2_by_layer": per_layer,
        "count_grid": list(count_grid),
        "curves": {
            "mean_difference": {"P_ten": md_t, "P_ace": md_a},
            "random_direction": {"P_ten": rand_t, "P_ace": rand_a},
            "mean_difference_last_layer": {"P_ten": last_t, "P_ace": last_a},
        },
        "empirical_true_composition": {"P_ten": emp_t, "P_ace": emp_a},
        "slopes_P_ten": {
            "mean_difference": slope(md_t),
            "random_direction": slope(rand_t),
            "empirical": slope(emp_t),
        },
    }


def plot_next_card(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = report["count_grid"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title in (
        (axes[0], "P_ten", "P(next card = T)"),
        (axes[1], "P_ace", "P(next card = A)"),
    ):
        ax.plot(
            g,
            report["curves"]["mean_difference"][key],
            marker="o",
            color="tab:blue",
            label="mean-diff patch",
        )
        ax.plot(
            g,
            report["curves"]["random_direction"][key],
            marker="x",
            ls=":",
            color="gray",
            label="random dir",
        )
        ax.plot(
            g,
            report["curves"]["mean_difference_last_layer"][key],
            marker="s",
            ls="--",
            color="tab:green",
            label="mean-diff @ last layer",
        )
        ax.plot(
            g,
            report["empirical_true_composition"][key],
            marker="D",
            color="tab:red",
            label="true composition vs count",
        )
        ax.set_xlabel("implied running-count shift (α)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"λ=0 next-card dose-response (layer {report['best_layer']})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="λ=0 next-card intervention.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--n-shoes", type=int, default=300)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    exp = load_config(args.config)
    run_dir = Path(args.run_dir)
    device = resolve_device(args.device or exp.train.device)
    report = run_next_card_intervention(exp, run_dir, device, n_shoes=args.n_shoes)
    out = run_dir / "probe" / "next_card_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_next_card(report, run_dir / "probe" / "next_card.png")
    s = report["slopes_P_ten"]
    print(
        f"[next-card] dP(ten)/dcount: mean-diff={s['mean_difference']:+.4f} "
        f"random={s['random_direction']:+.4f} empirical={s['empirical']:+.4f}"
    )
    print(f"[next-card] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

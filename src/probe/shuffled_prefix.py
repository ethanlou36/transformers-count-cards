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
from blackjackgpt.probe.counting_systems import COUNTING_SYSTEMS, RANKS
from blackjackgpt.probe.probes import split_positions_by_shoe
from blackjackgpt.probe.regression import fit_ridge
from blackjackgpt.train import resolve_device, seed_everything

_CARD_PREFIXES = ("P:", "D_UP:", "D_REVEAL:", "D_DRAW:")


def _build_card_maps(vocab):
    rank_values = [r.value for r in RANKS]
    id_to_pr: dict[int, tuple[str, int]] = {}
    pr_to_id: dict[tuple[str, int], int] = {}
    for text in vocab.tokens:
        for p in _CARD_PREFIXES:
            if text.startswith(p):
                rv = text[len(p) :]
                if rv in rank_values:
                    tid = int(vocab.encode(text))
                    ri = rank_values.index(rv)
                    id_to_pr[tid] = (p, ri)
                    pr_to_id[(p, ri)] = tid
    hilo = COUNTING_SYSTEMS["hi_lo"]
    return set(id_to_pr), id_to_pr, pr_to_id, hilo


def _permute_and_recount(seq, card_id_set, id_to_pr, pr_to_id, hilo, rng):
    card_pos = [i for i, t in enumerate(seq) if int(t) in card_id_set]
    prefixes = [id_to_pr[int(seq[i])][0] for i in card_pos]
    orig_ranks = [id_to_pr[int(seq[i])][1] for i in card_pos]
    perm_ranks = list(np.array(orig_ranks)[rng.permutation(len(orig_ranks))])

    new_seq = np.array(seq, dtype=np.int64).copy()
    for pos, pfx, ri in zip(card_pos, prefixes, perm_ranks):
        new_seq[pos] = pr_to_id[(pfx, int(ri))]

    orig_count = np.cumsum([hilo[r] for r in orig_ranks]).astype(np.float64)
    perm_count = np.cumsum([hilo[r] for r in perm_ranks]).astype(np.float64)
    return new_seq, np.array(card_pos), orig_count, perm_count


@torch.no_grad()
def _extract(model, seqs, card_id_set, device, pad_id):
    feats: list[np.ndarray] = []
    shoe_index: list[int] = []
    for start in range(0, len(seqs), 16):
        batch = seqs[start : start + 16]
        mx = max(len(s) for s in batch)
        idx = np.full((len(batch), mx), pad_id, dtype=np.int64)
        for i, s in enumerate(batch):
            idx[i, : len(s)] = s
        idx_t = torch.from_numpy(idx).to(device)
        acts = model(idx_t, return_activations=True)["activations"]
        stacked = torch.stack(acts, dim=2).to(torch.float32).cpu().numpy()
        for i, s in enumerate(batch):
            local = np.array(
                [j for j, t in enumerate(s) if int(t) in card_id_set], dtype=np.int64
            )
            if len(local):
                feats.append(stacked[i, local])
                shoe_index.extend([start + i] * len(local))
    return np.concatenate(feats, axis=0), np.array(shoe_index)


def _best_layer_r2(feats_by_layer, y, sp, alphas):
    tr, ev = sp.train_mask, sp.eval_mask
    r2 = [fit_ridge(f[tr], y[tr], f[ev], y[ev], alphas).r2 for f in feats_by_layer]
    return int(np.nanargmax(r2)), r2


def run_shuffled_prefix(
    exp, run_dir: Path, device: torch.device, n_shoes: int = 500, seed: int = 20260722
) -> dict:
    vocab, pad_id = load_vocabulary(exp.data.dataset_dir)
    tokens, offsets = load_stream(exp.data.dataset_dir)
    split = split_by_shoe(offsets, exp.train.val_frac, exp.train.split_seed)
    shoe_ids = split.val[:n_shoes]
    alphas = exp.probe.alphas
    card_id_set, id_to_pr, pr_to_id, hilo = _build_card_maps(vocab)
    rng = np.random.default_rng(seed)

    orig_seqs, perm_seqs = [], []
    orig_counts, perm_counts = [], []
    for s in shoe_ids:
        seq = tokens[offsets[s] : offsets[s + 1]]
        new_seq, _pos, oc, pc = _permute_and_recount(
            seq, card_id_set, id_to_pr, pr_to_id, hilo, rng
        )
        orig_seqs.append(np.array(seq, dtype=np.int64))
        perm_seqs.append(new_seq)
        orig_counts.append(oc)
        perm_counts.append(pc)
    orig_count = np.concatenate(orig_counts)
    perm_count = np.concatenate(perm_counts)

    trained, gpt_cfg = load_model(resolve_checkpoint(run_dir), device)
    seed_everything(seed)
    random_model = GPT(gpt_cfg).to(device).eval()

    orig_feats, orig_idx = _extract(trained, orig_seqs, card_id_set, device, pad_id)
    sp_o = split_positions_by_shoe(orig_idx, exp.probe.probe_eval_frac, seed)
    layers_o = [orig_feats[:, i] for i in range(orig_feats.shape[1])]
    best_layer, ref_r2_by_layer = _best_layer_r2(layers_o, orig_count, sp_o, alphas)

    perm_feats, perm_idx = _extract(trained, perm_seqs, card_id_set, device, pad_id)
    sp_p = split_positions_by_shoe(perm_idx, exp.probe.probe_eval_frac, seed)
    tr, ev = sp_p.train_mask, sp_p.eval_mask
    f_best = perm_feats[:, best_layer]
    test_r2 = fit_ridge(
        f_best[tr], perm_count[tr], f_best[ev], perm_count[ev], alphas
    ).r2
    falsify_r2 = fit_ridge(
        f_best[tr], orig_count[tr], f_best[ev], orig_count[ev], alphas
    ).r2

    rand_feats, _rand_idx = _extract(
        random_model, perm_seqs, card_id_set, device, pad_id
    )
    rf = rand_feats[:, best_layer]
    control_r2 = fit_ridge(rf[tr], perm_count[tr], rf[ev], perm_count[ev], alphas).r2

    return {
        "run_dir": str(run_dir),
        "n_held_out_shoes": len(shoe_ids),
        "best_layer": best_layer + 1,
        "seed": seed,
        "reference_original_input_original_count_r2": float(
            ref_r2_by_layer[best_layer]
        ),
        "test_permuted_input_permuted_count_r2": float(test_r2),
        "falsification_permuted_input_original_count_r2": float(falsify_r2),
        "control_randominit_permuted_input_permuted_count_r2": float(control_r2),
        "verdict": (
            "computes count from input"
            if test_r2 > 0.8 * ref_r2_by_layer[best_layer]
            and falsify_r2 < 0.5 * test_r2
            and test_r2 > control_r2 + 0.2
            else "inconclusive"
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Shuffled-prefix count control (B2).")
    p.add_argument("--config", default="configs/train_lambda0.yaml")
    p.add_argument("--run-dir", default="runs/lambda0-v1")
    p.add_argument("--n-shoes", type=int, default=500)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    exp = load_config(args.config)
    run_dir = Path(args.run_dir)
    device = resolve_device(args.device or exp.train.device)
    report = run_shuffled_prefix(exp, run_dir, device, n_shoes=args.n_shoes)
    out = Path(args.out) if args.out else run_dir / "probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / "shuffled_prefix.json").write_text(json.dumps(report, indent=2))
    print(
        f"[b2] ref {report['reference_original_input_original_count_r2']:.3f}  "
        f"test(perm/perm) {report['test_permuted_input_permuted_count_r2']:.3f}  "
        f"falsify(perm/orig) {report['falsification_permuted_input_original_count_r2']:.3f}  "
        f"control(rand) {report['control_randominit_permuted_input_permuted_count_r2']:.3f}"
    )
    print(f"[b2] verdict: {report['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

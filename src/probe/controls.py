from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from blackjackgpt.checkpoint import load_model, resolve_checkpoint
from blackjackgpt.config import load_config
from blackjackgpt.dataset import load_stream, split_by_shoe
from blackjackgpt.model import GPT, load_vocabulary
from blackjackgpt.probe.activations import build_activation_cache
from blackjackgpt.probe.alignment import load_labels, verify_alignment
from blackjackgpt.probe.probes import split_positions_by_shoe
from blackjackgpt.probe.regression import held_out_r2
from blackjackgpt.sidecar import load_sidecar
from blackjackgpt.tokens import TokenTables
from blackjackgpt.train import resolve_device, seed_everything

_KIND_STATE = {"card": "after", "bet": "before"}


def _running_count_r2_by_layer(cache, labels, split, alphas):
    tr, ev = split.train_mask, split.eval_mask
    y = labels.running_count
    return [
        held_out_r2(cache.layer(i)[tr], y[tr], cache.layer(i)[ev], y[ev], alphas)
        for i in range(cache.n_layer)
    ]


def _cache(model, tokens, offsets, shoe_ids, tables, out, device, kind):
    return build_activation_cache(
        model,
        tokens,
        offsets,
        shoe_ids,
        tables,
        out,
        device,
        kind,
        include_embedding=True,
    )


def run_controls(
    exp,
    trained_run_dirs: list[Path],
    device: torch.device,
    seed: int,
    out_root: Path,
) -> dict:
    vocab, _pad_id = load_vocabulary(exp.data.dataset_dir)
    tables = TokenTables(vocab)
    tokens, offsets = load_stream(exp.data.dataset_dir)
    split = split_by_shoe(offsets, exp.train.val_frac, exp.train.split_seed)
    shoe_ids = split.val
    if exp.probe.max_shoes:
        shoe_ids = shoe_ids[: exp.probe.max_shoes]
    sidecar = load_sidecar(exp.data.dataset_dir)
    alphas = exp.probe.alphas
    out_root.mkdir(parents=True, exist_ok=True)

    trained_models = [
        load_model(resolve_checkpoint(rd), device) for rd in trained_run_dirs
    ]
    _, gpt_cfg = trained_models[0]
    seed_everything(seed)
    random_model = GPT(gpt_cfg).to(device).eval()
    torch.save(
        {
            "model_state": random_model.state_dict(),
            "gpt_config": asdict(gpt_cfg),
            "step": -1,
            "metrics": {},
        },
        out_root / "ckpt_random_init.pt",
    )

    report: dict = {
        "trained_run_dirs": [str(p) for p in trained_run_dirs],
        "n_seeds": len(trained_run_dirs),
        "n_held_out_shoes": len(shoe_ids),
        "position_classes": {},
    }
    for kind in _KIND_STATE:
        verify_alignment(tokens, offsets, shoe_ids, vocab, tables, sidecar, kind)
        state = _KIND_STATE[kind]

        rand_cache = _cache(
            random_model,
            tokens,
            offsets,
            shoe_ids,
            tables,
            out_root / f"{kind}_random",
            device,
            kind,
        )
        split_p = split_positions_by_shoe(
            rand_cache.positions.shoe_index,
            exp.probe.probe_eval_frac,
            exp.probe.probe_split_seed,
        )
        labels = load_labels(rand_cache.positions, sidecar, state)
        random_by_layer = _running_count_r2_by_layer(
            rand_cache, labels, split_p, alphas
        )

        trained_by_seed = []
        emb_only_trained = []
        shuffled = None
        for i, (model, _) in enumerate(trained_models):
            tc = _cache(
                model,
                tokens,
                offsets,
                shoe_ids,
                tables,
                out_root / f"{kind}_trained_{i}",
                device,
                kind,
            )
            r2 = _running_count_r2_by_layer(tc, labels, split_p, alphas)
            trained_by_seed.append(r2)
            emb_only_trained.append(r2[0])
            if i == 0:
                best = int(np.nanargmax(r2[1:])) + 1
                tr, ev = split_p.train_mask, split_p.eval_mask
                y_shuf = np.random.default_rng(seed).permutation(labels.running_count)
                shuffled = held_out_r2(
                    tc.layer(best)[tr],
                    y_shuf[tr],
                    tc.layer(best)[ev],
                    y_shuf[ev],
                    alphas,
                )

        trained_arr = np.array(trained_by_seed)
        report["position_classes"][kind] = {
            "layers_include_embedding_at_index_0": True,
            "trained_r2_by_layer_per_seed": trained_by_seed,
            "trained_mean_by_layer": trained_arr.mean(axis=0).tolist(),
            "trained_min_by_layer": trained_arr.min(axis=0).tolist(),
            "trained_max_by_layer": trained_arr.max(axis=0).tolist(),
            "random_init_r2_by_layer": random_by_layer,
            "embeddings_only_trained_r2": float(np.mean(emb_only_trained)),
            "embeddings_only_random_r2": float(random_by_layer[0]),
            "shuffled_label_r2": float(shuffled) if shuffled is not None else None,
        }
        print(
            f"[controls] {kind}: trained(block max, mean over seeds)="
            f"{trained_arr[:, 1:].max(axis=1).mean():.3f} "
            f"random={max(random_by_layer[1:]):.3f} "
            f"emb_only={np.mean(emb_only_trained):.3f} shuffled={shuffled:.3f}"
        )
    return report


def plot_controls(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = report["position_classes"]
    names = list(classes)
    fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 5), squeeze=False)
    for ax, name in zip(axes[0], names):
        p = classes[name]

        mean = p["trained_mean_by_layer"][1:]
        lo = p["trained_min_by_layer"][1:]
        hi = p["trained_max_by_layer"][1:]
        rand = p["random_init_r2_by_layer"][1:]
        layers = list(range(1, len(mean) + 1))
        ax.plot(
            layers,
            mean,
            marker="o",
            color="tab:blue",
            label=f"trained (mean of {report['n_seeds']} seeds)",
        )
        ax.fill_between(layers, lo, hi, color="tab:blue", alpha=0.2, label="seed range")
        ax.plot(layers, rand, marker="s", color="tab:red", label="random-init")
        ax.axhline(
            p["embeddings_only_trained_r2"],
            ls="--",
            color="tab:green",
            label="embeddings-only",
        )
        ax.axhline(p["shuffled_label_r2"], ls=":", color="gray", label="shuffled-label")
        ax.set_title(f"{name} positions")
        ax.set_xlabel("layer (post-block residual stream)")
        ax.set_ylabel("running-count held-out R²")
        ax.set_xticks(layers)
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Probe validity controls")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe validity controls.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--run-dirs",
        type=str,
        required=True,
        help="comma-separated trained run dirs (seeds)",
    )
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    exp = load_config(args.config)
    device = resolve_device(args.device or exp.train.device)
    run_dirs = [Path(p) for p in args.run_dirs.split(",") if p]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    report = run_controls(exp, run_dirs, device, args.seed, out_root)
    out = out_root / "controls_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_controls(report, out_root / "controls_r2.png")
    print(f"[controls] wrote {out} and controls_r2.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

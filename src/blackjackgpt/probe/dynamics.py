from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch

from blackjackgpt.checkpoint import load_model
from blackjackgpt.config import load_config
from blackjackgpt.dataset import load_stream, split_by_shoe
from blackjackgpt.model import load_vocabulary
from blackjackgpt.probe.activations import build_activation_cache
from blackjackgpt.probe.alignment import load_labels, verify_alignment
from blackjackgpt.probe.probes import split_positions_by_shoe
from blackjackgpt.probe.regression import held_out_r2
from blackjackgpt.sidecar import load_sidecar
from blackjackgpt.tokens import TokenTables
from blackjackgpt.train import resolve_device

_STEP_RE = re.compile(r"ckpt_step_(\d+)\.pt$")


def _checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in run_dir.glob("ckpt_step_*.pt"):
        m = _STEP_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    last = run_dir / "ckpt_last.pt"
    if last.exists():
        payload = torch.load(last, map_location="cpu", weights_only=False)
        out.append((int(payload.get("step", 10**9)), last))
    return sorted(out)


def _loss_by_step(run_dir: Path, column: str) -> dict[int, float]:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return {}
    out: dict[int, float] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            val = row.get(column, "")
            if val not in ("", None):
                out[int(row["step"])] = float(val)
    return out


def run_dynamics(
    exp,
    run_dir: Path,
    device: torch.device,
    kind: str,
    loss_column: str,
    max_shoes: int,
) -> dict:
    vocab, _pad_id = load_vocabulary(exp.data.dataset_dir)
    tables = TokenTables(vocab)
    tokens, offsets = load_stream(exp.data.dataset_dir)
    split = split_by_shoe(offsets, exp.train.val_frac, exp.train.split_seed)
    shoe_ids = split.val[:max_shoes] if max_shoes else split.val
    sidecar = load_sidecar(exp.data.dataset_dir)
    verify_alignment(tokens, offsets, shoe_ids, vocab, tables, sidecar, kind)
    state = "after" if kind == "card" else "before"
    alphas = exp.probe.alphas

    checkpoints = _checkpoints(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints under {run_dir}")

    final_model, _ = load_model(checkpoints[-1][1], device)
    cache = build_activation_cache(
        final_model,
        tokens,
        offsets,
        shoe_ids,
        tables,
        run_dir / "probe" / "dynamics" / "final",
        device,
        kind,
    )
    split_p = split_positions_by_shoe(
        cache.positions.shoe_index,
        exp.probe.probe_eval_frac,
        exp.probe.probe_split_seed,
    )
    tr, ev = split_p.train_mask, split_p.eval_mask
    labels = load_labels(cache.positions, sidecar, state)
    rc, comp = labels.running_count, labels.remaining_composition
    per_layer = [
        held_out_r2(cache.layer(i)[tr], rc[tr], cache.layer(i)[ev], rc[ev], alphas)
        for i in range(cache.n_layer)
    ]
    best_layer = int(np.nanargmax(per_layer))

    loss_map = _loss_by_step(run_dir, loss_column)
    steps, rc_r2, comp_r2, losses = [], [], [], []
    for step, path in checkpoints:
        model, _ = load_model(path, device)
        c = build_activation_cache(
            model,
            tokens,
            offsets,
            shoe_ids,
            tables,
            run_dir / "probe" / "dynamics" / f"step_{step}",
            device,
            kind,
        )
        feats = c.layer(best_layer)
        rc_r2.append(held_out_r2(feats[tr], rc[tr], feats[ev], rc[ev], alphas))
        comp_r2.append(held_out_r2(feats[tr], comp[tr], feats[ev], comp[ev], alphas))
        steps.append(step)

        prior = [s for s in loss_map if s <= step]
        losses.append(loss_map[max(prior)] if prior else float("nan"))
        print(
            f"[dynamics] step {step}: rc_R2={rc_r2[-1]:.3f} comp_R2={comp_r2[-1]:.3f}"
        )

    return {
        "kind": kind,
        "best_layer": best_layer + 1,
        "loss_column": loss_column,
        "steps": steps,
        "running_count_r2": rc_r2,
        "composition_r2": comp_r2,
        "loss": losses,
    }


def plot_dynamics(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 5))
    steps = report["steps"]
    ax1.plot(
        steps,
        report["running_count_r2"],
        marker="o",
        color="tab:blue",
        label="running-count R²",
    )
    ax1.plot(
        steps,
        report["composition_r2"],
        marker="^",
        color="tab:purple",
        label="composition R²",
    )
    ax1.set_xlabel("training step")
    ax1.set_ylabel("held-out R²")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        steps,
        report["loss"],
        marker="s",
        color="tab:red",
        label=f"{report['loss_column']}",
    )
    ax2.set_ylabel(f"{report['loss_column']}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
    ax1.set_title(
        f"Count representation vs training ({report['kind']}, layer "
        f"{report['best_layer']})"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe training dynamics.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--kind", type=str, default="card")
    parser.add_argument("--loss-column", type=str, default="val_loss_bet")
    parser.add_argument("--max-shoes", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    exp = load_config(args.config)
    run_dir = Path(args.run_dir)
    device = resolve_device(args.device or exp.train.device)
    report = run_dynamics(
        exp, run_dir, device, args.kind, args.loss_column, args.max_shoes
    )
    out = run_dir / "probe" / "dynamics_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_dynamics(report, run_dir / "probe" / "dynamics.png")
    print(f"[dynamics] wrote {out} and dynamics.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

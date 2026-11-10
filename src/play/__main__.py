from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from blackjackgpt.checkpoint import load_model, resolve_checkpoint
from blackjackgpt.config import load_config
from blackjackgpt.expert.count_blind import count_blind_policy
from blackjackgpt.expert.policy import ExpertPolicy
from blackjackgpt.model import load_vocabulary
from blackjackgpt.play.harness import DecisionMaker
from blackjackgpt.provenance import dataset_manifest, git_commit
from blackjackgpt.train import resolve_device

from .ablation import AblationContext, load_specs, save_specs
from .decision import ModelDecisionMaker, ScriptedDecisionMaker
from .harness import play_shoes, summarize


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Model-in-the-loop play + ablation EV.")
    p.add_argument("--config-l1", default="configs/train_full.yaml")
    p.add_argument("--run-l1", default="runs/full-latest")
    p.add_argument("--config-l0", default="configs/train_lambda0.yaml")
    p.add_argument("--run-l0", default="runs/lambda0-v1")
    p.add_argument("--shoes", type=int, default=150_000)
    p.add_argument("--seed-base", type=int, default=1_000_000_000)
    p.add_argument("--concurrency", type=int, default=512)
    p.add_argument("--device", default=None)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--only",
        default=None,
        help="comma-separated condition names to run (default: all)",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="suffix for the output shard filename (for parallel groups)",
    )
    p.add_argument(
        "--ablation-specs",
        default=None,
        help="npz of precomputed ablation directions to load (parallel workers "
        "share one file instead of each rebuilding the activation cache)",
    )
    p.add_argument(
        "--build-specs-only",
        action="store_true",
        help="build the ablation specs, save them, and exit (no play)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch CPU thread cap (for parallel CPU workers)",
    )
    return p


def _row(name: str, dm, stats) -> dict:
    row = {
        "condition": name,
        "n_hands": stats.n_hands,
        "ev_on_action": stats.ev_on_action,
        "ev_action_se": stats.ev_action_se,
        "ev_on_initial": stats.ev_on_initial,
        "ev_se": stats.ev_se,
        "mean_bet": stats.mean_bet,
        "bet_count_correlation": stats.bet_count_correlation,
    }
    if isinstance(dm, ModelDecisionMaker):
        row["in_play_legality"] = dm.in_play_legality
        row["bet_match_expert"] = dm.bet_match / max(1, dm.bet_total)
        row["action_match_expert"] = dm.action_match / max(1, dm.action_total)
        row["deviation_compliance"] = dm.dev_match / max(1, dm.dev_total)
        row["basic_agreement_nondeviation"] = dm.nondev_match / max(1, dm.nondev_total)
    return row


def main(argv: list[str] | None = None) -> int:
    import torch

    args = build_arg_parser().parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device or "auto")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    exp_l1 = load_config(args.config_l1)
    exp_l0 = load_config(args.config_l0)
    run_l1 = Path(args.run_l1)
    run_l0 = Path(args.run_l0)
    vocab, _ = load_vocabulary(exp_l1.data.dataset_dir)

    print(f"[play] loading models on {device}")
    model_l1, _ = load_model(resolve_checkpoint(run_l1), device)
    model_l0, _ = load_model(resolve_checkpoint(run_l0), device)

    spec_meta: dict = {}
    if args.ablation_specs and Path(args.ablation_specs).exists():
        print(f"[play] loading ablation directions from {args.ablation_specs}")
        specs = load_specs(Path(args.ablation_specs))
    else:
        print(
            "[play] building ablation directions from the lambda=1 intervention layer"
        )
        actx = AblationContext(exp_l1, run_l1, device)
        print(
            f"[play] intervention layer (0-based block) = {actx.best_layer}; "
            f"running-count R^2 = {actx.running_count_r2:.3f}, "
            f"composition R^2 = {actx.composition_r2:.3f}"
        )
        specs = actx.all_specs()
        spec_meta = {
            "intervention_layer_block": actx.best_layer,
            "running_count_r2": actx.running_count_r2,
            "composition_r2": actx.composition_r2,
        }
        specs_path = Path(args.ablation_specs) if args.ablation_specs else out / "specs"
        save_specs(specs, specs_path.with_suffix(".npz"))
        (out / "spec_meta.json").write_text(json.dumps(spec_meta, indent=2))
        print(f"[play] saved specs to {specs_path.with_suffix('.npz')}")
        if args.build_specs_only:
            return 0

    def model_l1_dm(ablation=None):
        return ModelDecisionMaker(
            model_l1, vocab, device, ablation=ablation, audit_policy=ExpertPolicy()
        )

    conditions: list[tuple[str, Callable[[], DecisionMaker]]] = [
        ("hilo_expert", lambda: ScriptedDecisionMaker(ExpertPolicy())),
        ("basic_bot", lambda: ScriptedDecisionMaker(count_blind_policy())),
        ("l1_intact", lambda: model_l1_dm()),
        ("l1_count_ablated", lambda: model_l1_dm(specs["count"])),
        ("l1_random_ablated_k1", lambda: model_l1_dm(specs["random_k1"])),
        (
            "l0_intact",
            lambda: ModelDecisionMaker(
                model_l0, vocab, device, audit_policy=ExpertPolicy()
            ),
        ),
        ("l1_comp_ablated_k1", lambda: model_l1_dm(specs["comp_k1"])),
        ("l1_comp_ablated_k2", lambda: model_l1_dm(specs["comp_k2"])),
        ("l1_comp_ablated_k5", lambda: model_l1_dm(specs["comp_k5"])),
        ("l1_comp_ablated_k10", lambda: model_l1_dm(specs["comp_k10"])),
        ("l1_random_ablated_k10", lambda: model_l1_dm(specs["random_k10"])),
        ("l1_compvar_ablated_k1", lambda: model_l1_dm(specs["compvar_k1"])),
    ]

    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        conditions = [c for c in conditions if c[0] in wanted]
        missing = wanted - {c[0] for c in conditions}
        if missing:
            raise SystemExit(f"unknown conditions: {sorted(missing)}")

    if not spec_meta and (out / "spec_meta.json").exists():
        spec_meta = json.loads((out / "spec_meta.json").read_text())

    seeds = list(range(args.seed_base, args.seed_base + args.shoes))
    meta = {
        "shoes": args.shoes,
        "seed_base": args.seed_base,
        "seed_are_fresh": True,
        **spec_meta,
        "git": git_commit(),
        "run_l1": str(run_l1),
        "run_l0": str(run_l0),
        "dataset_l1": dataset_manifest(exp_l1.data.dataset_dir),
        "dataset_l0": dataset_manifest(exp_l0.data.dataset_dir),
        "checkpoint_l1": str(resolve_checkpoint(run_l1)),
        "checkpoint_l0": str(resolve_checkpoint(run_l0)),
    }

    shard = f"part_a_table{'_' + args.tag if args.tag else ''}.json"
    rows: list[dict] = []
    done: set[str] = set()
    if (out / shard).exists():
        prev = json.loads((out / shard).read_text())
        rows = prev.get("rows", [])
        done = {r["condition"] for r in rows}
        if done:
            print(f"[play] resuming; already have {sorted(done)}")

    for name, factory in conditions:
        if name in done:
            continue
        dm = factory()
        results = play_shoes(dm, seeds, vocab, args.concurrency)
        stats = summarize(results)
        row = _row(name, dm, stats)
        rows.append(row)
        print(
            f"[play] {name:24s} EV_action {row['ev_on_action']:+.4%} "
            f"+/- {row['ev_action_se']:.4%}  mean_bet {row['mean_bet']:.3f}  "
            f"bet~count r={row['bet_count_correlation']:+.3f}"
            + (
                f"  legal {row.get('in_play_legality', float('nan')):.4f}"
                if "in_play_legality" in row
                else ""
            )
        )

        (out / shard).write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))

    print(f"[play] wrote {out / shard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

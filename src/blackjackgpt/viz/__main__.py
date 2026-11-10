from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from blackjackgpt.train import resolve_device
from blackjackgpt.viz.deck_tracker import prepare_shoe_viz
from blackjackgpt.viz.render import (
    render_animation,
    render_filmstrip,
    render_infinite_contrast,
)

_L1_RUN = Path("runs/full-20260723-002040")
_L0_RUN = Path("runs/lambda0-v1")
_INF_RUN = Path("runs/infinite-deck-v1")
_L0_CAPTION = (
    "This model never saw anyone count or vary their bets — it learned to track "
    "the deck anyway, just from predicting the next card."
)


def _frame_near(frames, cards: int) -> int:
    return min(range(len(frames)), key=lambda i: abs(frames[i].cards_seen - cards))


def build(out: Path, device: torch.device) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    prov: dict = {}

    sv1 = prepare_shoe_viz("data/merged", _L1_RUN, device, "lambda1")
    render_animation(
        sv1, out / "shoe_animation_lambda1", title="Watch the model count the deck"
    )
    render_filmstrip(sv1, out / "filmstrip_lambda1.png")
    prov["lambda1"] = sv1.provenance | {
        "checkpoint": sv1.checkpoint,
        "bet_varies": sv1.has_bet_variation,
    }

    sv0 = prepare_shoe_viz("data/merged_lambda0", _L0_RUN, device, "lambda0")
    render_animation(
        sv0,
        out / "shoe_animation_lambda0",
        title="A model that never learned to bet — still counts",
        caption=_L0_CAPTION,
    )
    prov["lambda0"] = sv0.provenance | {
        "checkpoint": sv0.checkpoint,
        "bet_varies": sv0.has_bet_variation,
    }

    sv_inf = prepare_shoe_viz(
        "data/merged", _INF_RUN, device, "infinite", force_shoe=sv1.shoe_id
    )

    k = min(_frame_near(sv1.frames, 34), len(sv1.frames) - 2)
    render_infinite_contrast(
        sv1.frames[k].true_remaining,
        sv1.frames[k].model_remaining,
        sv_inf.frames[k].model_remaining,
        cards_seen=sv1.frames[k].cards_seen,
        out_path=out / "infinite_deck_contrast.png",
    )
    prov["infinite_contrast"] = sv_inf.provenance | {
        "checkpoint": sv_inf.checkpoint,
        "snapshot_card": sv1.frames[k].cards_seen,
    }

    (out / "provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n", encoding="utf-8"
    )
    return prov


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build deck-tracker visualizations.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    np.random.seed(0)
    torch.manual_seed(0)
    prov = build(args.out, resolve_device(args.device))
    print(
        f"[viz] shoe λ=1={prov['lambda1']['shoe_index']} "
        f"λ=0={prov['lambda0']['shoe_index']}; wrote artifacts to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

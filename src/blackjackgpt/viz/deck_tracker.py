from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from blackjackgpt.checkpoint import load_model, resolve_checkpoint
from blackjackgpt.dataset import load_stream, split_by_shoe
from blackjackgpt.model import load_vocabulary
from blackjackgpt.probe.activations import build_activation_cache
from blackjackgpt.probe.alignment import load_labels
from blackjackgpt.sidecar import load_sidecar
from blackjackgpt.tokens import TokenTables

_BAR_ORDER = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "A")
_RANK_INDEX = {
    "A": 0,
    "2": 1,
    "3": 2,
    "4": 3,
    "5": 4,
    "6": 5,
    "7": 6,
    "8": 7,
    "9": 8,
    "T": 9,
}


@dataclass
class Frame:
    true_remaining: np.ndarray
    model_remaining: np.ndarray
    player_cards: list[str]
    dealer_cards: list[str]
    just_dealt: str
    dealt_to: str
    bet: int
    hand_number: int
    cards_seen: int


@dataclass
class ShoeViz:
    frames: list[Frame]
    shoe_id: int
    checkpoint: str
    best_layer: int
    label: str
    has_bet_variation: bool
    provenance: dict = field(default_factory=dict)


def _fit_composition_probe(feats_tr, comp_tr, feats_ev, comp_ev, alphas):
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(feats_tr)
    model = RidgeCV(alphas=alphas).fit(scaler.transform(feats_tr), comp_tr)
    r2 = r2_score(
        comp_ev,
        model.predict(scaler.transform(feats_ev)),
        multioutput="variance_weighted",
    )

    def predict(feats):
        return model.predict(scaler.transform(feats))

    return predict, float(r2)


def _to_bar_order(comp_sidecar_order: np.ndarray) -> np.ndarray:
    idx = [_RANK_INDEX[r] for r in _BAR_ORDER]
    return comp_sidecar_order[..., idx]


def prepare_shoe_viz(
    dataset_dir: str,
    run_dir: Path,
    device: torch.device,
    label: str,
    alphas=(0.1, 1.0, 10.0, 100.0, 1000.0),
    fit_shoes: int = 700,
    candidate_shoes: int = 200,
    val_frac: float = 0.05,
    split_seed: int = 0,
    force_shoe: int | None = None,
) -> ShoeViz:
    vocab, _ = load_vocabulary(dataset_dir)
    tables = TokenTables(vocab)
    tokens, offsets = load_stream(dataset_dir)
    sidecar = load_sidecar(dataset_dir)
    model, _ = load_model(resolve_checkpoint(run_dir), device)
    ctx = model.config.context_len
    split = split_by_shoe(offsets, val_frac, split_seed)
    lengths = np.diff(offsets)
    val = np.array([s for s in split.val if s != force_shoe and lengths[s] <= ctx])
    fit_ids = val[:fit_shoes]
    candidate_ids = val[fit_shoes : fit_shoes + candidate_shoes]

    fit_cache = build_activation_cache(
        model,
        tokens,
        offsets,
        fit_ids,
        tables,
        run_dir / "viz" / "fit_cache",
        device,
        "card",
    )
    fit_labels = load_labels(fit_cache.positions, sidecar, "after")
    comp = fit_labels.remaining_composition
    n = len(comp)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    tr, ev = perm[: int(0.7 * n)], perm[int(0.7 * n) :]
    best_layer, best_r2, best_predict = 0, -np.inf, None
    for layer in range(fit_cache.n_layer):
        feats = fit_cache.layer(layer)
        predict, r2 = _fit_composition_probe(
            feats[tr], comp[tr], feats[ev], comp[ev], alphas
        )
        if r2 > best_r2:
            best_layer, best_r2, best_predict = layer, r2, predict
    assert best_predict is not None

    weight = np.array(tables.hilo_weight)
    running = np.cumsum(weight[tokens])
    if force_shoe is not None:
        s = force_shoe
        base = running[int(offsets[s])] - weight[tokens[int(offsets[s])]]
        best_shoe = int(s)
        best_excursion = float(
            np.abs(running[int(offsets[s]) : int(offsets[s + 1])] - base).max()
        )
    else:
        best_shoe, best_excursion = int(candidate_ids[0]), -1.0
        for s in candidate_ids:
            start, end = int(offsets[s]), int(offsets[s + 1])
            base = running[start] - weight[tokens[start]]
            excursion = float(np.abs(running[start:end] - base).max())
            if excursion > best_excursion:
                best_excursion, best_shoe = excursion, int(s)

    shoe_cache = build_activation_cache(
        model,
        tokens,
        offsets,
        np.array([best_shoe]),
        tables,
        run_dir / "viz" / "shoe_cache",
        device,
        "card",
    )
    decoded = best_predict(shoe_cache.layer(best_layer))
    true = load_labels(shoe_cache.positions, sidecar, "after").remaining_composition
    offset_to_row = {
        int(g): i for i, g in enumerate(shoe_cache.positions.global_offset)
    }

    frames = _walk_shoe_frames(
        tokens, offsets, best_shoe, vocab, decoded, true, offset_to_row
    )
    bets = {f.bet for f in frames}
    return ShoeViz(
        frames=frames,
        shoe_id=best_shoe,
        checkpoint=str(resolve_checkpoint(run_dir)),
        best_layer=best_layer,
        label=label,
        has_bet_variation=len(bets) > 1,
        provenance={
            "dataset_dir": dataset_dir,
            "shoe_index": best_shoe,
            "max_running_count_excursion": best_excursion,
            "read_out_layer_1indexed": best_layer + 1,
            "composition_read_out_r2_on_held_out": round(best_r2, 4),
            "selection": "max abs running-count excursion over candidate shoes",
        },
    )


def _walk_shoe_frames(
    tokens, offsets, shoe_id, vocab, decoded, true, offset_to_row
) -> list[Frame]:
    start, end = int(offsets[shoe_id]), int(offsets[shoe_id + 1])
    player: list[str] = []
    dealer: list[str] = []
    bet, hand = 0, 0
    frames: list[Frame] = []
    for g in range(start, end):
        text = vocab.decode(int(tokens[g]))
        if text == "ROUND_START":
            player, dealer, hand = [], [], hand + 1
            continue
        if text.startswith("BET_"):
            bet = int(text.split("_")[1])
            continue
        if text == "D_HOLE":
            dealer.append("?")
            continue
        if text.startswith(("P:", "D_UP:", "D_REVEAL:", "D_DRAW:")):
            rank = text.split(":")[1]
            to = "player" if text.startswith("P:") else "dealer"
            if text.startswith("D_UP:") or text.startswith("D_DRAW:"):
                dealer.append(rank)
            elif text.startswith("D_REVEAL:"):
                if "?" in dealer:
                    dealer[dealer.index("?")] = rank
                else:
                    dealer.append(rank)
            else:
                player.append(rank)
            row = offset_to_row.get(g)
            if row is None:
                continue
            frames.append(
                Frame(
                    true_remaining=_to_bar_order(true[row]),
                    model_remaining=_to_bar_order(np.clip(decoded[row], 0, None)),
                    player_cards=list(player),
                    dealer_cards=list(dealer),
                    just_dealt=rank,
                    dealt_to=to,
                    bet=bet,
                    hand_number=hand,
                    cards_seen=len(frames) + 1,
                )
            )
    return frames

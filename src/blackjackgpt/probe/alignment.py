from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blackjackgpt.data.vocabulary import Vocabulary
from blackjackgpt.sidecar import Sidecar
from blackjackgpt.tokens import TokenClass, TokenTables

_KIND_SPEC = {
    "card": (TokenClass.CARD, "after"),
    "bet": (TokenClass.BET, "before"),
}


@dataclass(frozen=True)
class Positions:
    shoe_index: np.ndarray
    within_pos: np.ndarray
    global_offset: np.ndarray
    token_id: np.ndarray

    def __len__(self) -> int:
        return len(self.global_offset)


CardPositions = Positions


@dataclass(frozen=True)
class ProbeLabels:
    running_count: np.ndarray
    raw_true_count: np.ndarray
    cards_dealt: np.ndarray
    remaining_composition: np.ndarray

    def as_targets(self) -> dict[str, np.ndarray]:
        return {
            "running_count": self.running_count,
            "raw_true_count": self.raw_true_count,
            "cards_dealt": self.cards_dealt,
            "remaining_composition": self.remaining_composition,
        }


def _token_ids_for(tables: TokenTables, token_class: TokenClass) -> np.ndarray:
    return np.array(sorted(tables.ids_in_class(token_class)), dtype=np.int64)


def select_positions(
    tokens: np.ndarray,
    offsets: np.ndarray,
    shoe_ids: np.ndarray,
    select_ids: np.ndarray,
) -> Positions:
    vocab_size = int(tokens.max()) + 1 if len(tokens) else 1
    lookup_size = max(int(select_ids.max()) + 1 if len(select_ids) else 1, vocab_size)
    lookup = np.zeros(lookup_size, dtype=bool)
    lookup[select_ids] = True

    shoe_out: list[np.ndarray] = []
    pos_out: list[np.ndarray] = []
    goff_out: list[np.ndarray] = []
    for shoe_id in shoe_ids:
        start, end = int(offsets[shoe_id]), int(offsets[shoe_id + 1])
        seq = tokens[start:end]
        local = np.nonzero(lookup[seq])[0]
        shoe_out.append(np.full(local.shape, shoe_id, dtype=np.int64))
        pos_out.append(local.astype(np.int64))
        goff_out.append((local + start).astype(np.int64))

    if shoe_out:
        shoe_index = np.concatenate(shoe_out)
        within_pos = np.concatenate(pos_out)
        global_offset = np.concatenate(goff_out)
    else:
        shoe_index = within_pos = global_offset = np.empty(0, dtype=np.int64)
    return Positions(
        shoe_index=shoe_index,
        within_pos=within_pos,
        global_offset=global_offset,
        token_id=tokens[global_offset],
    )


def positions_for_kind(
    tokens: np.ndarray,
    offsets: np.ndarray,
    shoe_ids: np.ndarray,
    tables: TokenTables,
    kind: str,
) -> Positions:
    token_class, _ = _KIND_SPEC[kind]
    return select_positions(
        tokens, offsets, shoe_ids, _token_ids_for(tables, token_class)
    )


def card_positions(
    tokens: np.ndarray,
    offsets: np.ndarray,
    shoe_ids: np.ndarray,
    tables: TokenTables,
) -> Positions:
    return positions_for_kind(tokens, offsets, shoe_ids, tables, "card")


def load_labels(
    positions: Positions, sidecar: Sidecar, state: str = "after"
) -> ProbeLabels:
    offsets = positions.global_offset
    sidecar_tok = sidecar.take("token_id", offsets)
    if not np.array_equal(sidecar_tok, positions.token_id):
        raise ValueError("sidecar token_id disagrees with tokens.npy at positions")
    return ProbeLabels(
        running_count=sidecar.take(f"{state}_running_count", offsets).astype(
            np.float64
        ),
        raw_true_count=sidecar.take(f"{state}_raw_true_count", offsets).astype(
            np.float64
        ),
        cards_dealt=sidecar.take(f"{state}_cards_dealt", offsets).astype(np.float64),
        remaining_composition=sidecar.remaining_composition(offsets, state).astype(
            np.float64
        ),
    )


def verify_alignment(
    tokens: np.ndarray,
    offsets: np.ndarray,
    shoe_ids: np.ndarray,
    vocab: Vocabulary,
    tables: TokenTables,
    sidecar: Sidecar,
    kind: str = "card",
) -> Positions:
    _, state = _KIND_SPEC[kind]
    positions = positions_for_kind(tokens, offsets, shoe_ids, tables, kind)
    labels = load_labels(positions, sidecar, state)

    weight = np.array(tables.hilo_weight, dtype=np.int64)
    deals = np.array(tables.deals_card, dtype=np.int64)
    running = np.cumsum(weight[tokens])
    dealt = np.cumsum(deals[tokens])
    recomputed_rc = running[positions.global_offset]
    recomputed_dealt = dealt[positions.global_offset]

    shuffle_id = vocab.encode("SHUFFLE")
    for shoe_id in np.unique(positions.shoe_index):
        start = int(offsets[shoe_id])
        assert tokens[start] == shuffle_id, "each shoe must begin with SHUFFLE"
        sel = positions.shoe_index == shoe_id
        base_rc = running[start] - weight[tokens[start]]
        base_dealt = dealt[start] - deals[tokens[start]]
        got_rc = recomputed_rc[sel] - base_rc
        got_dealt = recomputed_dealt[sel] - base_dealt
        assert np.array_equal(got_rc, labels.running_count[sel].astype(np.int64)), (
            f"running-count misalignment ({kind}) in shoe {shoe_id}"
        )
        assert np.array_equal(got_dealt, labels.cards_dealt[sel].astype(np.int64)), (
            f"cards-dealt misalignment ({kind}) in shoe {shoe_id}"
        )
    return positions

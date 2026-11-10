from __future__ import annotations

import numpy as np

from blackjackgpt.blackjack.cards import RANK_MULTIPLICITIES, Rank

RANKS: tuple[Rank, ...] = tuple(Rank)

FULL_DECK = np.array([RANK_MULTIPLICITIES[r] for r in RANKS], dtype=np.float64)


COUNTING_SYSTEMS: dict[str, np.ndarray] = {
    "hi_lo": np.array([-1, 1, 1, 1, 1, 1, 0, 0, 0, -1], dtype=np.float64),
    "ko": np.array([-1, 1, 1, 1, 1, 1, 1, 0, 0, -1], dtype=np.float64),
    "hi_opt_i": np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, -1], dtype=np.float64),
    "ace_side": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
}


def system_running_count(
    remaining_composition: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    dealt = FULL_DECK[None, :] - remaining_composition
    return dealt @ weights

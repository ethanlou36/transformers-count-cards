from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from blackjackgpt.blackjack.cards import RANK_MULTIPLICITIES, Rank

RANKS: tuple[Rank, ...] = tuple(Rank)
_COMP_COLS = tuple(f"latent_before_remaining_{r.value}" for r in RANKS)


def eor_ev_weights(
    dataset_dir: str | Path, alphas=(1, 10, 100, 1000, 1e4)
) -> np.ndarray:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    table = pq.read_table(
        Path(dataset_dir) / "events.parquet",
        columns=[
            "shoe_index",
            "round_id",
            "token_category",
            "profit_half_units",
            *_COMP_COLS,
        ],
    ).to_pandas()
    bets = table[table.token_category == "BET"]
    prof = (
        table.dropna(subset=["profit_half_units"])
        .groupby(["shoe_index", "round_id"])["profit_half_units"]
        .last()
        .reset_index()
    )
    merged = bets.merge(prof, on=["shoe_index", "round_id"], suffixes=("", "_r"))
    x = merged[list(_COMP_COLS)].to_numpy(float)
    y = merged["profit_half_units_r"].to_numpy(float) / 2.0
    xs = StandardScaler().fit_transform(x)
    model = RidgeCV(alphas=list(alphas)).fit(xs, y)

    return -np.asarray(model.coef_, dtype=np.float64)


def eor_entropy_weights() -> np.ndarray:
    full = np.array([RANK_MULTIPLICITIES[r] for r in RANKS], dtype=np.float64)

    def entropy(counts: np.ndarray) -> float:
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    base = entropy(full)
    out = np.zeros(len(RANKS))
    for i in range(len(RANKS)):
        c = full.copy()
        c[i] -= 1
        out[i] = entropy(c) - base
    return out

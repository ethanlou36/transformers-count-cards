from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from blackjackgpt.blackjack.cards import Rank


def remaining_composition_columns(state: str = "after") -> tuple[str, ...]:
    return tuple(f"latent_{state}_remaining_{r.value}" for r in Rank)


class Sidecar:
    def __init__(self, columns: dict[str, np.ndarray]) -> None:
        self._columns = columns
        offsets = columns["global_token_offset"]
        expected = np.arange(len(offsets))
        if not np.array_equal(offsets, expected):
            raise ValueError(
                "sidecar global_token_offset must be a dense 0..N-1 range after "
                "sorting; alignment cannot be trusted otherwise"
            )

    def __len__(self) -> int:
        return len(self._columns["global_token_offset"])

    def column(self, name: str) -> np.ndarray:
        return self._columns[name]

    def take(self, name: str, offsets: np.ndarray) -> np.ndarray:
        return self._columns[name][offsets]

    def remaining_composition(
        self, offsets: np.ndarray, state: str = "after"
    ) -> np.ndarray:
        return np.stack(
            [self._columns[c][offsets] for c in remaining_composition_columns(state)],
            axis=1,
        )


_STATE_COLUMNS = (
    tuple(
        f"{state}_{field}"
        for state in ("before", "after")
        for field in ("running_count", "raw_true_count", "cards_dealt")
    )
    + remaining_composition_columns("before")
    + remaining_composition_columns("after")
)

_DEFAULT_COLUMNS = (
    "global_token_offset",
    "shoe_index",
    "token_id",
    "token_category",
    "event_token",
    "decision_bet_units",
    "decision_raw_true_count",
    *_STATE_COLUMNS,
)


def load_sidecar(
    dataset_dir: str | Path, columns: tuple[str, ...] = _DEFAULT_COLUMNS
) -> Sidecar:
    table = pq.read_table(Path(dataset_dir) / "events.parquet", columns=list(columns))
    order = np.argsort(table.column("global_token_offset").to_numpy())
    data: dict[str, np.ndarray] = {}
    for name in columns:
        arr = table.column(name).to_numpy(zero_copy_only=False)
        data[name] = arr[order]
    return Sidecar(data)

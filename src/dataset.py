from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ShoeSplit:
    train: np.ndarray
    val: np.ndarray

    def __post_init__(self) -> None:
        overlap = np.intersect1d(self.train, self.val)
        if overlap.size:
            raise ValueError(f"shoe split leaks {overlap.size} shoes across sides")


def load_stream(dataset_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    root = Path(dataset_dir)
    tokens = np.load(root / "tokens.npy").astype(np.int64)
    offsets = np.load(root / "shoe_offsets.npy").astype(np.int64)
    if offsets[-1] != len(tokens):
        raise ValueError(
            "shoe_offsets last entry must equal len(tokens) "
            f"({offsets[-1]} != {len(tokens)})"
        )
    if np.any(np.diff(offsets) < 0):
        raise ValueError("shoe_offsets must be non-decreasing")
    return tokens, offsets


def num_shoes(offsets: np.ndarray) -> int:
    return len(offsets) - 1


def shoe_lengths(offsets: np.ndarray) -> np.ndarray:
    return np.diff(offsets)


def max_shoe_length(offsets: np.ndarray) -> int:
    return int(shoe_lengths(offsets).max())


def split_by_shoe(offsets: np.ndarray, val_frac: float, seed: int) -> ShoeSplit:
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be in [0, 1)")
    total = num_shoes(offsets)
    perm = np.random.default_rng(seed).permutation(total)
    n_val = round(total * val_frac)
    val = np.sort(perm[:n_val])
    train = np.sort(perm[n_val:])
    return ShoeSplit(train=train, val=val)


class ShoeDataset(Dataset):
    def __init__(
        self,
        tokens: np.ndarray,
        offsets: np.ndarray,
        shoe_ids: np.ndarray,
        context_len: int,
        pad_id: int,
    ) -> None:
        max_len = max_shoe_length(offsets)
        if context_len < max_len:
            raise ValueError(
                f"context_len ({context_len}) must be >= max shoe length "
                f"({max_len}); a shoe would be truncated"
            )
        self.tokens = tokens
        self.offsets = offsets
        self.shoe_ids = np.asarray(shoe_ids, dtype=np.int64)
        self.context_len = context_len
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.shoe_ids)

    def shoe_slice(self, shoe_id: int) -> np.ndarray:
        start, end = self.offsets[shoe_id], self.offsets[shoe_id + 1]
        return self.tokens[start:end]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shoe_id = int(self.shoe_ids[index])
        seq = self.shoe_slice(shoe_id)
        length = len(seq)

        padded = np.full(self.context_len, self.pad_id, dtype=np.int64)
        padded[:length] = seq
        x = torch.from_numpy(padded[:-1].copy())
        y = torch.from_numpy(padded[1:].copy())

        loss_mask = y != self.pad_id
        return {
            "input_ids": x,
            "targets": y,
            "loss_mask": loss_mask,
            "shoe_id": torch.tensor(shoe_id, dtype=torch.int64),
            "length": torch.tensor(length, dtype=torch.int64),
        }

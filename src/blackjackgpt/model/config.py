from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from blackjackgpt.data.vocabulary import Vocabulary


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    context_len: int
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    dropout: float = 0.0
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
            )
        if self.vocab_size <= 0 or self.context_len <= 0:
            raise ValueError("vocab_size and context_len must be positive")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id out of range")


def load_vocabulary(dataset_dir: str | Path) -> tuple[Vocabulary, int]:
    payload = json.loads((Path(dataset_dir) / "vocab.json").read_text(encoding="utf-8"))
    vocab = Vocabulary(version=payload["version"], tokens=tuple(payload["tokens"]))
    if vocab.sha256 != payload["sha256"]:
        raise ValueError(
            "vocab.json sha256 does not match its token list; refusing to load"
        )
    pad_id = int(payload["pad_token_id"])
    if vocab.tokens[pad_id] != payload["pad_token"]:
        raise ValueError("vocab.json pad_token_id does not point at pad_token")
    return vocab, pad_id

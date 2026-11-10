from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

VOCABULARY_VERSION = "blackjack-public-v1"


VERSION1_TOKEN_TEXTS: tuple[str, ...] = (
    "PAD",
    "SHUFFLE",
    "ROUND_START",
    "ROUND_END",
    "SHOE_END",
    "PEEK_CLEAR",
    "PEEK_BLACKJACK",
    "D_HOLE",
    "BET_1",
    "BET_2",
    "BET_4",
    "BET_6",
    "BET_8",
    "P:A",
    "P:2",
    "P:3",
    "P:4",
    "P:5",
    "P:6",
    "P:7",
    "P:8",
    "P:9",
    "P:T",
    "D_UP:A",
    "D_UP:2",
    "D_UP:3",
    "D_UP:4",
    "D_UP:5",
    "D_UP:6",
    "D_UP:7",
    "D_UP:8",
    "D_UP:9",
    "D_UP:T",
    "D_REVEAL:A",
    "D_REVEAL:2",
    "D_REVEAL:3",
    "D_REVEAL:4",
    "D_REVEAL:5",
    "D_REVEAL:6",
    "D_REVEAL:7",
    "D_REVEAL:8",
    "D_REVEAL:9",
    "D_REVEAL:T",
    "D_DRAW:A",
    "D_DRAW:2",
    "D_DRAW:3",
    "D_DRAW:4",
    "D_DRAW:5",
    "D_DRAW:6",
    "D_DRAW:7",
    "D_DRAW:8",
    "D_DRAW:9",
    "D_DRAW:T",
    "ACTION:HIT",
    "ACTION:STAND",
    "ACTION:DOUBLE",
    "OUTCOME:PLAYER_BLACKJACK",
    "OUTCOME:DEALER_BLACKJACK",
    "OUTCOME:BLACKJACK_PUSH",
    "OUTCOME:PLAYER_BUST",
    "OUTCOME:DEALER_BUST",
    "OUTCOME:WIN",
    "OUTCOME:LOSS",
    "OUTCOME:PUSH",
)


@dataclass(frozen=True, slots=True)
class Vocabulary:
    version: str
    tokens: tuple[str, ...]
    _token_to_id: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("vocabulary version must be a non-empty string")

        immutable_tokens = tuple(self.tokens)
        if not immutable_tokens:
            raise ValueError("vocabulary must contain at least one token")
        if any(not isinstance(token, str) or not token for token in immutable_tokens):
            raise ValueError("vocabulary tokens must be non-empty strings")
        if len(set(immutable_tokens)) != len(immutable_tokens):
            duplicates = sorted(
                token
                for token in set(immutable_tokens)
                if immutable_tokens.count(token) > 1
            )
            raise ValueError(f"vocabulary contains duplicate tokens: {duplicates}")

        object.__setattr__(self, "tokens", immutable_tokens)
        object.__setattr__(
            self,
            "_token_to_id",
            MappingProxyType(
                {token: token_id for token_id, token in enumerate(immutable_tokens)}
            ),
        )

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def token_to_id(self) -> Mapping[str, int]:
        return self._token_to_id

    def encode(self, token: str) -> int:
        if not isinstance(token, str):
            raise TypeError("token must be a string")
        try:
            return self._token_to_id[token]
        except KeyError as error:
            raise ValueError(f"unknown token: {token!r}") from error

    def decode(self, token_id: int) -> str:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("token ID must be an integer")
        if not 0 <= token_id < len(self.tokens):
            raise ValueError(f"unknown token ID: {token_id}")
        return self.tokens[token_id]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "tokens": list(self.tokens),
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_metadata(self) -> dict[str, object]:
        payload = self.canonical_payload()
        payload.update(
            {
                "size": len(self),
                "pad_token": "PAD",
                "pad_token_id": self._token_to_id.get("PAD"),
                "sha256": self.sha256,
            }
        )
        return payload


VERSION1_VOCABULARY = Vocabulary(
    version=VOCABULARY_VERSION,
    tokens=VERSION1_TOKEN_TEXTS,
)

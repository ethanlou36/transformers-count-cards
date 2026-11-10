from __future__ import annotations

from enum import Enum

from blackjackgpt.blackjack.cards import Rank
from blackjackgpt.data.vocabulary import Vocabulary
from blackjackgpt.expert.hilo import hi_lo_weight


class TokenClass(str, Enum):
    PAD = "PAD"
    BOUNDARY = "BOUNDARY"
    BET = "BET"
    CARD = "CARD"
    HIDDEN = "HIDDEN"
    PEEK = "PEEK"
    ACTION = "ACTION"
    OUTCOME = "OUTCOME"


_COUNTED_CARD_PREFIXES = ("P:", "D_UP:", "D_REVEAL:", "D_DRAW:")


_DEALT_CARD_PREFIXES = ("P:", "D_UP:", "D_DRAW:")


def classify_token(text: str) -> TokenClass:
    if text == "PAD":
        return TokenClass.PAD
    if text == "D_HOLE":
        return TokenClass.HIDDEN
    if text.startswith("BET_"):
        return TokenClass.BET
    if text.startswith("P:") or text.startswith(("D_UP:", "D_REVEAL:", "D_DRAW:")):
        return TokenClass.CARD
    if text.startswith("ACTION:"):
        return TokenClass.ACTION
    if text.startswith("OUTCOME:"):
        return TokenClass.OUTCOME
    if text.startswith("PEEK"):
        return TokenClass.PEEK
    return TokenClass.BOUNDARY


def rank_of(text: str) -> Rank | None:
    if ":" not in text:
        return None
    prefix, _, payload = text.partition(":")
    if prefix in {"P", "D_UP", "D_REVEAL", "D_DRAW"}:
        return Rank(payload)
    return None


def hilo_weight_of(text: str) -> int:
    if not text.startswith(_COUNTED_CARD_PREFIXES):
        return 0
    rank = rank_of(text)
    assert rank is not None
    return hi_lo_weight(rank)


def deals_physical_card(text: str) -> bool:
    return text == "D_HOLE" or text.startswith(_DEALT_CARD_PREFIXES)


class TokenTables:
    def __init__(self, vocab: Vocabulary) -> None:
        self.vocab = vocab
        self.texts: tuple[str, ...] = tuple(vocab.tokens)
        self.token_class: tuple[TokenClass, ...] = tuple(
            classify_token(text) for text in self.texts
        )
        self.hilo_weight: tuple[int, ...] = tuple(
            hilo_weight_of(text) for text in self.texts
        )
        self.deals_card: tuple[bool, ...] = tuple(
            deals_physical_card(text) for text in self.texts
        )

    def ids_in_class(self, cls: TokenClass) -> tuple[int, ...]:
        return tuple(i for i, c in enumerate(self.token_class) if c is cls)

    def card_position_ids(self) -> frozenset[int]:
        return frozenset(self.ids_in_class(TokenClass.CARD))

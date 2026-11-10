from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .cards import Card, Rank


@dataclass(frozen=True, slots=True)
class HandValue:
    hard_total: int
    best_legal_total: int | None
    usable_ace: bool
    is_bust: bool
    is_blackjack: bool

    @property
    def is_soft(self) -> bool:
        return self.usable_ace


@dataclass(frozen=True, slots=True)
class Hand:
    cards: tuple[Card, ...] = ()

    def add(self, card: Card) -> Hand:
        return Hand((*self.cards, card))

    @property
    def ranks(self) -> tuple[Rank, ...]:
        return tuple(card.rank for card in self.cards)

    @property
    def value(self) -> HandValue:
        return evaluate_hand(self.cards)


def evaluate_hand(cards: Iterable[Card]) -> HandValue:
    card_tuple = tuple(cards)
    hard_total = sum(card.rank.hard_value for card in card_tuple)
    contains_ace = any(card.rank is Rank.ACE for card in card_tuple)
    usable_ace = contains_ace and hard_total + 10 <= 21

    if usable_ace:
        best_legal_total: int | None = hard_total + 10
    elif hard_total <= 21:
        best_legal_total = hard_total
    else:
        best_legal_total = None

    is_bust = best_legal_total is None
    is_blackjack = len(card_tuple) == 2 and best_legal_total == 21
    return HandValue(
        hard_total=hard_total,
        best_legal_total=best_legal_total,
        usable_ace=usable_ace,
        is_bust=is_bust,
        is_blackjack=is_blackjack,
    )

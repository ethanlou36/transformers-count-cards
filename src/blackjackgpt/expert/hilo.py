from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from blackjackgpt.blackjack.cards import Rank
from blackjackgpt.blackjack.events import (
    DealerCardDrawn,
    DealerHoleCardRevealed,
    DealerUpcardExposed,
    EngineEvent,
    PlayerCardExposed,
)

HI_LO_WEIGHTS: Final[Mapping[Rank, int]] = MappingProxyType(
    {
        Rank.ACE: -1,
        Rank.TWO: 1,
        Rank.THREE: 1,
        Rank.FOUR: 1,
        Rank.FIVE: 1,
        Rank.SIX: 1,
        Rank.SEVEN: 0,
        Rank.EIGHT: 0,
        Rank.NINE: 0,
        Rank.TEN: -1,
    }
)


@dataclass(frozen=True, slots=True)
class TrueCountMetrics:
    running_count: int
    physical_cards_remaining: int
    numerator: int
    denominator: int
    raw_true_count: float
    floor: int
    ceiling: int
    truncation_toward_zero: int


@dataclass(frozen=True, slots=True)
class CountTransition:
    running_count_before: int
    running_count_after: int
    exposed_rank: Rank | None


def hi_lo_weight(rank: Rank | str) -> int:
    return HI_LO_WEIGHTS[Rank(rank)]


def exposed_rank_for_count(event: EngineEvent) -> Rank | None:
    if isinstance(
        event,
        (
            PlayerCardExposed,
            DealerUpcardExposed,
            DealerHoleCardRevealed,
            DealerCardDrawn,
        ),
    ):
        return event.card.rank
    return None


def update_running_count(running_count: int, event: EngineEvent) -> CountTransition:
    if isinstance(running_count, bool) or not isinstance(running_count, int):
        raise TypeError("running_count must be an integer")
    rank = exposed_rank_for_count(event)
    updated = running_count if rank is None else running_count + hi_lo_weight(rank)
    return CountTransition(
        running_count_before=running_count,
        running_count_after=updated,
        exposed_rank=rank,
    )


def true_count_metrics(
    running_count: int,
    physical_cards_remaining: int,
) -> TrueCountMetrics:
    if isinstance(running_count, bool) or not isinstance(running_count, int):
        raise TypeError("running_count must be an integer")
    if isinstance(physical_cards_remaining, bool) or not isinstance(
        physical_cards_remaining, int
    ):
        raise TypeError("physical_cards_remaining must be an integer")
    if physical_cards_remaining <= 0:
        raise ValueError("physical_cards_remaining must be positive")

    numerator = running_count * 52
    denominator = physical_cards_remaining
    raw = numerator / denominator
    return TrueCountMetrics(
        running_count=running_count,
        physical_cards_remaining=physical_cards_remaining,
        numerator=numerator,
        denominator=denominator,
        raw_true_count=raw,
        floor=math.floor(raw),
        ceiling=math.ceil(raw),
        truncation_toward_zero=math.trunc(raw),
    )


@dataclass(slots=True)
class HiLoCounter:
    running_count: int = 0
    exposed_cards_seen: int = 0

    def reset(self) -> None:
        self.running_count = 0
        self.exposed_cards_seen = 0

    def observe(self, event: EngineEvent) -> CountTransition:
        transition = update_running_count(self.running_count, event)
        self.running_count = transition.running_count_after
        if transition.exposed_rank is not None:
            self.exposed_cards_seen += 1
        return transition

    def true_count(self, physical_cards_remaining: int) -> TrueCountMetrics:
        return true_count_metrics(self.running_count, physical_cards_remaining)

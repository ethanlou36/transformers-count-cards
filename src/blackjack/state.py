from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum

from .cards import Card, Rank
from .hand import Hand


class Action(str, Enum):
    HIT = "HIT"
    STAND = "STAND"
    DOUBLE = "DOUBLE"


class Outcome(str, Enum):
    PLAYER_BLACKJACK = "PLAYER_BLACKJACK"
    DEALER_BLACKJACK = "DEALER_BLACKJACK"
    BLACKJACK_PUSH = "BLACKJACK_PUSH"
    PLAYER_BUST = "PLAYER_BUST"
    DEALER_BUST = "DEALER_BUST"
    WIN = "WIN"
    LOSS = "LOSS"
    PUSH = "PUSH"


class EnginePhase(str, Enum):
    IDLE = "IDLE"
    READY_FOR_ROUND = "READY_FOR_ROUND"
    PLAYER_TURN = "PLAYER_TURN"
    DEALER_TURN = "DEALER_TURN"
    SHOE_COMPLETE = "SHOE_COMPLETE"


class ShoeEndReason(str, Enum):
    CUT_CARD = "CUT_CARD"
    RESERVE_GUARD = "RESERVE_GUARD"


@dataclass(slots=True)
class ShoeState:
    shoe_id: str
    cards: tuple[Card, ...]
    position: int = 0

    @property
    def cards_dealt(self) -> int:
        return self.position

    @property
    def cards_remaining(self) -> int:
        return len(self.cards) - self.position

    @property
    def remaining_cards(self) -> tuple[Card, ...]:
        return self.cards[self.position :]

    def draw(self) -> Card:
        if self.position >= len(self.cards):
            raise IndexError("shoe is exhausted")
        card = self.cards[self.position]
        self.position += 1
        return card


@dataclass(slots=True)
class RoundState:
    round_id: int
    initial_bet_units: int
    current_wager_units: int
    player_hand: Hand
    dealer_hand: Hand
    dealer_hole_revealed: bool = False
    peek_completed: bool = False
    doubled: bool = False
    outcome: Outcome | None = None
    profit_half_units: int | None = None


@dataclass(frozen=True, slots=True)
class PublicRoundSnapshot:
    round_id: int
    player_cards: tuple[Rank, ...]
    dealer_cards: tuple[Rank, ...]
    dealer_upcard: Rank
    dealer_hole_revealed: bool
    initial_bet_units: int
    current_wager_units: int
    doubled: bool
    legal_actions: frozenset[Action]
    outcome: Outcome | None
    profit_half_units: int | None


@dataclass(frozen=True, slots=True)
class PrivateRoundSnapshot:
    round_id: int
    player_cards: tuple[Card, ...]
    dealer_cards: tuple[Card, ...]
    dealer_hole_revealed: bool
    initial_bet_units: int
    current_wager_units: int
    doubled: bool
    peek_completed: bool
    outcome: Outcome | None
    profit_half_units: int | None


@dataclass(frozen=True, slots=True)
class PublicEngineSnapshot:
    phase: EnginePhase
    shoe_id: str | None
    cards_dealt: int
    cards_remaining: int
    cumulative_profit_half_units: int
    round: PublicRoundSnapshot | None


@dataclass(frozen=True, slots=True)
class PrivateEngineSnapshot:
    phase: EnginePhase
    shoe_id: str | None
    cards_dealt: int
    cards_remaining: int
    cumulative_profit_half_units: int
    remaining_cards: tuple[Card, ...]
    remaining_rank_counts: tuple[tuple[Rank, int], ...]
    round: PrivateRoundSnapshot | None


def remaining_rank_counts(cards: tuple[Card, ...]) -> tuple[tuple[Rank, int], ...]:
    counts = Counter(card.rank for card in cards)
    return tuple((rank, counts[rank]) for rank in Rank)

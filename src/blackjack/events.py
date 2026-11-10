from __future__ import annotations

from dataclasses import dataclass

from .cards import Card
from .state import Action, Outcome, ShoeEndReason


@dataclass(frozen=True, slots=True)
class ShoeStarted:
    shoe_id: str
    total_cards: int


@dataclass(frozen=True, slots=True)
class RoundStarted:
    round_id: int


@dataclass(frozen=True, slots=True)
class BetPlaced:
    round_id: int
    bet_units: int


@dataclass(frozen=True, slots=True)
class PlayerCardExposed:
    round_id: int
    card: Card


@dataclass(frozen=True, slots=True)
class DealerUpcardExposed:
    round_id: int
    card: Card


@dataclass(frozen=True, slots=True)
class DealerHoleCardDealtHidden:
    round_id: int


@dataclass(frozen=True, slots=True)
class PeekCompleted:
    round_id: int
    has_blackjack: bool


@dataclass(frozen=True, slots=True)
class PlayerActionTaken:
    round_id: int
    action: Action


@dataclass(frozen=True, slots=True)
class DealerHoleCardRevealed:
    round_id: int
    card: Card


@dataclass(frozen=True, slots=True)
class DealerCardDrawn:
    round_id: int
    card: Card


@dataclass(frozen=True, slots=True)
class RoundSettled:
    round_id: int
    outcome: Outcome
    profit_half_units: int
    initial_bet_units: int
    current_wager_units: int


@dataclass(frozen=True, slots=True)
class RoundEnded:
    round_id: int


@dataclass(frozen=True, slots=True)
class ShoeEnded:
    shoe_id: str
    reason: ShoeEndReason
    cards_dealt: int


EngineEvent = (
    ShoeStarted
    | RoundStarted
    | BetPlaced
    | PlayerCardExposed
    | DealerUpcardExposed
    | DealerHoleCardDealtHidden
    | PeekCompleted
    | PlayerActionTaken
    | DealerHoleCardRevealed
    | DealerCardDrawn
    | RoundSettled
    | RoundEnded
    | ShoeEnded
)

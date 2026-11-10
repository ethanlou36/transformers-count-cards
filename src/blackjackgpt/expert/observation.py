from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from blackjackgpt.blackjack.cards import Card, Rank
from blackjackgpt.blackjack.hand import Hand
from blackjackgpt.blackjack.state import (
    Action,
    EnginePhase,
    PublicEngineSnapshot,
)

from .hilo import TrueCountMetrics, true_count_metrics


def _validate_common(
    running_count: int,
    physical_cards_remaining: int,
) -> None:
    true_count_metrics(running_count, physical_cards_remaining)


@dataclass(frozen=True, slots=True)
class BettingObservation:
    running_count: int
    physical_cards_remaining: int

    def __post_init__(self) -> None:
        _validate_common(
            self.running_count,
            self.physical_cards_remaining,
        )

    @property
    def true_count(self) -> TrueCountMetrics:
        return true_count_metrics(
            self.running_count,
            self.physical_cards_remaining,
        )

    @classmethod
    def from_public_snapshot(
        cls,
        snapshot: PublicEngineSnapshot,
        *,
        running_count: int,
    ) -> BettingObservation:
        if snapshot.phase is not EnginePhase.READY_FOR_ROUND:
            raise ValueError("betting observations require READY_FOR_ROUND")
        if snapshot.shoe_id is None:
            raise ValueError("public snapshot has no active shoe")
        return cls(
            running_count=running_count,
            physical_cards_remaining=snapshot.cards_remaining,
        )


@dataclass(frozen=True, slots=True)
class ActionObservation:
    player_cards: tuple[Rank, ...]
    dealer_upcard: Rank
    running_count: int
    physical_cards_remaining: int
    legal_actions: frozenset[Action]

    def __post_init__(self) -> None:
        _validate_common(
            self.running_count,
            self.physical_cards_remaining,
        )

        normalized_cards = tuple(Rank(rank) for rank in self.player_cards)
        if not normalized_cards:
            raise ValueError("player_cards must be non-empty")
        normalized_upcard = Rank(self.dealer_upcard)
        counts = Counter((*normalized_cards, normalized_upcard))
        if counts[Rank.TEN] > 16 or any(
            count > 4 for rank, count in counts.items() if rank is not Rank.TEN
        ):
            raise ValueError("visible cards exceed physical deck multiplicity")
        object.__setattr__(self, "player_cards", normalized_cards)
        object.__setattr__(self, "dealer_upcard", normalized_upcard)
        object.__setattr__(
            self,
            "legal_actions",
            frozenset(Action(action) for action in self.legal_actions),
        )

    @property
    def true_count(self) -> TrueCountMetrics:
        return true_count_metrics(
            self.running_count,
            self.physical_cards_remaining,
        )

    @property
    def player_hand(self) -> Hand:
        next_copy_index: Counter[Rank] = Counter()
        cards: list[Card] = []
        for rank in self.player_cards:
            copy_index = next_copy_index[rank]
            cards.append(Card(rank=rank, copy_index=copy_index))
            next_copy_index[rank] += 1
        return Hand(tuple(cards))

    @classmethod
    def from_public_snapshot(
        cls,
        snapshot: PublicEngineSnapshot,
        *,
        running_count: int,
    ) -> ActionObservation:
        if snapshot.phase is not EnginePhase.PLAYER_TURN:
            raise ValueError("action observations require PLAYER_TURN")
        if snapshot.shoe_id is None or snapshot.round is None:
            raise ValueError("public snapshot has no active player round")
        return cls(
            player_cards=snapshot.round.player_cards,
            dealer_upcard=snapshot.round.dealer_upcard,
            running_count=running_count,
            physical_cards_remaining=snapshot.cards_remaining,
            legal_actions=snapshot.round.legal_actions,
        )

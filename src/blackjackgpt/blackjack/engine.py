from __future__ import annotations

from collections.abc import Sequence

from .cards import Card, Rank, shuffled_deck, validate_standard_deck
from .events import (
    BetPlaced,
    DealerCardDrawn,
    DealerHoleCardDealtHidden,
    DealerHoleCardRevealed,
    DealerUpcardExposed,
    EngineEvent,
    PeekCompleted,
    PlayerActionTaken,
    PlayerCardExposed,
    RoundEnded,
    RoundSettled,
    RoundStarted,
    ShoeEnded,
    ShoeStarted,
)
from .hand import Hand
from .rules import VERSION1_RULES, RulesConfig
from .settlement import settle_hands
from .state import (
    Action,
    EnginePhase,
    PrivateEngineSnapshot,
    PrivateRoundSnapshot,
    PublicEngineSnapshot,
    PublicRoundSnapshot,
    RoundState,
    ShoeEndReason,
    ShoeState,
    remaining_rank_counts,
)


class BlackjackEngineError(RuntimeError):
    pass


class InvalidStateError(BlackjackEngineError):
    pass


class IllegalActionError(BlackjackEngineError):
    pass


class RoundCannotStartError(BlackjackEngineError):
    pass


class ShoeExhaustedError(BlackjackEngineError):
    pass


class BlackjackEngine:
    def __init__(self, rules: RulesConfig = VERSION1_RULES) -> None:
        if not rules.dealer_peeks:
            raise ValueError("version 1 requires dealer peek rules")
        self._rules = rules
        self._phase = EnginePhase.IDLE
        self._shoe: ShoeState | None = None
        self._round: RoundState | None = None
        self._next_round_id = 1
        self._cumulative_profit_half_units = 0

    @property
    def rules(self) -> RulesConfig:
        return self._rules

    @property
    def phase(self) -> EnginePhase:
        return self._phase

    @property
    def cumulative_profit_half_units(self) -> int:
        return self._cumulative_profit_half_units

    def start_shoe(
        self,
        ordered_deck: Sequence[Card],
        *,
        shoe_id: str = "shoe",
    ) -> tuple[EngineEvent, ...]:
        if self._phase in {EnginePhase.PLAYER_TURN, EnginePhase.DEALER_TURN}:
            raise InvalidStateError("cannot replace a shoe during an active round")
        if self._phase is EnginePhase.READY_FOR_ROUND:
            raise InvalidStateError("cannot replace a shoe before it is complete")
        if not shoe_id:
            raise ValueError("shoe_id must be non-empty")

        deck = tuple(ordered_deck)
        validate_standard_deck(deck)

        self._shoe = ShoeState(shoe_id=shoe_id, cards=deck)
        self._round = None
        self._next_round_id = 1
        self._cumulative_profit_half_units = 0
        self._phase = EnginePhase.READY_FOR_ROUND
        return (ShoeStarted(shoe_id=shoe_id, total_cards=len(deck)),)

    def start_shuffled_shoe(
        self,
        seed: int,
        *,
        stream: int = 54,
        shoe_id: str | None = None,
    ) -> tuple[EngineEvent, ...]:
        return self.start_shoe(
            shuffled_deck(seed, stream=stream),
            shoe_id=shoe_id or f"seed-{seed}",
        )

    def can_start_round(self) -> bool:
        if self._phase is not EnginePhase.READY_FOR_ROUND or self._shoe is None:
            return False
        if self._shoe.cards_dealt >= self._rules.cut_card_index:
            return False
        return self._shoe.cards_remaining >= self._rules.minimum_cards_to_start_round

    def start_round(self, bet_units: int) -> tuple[EngineEvent, ...]:
        self._require_phase(EnginePhase.READY_FOR_ROUND)
        if isinstance(bet_units, bool) or not isinstance(bet_units, int):
            raise ValueError("bet_units must be an integer")
        if bet_units <= 0:
            raise ValueError("bet_units must be positive")
        if not self.can_start_round():
            raise RoundCannotStartError(
                "cut card or minimum-card reserve prevents another round"
            )

        assert self._shoe is not None
        round_id = self._next_round_id
        self._next_round_id += 1
        round_state = RoundState(
            round_id=round_id,
            initial_bet_units=bet_units,
            current_wager_units=bet_units,
            player_hand=Hand(),
            dealer_hand=Hand(),
        )
        self._round = round_state

        events: list[EngineEvent] = [
            RoundStarted(round_id=round_id),
            BetPlaced(round_id=round_id, bet_units=bet_units),
        ]

        player_first = self._draw_card()
        round_state.player_hand = round_state.player_hand.add(player_first)
        events.append(PlayerCardExposed(round_id=round_id, card=player_first))

        dealer_upcard = self._draw_card()
        round_state.dealer_hand = round_state.dealer_hand.add(dealer_upcard)
        events.append(DealerUpcardExposed(round_id=round_id, card=dealer_upcard))

        player_second = self._draw_card()
        round_state.player_hand = round_state.player_hand.add(player_second)
        events.append(PlayerCardExposed(round_id=round_id, card=player_second))

        dealer_hole = self._draw_card()
        round_state.dealer_hand = round_state.dealer_hand.add(dealer_hole)
        events.append(DealerHoleCardDealtHidden(round_id=round_id))

        self._phase = EnginePhase.PLAYER_TURN
        upcard_requires_peek = dealer_upcard.rank in {Rank.ACE, Rank.TEN}
        if upcard_requires_peek:
            round_state.peek_completed = True
            dealer_has_blackjack = round_state.dealer_hand.value.is_blackjack
            events.append(
                PeekCompleted(
                    round_id=round_id,
                    has_blackjack=dealer_has_blackjack,
                )
            )
            if dealer_has_blackjack:
                events.extend(self._reveal_dealer_hole())
                events.extend(self._settle_round())
                return tuple(events)

        if round_state.player_hand.value.is_blackjack:
            if self._rules.reveal_hole_on_player_blackjack:
                events.extend(self._reveal_dealer_hole())
            events.extend(self._settle_round())
            return tuple(events)

        return tuple(events)

    def legal_actions(self) -> frozenset[Action]:
        if self._phase is not EnginePhase.PLAYER_TURN or self._round is None:
            return frozenset()

        value = self._round.player_hand.value
        if value.is_bust or value.best_legal_total == 21:
            return frozenset()

        actions = {Action.STAND}
        assert self._shoe is not None
        if self._shoe.cards_remaining > 0:
            actions.add(Action.HIT)
            if self._rules.double_allowed and len(self._round.player_hand.cards) == 2:
                actions.add(Action.DOUBLE)
        return frozenset(actions)

    def apply_action(self, action: Action | str) -> tuple[EngineEvent, ...]:
        self._require_phase(EnginePhase.PLAYER_TURN)
        try:
            normalized_action = Action(action)
        except ValueError as error:
            raise IllegalActionError(f"unknown player action: {action!r}") from error

        legal = self.legal_actions()
        if normalized_action not in legal:
            raise IllegalActionError(
                f"{normalized_action.value} is illegal; legal actions are "
                f"{sorted(item.value for item in legal)}"
            )

        assert self._round is not None
        events: list[EngineEvent] = [
            PlayerActionTaken(
                round_id=self._round.round_id,
                action=normalized_action,
            )
        ]

        if normalized_action is Action.STAND:
            self._phase = EnginePhase.DEALER_TURN
            return tuple(events)

        if normalized_action is Action.DOUBLE:
            self._round.current_wager_units *= 2
            self._round.doubled = True

        card = self._draw_card()
        self._round.player_hand = self._round.player_hand.add(card)
        events.append(PlayerCardExposed(round_id=self._round.round_id, card=card))

        if self._round.player_hand.value.is_bust:
            if self._rules.reveal_hole_on_player_bust:
                events.extend(self._reveal_dealer_hole())
            events.extend(self._settle_round())
            return tuple(events)

        if (
            normalized_action is Action.DOUBLE
            or self._round.player_hand.value.best_legal_total == 21
        ):
            self._phase = EnginePhase.DEALER_TURN
        return tuple(events)

    def play_dealer_and_settle(self) -> tuple[EngineEvent, ...]:
        self._require_phase(EnginePhase.DEALER_TURN)
        assert self._round is not None

        events: list[EngineEvent] = list(self._reveal_dealer_hole())
        while self._dealer_should_hit():
            card = self._draw_card()
            self._round.dealer_hand = self._round.dealer_hand.add(card)
            events.append(DealerCardDrawn(round_id=self._round.round_id, card=card))

        events.extend(self._settle_round())
        return tuple(events)

    def public_snapshot(self) -> PublicEngineSnapshot:
        shoe = self._shoe
        round_snapshot: PublicRoundSnapshot | None = None
        if self._round is not None:
            dealer_cards = self._round.dealer_hand.cards
            public_dealer_cards = (
                tuple(card.rank for card in dealer_cards)
                if self._round.dealer_hole_revealed
                else tuple(card.rank for card in dealer_cards[:1])
            )
            round_snapshot = PublicRoundSnapshot(
                round_id=self._round.round_id,
                player_cards=self._round.player_hand.ranks,
                dealer_cards=public_dealer_cards,
                dealer_upcard=dealer_cards[0].rank,
                dealer_hole_revealed=self._round.dealer_hole_revealed,
                initial_bet_units=self._round.initial_bet_units,
                current_wager_units=self._round.current_wager_units,
                doubled=self._round.doubled,
                legal_actions=self.legal_actions(),
                outcome=self._round.outcome,
                profit_half_units=self._round.profit_half_units,
            )

        return PublicEngineSnapshot(
            phase=self._phase,
            shoe_id=shoe.shoe_id if shoe else None,
            cards_dealt=shoe.cards_dealt if shoe else 0,
            cards_remaining=shoe.cards_remaining if shoe else 0,
            cumulative_profit_half_units=self._cumulative_profit_half_units,
            round=round_snapshot,
        )

    def private_snapshot(self) -> PrivateEngineSnapshot:
        shoe = self._shoe
        round_snapshot: PrivateRoundSnapshot | None = None
        if self._round is not None:
            round_snapshot = PrivateRoundSnapshot(
                round_id=self._round.round_id,
                player_cards=self._round.player_hand.cards,
                dealer_cards=self._round.dealer_hand.cards,
                dealer_hole_revealed=self._round.dealer_hole_revealed,
                initial_bet_units=self._round.initial_bet_units,
                current_wager_units=self._round.current_wager_units,
                doubled=self._round.doubled,
                peek_completed=self._round.peek_completed,
                outcome=self._round.outcome,
                profit_half_units=self._round.profit_half_units,
            )

        remaining_cards = shoe.remaining_cards if shoe else ()
        return PrivateEngineSnapshot(
            phase=self._phase,
            shoe_id=shoe.shoe_id if shoe else None,
            cards_dealt=shoe.cards_dealt if shoe else 0,
            cards_remaining=shoe.cards_remaining if shoe else 0,
            cumulative_profit_half_units=self._cumulative_profit_half_units,
            remaining_cards=remaining_cards,
            remaining_rank_counts=remaining_rank_counts(remaining_cards),
            round=round_snapshot,
        )

    def _require_phase(self, expected: EnginePhase) -> None:
        if self._phase is not expected:
            raise InvalidStateError(
                f"expected engine phase {expected.value}, got {self._phase.value}"
            )

    def _draw_card(self) -> Card:
        assert self._shoe is not None
        try:
            return self._shoe.draw()
        except IndexError as error:
            raise ShoeExhaustedError(
                "active round exhausted the shoe; no reshuffle was performed"
            ) from error

    def _reveal_dealer_hole(self) -> tuple[EngineEvent, ...]:
        assert self._round is not None
        if self._round.dealer_hole_revealed:
            return ()
        self._round.dealer_hole_revealed = True
        hole_card = self._round.dealer_hand.cards[1]
        return (
            DealerHoleCardRevealed(
                round_id=self._round.round_id,
                card=hole_card,
            ),
        )

    def _dealer_should_hit(self) -> bool:
        assert self._round is not None
        value = self._round.dealer_hand.value
        if value.is_bust:
            return False
        assert value.best_legal_total is not None
        if value.best_legal_total < 17:
            return True
        if value.best_legal_total > 17:
            return False
        return value.is_soft and not self._rules.dealer_stands_soft_17

    def _settle_round(self) -> tuple[EngineEvent, ...]:
        assert self._round is not None
        if self._round.outcome is not None:
            raise InvalidStateError("round has already been settled")

        settlement = settle_hands(
            player_hand=self._round.player_hand,
            dealer_hand=self._round.dealer_hand,
            initial_bet_units=self._round.initial_bet_units,
            current_wager_units=self._round.current_wager_units,
            rules=self._rules,
        )
        self._round.outcome = settlement.outcome
        self._round.profit_half_units = settlement.profit_half_units
        self._cumulative_profit_half_units += settlement.profit_half_units

        events: list[EngineEvent] = [
            RoundSettled(
                round_id=self._round.round_id,
                outcome=settlement.outcome,
                profit_half_units=settlement.profit_half_units,
                initial_bet_units=self._round.initial_bet_units,
                current_wager_units=self._round.current_wager_units,
            ),
            RoundEnded(round_id=self._round.round_id),
        ]
        events.extend(self._finish_shoe_if_needed())
        return tuple(events)

    def _finish_shoe_if_needed(self) -> tuple[EngineEvent, ...]:
        assert self._shoe is not None
        reason: ShoeEndReason | None = None
        if self._shoe.cards_dealt >= self._rules.cut_card_index:
            reason = ShoeEndReason.CUT_CARD
        elif self._shoe.cards_remaining < self._rules.minimum_cards_to_start_round:
            reason = ShoeEndReason.RESERVE_GUARD

        if reason is None:
            self._phase = EnginePhase.READY_FOR_ROUND
            return ()

        self._phase = EnginePhase.SHOE_COMPLETE
        return (
            ShoeEnded(
                shoe_id=self._shoe.shoe_id,
                reason=reason,
                cards_dealt=self._shoe.cards_dealt,
            ),
        )

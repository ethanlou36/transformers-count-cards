from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from blackjackgpt.blackjack.cards import Card, Rank, validate_standard_deck
from blackjackgpt.blackjack.engine import BlackjackEngine
from blackjackgpt.blackjack.events import (
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
from blackjackgpt.blackjack.state import Action, Outcome
from blackjackgpt.expert.hilo import HiLoCounter, true_count_metrics
from blackjackgpt.expert.policy import (
    ActionDecision,
    ActionSource,
    BetDecision,
)

from .seeds import DerivedShoeSeed
from .vocabulary import VERSION1_VOCABULARY, Vocabulary

SIDECAR_SCHEMA_VERSION = "blackjack-event-sidecar-v1"


class TraceInvariantError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SemanticEvent:
    event_type: str
    token: str
    category: str


@dataclass(frozen=True, slots=True)
class TraceSnapshot:
    cards_dealt: int
    cards_remaining: int
    running_count: int
    true_count_numerator: int
    true_count_denominator: int
    raw_true_count: float | None
    true_count_floor: int | None
    true_count_ceiling: int | None
    true_count_truncation: int | None
    exposed_cards: tuple[str, ...]
    player_hand: tuple[str, ...]
    dealer_visible_hand: tuple[str, ...]
    dealer_upcard: str | None
    latent_dealer_hole_rank: str | None
    dealer_hole_observable: bool
    initial_bet_units: int | None
    wager_units: int | None
    doubled: bool
    peek_completed: bool
    outcome: str | None
    round_profit_half_units: int | None
    cumulative_profit_half_units: int
    latent_remaining_rank_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RecordedShoe:
    token_texts: tuple[str, ...]
    token_ids: tuple[int, ...]
    event_rows: tuple[dict[str, Any], ...]


def semantic_event_for(event: EngineEvent) -> SemanticEvent:
    if isinstance(event, ShoeStarted):
        return SemanticEvent(type(event).__name__, "SHUFFLE", "BOUNDARY")
    if isinstance(event, RoundStarted):
        return SemanticEvent(type(event).__name__, "ROUND_START", "BOUNDARY")
    if isinstance(event, BetPlaced):
        return SemanticEvent(type(event).__name__, f"BET_{event.bet_units}", "BET")
    if isinstance(event, PlayerCardExposed):
        return SemanticEvent(
            type(event).__name__, f"P:{event.card.rank.value}", "PLAYER_CARD"
        )
    if isinstance(event, DealerUpcardExposed):
        return SemanticEvent(
            type(event).__name__,
            f"D_UP:{event.card.rank.value}",
            "DEALER_CARD",
        )
    if isinstance(event, DealerHoleCardDealtHidden):
        return SemanticEvent(type(event).__name__, "D_HOLE", "HIDDEN_MARKER")
    if isinstance(event, PeekCompleted):
        token = "PEEK_BLACKJACK" if event.has_blackjack else "PEEK_CLEAR"
        return SemanticEvent(type(event).__name__, token, "PEEK")
    if isinstance(event, PlayerActionTaken):
        return SemanticEvent(
            type(event).__name__, f"ACTION:{event.action.value}", "ACTION"
        )
    if isinstance(event, DealerHoleCardRevealed):
        return SemanticEvent(
            type(event).__name__,
            f"D_REVEAL:{event.card.rank.value}",
            "DEALER_CARD",
        )
    if isinstance(event, DealerCardDrawn):
        return SemanticEvent(
            type(event).__name__,
            f"D_DRAW:{event.card.rank.value}",
            "DEALER_CARD",
        )
    if isinstance(event, RoundSettled):
        return SemanticEvent(
            type(event).__name__, f"OUTCOME:{event.outcome.value}", "OUTCOME"
        )
    if isinstance(event, RoundEnded):
        return SemanticEvent(type(event).__name__, "ROUND_END", "BOUNDARY")
    if isinstance(event, ShoeEnded):
        return SemanticEvent(type(event).__name__, "SHOE_END", "BOUNDARY")
    raise TypeError(f"unsupported engine event type: {type(event).__name__}")


class TraceReducer:
    def __init__(self, ordered_deck: Sequence[Card]) -> None:
        deck = tuple(ordered_deck)
        validate_standard_deck(deck)
        self._deck = deck
        self._position = 0
        self._counter = HiLoCounter()
        self._shoe_id: str | None = None
        self._round_id: int | None = None
        self._player_cards: list[Card] = []
        self._dealer_cards: list[Card] = []
        self._dealer_hole_observable = False
        self._initial_bet_units: int | None = None
        self._wager_units: int | None = None
        self._doubled = False
        self._peek_completed = False
        self._outcome: Outcome | None = None
        self._round_profit_half_units: int | None = None
        self._cumulative_profit_half_units = 0
        self._exposed_cards: list[Rank] = []
        self._shoe_end_reason: str | None = None

    @property
    def running_count(self) -> int:
        return self._counter.running_count

    @property
    def cards_remaining(self) -> int:
        return len(self._deck) - self._position

    @property
    def cards_dealt(self) -> int:
        return self._position

    @property
    def round_id(self) -> int | None:
        return self._round_id

    @property
    def shoe_end_reason(self) -> str | None:
        return self._shoe_end_reason

    def freeze(self) -> TraceSnapshot:
        remaining_cards = self._deck[self._position :]
        counts = Counter(card.rank for card in remaining_cards)
        count_tuple = tuple(counts[rank] for rank in Rank)
        numerator = self.running_count * 52
        denominator = self.cards_remaining
        if denominator:
            metrics = true_count_metrics(self.running_count, denominator)
            raw: float | None = metrics.raw_true_count
            floor: int | None = metrics.floor
            ceiling: int | None = metrics.ceiling
            truncation: int | None = metrics.truncation_toward_zero
        else:
            raw = None
            floor = None
            ceiling = None
            truncation = None

        dealer_upcard = self._dealer_cards[0].rank.value if self._dealer_cards else None
        dealer_hole = (
            self._dealer_cards[1].rank.value if len(self._dealer_cards) >= 2 else None
        )
        if self._dealer_hole_observable:
            visible_dealer = tuple(card.rank.value for card in self._dealer_cards)
        else:
            visible_dealer = (
                (self._dealer_cards[0].rank.value,) if self._dealer_cards else ()
            )

        return TraceSnapshot(
            cards_dealt=self.cards_dealt,
            cards_remaining=self.cards_remaining,
            running_count=self.running_count,
            true_count_numerator=numerator,
            true_count_denominator=denominator,
            raw_true_count=raw,
            true_count_floor=floor,
            true_count_ceiling=ceiling,
            true_count_truncation=truncation,
            exposed_cards=tuple(rank.value for rank in self._exposed_cards),
            player_hand=tuple(card.rank.value for card in self._player_cards),
            dealer_visible_hand=visible_dealer,
            dealer_upcard=dealer_upcard,
            latent_dealer_hole_rank=dealer_hole,
            dealer_hole_observable=self._dealer_hole_observable,
            initial_bet_units=self._initial_bet_units,
            wager_units=self._wager_units,
            doubled=self._doubled,
            peek_completed=self._peek_completed,
            outcome=self._outcome.value if self._outcome else None,
            round_profit_half_units=self._round_profit_half_units,
            cumulative_profit_half_units=self._cumulative_profit_half_units,
            latent_remaining_rank_counts=count_tuple,
        )

    def apply(self, event: EngineEvent) -> None:
        if isinstance(event, ShoeStarted):
            if event.total_cards != len(self._deck):
                raise TraceInvariantError("shoe event total does not match trace deck")
            self._shoe_id = event.shoe_id
            self._position = 0
            self._counter.reset()
            self._round_id = None
            self._player_cards = []
            self._dealer_cards = []
            self._dealer_hole_observable = False
            self._initial_bet_units = None
            self._wager_units = None
            self._doubled = False
            self._peek_completed = False
            self._outcome = None
            self._round_profit_half_units = None
            self._cumulative_profit_half_units = 0
            self._exposed_cards = []
            self._shoe_end_reason = None
            return

        if self._shoe_id is None:
            raise TraceInvariantError("event arrived before ShoeStarted")

        if isinstance(event, RoundStarted):
            self._round_id = event.round_id
            self._player_cards = []
            self._dealer_cards = []
            self._dealer_hole_observable = False
            self._initial_bet_units = None
            self._wager_units = None
            self._doubled = False
            self._peek_completed = False
            self._outcome = None
            self._round_profit_half_units = None
        elif isinstance(event, BetPlaced):
            self._require_round(event.round_id)
            self._initial_bet_units = event.bet_units
            self._wager_units = event.bet_units
        elif isinstance(event, PlayerCardExposed):
            self._require_round(event.round_id)
            card = self._consume(event.card)
            self._player_cards.append(card)
        elif isinstance(event, DealerUpcardExposed):
            self._require_round(event.round_id)
            card = self._consume(event.card)
            if self._dealer_cards:
                raise TraceInvariantError("dealer upcard was dealt more than once")
            self._dealer_cards.append(card)
        elif isinstance(event, DealerHoleCardDealtHidden):
            self._require_round(event.round_id)
            if len(self._dealer_cards) != 1:
                raise TraceInvariantError("hidden hole marker has invalid deal order")
            self._dealer_cards.append(self._consume())
        elif isinstance(event, PeekCompleted):
            self._require_round(event.round_id)
            self._peek_completed = True
        elif isinstance(event, PlayerActionTaken):
            self._require_round(event.round_id)
            if event.action is Action.DOUBLE:
                if self._wager_units is None:
                    raise TraceInvariantError("double occurred before bet")
                self._wager_units *= 2
                self._doubled = True
        elif isinstance(event, DealerHoleCardRevealed):
            self._require_round(event.round_id)
            if len(self._dealer_cards) < 2 or self._dealer_cards[1] != event.card:
                raise TraceInvariantError("revealed dealer hole does not match trace")
            if self._dealer_hole_observable:
                raise TraceInvariantError("dealer hole was revealed more than once")
            self._dealer_hole_observable = True
        elif isinstance(event, DealerCardDrawn):
            self._require_round(event.round_id)
            self._dealer_cards.append(self._consume(event.card))
        elif isinstance(event, RoundSettled):
            self._require_round(event.round_id)
            if event.initial_bet_units != self._initial_bet_units:
                raise TraceInvariantError("settlement initial bet disagrees with trace")
            if event.current_wager_units != self._wager_units:
                raise TraceInvariantError("settlement wager disagrees with trace")
            self._outcome = event.outcome
            self._round_profit_half_units = event.profit_half_units
            self._cumulative_profit_half_units += event.profit_half_units
        elif isinstance(event, RoundEnded):
            self._require_round(event.round_id)
            if self._outcome is None:
                raise TraceInvariantError("round ended before settlement")
        elif isinstance(event, ShoeEnded):
            if event.shoe_id != self._shoe_id:
                raise TraceInvariantError("shoe end ID disagrees with trace")
            if event.cards_dealt != self.cards_dealt:
                raise TraceInvariantError("shoe end position disagrees with trace")
            self._shoe_end_reason = event.reason.value
        else:
            raise TypeError(f"unsupported engine event type: {type(event).__name__}")

        transition = self._counter.observe(event)
        if transition.exposed_rank is not None:
            self._exposed_cards.append(transition.exposed_rank)

    def assert_matches_engine(self, engine: BlackjackEngine) -> None:
        public = engine.public_snapshot()
        private = engine.private_snapshot()
        failures: list[str] = []
        if self._shoe_id != private.shoe_id:
            failures.append("shoe_id")
        if self.cards_dealt != private.cards_dealt:
            failures.append("cards_dealt")
        if self.cards_remaining != private.cards_remaining:
            failures.append("cards_remaining")
        if self._cumulative_profit_half_units != private.cumulative_profit_half_units:
            failures.append("cumulative_profit")
        if tuple(self._deck[self._position :]) != private.remaining_cards:
            failures.append("remaining_cards")
        expected_counts = tuple(
            (rank, self.freeze().latent_remaining_rank_counts[index])
            for index, rank in enumerate(Rank)
        )
        if expected_counts != private.remaining_rank_counts:
            failures.append("remaining_rank_counts")

        if private.round is None:
            if self._round_id is not None:
                failures.append("round_presence")
        else:
            if self._round_id != private.round.round_id:
                failures.append("round_id")
            if tuple(self._player_cards) != private.round.player_cards:
                failures.append("player_cards")
            if tuple(self._dealer_cards) != private.round.dealer_cards:
                failures.append("dealer_cards")
            if self._dealer_hole_observable != private.round.dealer_hole_revealed:
                failures.append("dealer_hole_revealed")
            if self._initial_bet_units != private.round.initial_bet_units:
                failures.append("initial_bet")
            if self._wager_units != private.round.current_wager_units:
                failures.append("wager")
            if self._doubled != private.round.doubled:
                failures.append("doubled")
            if self._peek_completed != private.round.peek_completed:
                failures.append("peek_completed")
            if self._outcome != private.round.outcome:
                failures.append("outcome")
            if self._round_profit_half_units != private.round.profit_half_units:
                failures.append("round_profit")

        if public.cards_dealt != self.cards_dealt:
            failures.append("public_cards_dealt")
        if public.cards_remaining != self.cards_remaining:
            failures.append("public_cards_remaining")
        if failures:
            raise TraceInvariantError(
                "trace/engine mismatch after event batch: " + ", ".join(failures)
            )

    def _consume(self, expected: Card | None = None) -> Card:
        if self._position >= len(self._deck):
            raise TraceInvariantError("trace attempted to consume past deck end")
        card = self._deck[self._position]
        if expected is not None and card != expected:
            raise TraceInvariantError(
                f"event card {expected.instance_id} does not match "
                f"deck card {card.instance_id} at position {self._position}"
            )
        self._position += 1
        return card

    def _require_round(self, round_id: int) -> None:
        if self._round_id != round_id:
            raise TraceInvariantError(
                f"event round {round_id} does not match active {self._round_id}"
            )


class ShoeRecorder:
    def __init__(
        self,
        *,
        ordered_deck: Sequence[Card],
        shoe_id: str,
        shoe_index: int,
        split: str,
        derived_seed: DerivedShoeSeed,
        global_token_start: int,
        vocabulary: Vocabulary = VERSION1_VOCABULARY,
    ) -> None:
        self.trace = TraceReducer(ordered_deck)
        self._shoe_id = shoe_id
        self._shoe_index = shoe_index
        self._split = split
        self._derived_seed = derived_seed
        self._global_token_start = global_token_start
        self._vocabulary = vocabulary
        self._token_texts: list[str] = []
        self._token_ids: list[int] = []
        self._rows: list[dict[str, Any]] = []

    def process_batch(
        self,
        events: Sequence[EngineEvent],
        *,
        engine: BlackjackEngine,
        bet_decision: BetDecision | None = None,
        action_decision: ActionDecision | None = None,
    ) -> None:
        for event in events:
            before = self.trace.freeze()
            self.trace.apply(event)
            after = self.trace.freeze()
            semantic = semantic_event_for(event)

            attached_bet = bet_decision if isinstance(event, BetPlaced) else None
            attached_action = (
                action_decision if isinstance(event, PlayerActionTaken) else None
            )
            if isinstance(event, BetPlaced) and (
                attached_bet is None or attached_bet.bet_units != event.bet_units
            ):
                raise TraceInvariantError("bet event lacks its matching decision")
            if isinstance(event, PlayerActionTaken) and (
                attached_action is None
                or attached_action.selected_action is not event.action
            ):
                raise TraceInvariantError("action event lacks its matching decision")

            token_id = self._vocabulary.encode(semantic.token)
            row = self._build_row(
                event=event,
                semantic=semantic,
                token_id=token_id,
                before=before,
                after=after,
                bet_decision=attached_bet,
                action_decision=attached_action,
            )
            self._token_texts.append(semantic.token)
            self._token_ids.append(token_id)
            self._rows.append(row)

        self.trace.assert_matches_engine(engine)

    def finish(self) -> RecordedShoe:
        if not self._token_texts or self._token_texts[0] != "SHUFFLE":
            raise TraceInvariantError("recorded shoe does not begin with SHUFFLE")
        if self._token_texts[-1] != "SHOE_END":
            raise TraceInvariantError("recorded shoe does not end with SHOE_END")
        return RecordedShoe(
            token_texts=tuple(self._token_texts),
            token_ids=tuple(self._token_ids),
            event_rows=tuple(self._rows),
        )

    def _build_row(
        self,
        *,
        event: EngineEvent,
        semantic: SemanticEvent,
        token_id: int,
        before: TraceSnapshot,
        after: TraceSnapshot,
        bet_decision: BetDecision | None,
        action_decision: ActionDecision | None,
    ) -> dict[str, Any]:
        event_index = len(self._rows)
        round_id = getattr(event, "round_id", None)
        row: dict[str, Any] = {
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "split": self._split,
            "shoe_id": self._shoe_id,
            "shoe_index": self._shoe_index,
            "round_id": round_id,
            "event_index": event_index,
            "global_token_offset": self._global_token_start + event_index,
            "derived_seed_hex": self._derived_seed.digest_hex,
            "pcg_seed": self._derived_seed.pcg_seed,
            "pcg_stream": self._derived_seed.pcg_stream,
            "event_type": semantic.event_type,
            "event_token": semantic.token,
            "token_id": token_id,
            "token_category": semantic.category,
            **_snapshot_columns("before", before),
            **_snapshot_columns("after", after),
            "physical_cards_remaining": after.cards_remaining,
            "exposed_cards_seen": list(after.exposed_cards),
            "player_hand": list(after.player_hand),
            "dealer_upcard": after.dealer_upcard,
            "latent_dealer_hole_rank": after.latent_dealer_hole_rank,
            "dealer_hole_observable": after.dealer_hole_observable,
            "legal_actions": [],
            "basic_strategy_recommendation": None,
            "basic_strategy_action": None,
            "matched_deviation_id": None,
            "applied_deviation_id": None,
            "deviation_index_threshold": None,
            "deviation_threshold_met": None,
            "deviation_preferred_action": None,
            "deviation_fallback_used": False,
            "deviation_activated": False,
            "final_expert_action": None,
            "action_source": None,
            "bet_ramp_entry_index": None,
            "bet_ramp_minimum_raw_true_count": None,
            "decision_bet_units": None,
            "decision_running_count": None,
            "decision_cards_remaining": None,
            "decision_true_count_numerator": None,
            "decision_true_count_denominator": None,
            "decision_raw_true_count": None,
            "decision_true_count_floor": None,
            "decision_true_count_ceiling": None,
            "decision_true_count_truncation": None,
            "bet_units": after.initial_bet_units,
            "wager_units": after.wager_units,
            "outcome": after.outcome,
            "profit_half_units": (
                event.profit_half_units if isinstance(event, RoundSettled) else None
            ),
            "cumulative_shoe_profit_half_units": (after.cumulative_profit_half_units),
        }
        if bet_decision is not None:
            row.update(_true_count_decision_columns(bet_decision.true_count))
            row.update(
                {
                    "bet_ramp_entry_index": bet_decision.ramp_entry_index,
                    "bet_ramp_minimum_raw_true_count": (
                        bet_decision.ramp_entry.minimum_raw_true_count
                    ),
                    "decision_bet_units": bet_decision.bet_units,
                }
            )
        if action_decision is not None:
            row.update(_true_count_decision_columns(action_decision.true_count))
            row.update(
                {
                    "legal_actions": sorted(
                        action.value for action in action_decision.legal_actions
                    ),
                    "basic_strategy_recommendation": (
                        action_decision.basic_recommendation.value
                        if action_decision.basic_recommendation
                        else None
                    ),
                    "basic_strategy_action": (
                        action_decision.basic_strategy_action.value
                        if action_decision.basic_strategy_action
                        else None
                    ),
                    "matched_deviation_id": action_decision.matched_deviation_id,
                    "applied_deviation_id": action_decision.applied_deviation_id,
                    "deviation_index_threshold": (
                        action_decision.deviation_index_threshold
                    ),
                    "deviation_threshold_met": (
                        action_decision.deviation_threshold_met
                    ),
                    "deviation_preferred_action": (
                        action_decision.deviation_preferred_action.value
                        if action_decision.deviation_preferred_action
                        else None
                    ),
                    "deviation_fallback_used": (
                        action_decision.deviation_fallback_used
                    ),
                    "deviation_activated": action_decision.action_changed_from_basic,
                    "final_expert_action": (
                        action_decision.selected_action.value
                        if action_decision.selected_action
                        else None
                    ),
                    "action_source": action_decision.action_source.value,
                }
            )
        if isinstance(event, RoundSettled) and event.outcome in {
            Outcome.PLAYER_BLACKJACK,
            Outcome.DEALER_BLACKJACK,
            Outcome.BLACKJACK_PUSH,
        }:
            row["action_source"] = ActionSource.NATURAL_RESOLUTION.value
        return row


def _snapshot_columns(prefix: str, snapshot: TraceSnapshot) -> dict[str, Any]:
    columns: dict[str, Any] = {
        f"{prefix}_cards_dealt": snapshot.cards_dealt,
        f"{prefix}_cards_remaining": snapshot.cards_remaining,
        f"{prefix}_running_count": snapshot.running_count,
        f"{prefix}_true_count_numerator": snapshot.true_count_numerator,
        f"{prefix}_true_count_denominator": snapshot.true_count_denominator,
        f"{prefix}_raw_true_count": snapshot.raw_true_count,
        f"{prefix}_true_count_floor": snapshot.true_count_floor,
        f"{prefix}_true_count_ceiling": snapshot.true_count_ceiling,
        f"{prefix}_true_count_truncation": snapshot.true_count_truncation,
        f"{prefix}_exposed_cards": list(snapshot.exposed_cards),
        f"{prefix}_player_hand": list(snapshot.player_hand),
        f"{prefix}_dealer_visible_hand": list(snapshot.dealer_visible_hand),
        f"{prefix}_dealer_upcard": snapshot.dealer_upcard,
        f"latent_{prefix}_dealer_hole_rank": snapshot.latent_dealer_hole_rank,
        f"{prefix}_dealer_hole_observable": snapshot.dealer_hole_observable,
        f"{prefix}_initial_bet_units": snapshot.initial_bet_units,
        f"{prefix}_wager_units": snapshot.wager_units,
        f"{prefix}_doubled": snapshot.doubled,
        f"{prefix}_peek_completed": snapshot.peek_completed,
        f"{prefix}_outcome": snapshot.outcome,
        f"{prefix}_round_profit_half_units": snapshot.round_profit_half_units,
        f"{prefix}_cumulative_profit_half_units": (
            snapshot.cumulative_profit_half_units
        ),
    }
    for rank, count in zip(
        Rank,
        snapshot.latent_remaining_rank_counts,
        strict=True,
    ):
        columns[f"latent_{prefix}_remaining_{rank.value}"] = count
    return columns


def _true_count_decision_columns(metrics: Any) -> dict[str, Any]:
    return {
        "decision_running_count": metrics.running_count,
        "decision_cards_remaining": metrics.physical_cards_remaining,
        "decision_true_count_numerator": metrics.numerator,
        "decision_true_count_denominator": metrics.denominator,
        "decision_raw_true_count": metrics.raw_true_count,
        "decision_true_count_floor": metrics.floor,
        "decision_true_count_ceiling": metrics.ceiling,
        "decision_true_count_truncation": metrics.truncation_toward_zero,
    }

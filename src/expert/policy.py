from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Final

from blackjackgpt.blackjack.state import Action

from .basic_strategy import (
    StrategyRecommendation,
    basic_strategy_recommendation,
    resolve_recommendation,
)
from .deviations import (
    VERSION1_DEVIATIONS,
    DeviationRule,
    find_matching_deviation,
    resolve_deviation,
)
from .hilo import TrueCountMetrics
from .observation import ActionObservation, BettingObservation


class ActionSource(str, Enum):
    BASIC_STRATEGY = "BASIC_STRATEGY"
    COUNT_DEVIATION = "COUNT_DEVIATION"
    FORCED_ACTION = "FORCED_ACTION"
    NATURAL_RESOLUTION = "NATURAL_RESOLUTION"


@dataclass(frozen=True, slots=True)
class BetRampEntry:
    minimum_raw_true_count: float | None
    bet_units: int

    def __post_init__(self) -> None:
        if self.minimum_raw_true_count is not None and not math.isfinite(
            self.minimum_raw_true_count
        ):
            raise ValueError("minimum_raw_true_count must be finite or None")
        if isinstance(self.bet_units, bool) or not isinstance(self.bet_units, int):
            raise TypeError("bet_units must be an integer")
        if self.bet_units <= 0:
            raise ValueError("bet_units must be positive")


VERSION1_BET_RAMP: Final[tuple[BetRampEntry, ...]] = (
    BetRampEntry(None, 1),
    BetRampEntry(1.0, 2),
    BetRampEntry(2.0, 4),
    BetRampEntry(3.0, 6),
    BetRampEntry(4.0, 8),
)


@dataclass(frozen=True, slots=True)
class ExpertPolicyConfig:
    bet_ramp: tuple[BetRampEntry, ...] = VERSION1_BET_RAMP
    deviations: tuple[DeviationRule, ...] = VERSION1_DEVIATIONS

    def __post_init__(self) -> None:
        if not self.bet_ramp or self.bet_ramp[0].minimum_raw_true_count is not None:
            raise ValueError("bet ramp must begin with an unbounded base entry")
        finite_bounds = [
            entry.minimum_raw_true_count
            for entry in self.bet_ramp[1:]
            if entry.minimum_raw_true_count is not None
        ]
        if len(finite_bounds) != len(self.bet_ramp) - 1:
            raise ValueError("only the first bet-ramp entry may be unbounded")
        if finite_bounds != sorted(finite_bounds) or len(finite_bounds) != len(
            set(finite_bounds)
        ):
            raise ValueError("bet-ramp lower bounds must be unique and increasing")


VERSION1_POLICY_CONFIG: Final[ExpertPolicyConfig] = ExpertPolicyConfig()


@dataclass(frozen=True, slots=True)
class BetDecision:
    bet_units: int
    ramp_entry_index: int
    ramp_entry: BetRampEntry
    true_count: TrueCountMetrics

    @property
    def token(self) -> str:
        return f"BET_{self.bet_units}"


@dataclass(frozen=True, slots=True)
class ActionDecision:
    legal_actions: frozenset[Action]
    selected_action: Action | None
    action_source: ActionSource
    true_count: TrueCountMetrics
    basic_recommendation: StrategyRecommendation | None = None
    basic_strategy_action: Action | None = None
    matched_deviation_id: str | None = None
    deviation_index_threshold: float | None = None
    deviation_threshold_met: bool | None = None
    deviation_preferred_action: Action | None = None
    deviation_fallback_used: bool = False
    action_changed_from_basic: bool = False

    @property
    def applied_deviation_id(self) -> str | None:
        if self.action_changed_from_basic:
            return self.matched_deviation_id
        return None


def bet_ramp_entry_for_raw_true_count(
    raw_true_count: float,
    ramp: tuple[BetRampEntry, ...] = VERSION1_BET_RAMP,
) -> tuple[int, BetRampEntry]:
    if not math.isfinite(raw_true_count):
        raise ValueError("raw_true_count must be finite")
    if not ramp or ramp[0].minimum_raw_true_count is not None:
        raise ValueError("bet ramp must have an unbounded base entry")
    for index in range(len(ramp) - 1, -1, -1):
        entry = ramp[index]
        lower = entry.minimum_raw_true_count
        if lower is None or raw_true_count >= lower:
            return index, entry
    raise AssertionError("unreachable: the base bet entry matches every raw count")


def bet_units_for_raw_true_count(
    raw_true_count: float,
    ramp: tuple[BetRampEntry, ...] = VERSION1_BET_RAMP,
) -> int:
    return bet_ramp_entry_for_raw_true_count(raw_true_count, ramp)[1].bet_units


class ExpertPolicy:
    def __init__(self, config: ExpertPolicyConfig = VERSION1_POLICY_CONFIG) -> None:
        self._config = config

    @property
    def config(self) -> ExpertPolicyConfig:
        return self._config

    def choose_bet(self, observation: BettingObservation) -> BetDecision:
        true_count = observation.true_count
        index, entry = bet_ramp_entry_for_raw_true_count(
            true_count.raw_true_count,
            self._config.bet_ramp,
        )
        return BetDecision(
            bet_units=entry.bet_units,
            ramp_entry_index=index,
            ramp_entry=entry,
            true_count=true_count,
        )

    def choose_action(self, observation: ActionObservation) -> ActionDecision:
        hand = observation.player_hand
        legal_actions = observation.legal_actions
        true_count = observation.true_count

        if not legal_actions:
            return ActionDecision(
                legal_actions=legal_actions,
                selected_action=None,
                action_source=ActionSource.NATURAL_RESOLUTION,
                true_count=true_count,
            )

        if len(legal_actions) == 1:
            return ActionDecision(
                legal_actions=legal_actions,
                selected_action=next(iter(legal_actions)),
                action_source=ActionSource.FORCED_ACTION,
                true_count=true_count,
            )

        recommendation = basic_strategy_recommendation(
            hand,
            observation.dealer_upcard,
        )
        basic_action = resolve_recommendation(recommendation, legal_actions)
        deviation = find_matching_deviation(
            hand,
            observation.dealer_upcard,
            self._config.deviations,
        )
        if deviation is None:
            return ActionDecision(
                legal_actions=legal_actions,
                selected_action=basic_action,
                action_source=ActionSource.BASIC_STRATEGY,
                true_count=true_count,
                basic_recommendation=recommendation,
                basic_strategy_action=basic_action,
            )

        resolution = resolve_deviation(
            deviation,
            raw_true_count=true_count.raw_true_count,
            legal_actions=legal_actions,
        )
        changed_from_basic = resolution.selected_action != basic_action
        return ActionDecision(
            legal_actions=legal_actions,
            selected_action=resolution.selected_action,
            action_source=(
                ActionSource.COUNT_DEVIATION
                if changed_from_basic
                else ActionSource.BASIC_STRATEGY
            ),
            true_count=true_count,
            basic_recommendation=recommendation,
            basic_strategy_action=basic_action,
            matched_deviation_id=deviation.deviation_id,
            deviation_index_threshold=deviation.index_threshold,
            deviation_threshold_met=resolution.threshold_met,
            deviation_preferred_action=resolution.preferred_action,
            deviation_fallback_used=resolution.fallback_used,
            action_changed_from_basic=changed_from_basic,
        )

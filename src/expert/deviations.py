from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Final

from blackjackgpt.blackjack.cards import Rank
from blackjackgpt.blackjack.hand import Hand
from blackjackgpt.blackjack.state import Action


class HandType(str, Enum):
    HARD = "HARD"
    SOFT = "SOFT"


@dataclass(frozen=True, slots=True)
class DeviationRule:
    deviation_id: str
    hand_type: HandType
    player_total: int
    dealer_upcard: Rank
    index_threshold: float
    action_at_or_above: Action
    action_below: Action
    illegal_action_fallback: Action

    def __post_init__(self) -> None:
        if not self.deviation_id:
            raise ValueError("deviation_id must be non-empty")
        if not 4 <= self.player_total <= 21:
            raise ValueError("player_total must be in [4, 21]")
        if not math.isfinite(self.index_threshold):
            raise ValueError("index_threshold must be finite")

    def matches(self, hand: Hand, dealer_upcard: Rank) -> bool:
        value = hand.value
        if value.is_bust or value.best_legal_total != self.player_total:
            return False
        actual_type = HandType.SOFT if value.is_soft else HandType.HARD
        return actual_type is self.hand_type and dealer_upcard is self.dealer_upcard


@dataclass(frozen=True, slots=True)
class DeviationResolution:
    rule: DeviationRule
    raw_true_count: float
    threshold_met: bool
    preferred_action: Action
    selected_action: Action
    fallback_used: bool


def resolve_deviation(
    rule: DeviationRule,
    *,
    raw_true_count: float,
    legal_actions: frozenset[Action],
) -> DeviationResolution:
    if not math.isfinite(raw_true_count):
        raise ValueError("raw_true_count must be finite")
    threshold_met = raw_true_count >= rule.index_threshold
    preferred = rule.action_at_or_above if threshold_met else rule.action_below
    if preferred in legal_actions:
        selected = preferred
        fallback_used = False
    elif rule.illegal_action_fallback in legal_actions:
        selected = rule.illegal_action_fallback
        fallback_used = True
    else:
        raise ValueError(
            f"deviation {rule.deviation_id!r} selected {preferred.value}, "
            f"but neither it nor fallback {rule.illegal_action_fallback.value} "
            "is legal"
        )
    return DeviationResolution(
        rule=rule,
        raw_true_count=raw_true_count,
        threshold_met=threshold_met,
        preferred_action=preferred,
        selected_action=selected,
        fallback_used=fallback_used,
    )


def find_matching_deviation(
    hand: Hand,
    dealer_upcard: Rank,
    deviations: tuple[DeviationRule, ...],
) -> DeviationRule | None:
    matches = tuple(rule for rule in deviations if rule.matches(hand, dealer_upcard))
    if len(matches) > 1:
        ids = ", ".join(rule.deviation_id for rule in matches)
        raise ValueError(f"multiple deviations match one decision: {ids}")
    return matches[0] if matches else None


def _hard_rule(
    deviation_id: str,
    total: int,
    upcard: Rank,
    threshold: int,
    action_at_or_above: Action,
) -> DeviationRule:
    return DeviationRule(
        deviation_id=deviation_id,
        hand_type=HandType.HARD,
        player_total=total,
        dealer_upcard=upcard,
        index_threshold=float(threshold),
        action_at_or_above=action_at_or_above,
        action_below=Action.HIT,
        illegal_action_fallback=Action.HIT,
    )


VERSION1_DEVIATIONS: Final[tuple[DeviationRule, ...]] = (
    _hard_rule("hard_16_vs_T", 16, Rank.TEN, 0, Action.STAND),
    _hard_rule("hard_15_vs_T", 15, Rank.TEN, 4, Action.STAND),
    _hard_rule("hard_12_vs_3", 12, Rank.THREE, 2, Action.STAND),
    _hard_rule("hard_12_vs_2", 12, Rank.TWO, 3, Action.STAND),
    _hard_rule("hard_10_vs_T", 10, Rank.TEN, 4, Action.DOUBLE),
    _hard_rule("hard_11_vs_A", 11, Rank.ACE, 1, Action.DOUBLE),
    _hard_rule("hard_9_vs_2", 9, Rank.TWO, 1, Action.DOUBLE),
    _hard_rule("hard_10_vs_A", 10, Rank.ACE, 4, Action.DOUBLE),
    _hard_rule("hard_9_vs_7", 9, Rank.SEVEN, 3, Action.DOUBLE),
    _hard_rule("hard_16_vs_9", 16, Rank.NINE, 5, Action.STAND),
    _hard_rule("hard_13_vs_2", 13, Rank.TWO, -1, Action.STAND),
    _hard_rule("hard_12_vs_4", 12, Rank.FOUR, 0, Action.STAND),
    _hard_rule("hard_12_vs_5", 12, Rank.FIVE, -2, Action.STAND),
    _hard_rule("hard_12_vs_6", 12, Rank.SIX, -1, Action.STAND),
    _hard_rule("hard_13_vs_3", 13, Rank.THREE, -2, Action.STAND),
)


def _validate_version1_deviations() -> None:
    ids = [rule.deviation_id for rule in VERSION1_DEVIATIONS]
    if len(ids) != len(set(ids)):
        raise RuntimeError("version-1 deviation IDs must be unique")
    keys = [
        (rule.hand_type, rule.player_total, rule.dealer_upcard)
        for rule in VERSION1_DEVIATIONS
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("version-1 deviation match keys must be unique")


_validate_version1_deviations()

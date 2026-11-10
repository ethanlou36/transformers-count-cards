from __future__ import annotations

import hashlib
import json
from enum import Enum

from blackjackgpt.blackjack.cards import Rank
from blackjackgpt.blackjack.hand import Hand
from blackjackgpt.blackjack.state import Action


class StrategyRecommendation(str, Enum):
    HIT = "HIT"
    STAND = "STAND"
    DOUBLE_OR_HIT = "DOUBLE_OR_HIT"
    DOUBLE_OR_STAND = "DOUBLE_OR_STAND"


BASIC_STRATEGY_PROFILE_ID = "single_deck_s17_american_peek_total_dependent_v1"
BASIC_STRATEGY_RETRIEVED_ON = "2026-07-22"
BASIC_STRATEGY_PRIMARY_SOURCE_URL = (
    "https://wizardofodds.com/games/blackjack/strategy/calculator/"
)
BASIC_STRATEGY_CROSS_CHECK_SOURCE_URL = (
    "https://www.blackjackinfo.com/blackjack-basic-strategy-engine/"
    "?numdecks=1&soft17=s17&dbl=all&das=yes&surr=ns&peek=yes"
)
BASIC_STRATEGY_SOFT_12_SOURCE_URL = (
    "https://wizardofodds.com/games/blackjack/strategy/1-deck/"
)
BASIC_STRATEGY_SOURCE_URLS = (
    BASIC_STRATEGY_PRIMARY_SOURCE_URL,
    BASIC_STRATEGY_CROSS_CHECK_SOURCE_URL,
    BASIC_STRATEGY_SOFT_12_SOURCE_URL,
)

DEALER_UPCARD_ORDER: tuple[Rank, ...] = (
    Rank.TWO,
    Rank.THREE,
    Rank.FOUR,
    Rank.FIVE,
    Rank.SIX,
    Rank.SEVEN,
    Rank.EIGHT,
    Rank.NINE,
    Rank.TEN,
    Rank.ACE,
)
_DEALER_UPCARD_INDEX = {
    upcard: index for index, upcard in enumerate(DEALER_UPCARD_ORDER)
}

_H = StrategyRecommendation.HIT
_S = StrategyRecommendation.STAND
_DH = StrategyRecommendation.DOUBLE_OR_HIT
_DS = StrategyRecommendation.DOUBLE_OR_STAND


_HARD_TABLE: dict[int, tuple[StrategyRecommendation, ...]] = {
    4: (_H, _H, _H, _H, _H, _H, _H, _H, _H, _H),
    5: (_H, _H, _H, _H, _H, _H, _H, _H, _H, _H),
    6: (_H, _H, _H, _H, _H, _H, _H, _H, _H, _H),
    7: (_H, _H, _H, _H, _H, _H, _H, _H, _H, _H),
    8: (_H, _H, _H, _DH, _DH, _H, _H, _H, _H, _H),
    9: (_DH, _DH, _DH, _DH, _DH, _H, _H, _H, _H, _H),
    10: (_DH, _DH, _DH, _DH, _DH, _DH, _DH, _DH, _H, _H),
    11: (_DH, _DH, _DH, _DH, _DH, _DH, _DH, _DH, _DH, _DH),
    12: (_H, _H, _S, _S, _S, _H, _H, _H, _H, _H),
    13: (_S, _S, _S, _S, _S, _H, _H, _H, _H, _H),
    14: (_S, _S, _S, _S, _S, _H, _H, _H, _H, _H),
    15: (_S, _S, _S, _S, _S, _H, _H, _H, _H, _H),
    16: (_S, _S, _S, _S, _S, _H, _H, _H, _H, _H),
    17: (_S, _S, _S, _S, _S, _S, _S, _S, _S, _S),
    18: (_S, _S, _S, _S, _S, _S, _S, _S, _S, _S),
    19: (_S, _S, _S, _S, _S, _S, _S, _S, _S, _S),
    20: (_S, _S, _S, _S, _S, _S, _S, _S, _S, _S),
    21: (_S, _S, _S, _S, _S, _S, _S, _S, _S, _S),
}

_SOFT_TABLE: dict[int, tuple[StrategyRecommendation, ...]] = {
    12: (_H, _H, _H, _DH, _DH, _H, _H, _H, _H, _H),
    13: (_H, _H, _DH, _DH, _DH, _H, _H, _H, _H, _H),
    14: (_H, _H, _DH, _DH, _DH, _H, _H, _H, _H, _H),
    15: (_H, _H, _DH, _DH, _DH, _H, _H, _H, _H, _H),
    16: (_H, _H, _DH, _DH, _DH, _H, _H, _H, _H, _H),
    17: (_DH, _DH, _DH, _DH, _DH, _H, _H, _H, _H, _H),
    18: (_S, _DS, _DS, _DS, _DS, _S, _S, _H, _H, _S),
    19: (_S, _S, _S, _S, _DS, _S, _S, _S, _S, _S),
    20: (_S, _S, _S, _S, _S, _S, _S, _S, _S, _S),
    21: (_S, _S, _S, _S, _S, _S, _S, _S, _S, _S),
}


def canonical_basic_strategy_table_json() -> str:
    payload = {
        "dealer_upcards": [rank.value for rank in DEALER_UPCARD_ORDER],
        "hard": {
            str(total): [recommendation.value for recommendation in row]
            for total, row in sorted(_HARD_TABLE.items())
        },
        "profile_id": BASIC_STRATEGY_PROFILE_ID,
        "soft": {
            str(total): [recommendation.value for recommendation in row]
            for total, row in sorted(_SOFT_TABLE.items())
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def basic_strategy_table_sha256() -> str:
    canonical = canonical_basic_strategy_table_json().encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


BASIC_STRATEGY_TABLE_SHA256 = (
    "9cd7ffbe36e447aaa08c3129195819678d8478e41c8dc9e55ff1bbfa5ece6bbc"
)


def basic_strategy_recommendation(
    hand: Hand,
    dealer_upcard: Rank,
) -> StrategyRecommendation:
    if not isinstance(hand, Hand):
        raise TypeError("hand must be a Hand")
    if not isinstance(dealer_upcard, Rank):
        raise TypeError("dealer_upcard must be a Rank")
    if len(hand.cards) < 2:
        raise ValueError("basic strategy requires at least two player cards")

    value = hand.value
    if value.is_bust:
        raise ValueError("basic strategy is undefined for a bust hand")
    if value.best_legal_total is None:
        raise ValueError("basic strategy requires a legal player total")

    table = _SOFT_TABLE if value.usable_ace else _HARD_TABLE
    try:
        row = table[value.best_legal_total]
    except KeyError as error:
        hand_class = "soft" if value.usable_ace else "hard"
        raise ValueError(
            f"unsupported {hand_class} total: {value.best_legal_total}"
        ) from error

    return row[_DEALER_UPCARD_INDEX[dealer_upcard]]


_RECOMMENDATION_PREFERENCES: dict[StrategyRecommendation, tuple[Action, ...]] = {
    StrategyRecommendation.HIT: (Action.HIT,),
    StrategyRecommendation.STAND: (Action.STAND,),
    StrategyRecommendation.DOUBLE_OR_HIT: (Action.DOUBLE, Action.HIT),
    StrategyRecommendation.DOUBLE_OR_STAND: (Action.DOUBLE, Action.STAND),
}


def resolve_recommendation(
    recommendation: StrategyRecommendation,
    legal_actions: frozenset[Action],
) -> Action:
    if not isinstance(recommendation, StrategyRecommendation):
        raise TypeError("recommendation must be a StrategyRecommendation")
    if not isinstance(legal_actions, frozenset):
        raise TypeError("legal_actions must be a frozenset")
    if any(not isinstance(action, Action) for action in legal_actions):
        raise TypeError("legal_actions must contain only Action values")

    preferences = _RECOMMENDATION_PREFERENCES[recommendation]
    for action in preferences:
        if action in legal_actions:
            return action

    preferred_names = ", ".join(action.value for action in preferences)
    legal_names = ", ".join(sorted(action.value for action in legal_actions))
    raise ValueError(
        f"no legal fallback for {recommendation.value}; "
        f"preferred actions=[{preferred_names}], legal actions=[{legal_names}]"
    )

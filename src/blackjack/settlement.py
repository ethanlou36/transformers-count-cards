from __future__ import annotations

from dataclasses import dataclass

from .hand import Hand
from .rules import RulesConfig
from .state import Outcome


@dataclass(frozen=True, slots=True)
class Settlement:
    outcome: Outcome
    profit_half_units: int


def settle_hands(
    *,
    player_hand: Hand,
    dealer_hand: Hand,
    initial_bet_units: int,
    current_wager_units: int,
    rules: RulesConfig,
) -> Settlement:
    if initial_bet_units <= 0:
        raise ValueError("initial_bet_units must be positive")
    if current_wager_units not in {initial_bet_units, 2 * initial_bet_units}:
        raise ValueError("current wager must equal the initial or doubled wager")

    player_value = player_hand.value
    dealer_value = dealer_hand.value

    if player_value.is_blackjack and dealer_value.is_blackjack:
        return Settlement(Outcome.BLACKJACK_PUSH, 0)
    if dealer_value.is_blackjack:
        return Settlement(Outcome.DEALER_BLACKJACK, -2 * initial_bet_units)
    if player_value.is_blackjack:
        return Settlement(
            Outcome.PLAYER_BLACKJACK,
            rules.blackjack_profit_half_units_per_bet_unit * initial_bet_units,
        )
    if player_value.is_bust:
        return Settlement(Outcome.PLAYER_BUST, -2 * current_wager_units)
    if dealer_value.is_bust:
        return Settlement(Outcome.DEALER_BUST, 2 * current_wager_units)

    assert player_value.best_legal_total is not None
    assert dealer_value.best_legal_total is not None
    if player_value.best_legal_total > dealer_value.best_legal_total:
        return Settlement(Outcome.WIN, 2 * current_wager_units)
    if player_value.best_legal_total < dealer_value.best_legal_total:
        return Settlement(Outcome.LOSS, -2 * current_wager_units)
    return Settlement(Outcome.PUSH, 0)

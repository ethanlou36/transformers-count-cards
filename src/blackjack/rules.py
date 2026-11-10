from __future__ import annotations

from dataclasses import dataclass

from .cards import (
    RANK_MULTIPLICITIES,
    RANK_ORDER,
    SHUFFLE_ALGORITHM_ID,
)

STANDARD_RANK_ORDER = tuple(rank.value for rank in RANK_ORDER)
STANDARD_RANK_MULTIPLICITIES = tuple(RANK_MULTIPLICITIES[rank] for rank in RANK_ORDER)
VERSION1_INITIAL_DEAL_ORDER = (
    "PLAYER_FIRST",
    "DEALER_UPCARD",
    "PLAYER_SECOND",
    "DEALER_HOLE",
)


@dataclass(frozen=True, slots=True)
class RulesConfig:
    num_decks: int = 1
    num_players: int = 1
    american_hole_card: bool = True
    ranks_only: bool = True
    suits_affect_play: bool = False
    deal_without_replacement: bool = True
    rank_order: tuple[str, ...] = STANDARD_RANK_ORDER
    rank_multiplicities: tuple[int, ...] = STANDARD_RANK_MULTIPLICITIES
    initial_deal_order: tuple[str, ...] = VERSION1_INITIAL_DEAL_ORDER
    shuffle_algorithm: str = SHUFFLE_ALGORITHM_ID
    dealer_stands_soft_17: bool = True
    blackjack_profit_half_units_per_bet_unit: int = 3
    dealer_peeks: bool = True
    double_allowed: bool = True
    double_initial_two_only: bool = True
    split_allowed: bool = False
    surrender_allowed: bool = False
    insurance_allowed: bool = False
    cut_card_index: int = 39
    minimum_cards_to_start_round: int = 16
    reveal_hole_on_player_bust: bool = False
    reveal_hole_on_player_blackjack: bool = False

    def __post_init__(self) -> None:
        if self.num_decks != 1:
            raise ValueError("version 1 supports exactly one deck")
        if self.num_players != 1:
            raise ValueError("version 1 supports exactly one player")
        if not self.american_hole_card:
            raise ValueError("version 1 requires an American hole card")
        if not self.ranks_only or self.suits_affect_play:
            raise ValueError("version 1 is rank-only and suit-insensitive")
        if not self.deal_without_replacement:
            raise ValueError("version 1 deals without replacement")
        if self.rank_order != STANDARD_RANK_ORDER:
            raise ValueError("rank_order does not match the version-1 deck")
        if self.rank_multiplicities != STANDARD_RANK_MULTIPLICITIES:
            raise ValueError("rank_multiplicities do not match a standard deck")
        if self.initial_deal_order != VERSION1_INITIAL_DEAL_ORDER:
            raise ValueError("initial_deal_order does not match version 1")
        if self.shuffle_algorithm != SHUFFLE_ALGORITHM_ID:
            raise ValueError("shuffle_algorithm does not match version 1")
        if self.blackjack_profit_half_units_per_bet_unit <= 0:
            raise ValueError("blackjack payout must be positive")
        if isinstance(self.cut_card_index, bool) or not isinstance(
            self.cut_card_index, int
        ):
            raise TypeError("cut_card_index must be an integer")
        if not 4 <= self.cut_card_index <= 52:
            raise ValueError("cut_card_index must be in [4, 52]")
        if isinstance(self.minimum_cards_to_start_round, bool) or not isinstance(
            self.minimum_cards_to_start_round, int
        ):
            raise TypeError("minimum_cards_to_start_round must be an integer")
        if not 16 <= self.minimum_cards_to_start_round <= 52:
            raise ValueError("minimum_cards_to_start_round must be in [16, 52]")
        if self.split_allowed or self.surrender_allowed or self.insurance_allowed:
            raise ValueError(
                "splitting, surrender, and insurance are not implemented in version 1"
            )
        if not self.double_initial_two_only:
            raise ValueError(
                "version 1 only supports doubling on the initial two cards"
            )


VERSION1_RULES = RulesConfig()

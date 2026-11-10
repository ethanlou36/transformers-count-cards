from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum


class Rank(str, Enum):
    ACE = "A"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "T"

    @property
    def hard_value(self) -> int:
        if self is Rank.ACE:
            return 1
        if self is Rank.TEN:
            return 10
        return int(self.value)


RANK_ORDER: tuple[Rank, ...] = tuple(Rank)
SHUFFLE_ALGORITHM_ID = "pcg32-xsh-rr-rejection-fisher-yates-v1"
RANK_MULTIPLICITIES: dict[Rank, int] = {
    Rank.ACE: 4,
    Rank.TWO: 4,
    Rank.THREE: 4,
    Rank.FOUR: 4,
    Rank.FIVE: 4,
    Rank.SIX: 4,
    Rank.SEVEN: 4,
    Rank.EIGHT: 4,
    Rank.NINE: 4,
    Rank.TEN: 16,
}


@dataclass(frozen=True, slots=True)
class Card:
    rank: Rank
    copy_index: int

    def __post_init__(self) -> None:
        maximum = RANK_MULTIPLICITIES[self.rank]
        if not 0 <= self.copy_index < maximum:
            raise ValueError(
                f"copy_index for {self.rank.value} must be in [0, {maximum})"
            )

    @property
    def instance_id(self) -> str:
        return f"{self.rank.value}{self.copy_index}"


def canonical_deck() -> tuple[Card, ...]:
    return tuple(
        Card(rank=rank, copy_index=copy_index)
        for rank in RANK_ORDER
        for copy_index in range(RANK_MULTIPLICITIES[rank])
    )


def validate_standard_deck(cards: Sequence[Card]) -> None:
    if len(cards) != 52:
        raise ValueError(f"a single deck must contain 52 cards, got {len(cards)}")

    actual_ids = [card.instance_id for card in cards]
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError("deck contains duplicate physical card instances")

    expected_ids = {card.instance_id for card in canonical_deck()}
    if set(actual_ids) != expected_ids:
        missing = sorted(expected_ids - set(actual_ids))
        unexpected = sorted(set(actual_ids) - expected_ids)
        raise ValueError(
            f"deck instances do not match the canonical deck; "
            f"missing={missing}, unexpected={unexpected}"
        )

    actual_counts = Counter(card.rank for card in cards)
    if actual_counts != Counter(RANK_MULTIPLICITIES):
        raise ValueError("deck rank multiplicities are invalid")


class PCG32:
    _MASK_32 = (1 << 32) - 1
    _MASK_64 = (1 << 64) - 1
    _MULTIPLIER = 6364136223846793005

    def __init__(self, seed: int, stream: int = 54) -> None:
        if not 0 <= seed <= self._MASK_64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if not 0 <= stream < (1 << 63):
            raise ValueError("stream must be an unsigned 63-bit integer")

        self._state = 0
        self._increment = ((stream << 1) | 1) & self._MASK_64
        self.next_u32()
        self._state = (self._state + seed) & self._MASK_64
        self.next_u32()

    def next_u32(self) -> int:
        old_state = self._state
        self._state = (old_state * self._MULTIPLIER + self._increment) & self._MASK_64
        xorshifted = (((old_state >> 18) ^ old_state) >> 27) & self._MASK_32
        rotation = old_state >> 59
        return (
            (xorshifted >> rotation) | (xorshifted << ((-rotation) & 31))
        ) & self._MASK_32

    def randbelow(self, upper_bound: int) -> int:
        if not 1 <= upper_bound <= 1 << 32:
            raise ValueError("upper_bound must be in [1, 2**32]")
        threshold = ((1 << 32) - upper_bound) % upper_bound
        while True:
            value = self.next_u32()
            if value >= threshold:
                return value % upper_bound


def fisher_yates_shuffle(cards: Iterable[Card], rng: PCG32) -> tuple[Card, ...]:
    shuffled = list(cards)
    for index in range(len(shuffled) - 1, 0, -1):
        swap_index = rng.randbelow(index + 1)
        shuffled[index], shuffled[swap_index] = (
            shuffled[swap_index],
            shuffled[index],
        )
    return tuple(shuffled)


def shuffled_deck(seed: int, *, stream: int = 54) -> tuple[Card, ...]:
    return fisher_yates_shuffle(canonical_deck(), PCG32(seed, stream=stream))

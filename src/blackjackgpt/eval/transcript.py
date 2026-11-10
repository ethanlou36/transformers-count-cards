from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from blackjackgpt.blackjack.cards import Rank
from blackjackgpt.blackjack.rules import VERSION1_RULES, RulesConfig
from blackjackgpt.data.vocabulary import Vocabulary
from blackjackgpt.tokens import hilo_weight_of, rank_of

_DECK_SIZE = 52


class _State(Enum):
    EXPECT_SHUFFLE = auto()
    BETWEEN = auto()
    EXPECT_BET = auto()
    EXPECT_P1 = auto()
    EXPECT_DUP = auto()
    EXPECT_P2 = auto()
    EXPECT_DHOLE = auto()
    EXPECT_PEEK = auto()
    EXPECT_PEEK_REVEAL = auto()
    EXPECT_PEEK_OUTCOME = auto()
    EXPECT_PLAYER_NAT_OUTCOME = auto()
    PLAYER_TURN = auto()
    EXPECT_HIT_CARD = auto()
    EXPECT_DOUBLE_CARD = auto()
    EXPECT_PLAYER_BUST_OUTCOME = auto()
    EXPECT_DEALER_REVEAL = auto()
    DEALER_DRAW = auto()
    EXPECT_ROUND_END = auto()
    DONE = auto()


@dataclass
class _HandValue:
    best_total: int | None
    is_bust: bool
    is_blackjack: bool


def _evaluate(ranks: list[Rank]) -> _HandValue:
    hard = sum(r.hard_value for r in ranks)
    usable_ace = any(r is Rank.ACE for r in ranks) and hard + 10 <= 21
    if usable_ace:
        best: int | None = hard + 10
    elif hard <= 21:
        best = hard
    else:
        best = None
    return _HandValue(
        best_total=best,
        is_bust=best is None,
        is_blackjack=len(ranks) == 2 and best == 21,
    )


class TranscriptTracker:
    def __init__(self, vocab: Vocabulary, rules: RulesConfig = VERSION1_RULES) -> None:
        self.vocab = vocab
        self.rules = rules
        self._id = vocab.token_to_id
        self._bet_ids = frozenset(
            self._id[t] for t in vocab.tokens if t.startswith("BET_")
        )
        self.reset()

    def reset(self) -> None:
        self.state = _State.EXPECT_SHUFFLE
        self.dealt = 0
        self.running_count = 0
        self.upcard: Rank | None = None
        self.player_ranks: list[Rank] = []

    def _ids(self, *texts: str) -> frozenset[int]:
        return frozenset(self._id[t] for t in texts)

    def _can_start_round(self) -> bool:
        return (
            self.dealt < self.rules.cut_card_index
            and (_DECK_SIZE - self.dealt) >= self.rules.minimum_cards_to_start_round
        )

    def _resolve_after_hole(self) -> _State:
        if _evaluate(self.player_ranks).is_blackjack:
            return _State.EXPECT_PLAYER_NAT_OUTCOME
        return _State.PLAYER_TURN

    def _player_turn_actions(self) -> frozenset[int]:
        actions = ["ACTION:STAND", "ACTION:HIT"]
        if self.rules.double_allowed and len(self.player_ranks) == 2:
            actions.append("ACTION:DOUBLE")
        return self._ids(*actions)

    def legal_next(self) -> frozenset[int]:
        s = self.state
        if s is _State.EXPECT_SHUFFLE:
            return self._ids("SHUFFLE")
        if s is _State.BETWEEN:
            return (
                self._ids("ROUND_START")
                if self._can_start_round()
                else self._ids("SHOE_END")
            )
        if s is _State.EXPECT_BET:
            return self._bet_ids
        if s in (_State.EXPECT_P1, _State.EXPECT_P2):
            return self._rank_ids("P:")
        if s is _State.EXPECT_DUP:
            return self._rank_ids("D_UP:")
        if s is _State.EXPECT_DHOLE:
            return self._ids("D_HOLE")
        if s is _State.EXPECT_PEEK:
            return self._ids("PEEK_CLEAR", "PEEK_BLACKJACK")
        if s in (_State.EXPECT_PEEK_REVEAL, _State.EXPECT_DEALER_REVEAL):
            return self._rank_ids("D_REVEAL:")
        if s is _State.EXPECT_PEEK_OUTCOME:
            return self._ids("OUTCOME:DEALER_BLACKJACK", "OUTCOME:BLACKJACK_PUSH")
        if s is _State.EXPECT_PLAYER_NAT_OUTCOME:
            return self._ids("OUTCOME:PLAYER_BLACKJACK")
        if s is _State.PLAYER_TURN:
            return self._player_turn_actions()
        if s in (_State.EXPECT_HIT_CARD, _State.EXPECT_DOUBLE_CARD):
            return self._rank_ids("P:")
        if s is _State.EXPECT_PLAYER_BUST_OUTCOME:
            return self._ids("OUTCOME:PLAYER_BUST")
        if s is _State.DEALER_DRAW:
            return self._rank_ids("D_DRAW:") | self._ids(
                "OUTCOME:WIN", "OUTCOME:LOSS", "OUTCOME:PUSH", "OUTCOME:DEALER_BUST"
            )
        if s is _State.EXPECT_ROUND_END:
            return self._ids("ROUND_END")
        return frozenset()

    def _rank_ids(self, prefix: str) -> frozenset[int]:
        return frozenset(self._id[f"{prefix}{r.value}"] for r in Rank)

    def advance(self, token_id: int) -> None:
        text = self.vocab.decode(token_id)
        if token_id not in self.legal_next():
            raise ValueError(f"illegal token {text!r} in state {self.state.name}")

        s = self.state
        if s is _State.EXPECT_SHUFFLE:
            self.dealt = 0
            self.running_count = 0
            self.state = _State.BETWEEN
        elif s is _State.BETWEEN:
            if text == "ROUND_START":
                self.player_ranks = []
                self.upcard = None
                self.state = _State.EXPECT_BET
            else:
                self.state = _State.DONE
        elif s is _State.EXPECT_BET:
            self.state = _State.EXPECT_P1
        elif s is _State.EXPECT_P1:
            self._add_card(text, to_player=True)
            self.state = _State.EXPECT_DUP
        elif s is _State.EXPECT_DUP:
            self.upcard = rank_of(text)
            self._add_card(text, to_player=False)
            self.state = _State.EXPECT_P2
        elif s is _State.EXPECT_P2:
            self._add_card(text, to_player=True)
            self.state = _State.EXPECT_DHOLE
        elif s is _State.EXPECT_DHOLE:
            self.dealt += 1
            if self.upcard in {Rank.ACE, Rank.TEN}:
                self.state = _State.EXPECT_PEEK
            else:
                self.state = self._resolve_after_hole()
        elif s is _State.EXPECT_PEEK:
            if text == "PEEK_BLACKJACK":
                self.state = _State.EXPECT_PEEK_REVEAL
            else:
                self.state = self._resolve_after_hole()
        elif s is _State.EXPECT_PEEK_REVEAL:
            self.running_count += hilo_weight_of(text)
            self.state = _State.EXPECT_PEEK_OUTCOME
        elif s in (_State.EXPECT_PEEK_OUTCOME, _State.EXPECT_PLAYER_NAT_OUTCOME):
            self.state = _State.EXPECT_ROUND_END
        elif s is _State.PLAYER_TURN:
            if text == "ACTION:STAND":
                self.state = _State.EXPECT_DEALER_REVEAL
            elif text == "ACTION:HIT":
                self.state = _State.EXPECT_HIT_CARD
            else:
                self.state = _State.EXPECT_DOUBLE_CARD
        elif s in (_State.EXPECT_HIT_CARD, _State.EXPECT_DOUBLE_CARD):
            self._add_card(text, to_player=True)
            value = _evaluate(self.player_ranks)
            if value.is_bust:
                self.state = _State.EXPECT_PLAYER_BUST_OUTCOME
            elif s is _State.EXPECT_DOUBLE_CARD or value.best_total == 21:
                self.state = _State.EXPECT_DEALER_REVEAL
            else:
                self.state = _State.PLAYER_TURN
        elif s is _State.EXPECT_PLAYER_BUST_OUTCOME:
            self.state = _State.EXPECT_ROUND_END
        elif s is _State.EXPECT_DEALER_REVEAL:
            self.running_count += hilo_weight_of(text)
            self.state = _State.DEALER_DRAW
        elif s is _State.DEALER_DRAW:
            if text.startswith("OUTCOME:"):
                self.state = _State.EXPECT_ROUND_END
            else:
                self._add_card(text, to_player=False)
                self.state = _State.DEALER_DRAW
        elif s is _State.EXPECT_ROUND_END:
            self.state = _State.BETWEEN
        else:
            raise ValueError("no tokens are legal after SHOE_END")

    def _add_card(self, text: str, *, to_player: bool) -> None:
        self.dealt += 1
        self.running_count += hilo_weight_of(text)
        if to_player:
            rank = rank_of(text)
            assert rank is not None
            self.player_ranks.append(rank)

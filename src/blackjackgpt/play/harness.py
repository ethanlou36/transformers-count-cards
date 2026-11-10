from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from blackjackgpt.blackjack.engine import BlackjackEngine
from blackjackgpt.blackjack.events import EngineEvent, RoundSettled
from blackjackgpt.blackjack.state import Action, EnginePhase, Rank
from blackjackgpt.data.serialization import semantic_event_for
from blackjackgpt.data.vocabulary import Vocabulary
from blackjackgpt.expert.hilo import HiLoCounter

_BET_UNITS = (1, 2, 4, 6, 8)


@dataclass
class Decision:
    bet_units: int | None = None
    action: Action | None = None


@dataclass
class Context:
    is_bet: bool
    transcript: list[int]
    running_count: int
    cards_remaining: int
    legal_token_ids: tuple[int, ...]
    player_ranks: tuple[Rank, ...] = ()
    dealer_upcard: Rank | None = None
    legal_actions: frozenset[Action] = frozenset()


@dataclass
class HandResult:
    initial_bet_units: int
    wager_units: int
    profit_half_units: int
    outcome: str
    running_count_at_bet: int


class DecisionMaker(Protocol):
    def decide(self, contexts: list[Context]) -> list[Decision]: ...


class ShoeGame:
    def __init__(self, seed: int, vocab: Vocabulary, shoe_id: str) -> None:
        self.engine = BlackjackEngine()
        self.vocab = vocab
        self.counter = HiLoCounter()
        self.transcript: list[int] = []
        self.hands: list[HandResult] = []
        self._rs_id = vocab.encode("ROUND_START")
        self._bet_ids = tuple(vocab.encode(f"BET_{u}") for u in _BET_UNITS)
        self._state = "between_rounds"
        self._pending_count: int = 0
        self._emit(self.engine.start_shuffled_shoe(seed, shoe_id=shoe_id))

    def _emit(self, events: tuple[EngineEvent, ...], skip_first: bool = False) -> None:
        for e in events[1:] if skip_first else events:
            self.transcript.append(self.vocab.encode(semantic_event_for(e).token))
            self.counter.observe(e)
            if isinstance(e, RoundSettled):
                self.hands.append(
                    HandResult(
                        initial_bet_units=e.initial_bet_units,
                        wager_units=e.current_wager_units,
                        profit_half_units=e.profit_half_units,
                        outcome=e.outcome.value,
                        running_count_at_bet=self._pending_count,
                    )
                )

    def _action_ids(self, legal: frozenset[Action]) -> tuple[int, ...]:
        return tuple(self.vocab.encode(f"ACTION:{a.value}") for a in legal)

    def advance(self) -> Context | None:
        if self._state == "between_rounds":
            if not self.engine.can_start_round():
                return None
            self.transcript.append(self._rs_id)
            self._state = "awaiting_bet"
            snap = self.engine.public_snapshot()
            return Context(
                is_bet=True,
                transcript=self.transcript,
                running_count=self.counter.running_count,
                cards_remaining=snap.cards_remaining,
                legal_token_ids=self._bet_ids,
            )

        legal = self.engine.legal_actions()
        rnd = self.engine.public_snapshot().round
        assert rnd is not None
        return Context(
            is_bet=False,
            transcript=self.transcript,
            running_count=self.counter.running_count,
            cards_remaining=self.engine.public_snapshot().cards_remaining,
            legal_token_ids=self._action_ids(legal),
            player_ranks=rnd.player_cards,
            dealer_upcard=rnd.dealer_upcard,
            legal_actions=legal,
        )

    def apply(self, decision: Decision) -> None:
        if self._state == "awaiting_bet":
            assert decision.bet_units is not None
            self._pending_count = self.counter.running_count
            events = self.engine.start_round(decision.bet_units)
            self._emit(events, skip_first=True)
            self._state = (
                "awaiting_action"
                if self.engine.phase is EnginePhase.PLAYER_TURN
                else "between_rounds"
            )
        else:
            assert decision.action is not None
            self._emit(self.engine.apply_action(decision.action))
            if self.engine.phase is EnginePhase.DEALER_TURN:
                self._emit(self.engine.play_dealer_and_settle())
                self._state = "between_rounds"
            elif self.engine.phase is not EnginePhase.PLAYER_TURN:
                self._state = "between_rounds"


def play_shoes(
    decision_maker: DecisionMaker,
    seeds: list[int],
    vocab: Vocabulary,
    concurrency: int = 256,
) -> list[HandResult]:
    results: list[HandResult] = []
    seed_iter = iter(seeds)

    def _new() -> tuple[ShoeGame, Context] | None:
        s = next(seed_iter, None)
        if s is None:
            return None
        g = ShoeGame(s, vocab, f"play-{s}")
        ctx = g.advance()
        assert ctx is not None
        return g, ctx

    active: list[tuple[ShoeGame, Context]] = []
    for _ in range(concurrency):
        item = _new()
        if item is None:
            break
        active.append(item)

    while active:
        decisions = decision_maker.decide([c for _, c in active])
        nxt: list[tuple[ShoeGame, Context]] = []
        for (game, _), decision in zip(active, decisions):
            game.apply(decision)
            ctx = game.advance()
            if ctx is None:
                results.extend(game.hands)
                repl = _new()
                if repl is not None:
                    nxt.append(repl)
            else:
                nxt.append((game, ctx))
        active = nxt
    return results


@dataclass
class PlayStats:
    n_hands: int
    ev_on_initial: float
    ev_se: float
    ev_on_action: float
    ev_action_se: float
    mean_bet: float
    bet_count_correlation: float
    extra: dict = field(default_factory=dict)


def _ratio_se(profit: np.ndarray, stake: np.ndarray) -> float:
    n = len(profit)
    total = stake.sum()
    if n < 2 or total <= 0:
        return float("nan")
    r = profit.sum() / total
    resid = profit - r * stake
    return float(np.sqrt(np.sum(resid**2)) / total)


def summarize(results: list[HandResult]) -> PlayStats:
    profit = np.array([h.profit_half_units for h in results], dtype=np.float64) / 2.0
    init = np.array([h.initial_bet_units for h in results], dtype=np.float64)
    wager = np.array([h.wager_units for h in results], dtype=np.float64)
    counts = np.array([h.running_count_at_bet for h in results], dtype=np.float64)
    bets = init
    corr = (
        float(np.corrcoef(bets, counts)[0, 1])
        if bets.std() > 0 and counts.std() > 0
        else 0.0
    )
    return PlayStats(
        n_hands=len(results),
        ev_on_initial=float(profit.sum() / init.sum()),
        ev_se=_ratio_se(profit, init),
        ev_on_action=float(profit.sum() / wager.sum()),
        ev_action_se=_ratio_se(profit, wager),
        mean_bet=float(bets.mean()),
        bet_count_correlation=corr,
    )

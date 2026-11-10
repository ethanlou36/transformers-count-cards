from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from blackjackgpt.blackjack.cards import RANK_MULTIPLICITIES, Card, Rank
from blackjackgpt.blackjack.hand import Hand
from blackjackgpt.blackjack.rules import VERSION1_RULES
from blackjackgpt.blackjack.settlement import settle_hands
from blackjackgpt.blackjack.state import Action, Outcome
from blackjackgpt.data.vocabulary import VERSION1_VOCABULARY
from blackjackgpt.eval.transcript import TranscriptTracker
from blackjackgpt.expert.basic_strategy import (
    basic_strategy_recommendation,
    resolve_recommendation,
)

_RANKS = tuple(Rank)
_PROBS = np.array([RANK_MULTIPLICITIES[r] for r in _RANKS], dtype=np.float64) / 52.0
_CUT = VERSION1_RULES.cut_card_index
_RESERVE = VERSION1_RULES.minimum_cards_to_start_round
_DECK = 52
_OUTCOME_TOKEN = {
    Outcome.PLAYER_BLACKJACK: "OUTCOME:PLAYER_BLACKJACK",
    Outcome.DEALER_BLACKJACK: "OUTCOME:DEALER_BLACKJACK",
    Outcome.BLACKJACK_PUSH: "OUTCOME:BLACKJACK_PUSH",
    Outcome.PLAYER_BUST: "OUTCOME:PLAYER_BUST",
    Outcome.DEALER_BUST: "OUTCOME:DEALER_BUST",
    Outcome.WIN: "OUTCOME:WIN",
    Outcome.LOSS: "OUTCOME:LOSS",
    Outcome.PUSH: "OUTCOME:PUSH",
}


class _Dealer:
    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._n = dict.fromkeys(_RANKS, 0)

    def draw(self) -> Card:
        rank = _RANKS[self._rng.choice(len(_RANKS), p=_PROBS)]
        idx = self._n[rank] % RANK_MULTIPLICITIES[rank]
        self._n[rank] += 1
        return Card(rank=rank, copy_index=idx)


def _legal_actions(hand: Hand) -> frozenset[Action]:
    value = hand.value
    if value.is_bust or value.best_legal_total == 21:
        return frozenset()
    actions = {Action.STAND, Action.HIT}
    if len(hand.cards) == 2:
        actions.add(Action.DOUBLE)
    return frozenset(actions)


def _dealer_hits(hand: Hand) -> bool:
    value = hand.value
    if value.is_bust:
        return False
    assert value.best_legal_total is not None

    return value.best_legal_total < 17


def simulate_shoe(rng: np.random.Generator) -> list[str]:
    dealer_cards = _Dealer(rng)
    tokens: list[str] = ["SHUFFLE"]
    dealt = 0

    def can_start() -> bool:
        return dealt < _CUT and (_DECK - dealt) >= _RESERVE

    while can_start():
        tokens.append("ROUND_START")
        tokens.append("BET_1")

        p1 = dealer_cards.draw()
        up = dealer_cards.draw()
        p2 = dealer_cards.draw()
        hole = dealer_cards.draw()
        player = Hand((p1, p2))
        dealer = Hand((up, hole))
        tokens += [
            f"P:{p1.rank.value}",
            f"D_UP:{up.rank.value}",
            f"P:{p2.rank.value}",
            "D_HOLE",
        ]
        dealt += 4

        resolved = False
        if up.rank in {Rank.ACE, Rank.TEN}:
            if dealer.value.is_blackjack:
                tokens.append("PEEK_BLACKJACK")
                tokens.append(f"D_REVEAL:{hole.rank.value}")
                outcome = settle_hands(
                    player_hand=player,
                    dealer_hand=dealer,
                    initial_bet_units=1,
                    current_wager_units=1,
                    rules=VERSION1_RULES,
                ).outcome
                tokens.append(_OUTCOME_TOKEN[outcome])
                resolved = True
            else:
                tokens.append("PEEK_CLEAR")

        if not resolved and player.value.is_blackjack:
            tokens.append("OUTCOME:PLAYER_BLACKJACK")
            resolved = True

        if not resolved:
            _play_player_and_dealer(tokens, player, up, hole, dealer_cards)
        tokens.append("ROUND_END")
        dealt = _count_dealt(tokens)

    tokens.append("SHOE_END")
    return tokens


def _count_dealt(tokens: list[str]) -> int:
    n = 0
    for t in tokens:
        if t.startswith(("P:", "D_UP:", "D_DRAW:")) or t == "D_HOLE":
            n += 1
    return n


def _play_player_and_dealer(tokens, player, up, hole, dealer_cards) -> None:
    dealer = Hand((up, hole))
    while True:
        legal = _legal_actions(player)
        if not legal:
            break
        recommendation = basic_strategy_recommendation(player, up.rank)
        action = resolve_recommendation(recommendation, legal)
        tokens.append(f"ACTION:{action.value}")
        if action is Action.STAND:
            break
        c = dealer_cards.draw()
        player = player.add(c)
        tokens.append(f"P:{c.rank.value}")
        if player.value.is_bust:
            tokens.append("OUTCOME:PLAYER_BUST")
            return
        if action is Action.DOUBLE or player.value.best_legal_total == 21:
            break

    tokens.append(f"D_REVEAL:{hole.rank.value}")
    while _dealer_hits(dealer):
        c = dealer_cards.draw()
        dealer = dealer.add(c)
        tokens.append(f"D_DRAW:{c.rank.value}")
    outcome = settle_hands(
        player_hand=player,
        dealer_hand=dealer,
        initial_bet_units=1,
        current_wager_units=1,
        rules=VERSION1_RULES,
    ).outcome
    tokens.append(_OUTCOME_TOKEN[outcome])


def generate(num_shoes: int, seed: int, output_dir: Path) -> Path:
    vocab = VERSION1_VOCABULARY
    tracker = TranscriptTracker(vocab)
    rng = np.random.default_rng(seed)
    all_ids: list[int] = []
    offsets = [0]
    for _ in range(num_shoes):
        toks = simulate_shoe(rng)

        tracker.reset()
        for t in toks:
            tid = vocab.encode(t)
            tracker.advance(tid)
            all_ids.append(tid)
        offsets.append(len(all_ids))

    output_dir.mkdir(parents=True, exist_ok=True)
    tokens = np.array(all_ids, dtype=np.uint16)
    np.save(output_dir / "tokens.npy", tokens, allow_pickle=False)
    np.save(
        output_dir / "shoe_offsets.npy",
        np.array(offsets, dtype=np.int64),
        allow_pickle=False,
    )
    (output_dir / "vocab.json").write_text(json.dumps(vocab.to_metadata(), indent=2))
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "infinite-deck-control",
                "num_shoes": num_shoes,
                "seed": seed,
                "tokens": len(tokens),
                "note": "i.i.d. draws; composition uninformative",
            },
            indent=2,
        )
    )
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an infinite-deck control.")
    parser.add_argument("--num-shoes", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=90001)
    parser.add_argument("--output", type=Path, default=Path("data/infinite_deck"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out = generate(args.num_shoes, args.seed, args.output)
    tok = np.load(out / "tokens.npy")
    print(f"[inf-deck] wrote {args.num_shoes} shoes / {len(tok)} tokens to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

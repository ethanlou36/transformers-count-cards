from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from blackjackgpt import __version__
from blackjackgpt.blackjack.cards import Card, shuffled_deck
from blackjackgpt.blackjack.engine import BlackjackEngine
from blackjackgpt.blackjack.events import EngineEvent, RoundSettled
from blackjackgpt.blackjack.rules import VERSION1_RULES, RulesConfig
from blackjackgpt.blackjack.state import EnginePhase
from blackjackgpt.expert.basic_strategy import (
    BASIC_STRATEGY_CROSS_CHECK_SOURCE_URL,
    BASIC_STRATEGY_PRIMARY_SOURCE_URL,
    BASIC_STRATEGY_PROFILE_ID,
    BASIC_STRATEGY_RETRIEVED_ON,
    BASIC_STRATEGY_SOFT_12_SOURCE_URL,
    BASIC_STRATEGY_TABLE_SHA256,
    canonical_basic_strategy_table_json,
)
from blackjackgpt.expert.hilo import HI_LO_WEIGHTS
from blackjackgpt.expert.observation import ActionObservation, BettingObservation
from blackjackgpt.expert.policy import (
    VERSION1_POLICY_CONFIG,
    ExpertPolicy,
    ExpertPolicyConfig,
)

from .seeds import SEED_DERIVATION_VERSION, DerivedShoeSeed, derive_shoe_seed
from .serialization import RecordedShoe, ShoeRecorder
from .validation import validate_dataset
from .vocabulary import VERSION1_VOCABULARY, Vocabulary

GENERATOR_VERSION = "full-shoe-generator-v1"
DATASET_SCHEMA_VERSION = "blackjack-full-shoe-v1"


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    num_shoes: int
    root_seed: int
    split: str = "debug"
    dataset_version: str = DATASET_SCHEMA_VERSION
    rules: RulesConfig = VERSION1_RULES

    def __post_init__(self) -> None:
        if isinstance(self.num_shoes, bool) or not isinstance(self.num_shoes, int):
            raise TypeError("num_shoes must be an integer")
        if self.num_shoes <= 0:
            raise ValueError("num_shoes must be positive")
        if isinstance(self.root_seed, bool) or not isinstance(self.root_seed, int):
            raise TypeError("root_seed must be an integer")
        if not 0 <= self.root_seed < 2**64:
            raise ValueError("root_seed must be an unsigned 64-bit integer")
        if not isinstance(self.split, str) or not self.split.strip():
            raise ValueError("split must be a non-empty string")
        if not isinstance(self.dataset_version, str) or not self.dataset_version:
            raise ValueError("dataset_version must be a non-empty string")
        if self.rules != VERSION1_RULES:
            raise ValueError(
                "version-1 data generation requires the exact VERSION1_RULES; "
                "engine-only rule overrides are not compatible with the frozen expert"
            )

    @property
    def seed_domain_version(self) -> str:
        return f"{self.dataset_version}/split/{self.split}"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "num_shoes": self.num_shoes,
            "root_seed": self.root_seed,
            "split": self.split,
            "dataset_version": self.dataset_version,
            "seed_domain_version": self.seed_domain_version,
            "rules": asdict(self.rules),
        }


@dataclass(frozen=True, slots=True)
class GeneratedDataset:
    config: GenerationConfig
    policy_config: ExpertPolicyConfig
    vocabulary: Vocabulary
    token_ids: tuple[int, ...]
    event_rows: tuple[dict[str, Any], ...]
    shoe_rows: tuple[dict[str, Any], ...]
    transcript_tokens_by_shoe: tuple[tuple[str, ...], ...]
    validation_report: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GeneratedShoe:
    recorded: RecordedShoe
    shoe_row: dict[str, Any]


def generate_one_shoe(
    config: GenerationConfig,
    shoe_index: int,
    *,
    policy: ExpertPolicy | None = None,
    vocabulary: Vocabulary = VERSION1_VOCABULARY,
) -> GeneratedShoe:
    _validate_shoe_range(shoe_index, 1)
    expert = policy or ExpertPolicy()
    retry_index = 0
    derived = derive_shoe_seed(
        config.root_seed,
        config.seed_domain_version,
        shoe_index,
        retry_index,
    )
    deck = shuffled_deck(derived.pcg_seed, stream=derived.pcg_stream)
    shoe_id = f"{config.split}-{shoe_index:08d}"
    recorded, shoe_row = _generate_one_shoe(
        config=config,
        policy=expert,
        vocabulary=vocabulary,
        deck=deck,
        shoe_id=shoe_id,
        shoe_index=shoe_index,
        retry_index=retry_index,
        derived_seed=derived,
        global_token_start=0,
    )
    shoe_row["observable_token_sha256"] = _sequence_sha256(recorded.token_texts)
    return GeneratedShoe(recorded=recorded, shoe_row=shoe_row)


def iter_generated_shoes(
    config: GenerationConfig,
    *,
    start_shoe_index: int = 0,
    shoe_count: int | None = None,
    policy: ExpertPolicy | None = None,
    vocabulary: Vocabulary = VERSION1_VOCABULARY,
) -> Iterator[GeneratedShoe]:
    resolved_count = config.num_shoes if shoe_count is None else shoe_count
    _validate_shoe_range(start_shoe_index, resolved_count)
    expert = policy or ExpertPolicy()
    for shoe_index in range(start_shoe_index, start_shoe_index + resolved_count):
        yield generate_one_shoe(
            config,
            shoe_index,
            policy=expert,
            vocabulary=vocabulary,
        )


def _validate_shoe_range(start_shoe_index: int, shoe_count: int) -> None:
    for name, value in (
        ("start_shoe_index", start_shoe_index),
        ("shoe_count", shoe_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if start_shoe_index < 0:
        raise ValueError("start_shoe_index must be nonnegative")
    if shoe_count <= 0:
        raise ValueError("shoe_count must be positive")
    if start_shoe_index + shoe_count > 2**63:
        raise ValueError("shoe range exceeds signed 64-bit storage")


def generate_dataset(
    config: GenerationConfig,
    *,
    policy: ExpertPolicy | None = None,
    vocabulary: Vocabulary = VERSION1_VOCABULARY,
) -> GeneratedDataset:
    expert = policy or ExpertPolicy()
    all_token_ids: list[int] = []
    all_event_rows: list[dict[str, Any]] = []
    shoe_rows: list[dict[str, Any]] = []
    transcripts: list[tuple[str, ...]] = []
    rank_order_hashes: set[str] = set()
    observable_hashes: set[str] = set()

    for shoe_index in range(config.num_shoes):
        retry_index = 0
        derived = derive_shoe_seed(
            config.root_seed,
            config.seed_domain_version,
            shoe_index,
            retry_index,
        )
        deck = shuffled_deck(derived.pcg_seed, stream=derived.pcg_stream)
        rank_hash = _sequence_sha256([card.rank.value for card in deck])
        if rank_hash in rank_order_hashes:
            raise RuntimeError(
                "two independently derived shoes produced the same rank order; "
                "use another root seed"
            )

        shoe_id = f"{config.split}-{shoe_index:08d}"
        recorded, shoe_row = _generate_one_shoe(
            config=config,
            policy=expert,
            vocabulary=vocabulary,
            deck=deck,
            shoe_id=shoe_id,
            shoe_index=shoe_index,
            retry_index=retry_index,
            derived_seed=derived,
            global_token_start=len(all_token_ids),
        )
        observable_hash = _sequence_sha256(recorded.token_texts)
        if observable_hash in observable_hashes:
            raise RuntimeError(
                "two distinct shoes produced the same observable transcript; "
                "use another root seed or a larger deduplication registry"
            )

        rank_order_hashes.add(rank_hash)
        observable_hashes.add(observable_hash)
        shoe_row["observable_token_sha256"] = observable_hash
        all_token_ids.extend(recorded.token_ids)
        all_event_rows.extend(recorded.event_rows)
        shoe_rows.append(shoe_row)
        transcripts.append(recorded.token_texts)

    validation = validate_dataset(
        tokens=all_token_ids,
        event_rows=all_event_rows,
        shoe_rows=shoe_rows,
        generation_config=config,
        vocabulary=vocabulary,
        policy_config=expert.config,
    )
    if not validation["passed"]:
        joined = "; ".join(str(item) for item in validation["errors"])
        raise RuntimeError(f"generated dataset failed validation: {joined}")

    return GeneratedDataset(
        config=config,
        policy_config=expert.config,
        vocabulary=vocabulary,
        token_ids=tuple(all_token_ids),
        event_rows=tuple(all_event_rows),
        shoe_rows=tuple(shoe_rows),
        transcript_tokens_by_shoe=tuple(transcripts),
        validation_report=validation,
    )


def _generate_one_shoe(
    *,
    config: GenerationConfig,
    policy: ExpertPolicy,
    vocabulary: Vocabulary,
    deck: Sequence[Card],
    shoe_id: str,
    shoe_index: int,
    retry_index: int,
    derived_seed: DerivedShoeSeed,
    global_token_start: int,
) -> tuple[RecordedShoe, dict[str, Any]]:
    engine = BlackjackEngine(config.rules)
    recorder = ShoeRecorder(
        ordered_deck=deck,
        shoe_id=shoe_id,
        shoe_index=shoe_index,
        split=config.split,
        derived_seed=derived_seed,
        global_token_start=global_token_start,
        vocabulary=vocabulary,
    )
    round_count = 0
    initial_units_wagered = 0
    total_units_wagered = 0
    outcome_counts: Counter[str] = Counter()
    action_count = 0
    deviation_match_count = 0
    deviation_activation_count = 0

    events = engine.start_shoe(deck, shoe_id=shoe_id)
    recorder.process_batch(events, engine=engine)

    transitions = 0
    while engine.phase is not EnginePhase.SHOE_COMPLETE:
        transitions += 1
        if transitions > 10_000:
            raise RuntimeError(f"shoe {shoe_id} exceeded transition safety bound")

        if engine.phase is EnginePhase.READY_FOR_ROUND:
            betting_observation = BettingObservation.from_public_snapshot(
                engine.public_snapshot(),
                running_count=recorder.trace.running_count,
            )
            bet_decision = policy.choose_bet(betting_observation)
            events = engine.start_round(bet_decision.bet_units)
            recorder.process_batch(
                events,
                engine=engine,
                bet_decision=bet_decision,
            )
            round_count += 1
            initial_units_wagered += bet_decision.bet_units
            _accumulate_settlements(events, outcome_counts)
            total_units_wagered += sum(
                event.current_wager_units
                for event in events
                if isinstance(event, RoundSettled)
            )
        elif engine.phase is EnginePhase.PLAYER_TURN:
            action_observation = ActionObservation.from_public_snapshot(
                engine.public_snapshot(),
                running_count=recorder.trace.running_count,
            )
            action_decision = policy.choose_action(action_observation)
            if action_decision.selected_action is None:
                raise RuntimeError("expert returned no action during PLAYER_TURN")
            events = engine.apply_action(action_decision.selected_action)
            recorder.process_batch(
                events,
                engine=engine,
                action_decision=action_decision,
            )
            action_count += 1
            if action_decision.matched_deviation_id is not None:
                deviation_match_count += 1
            if action_decision.action_changed_from_basic:
                deviation_activation_count += 1
            _accumulate_settlements(events, outcome_counts)
            total_units_wagered += sum(
                event.current_wager_units
                for event in events
                if isinstance(event, RoundSettled)
            )
        elif engine.phase is EnginePhase.DEALER_TURN:
            events = engine.play_dealer_and_settle()
            recorder.process_batch(events, engine=engine)
            _accumulate_settlements(events, outcome_counts)
            total_units_wagered += sum(
                event.current_wager_units
                for event in events
                if isinstance(event, RoundSettled)
            )
        else:
            raise RuntimeError(f"unexpected engine phase: {engine.phase.value}")

    recorded = recorder.finish()
    private = engine.private_snapshot()
    rank_order = [card.rank.value for card in deck]
    instance_order = [card.instance_id for card in deck]
    shoe_row: dict[str, Any] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "blackjackgpt_version": __version__,
        "dataset_version": config.dataset_version,
        "split": config.split,
        "shoe_id": shoe_id,
        "shoe_index": shoe_index,
        "retry_index": retry_index,
        "root_seed": config.root_seed,
        "seed_derivation_version": SEED_DERIVATION_VERSION,
        "seed_domain_version": config.seed_domain_version,
        "derived_seed_hex": derived_seed.digest_hex,
        "pcg_seed": derived_seed.pcg_seed,
        "pcg_stream": derived_seed.pcg_stream,
        "prng": "PCG32-XSH-RR + rejection-sampled Fisher-Yates",
        "token_start": global_token_start,
        "token_end_exclusive": global_token_start + len(recorded.token_ids),
        "token_count": len(recorded.token_ids),
        "round_count": round_count,
        "cards_dealt": private.cards_dealt,
        "cards_remaining": private.cards_remaining,
        "penetration": private.cards_dealt / 52,
        "end_reason": recorder.trace.shoe_end_reason,
        "initial_units_wagered": initial_units_wagered,
        "total_units_wagered": total_units_wagered,
        "profit_half_units": private.cumulative_profit_half_units,
        "action_count": action_count,
        "deviation_match_count": deviation_match_count,
        "deviation_activation_count": deviation_activation_count,
        "outcome_counts_json": json.dumps(
            dict(sorted(outcome_counts.items())),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "deck_rank_order": rank_order,
        "deck_instance_order": instance_order,
        "rank_order_sha256": _sequence_sha256(rank_order),
        "physical_order_sha256": _sequence_sha256(instance_order),
        "observable_token_sha256": None,
    }
    return recorded, shoe_row


def _accumulate_settlements(
    events: Sequence[EngineEvent],
    outcome_counts: Counter[str],
) -> None:
    for event in events:
        if isinstance(event, RoundSettled):
            outcome_counts[event.outcome.value] += 1


def _sequence_sha256(values: Sequence[str]) -> str:
    payload = json.dumps(
        list(values),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rules_metadata(rules: RulesConfig = VERSION1_RULES) -> dict[str, Any]:
    return {
        "rules_version": "single-deck-s17-v1",
        "rules": asdict(rules),
        "money_unit": "integer half-unit ticks",
    }


def expert_policy_metadata(
    config: ExpertPolicyConfig = VERSION1_POLICY_CONFIG,
) -> dict[str, Any]:
    return {
        "policy_version": "hilo-expert-v1",
        "information_boundary": "public observations only",
        "true_count": {
            "formula": "running_count * 52 / physical_cards_remaining",
            "threshold_convention": "compare raw true count directly",
            "stored_integer_conventions": ["floor", "ceiling", "truncation"],
        },
        "hi_lo_weights": {rank.value: weight for rank, weight in HI_LO_WEIGHTS.items()},
        "bet_ramp": [
            {
                "minimum_raw_true_count": entry.minimum_raw_true_count,
                "bet_units": entry.bet_units,
            }
            for entry in config.bet_ramp
        ],
        "deviations": [
            {
                "id": rule.deviation_id,
                "hand_type": rule.hand_type.value,
                "player_total": rule.player_total,
                "dealer_upcard": rule.dealer_upcard.value,
                "comparison": ">=",
                "index_threshold": rule.index_threshold,
                "action_at_or_above": rule.action_at_or_above.value,
                "action_below": rule.action_below.value,
                "illegal_action_fallback": rule.illegal_action_fallback.value,
            }
            for rule in config.deviations
        ],
        "basic_strategy": {
            "profile_id": BASIC_STRATEGY_PROFILE_ID,
            "table_sha256": BASIC_STRATEGY_TABLE_SHA256,
            "retrieved_on": BASIC_STRATEGY_RETRIEVED_ON,
            "primary_source": BASIC_STRATEGY_PRIMARY_SOURCE_URL,
            "cross_check_source": BASIC_STRATEGY_CROSS_CHECK_SOURCE_URL,
            "soft_12_source": BASIC_STRATEGY_SOFT_12_SOURCE_URL,
            "rules": (
                "one deck; S17; American peek; double any initial two; "
                "no split; no surrender"
            ),
            "strategy_type": "total-dependent",
            "table": json.loads(canonical_basic_strategy_table_json()),
        },
        "action_sources": [
            "BASIC_STRATEGY",
            "COUNT_DEVIATION",
            "FORCED_ACTION",
            "NATURAL_RESOLUTION",
        ],
    }


def _nonnegative_int(value: str) -> int:
    parsed = int(value, 0)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic expert-play blackjack shoes."
    )
    parser.add_argument("--num-shoes", type=int, required=True)
    parser.add_argument("--seed", type=_nonnegative_int, required=True)
    parser.add_argument("--split", default="debug")
    parser.add_argument("--dataset-version", default=DATASET_SCHEMA_VERSION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--debug-sample-count", type=int, default=3)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output directory after a successful staged build",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    config = GenerationConfig(
        num_shoes=args.num_shoes,
        root_seed=args.seed,
        split=args.split,
        dataset_version=args.dataset_version,
    )
    dataset = generate_dataset(config)

    from .writers import write_dataset

    output = write_dataset(
        dataset,
        args.output_dir,
        debug_sample_count=args.debug_sample_count,
        overwrite=args.overwrite,
    )
    report = dataset.validation_report
    print(f"wrote {report['number_of_shoes']} shoes to {output}")
    print(
        f"rounds={report['number_of_rounds']} tokens={report['number_of_tokens']} "
        f"illegal_actions={report['illegal_action_count']} "
        f"hidden_leaks={report['hidden_card_leak_count']}"
    )
    if dataset.transcript_tokens_by_shoe:
        print("sample transcript:")
        print(" ".join(dataset.transcript_tokens_by_shoe[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

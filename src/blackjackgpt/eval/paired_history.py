from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
import pyarrow.parquet as pq
import torch

from blackjackgpt.blackjack.cards import (
    Card,
    Rank,
    canonical_deck,
    validate_standard_deck,
)
from blackjackgpt.blackjack.engine import BlackjackEngine
from blackjackgpt.blackjack.events import EngineEvent, PlayerCardExposed
from blackjackgpt.blackjack.rules import VERSION1_RULES
from blackjackgpt.blackjack.state import Action, EnginePhase
from blackjackgpt.checkpoint import load_model, resolve_checkpoint
from blackjackgpt.data.generator import expert_policy_metadata, rules_metadata
from blackjackgpt.data.serialization import TraceReducer, semantic_event_for
from blackjackgpt.data.vocabulary import Vocabulary
from blackjackgpt.dataset import load_stream
from blackjackgpt.eval.transcript import TranscriptTracker
from blackjackgpt.expert.deviations import VERSION1_DEVIATIONS, DeviationRule
from blackjackgpt.expert.hilo import HI_LO_WEIGHTS
from blackjackgpt.expert.observation import ActionObservation, BettingObservation
from blackjackgpt.expert.policy import ExpertPolicy
from blackjackgpt.model import load_vocabulary
from blackjackgpt.provenance import git_commit, sha256_file
from blackjackgpt.tokens import TokenTables
from blackjackgpt.train import resolve_device

GENERATOR_VERSION = "evaluation-anchored-round-injection-v1"
VALIDATION_VERSION = "full-engine-trace-replay-v1"
REPORT_VERSION = "synthetic-target-matched-histories-v1"
SELECTION_VERSION = "balanced-boundary-pairing-v1"
_DECK_SIZE = 52

_REQUIRED_ARTIFACTS = (
    "tokens.npy",
    "shoe_offsets.npy",
    "vocab.json",
    "events.parquet",
    "shoes.parquet",
    "rules.json",
    "expert_policy.json",
)

_EVENT_COLUMNS = (
    "split",
    "shoe_id",
    "shoe_index",
    "round_id",
    "event_index",
    "global_token_offset",
    "event_token",
    "before_cards_dealt",
    "before_cards_remaining",
    "before_running_count",
    "latent_before_remaining_A",
    "latent_before_remaining_2",
    "latent_before_remaining_3",
    "latent_before_remaining_4",
    "latent_before_remaining_5",
    "latent_before_remaining_6",
    "latent_before_remaining_7",
    "latent_before_remaining_8",
    "latent_before_remaining_9",
    "latent_before_remaining_T",
)

_SHOE_COLUMNS = (
    "split",
    "shoe_id",
    "shoe_index",
    "token_start",
    "token_end_exclusive",
    "deck_rank_order",
    "deck_instance_order",
)


@dataclass(frozen=True, slots=True)
class PublicState:
    running_count: int
    cards_dealt: int

    @property
    def raw_true_count(self) -> float:
        remaining = _DECK_SIZE - self.cards_dealt
        if remaining <= 0:
            raise ValueError("true count is undefined with no physical cards remaining")
        return self.running_count * _DECK_SIZE / remaining


@dataclass(frozen=True, slots=True)
class TargetTemplate:
    deviation_id: str
    player_cards: tuple[Rank, ...]
    dealer_upcard: Rank
    dealer_hole: Rank

    @property
    def intermediate_hit_count(self) -> int:
        return len(self.player_cards) - 2

    @property
    def physical_draw_ranks(self) -> tuple[Rank, ...]:
        return (
            self.player_cards[0],
            self.dealer_upcard,
            self.player_cards[1],
            self.dealer_hole,
            *self.player_cards[2:],
        )

    @property
    def visible_count_delta(self) -> int:
        visible = (*self.player_cards, self.dealer_upcard)
        return sum(HI_LO_WEIGHTS[rank] for rank in visible)

    @property
    def action_offset_from_round_start(self) -> int:
        peek_tokens = int(self.dealer_upcard in {Rank.ACE, Rank.TEN})
        return 6 + peek_tokens + 2 * self.intermediate_hit_count

    def public_prefix_tokens(self, bet_units: int) -> tuple[str, ...]:
        tokens = [
            "ROUND_START",
            f"BET_{bet_units}",
            f"P:{self.player_cards[0].value}",
            f"D_UP:{self.dealer_upcard.value}",
            f"P:{self.player_cards[1].value}",
            "D_HOLE",
        ]
        if self.dealer_upcard in {Rank.ACE, Rank.TEN}:
            tokens.append("PEEK_CLEAR")
        for rank in self.player_cards[2:]:
            tokens.extend(("ACTION:HIT", f"P:{rank.value}"))
        return tuple(tokens)


TARGET_TEMPLATES: tuple[TargetTemplate, ...] = (
    TargetTemplate(
        "hard_16_vs_T",
        (Rank.TWO, Rank.THREE, Rank.FIVE, Rank.SIX),
        Rank.TEN,
        Rank.EIGHT,
    ),
    TargetTemplate(
        "hard_15_vs_T",
        (Rank.TWO, Rank.THREE, Rank.FOUR, Rank.SIX),
        Rank.TEN,
        Rank.EIGHT,
    ),
    TargetTemplate("hard_12_vs_3", (Rank.SIX, Rank.SIX), Rank.THREE, Rank.EIGHT),
    TargetTemplate("hard_12_vs_2", (Rank.SIX, Rank.SIX), Rank.TWO, Rank.EIGHT),
    TargetTemplate("hard_10_vs_T", (Rank.FOUR, Rank.SIX), Rank.TEN, Rank.EIGHT),
    TargetTemplate("hard_11_vs_A", (Rank.FIVE, Rank.SIX), Rank.ACE, Rank.EIGHT),
    TargetTemplate("hard_9_vs_2", (Rank.FOUR, Rank.FIVE), Rank.TWO, Rank.EIGHT),
    TargetTemplate("hard_10_vs_A", (Rank.FOUR, Rank.SIX), Rank.ACE, Rank.EIGHT),
    TargetTemplate("hard_9_vs_7", (Rank.FOUR, Rank.FIVE), Rank.SEVEN, Rank.EIGHT),
    TargetTemplate(
        "hard_16_vs_9",
        (Rank.TWO, Rank.THREE, Rank.FIVE, Rank.SIX),
        Rank.NINE,
        Rank.EIGHT,
    ),
    TargetTemplate(
        "hard_13_vs_2",
        (Rank.TWO, Rank.THREE, Rank.EIGHT),
        Rank.TWO,
        Rank.NINE,
    ),
    TargetTemplate("hard_12_vs_4", (Rank.SIX, Rank.SIX), Rank.FOUR, Rank.EIGHT),
    TargetTemplate("hard_12_vs_5", (Rank.SIX, Rank.SIX), Rank.FIVE, Rank.EIGHT),
    TargetTemplate("hard_12_vs_6", (Rank.SIX, Rank.SIX), Rank.SIX, Rank.EIGHT),
    TargetTemplate(
        "hard_13_vs_3",
        (Rank.TWO, Rank.THREE, Rank.EIGHT),
        Rank.THREE,
        Rank.NINE,
    ),
)


@dataclass(frozen=True, slots=True)
class SourceShoe:
    shoe_index: int
    shoe_ordinal: int
    shoe_id: str
    token_start: int
    token_end_exclusive: int
    ordered_deck: tuple[Card, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class SourceBoundary:
    shoe_index: int
    shoe_ordinal: int
    shoe_id: str
    round_id: int
    event_index: int
    global_token_offset: int
    cards_dealt: int
    cards_remaining: int
    running_count: int
    bet_units: int
    remaining_rank_counts: tuple[int, ...]

    @property
    def stable_key(self) -> tuple[int, int]:
        return (self.shoe_index, self.event_index)


@dataclass(frozen=True)
class EvaluationInputs:
    dataset_dir: Path
    generation_manifest: dict[str, Any]
    generation_manifest_sha256: str
    artifact_sha256: dict[str, str]
    parent_dataset_manifest_path: Path | None
    parent_dataset_manifest_sha256: str | None
    vocab: Vocabulary
    tokens: np.ndarray
    offsets: np.ndarray
    shoes: dict[int, SourceShoe]
    boundaries: tuple[SourceBoundary, ...]


@dataclass(frozen=True, slots=True)
class InjectionMove:
    target_deck_position: int
    original_deck_position: int
    card_instance_id: str
    rank: str


@dataclass(frozen=True, slots=True)
class CounterfactualMember:
    source: SourceBoundary
    action_position: int
    prediction_position: int
    cards_dealt: int
    running_count: int
    raw_true_count: float
    expert_action_id: int
    expert_action: str
    legal_action_ids: tuple[int, ...]
    legal_actions: tuple[str, ...]
    prefix_sha256: str
    source_prefix_sha256: str
    current_round_prefix: tuple[int, ...]
    reconstructed_rank_order_sha256: str
    reconstructed_physical_order_sha256: str
    completed_transcript_sha256: str
    final_cards_dealt: int
    shoe_end_reason: str
    injection_moves: tuple[InjectionMove, ...]
    prefix_ids: tuple[int, ...] = field(repr=False)

    @property
    def stable_key(self) -> tuple[int, int, str]:
        return (*self.source.stable_key, self.prefix_sha256)


@dataclass(frozen=True, slots=True)
class CounterfactualPair:
    pair_id: int
    deviation_id: str
    deviation_threshold: float
    template: TargetTemplate
    low: CounterfactualMember
    high: CounterfactualMember


@dataclass(frozen=True, slots=True)
class _TemplateCandidate:
    boundary: SourceBoundary
    target_running_count: int
    target_raw_true_count: float
    side: str


@dataclass(frozen=True, slots=True)
class _CandidatePair:
    template: TargetTemplate
    rule: DeviationRule
    low: _TemplateCandidate
    high: _TemplateCandidate


class CounterfactualReplayError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code


def _fail(code: str, detail: str) -> NoReturn:
    raise CounterfactualReplayError(code, detail)


def _sha256_token_ids(token_ids: Sequence[int] | np.ndarray) -> str:
    payload = np.asarray(token_ids, dtype="<i8").tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _sha256_strings(values: Sequence[str]) -> str:
    payload = json.dumps(list(values), ensure_ascii=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def derive_public_states(
    token_ids: Sequence[int] | np.ndarray,
    vocab: Vocabulary,
    tables: TokenTables | None = None,
) -> tuple[PublicState, ...]:
    lookup = tables if tables is not None else TokenTables(vocab)
    running_count = 0
    cards_dealt = 0
    states: list[PublicState] = []
    for raw_token_id in token_ids:
        token_id = int(raw_token_id)
        text = vocab.decode(token_id)
        if text == "SHUFFLE":
            running_count = 0
            cards_dealt = 0
        else:
            running_count += lookup.hilo_weight[token_id]
            cards_dealt += int(lookup.deals_card[token_id])
        states.append(PublicState(running_count, cards_dealt))
    return tuple(states)


def require_evaluation_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    path = Path(dataset_dir) / "generation_manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(
            "counterfactual input requires generation_manifest.json"
        ) from error
    if not isinstance(value, dict):
        raise ValueError("generation_manifest.json must contain a JSON object")
    config = value.get("generation_config")
    split = config.get("split") if isinstance(config, dict) else None
    if split != "evaluation":
        raise ValueError(
            "counterfactual construction requires generation_config.split="
            f"'evaluation'; got {split!r}"
        )
    return value


def _find_parent_dataset_manifest(
    dataset_dir: Path, generation_manifest_sha256: str
) -> tuple[Path | None, str | None]:
    for parent in dataset_dir.parents:
        candidate = parent / "dataset_manifest.json"
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            relative = (dataset_dir / "generation_manifest.json").relative_to(parent)
        except (json.JSONDecodeError, OSError, ValueError):
            continue
        artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
        metadata = (
            artifacts.get(relative.as_posix()) if isinstance(artifacts, dict) else None
        )
        if (
            isinstance(metadata, dict)
            and metadata.get("sha256") == generation_manifest_sha256
        ):
            return candidate, sha256_file(candidate)
    return None, None


def _verified_artifact_hashes(
    dataset_dir: Path, manifest: Mapping[str, Any]
) -> dict[str, str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("generation manifest must describe input artifacts")
    verified: dict[str, str] = {}
    for name in _REQUIRED_ARTIFACTS:
        metadata = artifacts.get(name)
        expected = metadata.get("sha256") if isinstance(metadata, dict) else None
        if not isinstance(expected, str):
            raise ValueError(f"generation manifest has no SHA-256 for {name}")
        actual = sha256_file(dataset_dir / name)
        if actual != expected:
            raise ValueError(f"{name} SHA-256 disagrees with generation manifest")
        verified[name] = actual
    return verified


def _load_shoes(
    dataset_dir: Path,
    offsets: np.ndarray,
    *,
    start_shoe_index: int,
) -> dict[int, SourceShoe]:
    table = pq.read_table(dataset_dir / "shoes.parquet", columns=list(_SHOE_COLUMNS))
    columns = table.to_pydict()
    canonical_by_id = {card.instance_id: card for card in canonical_deck()}
    shoes: dict[int, SourceShoe] = {}
    for row_index in range(table.num_rows):
        if str(columns["split"][row_index]) != "evaluation":
            raise ValueError("shoes.parquet contains a non-evaluation row")
        shoe_index = int(columns["shoe_index"][row_index])
        shoe_ordinal = shoe_index - start_shoe_index
        if not 0 <= shoe_ordinal < len(offsets) - 1:
            raise ValueError(f"shoe index {shoe_index} falls outside shard range")
        instance_order = tuple(
            str(v) for v in columns["deck_instance_order"][row_index]
        )
        rank_order = tuple(str(v) for v in columns["deck_rank_order"][row_index])
        try:
            deck = tuple(canonical_by_id[instance_id] for instance_id in instance_order)
        except KeyError as error:
            raise ValueError(
                f"unknown physical card instance {error.args[0]!r}"
            ) from error
        validate_standard_deck(deck)
        if tuple(card.rank.value for card in deck) != rank_order:
            raise ValueError(f"shoe {shoe_index} rank and instance orders disagree")
        token_start = int(columns["token_start"][row_index])
        token_end = int(columns["token_end_exclusive"][row_index])
        if token_start != int(offsets[shoe_ordinal]):
            raise ValueError(f"shoe {shoe_index} token_start disagrees with offsets")
        if token_end != int(offsets[shoe_ordinal + 1]):
            raise ValueError(f"shoe {shoe_index} token_end disagrees with offsets")
        if shoe_index in shoes:
            raise ValueError(f"duplicate shoe index {shoe_index}")
        shoes[shoe_index] = SourceShoe(
            shoe_index=shoe_index,
            shoe_ordinal=shoe_ordinal,
            shoe_id=str(columns["shoe_id"][row_index]),
            token_start=token_start,
            token_end_exclusive=token_end,
            ordered_deck=deck,
        )
    if len(shoes) != len(offsets) - 1:
        raise ValueError("shoes.parquet row count disagrees with shoe offsets")
    return shoes


def _load_boundaries(
    dataset_dir: Path,
    shoes: Mapping[int, SourceShoe],
) -> tuple[SourceBoundary, ...]:
    table = pq.read_table(dataset_dir / "events.parquet", columns=list(_EVENT_COLUMNS))
    columns = table.to_pydict()
    policy = ExpertPolicy()
    boundaries: list[SourceBoundary] = []
    remaining_names = tuple(f"latent_before_remaining_{rank.value}" for rank in Rank)
    for row_index, raw_token in enumerate(columns["event_token"]):
        if str(raw_token) != "ROUND_START":
            continue
        if str(columns["split"][row_index]) != "evaluation":
            raise ValueError("events.parquet contains a non-evaluation boundary")
        shoe_index = int(columns["shoe_index"][row_index])
        shoe = shoes.get(shoe_index)
        if shoe is None:
            raise ValueError(f"events.parquet references missing shoe {shoe_index}")
        event_index = int(columns["event_index"][row_index])
        global_offset = int(columns["global_token_offset"][row_index])
        if global_offset != shoe.token_start + event_index:
            raise ValueError(
                f"shoe {shoe_index} boundary global/local offsets disagree"
            )
        cards_dealt = int(columns["before_cards_dealt"][row_index])
        cards_remaining = int(columns["before_cards_remaining"][row_index])
        running_count = int(columns["before_running_count"][row_index])
        if cards_dealt + cards_remaining != _DECK_SIZE:
            raise ValueError(f"shoe {shoe_index} boundary depth is inconsistent")
        bet = policy.choose_bet(
            BettingObservation(
                running_count=running_count,
                physical_cards_remaining=cards_remaining,
            )
        )
        round_id = int(cast(Any, columns["round_id"][row_index]))
        boundaries.append(
            SourceBoundary(
                shoe_index=shoe_index,
                shoe_ordinal=shoe.shoe_ordinal,
                shoe_id=str(columns["shoe_id"][row_index]),
                round_id=round_id,
                event_index=event_index,
                global_token_offset=global_offset,
                cards_dealt=cards_dealt,
                cards_remaining=cards_remaining,
                running_count=running_count,
                bet_units=bet.bet_units,
                remaining_rank_counts=tuple(
                    int(columns[name][row_index]) for name in remaining_names
                ),
            )
        )
    return tuple(sorted(boundaries, key=lambda item: item.stable_key))


def load_evaluation_inputs(dataset_dir: str | Path) -> EvaluationInputs:
    root = Path(dataset_dir)
    manifest = require_evaluation_manifest(root)
    manifest_path = root / "generation_manifest.json"
    generation_sha256 = sha256_file(manifest_path)
    artifact_sha256 = _verified_artifact_hashes(root, manifest)
    parent_path, parent_sha256 = _find_parent_dataset_manifest(root, generation_sha256)
    config = manifest.get("generation_config")
    rules = config.get("rules") if isinstance(config, dict) else None
    expected_rules = json.loads(json.dumps(asdict(VERSION1_RULES)))
    if rules != expected_rules:
        raise ValueError("evaluation shard rules do not match VERSION1_RULES")
    stored_rules = json.loads((root / "rules.json").read_text(encoding="utf-8"))
    expected_rules_metadata = json.loads(json.dumps(rules_metadata(VERSION1_RULES)))
    if (
        stored_rules != expected_rules_metadata
        or manifest.get("resolved_rules") != expected_rules_metadata
    ):
        raise ValueError("stored rules metadata does not match VERSION1_RULES")
    policy = ExpertPolicy()
    stored_policy = json.loads(
        (root / "expert_policy.json").read_text(encoding="utf-8")
    )
    expected_policy = json.loads(json.dumps(expert_policy_metadata(policy.config)))
    if (
        stored_policy != expected_policy
        or manifest.get("resolved_expert_policy") != expected_policy
    ):
        raise ValueError("stored expert policy does not match the replay policy")

    vocab, _ = load_vocabulary(root)
    tokens, offsets = load_stream(root)
    if offsets.ndim != 1 or not len(offsets) or int(offsets[0]) != 0:
        raise ValueError("shoe_offsets.npy must be one-dimensional and begin at zero")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("counterfactual input requires complete non-empty shoes")
    range_metadata = manifest.get("range")
    start_shoe_index = (
        int(range_metadata.get("start_shoe_index", 0))
        if isinstance(range_metadata, dict)
        else 0
    )
    shoes = _load_shoes(root, offsets, start_shoe_index=start_shoe_index)
    boundaries = _load_boundaries(root, shoes)
    if not boundaries:
        raise ValueError("evaluation sidecar contains no round boundaries")
    return EvaluationInputs(
        dataset_dir=root,
        generation_manifest=manifest,
        generation_manifest_sha256=generation_sha256,
        artifact_sha256=artifact_sha256,
        parent_dataset_manifest_path=parent_path,
        parent_dataset_manifest_sha256=parent_sha256,
        vocab=vocab,
        tokens=tokens,
        offsets=offsets,
        shoes=shoes,
        boundaries=boundaries,
    )


def _rules_by_id() -> dict[str, DeviationRule]:
    return {rule.deviation_id: rule for rule in VERSION1_DEVIATIONS}


def _validate_templates() -> None:
    rules = _rules_by_id()
    if {item.deviation_id for item in TARGET_TEMPLATES} != set(rules):
        raise RuntimeError("target templates must cover every version-1 deviation")
    for template in TARGET_TEMPLATES:
        rule = rules[template.deviation_id]
        if template.dealer_upcard is not rule.dealer_upcard:
            raise RuntimeError(f"{template.deviation_id} template upcard disagrees")
        if template.action_offset_from_round_start != len(
            template.public_prefix_tokens(1)
        ):
            raise RuntimeError(f"{template.deviation_id} action offset disagrees")


_validate_templates()


def _target_templates_sha256() -> str:
    metadata = [
        {
            "deviation_id": template.deviation_id,
            "player_cards": [rank.value for rank in template.player_cards],
            "dealer_upcard": template.dealer_upcard.value,
            "dealer_hole": template.dealer_hole.value,
        }
        for template in TARGET_TEMPLATES
    ]
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _has_required_ranks(boundary: SourceBoundary, template: TargetTemplate) -> bool:
    required = Counter(template.physical_draw_ranks)
    available = {
        rank: boundary.remaining_rank_counts[index] for index, rank in enumerate(Rank)
    }
    return all(required[rank] <= available[rank] for rank in Rank)


def _candidate_for(
    boundary: SourceBoundary,
    template: TargetTemplate,
    rule: DeviationRule,
) -> _TemplateCandidate | None:
    if not _has_required_ranks(boundary, template):
        return None
    physical_draw_count = len(template.physical_draw_ranks)
    target_remaining = boundary.cards_remaining - physical_draw_count
    if target_remaining <= 0:
        return None
    target_running = boundary.running_count + template.visible_count_delta
    raw_true_count = target_running * _DECK_SIZE / target_remaining
    side = "high" if raw_true_count >= rule.index_threshold else "low"
    return _TemplateCandidate(
        boundary=boundary,
        target_running_count=target_running,
        target_raw_true_count=raw_true_count,
        side=side,
    )


def _selection_key(
    seed: int,
    deviation_id: str,
    boundary: SourceBoundary,
) -> tuple[str, int, int]:
    payload = (
        f"{SELECTION_VERSION}\0{seed}\0{deviation_id}\0"
        f"{boundary.shoe_index}\0{boundary.event_index}"
    ).encode("ascii")
    return (
        hashlib.sha256(payload).hexdigest(),
        boundary.shoe_index,
        boundary.event_index,
    )


def candidate_pairs_for_template(
    inputs: EvaluationInputs,
    template: TargetTemplate,
    *,
    seed: int,
    limit: int,
) -> tuple[list[_CandidatePair], dict[str, int]]:
    rule = _rules_by_id()[template.deviation_id]
    grouped: dict[tuple[int, int, int], dict[str, list[_TemplateCandidate]]] = (
        defaultdict(lambda: {"low": [], "high": []})
    )
    counts = Counter[str]()
    for boundary in inputs.boundaries:
        candidate = _candidate_for(boundary, template, rule)
        if candidate is None:
            counts["rank_unavailable"] += 1
            continue
        counts[f"{candidate.side}_candidates"] += 1
        group_key = (
            boundary.cards_dealt,
            boundary.event_index,
            boundary.bet_units,
        )
        grouped[group_key][candidate.side].append(candidate)

    edges: list[_CandidatePair] = []
    for sides in grouped.values():
        lows = sorted(
            sides["low"],
            key=lambda item: (
                -item.target_raw_true_count,
                _selection_key(seed, template.deviation_id, item.boundary),
            ),
        )
        highs = sorted(
            sides["high"],
            key=lambda item: (
                item.target_raw_true_count,
                _selection_key(seed, template.deviation_id, item.boundary),
            ),
        )
        unused_highs = list(highs)
        for low in lows:
            high_index = next(
                (
                    index
                    for index, high in enumerate(unused_highs)
                    if high.boundary.shoe_index != low.boundary.shoe_index
                ),
                None,
            )
            if high_index is not None:
                high = unused_highs.pop(high_index)
                edges.append(
                    _CandidatePair(
                        template=template,
                        rule=rule,
                        low=low,
                        high=high,
                    )
                )
            if not unused_highs:
                break
    result = sorted(
        edges,
        key=lambda item: (
            item.high.target_raw_true_count - item.low.target_raw_true_count,
            hashlib.sha256(
                (
                    f"{seed}\0{template.deviation_id}\0"
                    f"{item.low.boundary.stable_key}\0"
                    f"{item.high.boundary.stable_key}"
                ).encode("ascii")
            ).hexdigest(),
        ),
    )[:limit]
    counts["prefilter_pairs"] = len(result)
    return result, dict(counts)


def _inject_target_round(
    source: SourceShoe,
    boundary: SourceBoundary,
    template: TargetTemplate,
) -> tuple[tuple[Card, ...], tuple[InjectionMove, ...]]:
    prefix = source.ordered_deck[: boundary.cards_dealt]
    tail = list(source.ordered_deck[boundary.cards_dealt :])
    original_positions = {
        card.instance_id: index for index, card in enumerate(source.ordered_deck)
    }
    selected: list[Card] = []
    moves: list[InjectionMove] = []
    for relative_position, rank in enumerate(template.physical_draw_ranks):
        tail_index = next(
            (index for index, card in enumerate(tail) if card.rank is rank),
            None,
        )
        if tail_index is None:
            _fail("rank_unavailable", f"remaining shoe has no {rank.value}")
        card = tail.pop(tail_index)
        selected.append(card)
        moves.append(
            InjectionMove(
                target_deck_position=boundary.cards_dealt + relative_position,
                original_deck_position=original_positions[card.instance_id],
                card_instance_id=card.instance_id,
                rank=card.rank.value,
            )
        )
    reconstructed = (*prefix, *selected, *tail)
    validate_standard_deck(reconstructed)
    if reconstructed[: boundary.cards_dealt] != prefix:
        _fail("consumed_prefix_changed", "target injection changed a consumed card")
    return reconstructed, tuple(moves)


@dataclass
class _Replay:
    engine: BlackjackEngine
    trace: TraceReducer
    vocab: Vocabulary
    token_texts: list[str] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)

    def process(self, events: Sequence[EngineEvent]) -> None:
        for event in events:
            self.trace.apply(event)
            token = semantic_event_for(event).token
            self.token_texts.append(token)
            self.token_ids.append(self.vocab.encode(token))
        self.trace.assert_matches_engine(self.engine)

    def betting_decision(self, policy: ExpertPolicy) -> Any:
        observation = BettingObservation.from_public_snapshot(
            self.engine.public_snapshot(),
            running_count=self.trace.running_count,
        )
        return policy.choose_bet(observation)

    def action_decision(self, policy: ExpertPolicy) -> Any:
        observation = ActionObservation.from_public_snapshot(
            self.engine.public_snapshot(),
            running_count=self.trace.running_count,
        )
        return policy.choose_action(observation)


def _advance_expert_once(replay: _Replay, policy: ExpertPolicy) -> None:
    phase = replay.engine.phase
    if phase is EnginePhase.READY_FOR_ROUND:
        bet = replay.betting_decision(policy)
        replay.process(replay.engine.start_round(bet.bet_units))
    elif phase is EnginePhase.PLAYER_TURN:
        decision = replay.action_decision(policy)
        if decision.selected_action is None:
            _fail("expert_no_action", "expert returned no PLAYER_TURN action")
        replay.process(replay.engine.apply_action(decision.selected_action))
    elif phase is EnginePhase.DEALER_TURN:
        replay.process(replay.engine.play_dealer_and_settle())
    else:
        _fail("unexpected_phase", f"cannot advance phase {phase.value}")


def _validate_complete_transcript(token_ids: Sequence[int], vocab: Vocabulary) -> None:
    tracker = TranscriptTracker(vocab)
    for position, token_id in enumerate(token_ids):
        legal = tracker.legal_next()
        if int(token_id) not in legal:
            token = vocab.decode(int(token_id))
            _fail(
                "transcript_illegal",
                f"token {token!r} at complete replay position {position}",
            )
        tracker.advance(int(token_id))
    if tracker.legal_next():
        _fail("transcript_incomplete", "complete replay did not end at SHOE_END")


def replay_counterfactual_member(
    inputs: EvaluationInputs,
    boundary: SourceBoundary,
    template: TargetTemplate,
    rule: DeviationRule,
) -> CounterfactualMember:
    source = inputs.shoes[boundary.shoe_index]
    reconstructed, moves = _inject_target_round(source, boundary, template)
    replay = _Replay(
        engine=BlackjackEngine(),
        trace=TraceReducer(reconstructed),
        vocab=inputs.vocab,
    )
    policy = ExpertPolicy()
    replay.process(
        replay.engine.start_shoe(
            reconstructed,
            shoe_id=f"counterfactual-{boundary.shoe_index}-{boundary.event_index}",
        )
    )

    transitions = 0
    while len(replay.token_ids) < boundary.event_index:
        transitions += 1
        if transitions > 10_000:
            _fail("transition_limit", "source-prefix replay exceeded safety bound")
        if replay.engine.phase is EnginePhase.SHOE_COMPLETE:
            _fail("source_ended_early", "shoe ended before source round boundary")
        _advance_expert_once(replay, policy)
        if len(replay.token_ids) > boundary.event_index:
            _fail("source_position_changed", "expert replay overshot source boundary")

    if replay.engine.phase is not EnginePhase.READY_FOR_ROUND:
        _fail("source_not_ready", "source boundary is not READY_FOR_ROUND")
    snapshot = replay.engine.public_snapshot()
    if snapshot.cards_dealt != boundary.cards_dealt:
        _fail("source_depth_changed", "replay depth disagrees with source boundary")
    if replay.trace.running_count != boundary.running_count:
        _fail("source_count_changed", "replay count disagrees with source boundary")
    source_prefix = inputs.tokens[
        source.token_start : source.token_start + boundary.event_index
    ]
    if tuple(replay.token_ids) != tuple(int(value) for value in source_prefix):
        _fail("source_prefix_changed", "replayed public history differs from source")

    target_round_start = len(replay.token_ids)
    bet = replay.betting_decision(policy)
    if bet.bet_units != boundary.bet_units:
        _fail("target_bet_changed", "replayed expert bet differs from paired bet")
    replay.process(replay.engine.start_round(bet.bet_units))
    if replay.engine.phase is not EnginePhase.PLAYER_TURN:
        _fail("target_auto_resolved", "target initial deal did not reach PLAYER_TURN")

    for expected_rank in template.player_cards[2:]:
        intermediate = replay.action_decision(policy)
        if intermediate.selected_action is not Action.HIT:
            _fail(
                "intermediate_action_changed",
                f"expected HIT, got {intermediate.selected_action}",
            )
        events = replay.engine.apply_action(Action.HIT)
        exposed = [
            event.card.rank for event in events if isinstance(event, PlayerCardExposed)
        ]
        if exposed != [expected_rank]:
            _fail(
                "intermediate_draw_changed",
                f"expected {expected_rank.value}, got {[rank.value for rank in exposed]}",
            )
        replay.process(events)
        if replay.engine.phase is not EnginePhase.PLAYER_TURN:
            _fail("target_ended_early", "intermediate HIT ended the target decision")

    decision = replay.action_decision(policy)
    if decision.selected_action is None:
        _fail("target_no_action", "expert returned no target action")
    if decision.matched_deviation_id != rule.deviation_id:
        _fail(
            "target_rule_mismatch",
            f"expected {rule.deviation_id}, got {decision.matched_deviation_id}",
        )
    expected_action = (
        rule.action_at_or_above
        if decision.true_count.raw_true_count >= rule.index_threshold
        else rule.action_below
    )
    if decision.selected_action is not expected_action:
        _fail(
            "target_action_mismatch",
            f"expected {expected_action.value}, got {decision.selected_action.value}",
        )

    prefix_ids = tuple(replay.token_ids)
    current_round_prefix = tuple(replay.token_ids[target_round_start:])
    expected_current = tuple(
        inputs.vocab.encode(token)
        for token in template.public_prefix_tokens(boundary.bet_units)
    )
    if current_round_prefix != expected_current:
        _fail("target_prefix_mismatch", "controlled public target prefix disagrees")
    if len(replay.token_ids) != (
        boundary.event_index + template.action_offset_from_round_start
    ):
        _fail("target_position_mismatch", "target action position disagrees")

    state = derive_public_states(prefix_ids, inputs.vocab)[-1]
    action_snapshot = replay.engine.public_snapshot()
    if state.cards_dealt != action_snapshot.cards_dealt:
        _fail("token_depth_mismatch", "token and engine target depths disagree")
    if state.running_count != replay.trace.running_count:
        _fail("token_count_mismatch", "token and engine target counts disagree")
    if not math.isclose(
        state.raw_true_count,
        decision.true_count.raw_true_count,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        _fail("token_true_count_mismatch", "token and expert true counts disagree")

    legal_actions = tuple(sorted(action.value for action in decision.legal_actions))
    legal_action_ids = tuple(
        sorted(inputs.vocab.encode(f"ACTION:{action}") for action in legal_actions)
    )
    target_action = decision.selected_action
    action_id = inputs.vocab.encode(f"ACTION:{target_action.value}")
    if action_id not in legal_action_ids:
        _fail("target_action_illegal", "expert action is absent from legal action set")

    saved = {
        "action_position": len(prefix_ids),
        "cards_dealt": state.cards_dealt,
        "running_count": state.running_count,
        "raw_true_count": state.raw_true_count,
        "expert_action_id": action_id,
        "expert_action": target_action.value,
        "legal_action_ids": legal_action_ids,
        "legal_actions": legal_actions,
        "prefix_sha256": _sha256_token_ids(prefix_ids),
        "source_prefix_sha256": _sha256_token_ids(source_prefix),
        "current_round_prefix": current_round_prefix,
    }

    replay.process(replay.engine.apply_action(target_action))
    transitions = 0
    while replay.engine.phase is not EnginePhase.SHOE_COMPLETE:
        transitions += 1
        if transitions > 10_000:
            _fail("completion_limit", "full-shoe replay exceeded safety bound")
        _advance_expert_once(replay, policy)
    _validate_complete_transcript(replay.token_ids, inputs.vocab)
    final_snapshot = replay.engine.private_snapshot()
    if replay.trace.shoe_end_reason is None:
        _fail("missing_shoe_end", "complete replay has no shoe-end reason")

    return CounterfactualMember(
        source=boundary,
        action_position=cast(int, saved["action_position"]),
        prediction_position=cast(int, saved["action_position"]) - 1,
        cards_dealt=cast(int, saved["cards_dealt"]),
        running_count=cast(int, saved["running_count"]),
        raw_true_count=cast(float, saved["raw_true_count"]),
        expert_action_id=cast(int, saved["expert_action_id"]),
        expert_action=cast(str, saved["expert_action"]),
        legal_action_ids=cast(tuple[int, ...], saved["legal_action_ids"]),
        legal_actions=cast(tuple[str, ...], saved["legal_actions"]),
        prefix_sha256=cast(str, saved["prefix_sha256"]),
        source_prefix_sha256=cast(str, saved["source_prefix_sha256"]),
        current_round_prefix=cast(tuple[int, ...], saved["current_round_prefix"]),
        reconstructed_rank_order_sha256=_sha256_strings(
            [card.rank.value for card in reconstructed]
        ),
        reconstructed_physical_order_sha256=_sha256_strings(
            [card.instance_id for card in reconstructed]
        ),
        completed_transcript_sha256=_sha256_token_ids(replay.token_ids),
        final_cards_dealt=final_snapshot.cards_dealt,
        shoe_end_reason=replay.trace.shoe_end_reason,
        injection_moves=moves,
        prefix_ids=prefix_ids,
    )


def _pair_from_candidate(
    pair_id: int,
    inputs: EvaluationInputs,
    candidate: _CandidatePair,
) -> CounterfactualPair:
    low = replay_counterfactual_member(
        inputs, candidate.low.boundary, candidate.template, candidate.rule
    )
    high = replay_counterfactual_member(
        inputs, candidate.high.boundary, candidate.template, candidate.rule
    )
    threshold = candidate.rule.index_threshold
    if not low.raw_true_count < threshold <= high.raw_true_count:
        _fail("threshold_not_straddled", "replayed pair does not straddle threshold")
    if low.expert_action == high.expert_action:
        _fail("expert_action_not_changed", "target expert actions are identical")
    if low.expert_action != candidate.rule.action_below.value:
        _fail("low_action_mismatch", "low action is not configured below action")
    if high.expert_action != candidate.rule.action_at_or_above.value:
        _fail("high_action_mismatch", "high action is not configured above action")
    if low.current_round_prefix != high.current_round_prefix:
        _fail("pair_prefix_mismatch", "current-round prefixes differ within pair")
    if low.cards_dealt != high.cards_dealt:
        _fail("pair_depth_mismatch", "physical target depths differ within pair")
    if low.action_position != high.action_position:
        _fail("pair_position_mismatch", "target action positions differ within pair")
    if low.legal_action_ids != high.legal_action_ids:
        _fail("pair_legality_mismatch", "legal action sets differ within pair")
    if low.source.shoe_index == high.source.shoe_index:
        _fail("same_source_shoe", "pair members use the same source shoe")
    return CounterfactualPair(
        pair_id=pair_id,
        deviation_id=candidate.rule.deviation_id,
        deviation_threshold=threshold,
        template=candidate.template,
        low=low,
        high=high,
    )


def generate_counterfactual_pairs(
    inputs: EvaluationInputs,
    *,
    pairs_per_deviation: int = 8,
    seed: int = 0,
    max_pairs: int | None = None,
    replay_attempts_per_deviation: int = 500,
) -> tuple[list[CounterfactualPair], dict[str, object]]:
    if pairs_per_deviation <= 0:
        raise ValueError("pairs_per_deviation must be positive")
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("max_pairs must be positive when provided")
    if replay_attempts_per_deviation <= 0:
        raise ValueError("replay_attempts_per_deviation must be positive")

    candidate_lists: dict[str, list[_CandidatePair]] = {}
    prefilter: dict[str, dict[str, int]] = {}
    effective_attempt_limit = max(replay_attempts_per_deviation, pairs_per_deviation)
    request_limit = effective_attempt_limit
    for template in TARGET_TEMPLATES:
        candidates, counts = candidate_pairs_for_template(
            inputs,
            template,
            seed=seed,
            limit=request_limit,
        )
        candidate_lists[template.deviation_id] = candidates
        prefilter[template.deviation_id] = counts

    cursors = Counter[str]()
    attempts = Counter[str]()
    accepted = Counter[str]()
    rejected = Counter[str]()
    pairs: list[CounterfactualPair] = []
    duplicate_hashes: set[tuple[str, str]] = set()
    used_source_shoes: set[int] = set()

    while True:
        progress = False
        for template in TARGET_TEMPLATES:
            deviation_id = template.deviation_id
            if accepted[deviation_id] >= pairs_per_deviation:
                continue
            candidates = candidate_lists[deviation_id]
            while (
                cursors[deviation_id] < len(candidates)
                and attempts[deviation_id] < effective_attempt_limit
            ):
                progress = True
                candidate = candidates[cursors[deviation_id]]
                cursors[deviation_id] += 1
                attempts[deviation_id] += 1
                candidate_shoes = {
                    candidate.low.boundary.shoe_index,
                    candidate.high.boundary.shoe_index,
                }
                if candidate_shoes & used_source_shoes:
                    rejected[f"{deviation_id}:source_shoe_reuse"] += 1
                    continue
                try:
                    pair = _pair_from_candidate(len(pairs) + 1, inputs, candidate)
                    pair_hash = (pair.low.prefix_sha256, pair.high.prefix_sha256)
                    if pair_hash in duplicate_hashes:
                        _fail("duplicate_pair", "generated public prefixes repeat")
                except CounterfactualReplayError as error:
                    rejected[f"{deviation_id}:{error.code}"] += 1
                    continue
                duplicate_hashes.add(pair_hash)
                used_source_shoes.update(candidate_shoes)
                pairs.append(pair)
                accepted[deviation_id] += 1
                break
            if max_pairs is not None and len(pairs) >= max_pairs:
                progress = False
                break
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
        if not progress or all(
            accepted[item.deviation_id] >= pairs_per_deviation
            for item in TARGET_TEMPLATES
        ):
            break

    generation: dict[str, object] = {
        "source_round_boundaries": len(inputs.boundaries),
        "requested_pairs_per_deviation": pairs_per_deviation,
        "maximum_pair_count": max_pairs,
        "effective_replay_attempt_limit_per_deviation": effective_attempt_limit,
        "accepted_pairs": len(pairs),
        "source_shoe_reuse_policy": "globally disallowed",
        "unique_source_shoes": len(used_source_shoes),
        "coverage_by_deviation": dict(sorted(accepted.items())),
        "replay_attempts_by_deviation": dict(sorted(attempts.items())),
        "rejections": dict(sorted(rejected.items())),
        "prefilter": prefilter,
    }
    return pairs, generation


def directional_margins(
    low_logits: Any,
    high_logits: Any,
    low_action_id: int,
    high_action_id: int,
) -> tuple[float, float, float]:
    low_margin = float(low_logits[high_action_id]) - float(low_logits[low_action_id])
    high_margin = float(high_logits[high_action_id]) - float(high_logits[low_action_id])
    return low_margin, high_margin, high_margin - low_margin


def _aggregate_score_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    count = len(records)
    if not count:
        return {
            "pair_count": 0,
            "individual_expert_action_accuracy": None,
            "both_members_correct_fraction": None,
            "directional_accuracy": None,
            "mean_directional_shift": None,
            "median_directional_shift": None,
            "margin_straddle_fraction": None,
            "oriented_prediction_flip_fraction": None,
        }
    shifts = [cast(float, record["directional_shift"]) for record in records]
    individual_correct = sum(
        int(bool(record["low_correct"])) + int(bool(record["high_correct"]))
        for record in records
    )
    return {
        "pair_count": count,
        "individual_expert_action_accuracy": individual_correct / (2 * count),
        "both_members_correct_fraction": sum(
            int(bool(record["both_correct"])) for record in records
        )
        / count,
        "directional_accuracy": sum(shift > 0.0 for shift in shifts) / count,
        "mean_directional_shift": float(np.mean(shifts)),
        "median_directional_shift": float(np.median(shifts)),
        "margin_straddle_fraction": sum(
            cast(float, record["low_margin"]) < 0.0 < cast(float, record["high_margin"])
            for record in records
        )
        / count,
        "oriented_prediction_flip_fraction": sum(
            bool(record["low_correct"])
            and bool(record["high_correct"])
            and record["low_predicted_action"] != record["high_predicted_action"]
            for record in records
        )
        / count,
    }


@torch.no_grad()
def score_counterfactual_pairs(
    model: torch.nn.Module,
    pairs: Sequence[CounterfactualPair],
    vocab: Vocabulary,
    device: torch.device,
) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    model.eval()
    members = [member for pair in pairs for member in (pair.low, pair.high)]
    by_length: dict[int, list[CounterfactualMember]] = defaultdict(list)
    for member in members:
        by_length[len(member.prefix_ids)].append(member)

    logits_by_hash: dict[str, np.ndarray] = {}
    for prefix_length in sorted(by_length):
        group = by_length[prefix_length]
        batch = torch.as_tensor(
            np.asarray([member.prefix_ids for member in group], dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        output = model(batch)
        if not isinstance(output, Mapping):
            raise TypeError("model output must be a mapping containing logits")
        logits = output.get("logits")
        if not isinstance(logits, torch.Tensor):
            raise TypeError("model output 'logits' must be a tensor")
        final_logits = logits[:, -1, :].float().cpu().numpy()
        for member, row in zip(group, final_logits, strict=True):
            logits_by_hash[member.prefix_sha256] = row

    scores: dict[int, dict[str, object]] = {}
    by_deviation: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for pair in pairs:
        low_logits = logits_by_hash[pair.low.prefix_sha256]
        high_logits = logits_by_hash[pair.high.prefix_sha256]
        common_legal_ids = tuple(
            sorted(set(pair.low.legal_action_ids) & set(pair.high.legal_action_ids))
        )
        if (
            pair.low.expert_action_id not in common_legal_ids
            or pair.high.expert_action_id not in common_legal_ids
        ):
            raise ValueError(f"pair {pair.pair_id} expert actions are not common-legal")
        low_prediction = max(common_legal_ids, key=lambda item: float(low_logits[item]))
        high_prediction = max(
            common_legal_ids, key=lambda item: float(high_logits[item])
        )
        low_correct = low_prediction == pair.low.expert_action_id
        high_correct = high_prediction == pair.high.expert_action_id
        low_margin, high_margin, shift = directional_margins(
            low_logits,
            high_logits,
            pair.low.expert_action_id,
            pair.high.expert_action_id,
        )
        record: dict[str, object] = {
            "pair_id": f"pair-{pair.pair_id:06d}",
            "deviation_id": pair.deviation_id,
            "common_legal_actions": [
                vocab.decode(item).split(":", 1)[1] for item in common_legal_ids
            ],
            "low_predicted_action": vocab.decode(low_prediction).split(":", 1)[1],
            "high_predicted_action": vocab.decode(high_prediction).split(":", 1)[1],
            "low_correct": low_correct,
            "high_correct": high_correct,
            "both_correct": low_correct and high_correct,
            "low_margin": low_margin,
            "high_margin": high_margin,
            "directional_shift": shift,
        }
        scores[pair.pair_id] = record
        by_deviation[pair.deviation_id].append(record)

    all_records = list(scores.values())
    metrics = {
        "performed": True,
        **_aggregate_score_records(all_records),
        "by_deviation": {
            key: _aggregate_score_records(value)
            for key, value in sorted(by_deviation.items())
        },
    }
    deviation_directional = [
        cast(float, item["directional_accuracy"])
        for item in cast(dict[str, dict[str, object]], metrics["by_deviation"]).values()
        if item["directional_accuracy"] is not None
    ]
    metrics["macro_directional_accuracy"] = (
        float(np.mean(deviation_directional)) if deviation_directional else None
    )
    return scores, metrics


def _member_record(
    member: CounterfactualMember,
    *,
    token_member_index: int,
    token_start: int,
    token_end: int,
) -> dict[str, object]:
    return {
        "source_shoe_index": member.source.shoe_index,
        "source_shoe_ordinal": member.source.shoe_ordinal,
        "source_round_id": member.source.round_id,
        "source_round_start_event_index": member.source.event_index,
        "source_boundary_cards_dealt": member.source.cards_dealt,
        "target_action_position": member.action_position,
        "prediction_position": member.prediction_position,
        "target_cards_dealt": member.cards_dealt,
        "running_count": member.running_count,
        "raw_true_count": member.raw_true_count,
        "expert_action": member.expert_action,
        "legal_actions": list(member.legal_actions),
        "prefix_sha256": member.prefix_sha256,
        "source_prefix_sha256": member.source_prefix_sha256,
        "token_member_index": token_member_index,
        "token_start": token_start,
        "token_end_exclusive": token_end,
    }


def _pair_record(
    pair: CounterfactualPair,
    vocab: Vocabulary,
    *,
    low_index: int,
    high_index: int,
    offsets: Sequence[int],
) -> dict[str, object]:
    return {
        "pair_id": f"pair-{pair.pair_id:06d}",
        "pair_kind": "threshold_crossing",
        "deviation_id": pair.deviation_id,
        "deviation_threshold": pair.deviation_threshold,
        "template": {
            "player_cards": [rank.value for rank in pair.template.player_cards],
            "dealer_upcard": pair.template.dealer_upcard.value,
            "intermediate_actions": ["HIT"] * pair.template.intermediate_hit_count,
        },
        "current_round_public_tokens": [
            vocab.decode(token_id) for token_id in pair.low.current_round_prefix
        ],
        "low": _member_record(
            pair.low,
            token_member_index=low_index,
            token_start=offsets[low_index],
            token_end=offsets[low_index + 1],
        ),
        "high": _member_record(
            pair.high,
            token_member_index=high_index,
            token_start=offsets[high_index],
            token_end=offsets[high_index + 1],
        ),
    }


def _audit_member(member: CounterfactualMember) -> dict[str, object]:
    return {
        "source_shoe_index": member.source.shoe_index,
        "source_round_start_event_index": member.source.event_index,
        "reconstructed_rank_order_sha256": member.reconstructed_rank_order_sha256,
        "reconstructed_physical_order_sha256": (
            member.reconstructed_physical_order_sha256
        ),
        "completed_transcript_sha256": member.completed_transcript_sha256,
        "full_shoe_completed": True,
        "final_cards_dealt": member.final_cards_dealt,
        "shoe_end_reason": member.shoe_end_reason,
        "injection_moves": [asdict(move) for move in member.injection_moves],
    }


def write_counterfactual_artifacts(
    pairs: Sequence[CounterfactualPair],
    inputs: EvaluationInputs,
    output_dir: str | Path,
) -> tuple[dict[str, Path], dict[str, str]]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    members = [member for pair in pairs for member in (pair.low, pair.high)]
    offsets = [0]
    flat_tokens: list[int] = []
    for member in members:
        flat_tokens.extend(member.prefix_ids)
        offsets.append(len(flat_tokens))

    tokens_path = root / "counterfactual_tokens.npy"
    offsets_path = root / "counterfactual_offsets.npy"
    np.save(tokens_path, np.asarray(flat_tokens, dtype=np.int64), allow_pickle=False)
    np.save(offsets_path, np.asarray(offsets, dtype=np.int64), allow_pickle=False)

    pairs_path = root / "counterfactual_pairs.jsonl"
    pair_records = [
        _pair_record(
            pair,
            inputs.vocab,
            low_index=2 * index,
            high_index=2 * index + 1,
            offsets=offsets,
        )
        for index, pair in enumerate(pairs)
    ]
    pairs_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in pair_records
        ),
        encoding="utf-8",
    )

    audit_path = root / "counterfactual_audit.jsonl"
    audit_records = [
        {
            "pair_id": f"pair-{pair.pair_id:06d}",
            "generator_version": GENERATOR_VERSION,
            "validation_version": VALIDATION_VERSION,
            "target_hidden_hole_rank": pair.template.dealer_hole.value,
            "low": _audit_member(pair.low),
            "high": _audit_member(pair.high),
        }
        for pair in pairs
    ]
    audit_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in audit_records
        ),
        encoding="utf-8",
    )
    paths = {
        "tokens": tokens_path,
        "offsets": offsets_path,
        "pairs": pairs_path,
        "audit": audit_path,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    return paths, hashes


def _verify_checkpoint_vocabulary(
    checkpoint_path: Path,
    inputs: EvaluationInputs,
) -> tuple[bool, object]:
    metadata_path = checkpoint_path.parent / "run_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(
            "checkpoint scoring requires adjacent run_metadata.json with the exact "
            "training vocab.json SHA-256"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(
            f"cannot read checkpoint metadata at {metadata_path}"
        ) from error
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint run_metadata.json must contain a JSON object")
    dataset_metadata = metadata.get("dataset_manifest")
    artifacts = (
        dataset_metadata.get("artifacts")
        if isinstance(dataset_metadata, dict)
        else None
    )
    checkpoint_vocab_sha256 = (
        artifacts.get("vocab.json") if isinstance(artifacts, dict) else None
    )
    if not isinstance(checkpoint_vocab_sha256, str):
        raise ValueError(
            "checkpoint run_metadata.json has no training vocab.json SHA-256"
        )
    if checkpoint_vocab_sha256 != inputs.artifact_sha256["vocab.json"]:
        raise ValueError("checkpoint vocabulary SHA-256 does not match evaluation data")
    return True, metadata.get("git")


def _numeric_summary(values: Sequence[int] | Sequence[float]) -> dict[str, object]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    array = np.asarray(values)
    return {
        "count": len(values),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
    }


def _build_report(
    *,
    inputs: EvaluationInputs,
    pairs: Sequence[CounterfactualPair],
    generation: Mapping[str, object],
    artifact_paths: Mapping[str, Path],
    artifact_hashes: Mapping[str, str],
    scores_path: Path,
    scores_sha256: str,
    scoring_metrics: Mapping[str, object],
    checkpoint_path: Path | None,
    checkpoint_sha256: str | None,
    checkpoint_vocab_verified: bool | None,
    checkpoint_run_git: object,
    runtimes: Mapping[str, float],
    pairs_per_deviation: int,
    seed: int,
    max_pairs: int | None,
    replay_attempts_per_deviation: int,
) -> dict[str, object]:
    return {
        "report_version": REPORT_VERSION,
        "evaluation": "engine_validated_synthetic_target_matched_histories",
        "terminology_note": (
            "Each member has a distinct real evaluation history and an injected "
            "common target round. Earlier cards, actions, outcomes, order, and "
            "remaining composition can differ, so this is a synthetic-target "
            "matched-history contrast rather than a single-history causal "
            "counterfactual or an intervention on scalar count alone."
        ),
        "parameters": {
            "generator_version": GENERATOR_VERSION,
            "validation_version": VALIDATION_VERSION,
            "selection_version": SELECTION_VERSION,
            "selection_objective": (
                "minimum raw-true-count gap, then seeded deterministic tie-break"
            ),
            "target_templates_sha256": _target_templates_sha256(),
            "source_shoe_reuse": "globally_disallowed",
            "pairs_per_deviation": pairs_per_deviation,
            "root_seed": seed,
            "maximum_pair_count": max_pairs,
            "replay_attempts_per_deviation": replay_attempts_per_deviation,
        },
        "generation": dict(generation),
        "validation": {
            "full_standard_deck_checked": True,
            "full_engine_replay_to_shoe_end": True,
            "trace_reducer_cross_check": True,
            "public_transcript_grammar_checked": True,
            "identical_current_round_prefix": True,
            "identical_target_depth_and_position": True,
            "expert_actions_legal_and_threshold_changed": True,
            "source_shoes_globally_unique": len(
                {
                    member.source.shoe_index
                    for pair in pairs
                    for member in (pair.low, pair.high)
                }
            )
            == 2 * len(pairs),
            "validated_member_count": 2 * len(pairs),
        },
        "summaries": {
            "target_depth": _numeric_summary([pair.low.cards_dealt for pair in pairs]),
            "action_position": _numeric_summary(
                [pair.low.action_position for pair in pairs]
            ),
            "running_count_gap": _numeric_summary(
                [pair.high.running_count - pair.low.running_count for pair in pairs]
            ),
            "raw_true_count_gap": _numeric_summary(
                [pair.high.raw_true_count - pair.low.raw_true_count for pair in pairs]
            ),
        },
        "scoring": dict(scoring_metrics),
        "runtime_seconds": dict(runtimes),
        "frozen_pair_artifacts": {
            name: {"path": str(path), "sha256": artifact_hashes[name]}
            for name, path in artifact_paths.items()
        },
        "scores_artifact": {
            "path": str(scores_path),
            "sha256": scores_sha256,
        },
        "dataset": {
            "directory": str(inputs.dataset_dir),
            "split": "evaluation",
            "generation_manifest_sha256": inputs.generation_manifest_sha256,
            "parent_dataset_manifest_path": (
                str(inputs.parent_dataset_manifest_path)
                if inputs.parent_dataset_manifest_path is not None
                else None
            ),
            "parent_dataset_manifest_sha256": inputs.parent_dataset_manifest_sha256,
            "consumed_artifact_sha256": dict(sorted(inputs.artifact_sha256.items())),
            "source_control": inputs.generation_manifest.get("source_control"),
        },
        "checkpoint": {
            "path": str(checkpoint_path) if checkpoint_path is not None else None,
            "sha256": checkpoint_sha256,
            "vocabulary_sha256_verified": checkpoint_vocab_verified,
            "run_git": checkpoint_run_git,
        },
        "source_provenance": git_commit(Path(__file__).resolve().parents[3]),
    }


def run_paired_history(
    *,
    dataset_dir: str | Path,
    run_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    device: torch.device | None = None,
    pairs_per_deviation: int = 8,
    seed: int = 0,
    max_pairs: int | None = None,
    replay_attempts_per_deviation: int = 500,
) -> tuple[dict[str, object], Path, Path]:
    started = time.perf_counter()
    inputs = load_evaluation_inputs(dataset_dir)
    generation_started = time.perf_counter()
    pairs, generation = generate_counterfactual_pairs(
        inputs,
        pairs_per_deviation=pairs_per_deviation,
        seed=seed,
        max_pairs=max_pairs,
        replay_attempts_per_deviation=replay_attempts_per_deviation,
    )
    generation_seconds = time.perf_counter() - generation_started
    if not pairs:
        raise RuntimeError("no valid synthetic target pairs were generated")

    output_dir = Path(run_dir)
    artifact_paths, artifact_hashes = write_counterfactual_artifacts(
        pairs, inputs, output_dir
    )
    evaluator_path = Path(__file__).resolve()
    frozen_manifest_path = output_dir / "counterfactual_generation_manifest.json"
    frozen_manifest_path.write_text(
        json.dumps(
            {
                "generator_version": GENERATOR_VERSION,
                "validation_version": VALIDATION_VERSION,
                "selection_version": SELECTION_VERSION,
                "pair_count": len(pairs),
                "parameters": {
                    "pairs_per_deviation": pairs_per_deviation,
                    "seed": seed,
                    "maximum_pair_count": max_pairs,
                    "replay_attempts_per_deviation": (replay_attempts_per_deviation),
                    "source_shoe_reuse": "globally_disallowed",
                },
                "source_generation_manifest_sha256": (
                    inputs.generation_manifest_sha256
                ),
                "source_artifact_sha256": dict(sorted(inputs.artifact_sha256.items())),
                "generator_source": {
                    "git": git_commit(evaluator_path.parents[3]),
                    "paired_history_py_sha256": sha256_file(evaluator_path),
                    "target_templates_sha256": _target_templates_sha256(),
                    "rules_json_sha256": inputs.artifact_sha256["rules.json"],
                    "expert_policy_json_sha256": (
                        inputs.artifact_sha256["expert_policy.json"]
                    ),
                },
                "frozen_artifact_sha256": dict(sorted(artifact_hashes.items())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_paths["generation_manifest"] = frozen_manifest_path
    artifact_hashes["generation_manifest"] = sha256_file(frozen_manifest_path)

    scoring_started = time.perf_counter()
    resolved_checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpoint_sha256: str | None = None
    checkpoint_vocab_verified: bool | None = None
    checkpoint_run_git: object = None
    score_records: dict[int, dict[str, object]] = {}
    if resolved_checkpoint is None:
        scoring_metrics: dict[str, object] = {
            "performed": False,
            "reason": "no checkpoint selected; generation and validation only",
            **_aggregate_score_records([]),
            "by_deviation": {},
            "macro_directional_accuracy": None,
        }
    else:
        if not resolved_checkpoint.is_file():
            raise FileNotFoundError(resolved_checkpoint)
        scoring_device = device if device is not None else resolve_device("auto")
        checkpoint_vocab_verified, checkpoint_run_git = _verify_checkpoint_vocabulary(
            resolved_checkpoint, inputs
        )
        model, model_config = load_model(resolved_checkpoint, scoring_device)
        if model_config.vocab_size != len(inputs.vocab):
            raise ValueError(
                "checkpoint vocabulary size does not match evaluation data"
            )
        if model_config.pad_token_id != inputs.vocab.encode("PAD"):
            raise ValueError("checkpoint PAD token ID does not match evaluation data")
        longest_prefix = max(
            len(member.prefix_ids) for pair in pairs for member in (pair.low, pair.high)
        )
        if longest_prefix > model_config.context_len:
            raise ValueError(
                "checkpoint context length cannot hold a generated public prefix"
            )
        score_records, scoring_metrics = score_counterfactual_pairs(
            model, pairs, inputs.vocab, scoring_device
        )
        checkpoint_sha256 = sha256_file(resolved_checkpoint)
    scoring_seconds = time.perf_counter() - scoring_started

    scores_path = output_dir / "counterfactual_scores.jsonl"
    scores_path.write_text(
        "".join(
            json.dumps(score_records[key], sort_keys=True, separators=(",", ":")) + "\n"
            for key in sorted(score_records)
        ),
        encoding="utf-8",
    )
    scores_sha256 = sha256_file(scores_path)
    runtimes = {
        "generation_and_validation": generation_seconds,
        "scoring": scoring_seconds,
        "total": time.perf_counter() - started,
    }
    report = _build_report(
        inputs=inputs,
        pairs=pairs,
        generation=generation,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        scores_path=scores_path,
        scores_sha256=scores_sha256,
        scoring_metrics=scoring_metrics,
        checkpoint_path=resolved_checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_vocab_verified=checkpoint_vocab_verified,
        checkpoint_run_git=checkpoint_run_git,
        runtimes=runtimes,
        pairs_per_deviation=pairs_per_deviation,
        seed=seed,
        max_pairs=max_pairs,
        replay_attempts_per_deviation=replay_attempts_per_deviation,
    )
    report_path = output_dir / "counterfactual_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report, report_path, artifact_paths["pairs"]


def resolve_evaluation_shard(dataset_dir: str | Path) -> Path:
    root = Path(dataset_dir)
    if (root / "generation_manifest.json").is_file():
        return root

    search_roots = (
        root / "splits" / "evaluation" / "shards",
        root / "shards",
    )
    candidates = sorted(
        manifest.parent
        for search_root in search_roots
        if search_root.is_dir()
        for manifest in search_root.glob("*/generation_manifest.json")
    )
    unique_candidates = tuple(dict.fromkeys(candidates))
    if len(unique_candidates) == 1:
        return unique_candidates[0]
    if not unique_candidates:
        raise ValueError(
            f"{root} is neither an evaluation shard nor a release/split "
            "containing one evaluation shard"
        )
    raise ValueError(
        f"{root} contains {len(unique_candidates)} evaluation shards; "
        "pass the exact shard directory"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and optionally score engine-validated synthetic-target "
            "matched blackjack histories."
        ),
        epilog=(
            "Provide --output-dir for generation, --checkpoint for scoring, "
            "or both to keep checkpoint evaluation artifacts separate."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help=(
            "exact evaluation shard, or a release/evaluation-split directory "
            "containing exactly one shard"
        ),
    )
    parser.add_argument(
        "--output-dir",
        "--run-dir",
        dest="run_dir",
        type=Path,
        metavar="OUTPUT_DIR",
        help=(
            "artifact directory; a best/last checkpoint there is scored "
            "automatically when no explicit checkpoint is supplied"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "explicit checkpoint to score; artifacts use --output-dir when "
            "provided, otherwise they are written beside the checkpoint"
        ),
    )
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument(
        "--pairs-per-deviation",
        type=int,
        default=8,
        help="target number of pairs for each version-1 deviation",
    )
    parser.add_argument("--seed", type=int, default=0, help="deterministic pair seed")
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="optional global cap after balanced round-robin selection",
    )
    parser.add_argument(
        "--replay-attempts-per-deviation",
        type=int,
        default=500,
        help=(
            "candidate-attempt budget per deviation; raised to at least "
            "--pairs-per-deviation"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.run_dir is None and args.checkpoint is None:
        parser.error("one of --output-dir/--run-dir or --checkpoint is required")
    try:
        dataset_dir = resolve_evaluation_shard(args.dataset_dir)
    except ValueError as error:
        parser.error(str(error))
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        run_dir = args.run_dir if args.run_dir is not None else checkpoint_path.parent
    else:
        run_dir = args.run_dir
        assert run_dir is not None
        try:
            checkpoint_path = resolve_checkpoint(run_dir)
        except FileNotFoundError:
            checkpoint_path = None

    report, report_path, pairs_path = run_paired_history(
        dataset_dir=dataset_dir,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        device=resolve_device(args.device),
        pairs_per_deviation=args.pairs_per_deviation,
        seed=args.seed,
        max_pairs=args.max_pairs,
        replay_attempts_per_deviation=args.replay_attempts_per_deviation,
    )
    generation = cast(dict[str, object], report["generation"])
    print(f"[counterfactual] evaluation shard: {dataset_dir}")
    print(
        "[counterfactual] "
        f"{generation['accepted_pairs']} engine-validated synthetic target pairs"
    )
    print(
        "[counterfactual] coverage: "
        f"{json.dumps(generation['coverage_by_deviation'], sort_keys=True)}"
    )
    print(f"[counterfactual] wrote {report_path}")
    print(f"[counterfactual] wrote {pairs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

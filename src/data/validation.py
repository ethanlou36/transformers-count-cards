from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from blackjackgpt.blackjack.cards import Rank, shuffled_deck
from blackjackgpt.blackjack.state import Action
from blackjackgpt.expert.observation import ActionObservation, BettingObservation
from blackjackgpt.expert.policy import (
    VERSION1_POLICY_CONFIG,
    ExpertPolicy,
    ExpertPolicyConfig,
)

from .seeds import SEED_DERIVATION_VERSION, derive_shoe_seed
from .vocabulary import VERSION1_VOCABULARY, Vocabulary

if TYPE_CHECKING:
    from .generator import GenerationConfig


VALIDATION_VERSION = "validation-v1"

_HI_LO_BY_VALUE: dict[str, int] = {
    Rank.ACE.value: -1,
    Rank.TWO.value: 1,
    Rank.THREE.value: 1,
    Rank.FOUR.value: 1,
    Rank.FIVE.value: 1,
    Rank.SIX.value: 1,
    Rank.SEVEN.value: 0,
    Rank.EIGHT.value: 0,
    Rank.NINE.value: 0,
    Rank.TEN.value: -1,
}
_VISIBLE_CARD_PREFIXES = ("P:", "D_UP:", "D_REVEAL:", "D_DRAW:")
_PHYSICAL_DRAW_PREFIXES = ("P:", "D_UP:", "D_DRAW:")
_SIDECAR_SCHEMA_VERSION = "blackjack-event-sidecar-v1"
_DATASET_SCHEMA_VERSION = "blackjack-full-shoe-v1"

_EXACT_EVENT_IDENTITIES: dict[str, tuple[str, str]] = {
    "SHUFFLE": ("ShoeStarted", "BOUNDARY"),
    "ROUND_START": ("RoundStarted", "BOUNDARY"),
    "ROUND_END": ("RoundEnded", "BOUNDARY"),
    "SHOE_END": ("ShoeEnded", "BOUNDARY"),
    "D_HOLE": ("DealerHoleCardDealtHidden", "HIDDEN_MARKER"),
    "PEEK_CLEAR": ("PeekCompleted", "PEEK"),
    "PEEK_BLACKJACK": ("PeekCompleted", "PEEK"),
}
_PREFIX_EVENT_IDENTITIES: tuple[tuple[str, str, str], ...] = (
    ("BET_", "BetPlaced", "BET"),
    ("P:", "PlayerCardExposed", "PLAYER_CARD"),
    ("D_UP:", "DealerUpcardExposed", "DEALER_CARD"),
    ("ACTION:", "PlayerActionTaken", "ACTION"),
    ("D_REVEAL:", "DealerHoleCardRevealed", "DEALER_CARD"),
    ("D_DRAW:", "DealerCardDrawn", "DEALER_CARD"),
    ("OUTCOME:", "RoundSettled", "OUTCOME"),
)


def _quantile(sorted_values: Sequence[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum": float(ordered[0]) if ordered else None,
        "p05": _quantile(ordered, 0.05),
        "p25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.50),
        "p75": _quantile(ordered, 0.75),
        "p95": _quantile(ordered, 0.95),
        "maximum": float(ordered[-1]) if ordered else None,
        "mean": statistics.fmean(ordered) if ordered else None,
    }


def _visible_rank(token: str) -> str | None:
    for prefix in _VISIBLE_CARD_PREFIXES:
        if token.startswith(prefix):
            return token.removeprefix(prefix)
    return None


def _expected_event_identity(token: str) -> tuple[str, str] | None:
    exact = _EXACT_EVENT_IDENTITIES.get(token)
    if exact is not None:
        return exact
    for prefix, event_type, category in _PREFIX_EVENT_IDENTITIES:
        if token.startswith(prefix):
            return event_type, category
    return None


def _clustered_ratio_confidence_interval(
    numerators: Sequence[float],
    denominators: Sequence[int],
) -> dict[str, Any]:
    total_denominator = sum(denominators)
    if not numerators or total_denominator == 0:
        return {
            "method": "shoe_clustered_ratio_normal_95",
            "sample_size": 0,
            "mean": None,
            "standard_error": None,
            "lower": None,
            "upper": None,
        }
    mean = sum(numerators) / total_denominator
    if len(numerators) < 2:
        return {
            "method": "shoe_clustered_ratio_normal_95",
            "sample_size": 1,
            "mean": mean,
            "standard_error": None,
            "lower": None,
            "upper": None,
        }
    residual_sum_squares = sum(
        (numerator - mean * denominator) ** 2
        for numerator, denominator in zip(
            numerators,
            denominators,
            strict=True,
        )
    )
    finite_sample_factor = len(numerators) / (len(numerators) - 1)
    standard_error = math.sqrt(
        finite_sample_factor * residual_sum_squares / total_denominator**2
    )
    margin = 1.96 * standard_error
    return {
        "method": "shoe_clustered_ratio_normal_95",
        "sample_size": len(numerators),
        "mean": mean,
        "standard_error": standard_error,
        "lower": mean - margin,
        "upper": mean + margin,
    }


def validate_dataset(
    *,
    tokens: Sequence[int],
    event_rows: Sequence[Mapping[str, Any]],
    shoe_rows: Sequence[Mapping[str, Any]],
    generation_config: GenerationConfig,
    vocabulary: Vocabulary = VERSION1_VOCABULARY,
    policy_config: ExpertPolicyConfig = VERSION1_POLICY_CONFIG,
    expected_start_shoe_index: int = 0,
) -> dict[str, Any]:
    if isinstance(expected_start_shoe_index, bool) or not isinstance(
        expected_start_shoe_index, int
    ):
        raise TypeError("expected_start_shoe_index must be an integer")
    if expected_start_shoe_index < 0:
        raise ValueError("expected_start_shoe_index must be nonnegative")

    errors: list[str] = []
    warnings: list[str] = []
    illegal_action_count = 0
    hidden_card_leak_count = 0
    count_transition_errors = 0
    physical_transition_errors = 0
    true_count_error_count = 0
    policy_error_count = 0
    summary_error_count = 0
    vocabulary_error_count = 0
    event_identity_error_count = 0

    if len(tokens) != len(event_rows):
        errors.append(
            "token/sidecar length mismatch: "
            f"{len(tokens)} tokens versus {len(event_rows)} rows"
        )

    action_frequencies: Counter[str] = Counter()
    bet_frequencies: Counter[str] = Counter()
    action_source_frequencies: Counter[str] = Counter()
    outcome_frequencies: Counter[str] = Counter()
    true_counts: list[float] = []
    deviation_matches = 0
    deviation_activations = 0
    current_shoe: str | None = None
    previous_after_count: int | None = None
    previous_after_remaining: int | None = None
    previous_after_cumulative: int | None = None
    shoe_first_token: dict[str, str] = {}
    shoe_last_token: dict[str, str] = {}
    hidden_marker_by_round: dict[tuple[str, int], bool] = {}
    reveal_by_round: dict[tuple[str, int], bool] = {}
    rows_by_shoe: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    expert = ExpertPolicy(policy_config)

    for row_index, row in enumerate(event_rows):
        token = str(row["event_token"])
        shoe_id = str(row["shoe_id"])
        token_id = int(row["token_id"])
        global_offset = int(row["global_token_offset"])
        rows_by_shoe[shoe_id].append(row)

        if row_index < len(tokens) and int(tokens[row_index]) != token_id:
            errors.append(f"token ID mismatch at global offset {row_index}")
        try:
            expected_token_id = vocabulary.encode(token)
            if expected_token_id != token_id or vocabulary.decode(token_id) != token:
                vocabulary_error_count += 1
        except (TypeError, ValueError):
            vocabulary_error_count += 1
        if global_offset != row_index:
            errors.append(
                f"non-contiguous global token offset {global_offset} at row {row_index}"
            )
        expected_identity = _expected_event_identity(token)
        if (
            row.get("schema_version") != _SIDECAR_SCHEMA_VERSION
            or expected_identity is None
            or (row.get("event_type"), row.get("token_category")) != expected_identity
        ):
            event_identity_error_count += 1

        shoe_first_token.setdefault(shoe_id, token)
        shoe_last_token[shoe_id] = token
        if current_shoe != shoe_id:
            current_shoe = shoe_id
            previous_after_count = None
            previous_after_remaining = None
            previous_after_cumulative = None

        before_count = int(row["before_running_count"])
        after_count = int(row["after_running_count"])
        before_remaining = int(row["before_cards_remaining"])
        after_remaining = int(row["after_cards_remaining"])
        before_cumulative = int(row["before_cumulative_profit_half_units"])
        after_cumulative = int(row["after_cumulative_profit_half_units"])
        if previous_after_count is not None and before_count != previous_after_count:
            count_transition_errors += 1
        if (
            previous_after_remaining is not None
            and before_remaining != previous_after_remaining
        ):
            physical_transition_errors += 1
        if (
            previous_after_cumulative is not None
            and before_cumulative != previous_after_cumulative
        ):
            summary_error_count += 1

        visible_rank = _visible_rank(token)
        expected_count_delta = _HI_LO_BY_VALUE.get(visible_rank or "", 0)
        if after_count - before_count != expected_count_delta:
            count_transition_errors += 1

        is_physical_draw = token == "D_HOLE" or token.startswith(
            _PHYSICAL_DRAW_PREFIXES
        )
        expected_remaining_delta = -1 if is_physical_draw else 0
        if after_remaining - before_remaining != expected_remaining_delta:
            physical_transition_errors += 1

        previous_after_count = after_count
        previous_after_remaining = after_remaining
        previous_after_cumulative = after_cumulative

        for prefix in ("before", "after"):
            if not _true_count_columns_are_valid(row, prefix):
                true_count_error_count += 1

        event_profit = int(row["profit_half_units"] or 0)
        expected_cumulative_delta = event_profit if token.startswith("OUTCOME:") else 0
        if after_cumulative - before_cumulative != expected_cumulative_delta:
            summary_error_count += 1

        round_id_raw = row.get("round_id")
        round_key: tuple[str, int] | None = None
        if round_id_raw is not None:
            round_key = (shoe_id, int(round_id_raw))
        if token.startswith("D_HOLE") and token != "D_HOLE":
            hidden_card_leak_count += 1
        if token == "D_HOLE" and round_key is not None:
            hidden_marker_by_round[round_key] = True
            if (
                row.get("latent_before_dealer_hole_rank") is not None
                or row.get("latent_after_dealer_hole_rank") is None
                or row.get("before_dealer_hole_observable")
                or row.get("after_dealer_hole_observable")
                or row.get("before_dealer_visible_hand")
                != row.get("after_dealer_visible_hand")
                or row.get("before_exposed_cards") != row.get("after_exposed_cards")
            ):
                hidden_card_leak_count += 1
        if token.startswith("D_REVEAL:") and round_key is not None:
            if not hidden_marker_by_round.get(round_key, False):
                hidden_card_leak_count += 1
            if reveal_by_round.get(round_key, False):
                hidden_card_leak_count += 1
            reveal_by_round[round_key] = True
            revealed_rank = token.removeprefix("D_REVEAL:")
            if (
                row.get("latent_before_dealer_hole_rank") != revealed_rank
                or row.get("latent_after_dealer_hole_rank") != revealed_rank
                or row.get("before_dealer_hole_observable")
                or not row.get("after_dealer_hole_observable")
            ):
                hidden_card_leak_count += 1

        if row.get("latent_after_dealer_hole_rank") is not None and not row.get(
            "after_dealer_hole_observable"
        ):
            upcard = row.get("after_dealer_upcard")
            expected_public_dealer = [upcard] if upcard is not None else []
            if row.get("after_dealer_visible_hand") != expected_public_dealer:
                hidden_card_leak_count += 1

        if not _latent_composition_is_valid(row):
            physical_transition_errors += 1

        if token.startswith("ACTION:"):
            action = token.removeprefix("ACTION:")
            action_frequencies[action] += 1
            legal = {str(value) for value in row.get("legal_actions") or ()}
            final_action = row.get("final_expert_action")
            if final_action != action or action not in legal:
                illegal_action_count += 1
            source = row.get("action_source")
            if source is not None:
                action_source_frequencies[str(source)] += 1
            if row.get("matched_deviation_id") is not None:
                deviation_matches += 1
            if bool(row.get("deviation_activated")):
                deviation_activations += 1
            if not _decision_true_count_is_valid(row):
                policy_error_count += 1
            if bool(row.get("deviation_activated")) != (source == "COUNT_DEVIATION"):
                policy_error_count += 1
            if not _recorded_action_matches_policy(row, expert):
                policy_error_count += 1
        elif token.startswith("BET_"):
            bet_frequencies[token] += 1
            if not _decision_true_count_is_valid(row):
                policy_error_count += 1
            if row.get(
                "decision_raw_true_count"
            ) is None or not _recorded_bet_matches_policy(row, expert):
                policy_error_count += 1

        if token.startswith("OUTCOME:"):
            outcome_name = token.removeprefix("OUTCOME:")
            outcome_frequencies[outcome_name] += 1
            if row.get("outcome") != outcome_name or not _profit_is_valid(row):
                summary_error_count += 1
            source = row.get("action_source")
            if source == "NATURAL_RESOLUTION":
                action_source_frequencies[str(source)] += 1

        decision_tc = row.get("decision_raw_true_count")
        if decision_tc is not None:
            true_counts.append(float(decision_tc))

    if count_transition_errors:
        errors.append(f"{count_transition_errors} Hi-Lo transition errors")
    if physical_transition_errors:
        errors.append(f"{physical_transition_errors} physical-card transition errors")
    if true_count_error_count:
        errors.append(f"{true_count_error_count} true-count field errors")
    if policy_error_count:
        errors.append(f"{policy_error_count} expert-policy record errors")
    if summary_error_count:
        errors.append(f"{summary_error_count} wager/profit summary errors")
    if vocabulary_error_count:
        errors.append(f"{vocabulary_error_count} vocabulary alignment errors")
    if event_identity_error_count:
        errors.append(f"{event_identity_error_count} event identity errors")
    if illegal_action_count:
        errors.append(f"{illegal_action_count} illegal or unexplained actions")
    if hidden_card_leak_count:
        errors.append(f"{hidden_card_leak_count} hidden-card token leaks")

    for shoe_id, first_token in shoe_first_token.items():
        if first_token != "SHUFFLE":
            errors.append(f"shoe {shoe_id} begins with {first_token}, not SHUFFLE")
        if shoe_last_token[shoe_id] != "SHOE_END":
            errors.append(
                f"shoe {shoe_id} ends with {shoe_last_token[shoe_id]}, not SHOE_END"
            )

    errors.extend(_stream_grammar_errors(event_rows))
    metadata_errors = _shoe_metadata_errors(
        tokens=tokens,
        rows_by_shoe=rows_by_shoe,
        shoe_rows=shoe_rows,
        generation_config=generation_config,
        expected_start_shoe_index=expected_start_shoe_index,
    )
    summary_error_count += len(metadata_errors)
    errors.extend(metadata_errors)

    number_of_shoes = len(shoe_rows)
    number_of_rounds = sum(outcome_frequencies.values())
    total_profit_half_units = sum(
        int(row["profit_half_units"] or 0)
        for row in event_rows
        if str(row["event_token"]).startswith("OUTCOME:")
    )
    total_initial_units = sum(
        int(row["decision_bet_units"] or 0)
        for row in event_rows
        if str(row["event_token"]).startswith("BET_")
    )
    total_wagered_units = sum(
        int(row["after_wager_units"] or 0)
        for row in event_rows
        if str(row["event_token"]).startswith("OUTCOME:")
    )
    shoe_profit_units: list[float] = []
    shoe_round_counts: list[int] = []
    for shoe_row in shoe_rows:
        shoe_id = str(shoe_row["shoe_id"])
        outcome_rows = [
            row
            for row in rows_by_shoe.get(shoe_id, [])
            if str(row["event_token"]).startswith("OUTCOME:")
        ]
        shoe_profit_units.append(
            sum(int(row["profit_half_units"] or 0) for row in outcome_rows) / 2
        )
        shoe_round_counts.append(len(outcome_rows))

    if number_of_shoes == 0:
        errors.append("dataset contains no shoes")
    if number_of_rounds == 0:
        errors.append("dataset contains no rounds")

    win_names = {"PLAYER_BLACKJACK", "DEALER_BUST", "WIN"}
    loss_names = {"DEALER_BLACKJACK", "PLAYER_BUST", "LOSS"}
    push_names = {"BLACKJACK_PUSH", "PUSH"}
    wins = sum(outcome_frequencies[name] for name in win_names)
    losses = sum(outcome_frequencies[name] for name in loss_names)
    pushes = sum(outcome_frequencies[name] for name in push_names)
    blackjack_outcomes = {
        name: outcome_frequencies[name]
        for name in ("PLAYER_BLACKJACK", "DEALER_BLACKJACK", "BLACKJACK_PUSH")
    }
    if wins + losses + pushes != number_of_rounds:
        errors.append(
            "outcome count does not equal round count: "
            f"{wins + losses + pushes} versus {number_of_rounds}"
        )

    if number_of_shoes and len(shoe_first_token) != number_of_shoes:
        errors.append(
            "sidecar shoe count does not equal shoes table count: "
            f"{len(shoe_first_token)} versus {number_of_shoes}"
        )
    if number_of_shoes < 30:
        warnings.append(
            "confidence interval is based on fewer than 30 independent shoes"
        )

    return {
        "validation_version": VALIDATION_VERSION,
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "number_of_shoes": number_of_shoes,
        "number_of_rounds": number_of_rounds,
        "number_of_tokens": len(tokens),
        "average_rounds_per_shoe": (
            number_of_rounds / number_of_shoes if number_of_shoes else None
        ),
        "average_tokens_per_shoe": (
            len(tokens) / number_of_shoes if number_of_shoes else None
        ),
        "action_frequencies": dict(sorted(action_frequencies.items())),
        "action_source_frequencies": dict(sorted(action_source_frequencies.items())),
        "bet_bucket_frequencies": dict(sorted(bet_frequencies.items())),
        "decision_true_count_distribution": _distribution(true_counts),
        "deviation_matches": deviation_matches,
        "deviation_activations": deviation_activations,
        "outcome_frequencies": dict(sorted(outcome_frequencies.items())),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "blackjack_outcomes": blackjack_outcomes,
        "total_profit_half_units": total_profit_half_units,
        "profit_per_round_units": (
            (total_profit_half_units / 2) / number_of_rounds
            if number_of_rounds
            else None
        ),
        "profit_per_initial_unit_wagered": (
            (total_profit_half_units / 2) / total_initial_units
            if total_initial_units
            else None
        ),
        "profit_per_total_unit_wagered": (
            (total_profit_half_units / 2) / total_wagered_units
            if total_wagered_units
            else None
        ),
        "profit_per_round_95_percent_confidence_interval": (
            _clustered_ratio_confidence_interval(
                shoe_profit_units,
                shoe_round_counts,
            )
        ),
        "illegal_action_count": illegal_action_count,
        "hidden_card_leak_count": hidden_card_leak_count,
        "count_transition_error_count": count_transition_errors,
        "physical_transition_error_count": physical_transition_errors,
        "true_count_error_count": true_count_error_count,
        "policy_error_count": policy_error_count,
        "summary_error_count": summary_error_count,
        "vocabulary_error_count": vocabulary_error_count,
        "event_identity_error_count": event_identity_error_count,
    }


def _true_count_columns_are_valid(
    row: Mapping[str, Any],
    prefix: str,
) -> bool:
    running_count = int(row[f"{prefix}_running_count"])
    cards_remaining = int(row[f"{prefix}_cards_remaining"])
    numerator = int(row[f"{prefix}_true_count_numerator"])
    denominator = int(row[f"{prefix}_true_count_denominator"])
    raw = row[f"{prefix}_raw_true_count"]
    floor = row[f"{prefix}_true_count_floor"]
    ceiling = row[f"{prefix}_true_count_ceiling"]
    truncation = row[f"{prefix}_true_count_truncation"]
    if numerator != running_count * 52 or denominator != cards_remaining:
        return False
    if cards_remaining == 0:
        return raw is None and floor is None and ceiling is None and truncation is None
    if cards_remaining < 0:
        return False
    expected = numerator / denominator
    return (
        raw is not None
        and math.isclose(float(raw), expected, rel_tol=1e-12, abs_tol=1e-12)
        and floor == math.floor(expected)
        and ceiling == math.ceil(expected)
        and truncation == math.trunc(expected)
    )


def _decision_true_count_is_valid(row: Mapping[str, Any]) -> bool:
    try:
        running_count = int(row["decision_running_count"])
        cards_remaining = int(row["decision_cards_remaining"])
        numerator = int(row["decision_true_count_numerator"])
        denominator = int(row["decision_true_count_denominator"])
        raw = float(row["decision_raw_true_count"])
    except (TypeError, ValueError):
        return False
    if (
        running_count != int(row["before_running_count"])
        or cards_remaining != int(row["before_cards_remaining"])
        or numerator != running_count * 52
        or denominator != cards_remaining
        or denominator <= 0
    ):
        return False
    expected = numerator / denominator
    return (
        math.isclose(raw, expected, rel_tol=1e-12, abs_tol=1e-12)
        and row["decision_true_count_floor"] == math.floor(expected)
        and row["decision_true_count_ceiling"] == math.ceil(expected)
        and row["decision_true_count_truncation"] == math.trunc(expected)
    )


def _recorded_bet_matches_policy(
    row: Mapping[str, Any],
    expert: ExpertPolicy,
) -> bool:
    try:
        observation = BettingObservation(
            running_count=int(row["before_running_count"]),
            physical_cards_remaining=int(row["before_cards_remaining"]),
        )
        decision = expert.choose_bet(observation)
    except (TypeError, ValueError):
        return False
    return (
        row["event_token"] == decision.token
        and row["decision_bet_units"] == decision.bet_units
        and row["bet_units"] == decision.bet_units
        and row["bet_ramp_entry_index"] == decision.ramp_entry_index
        and row["bet_ramp_minimum_raw_true_count"]
        == decision.ramp_entry.minimum_raw_true_count
    )


def _recorded_action_matches_policy(
    row: Mapping[str, Any],
    expert: ExpertPolicy,
) -> bool:
    try:
        observation = ActionObservation(
            player_cards=tuple(Rank(value) for value in row["before_player_hand"]),
            dealer_upcard=Rank(str(row["before_dealer_upcard"])),
            running_count=int(row["before_running_count"]),
            physical_cards_remaining=int(row["before_cards_remaining"]),
            legal_actions=frozenset(Action(value) for value in row["legal_actions"]),
        )
        decision = expert.choose_action(observation)
    except (TypeError, ValueError):
        return False
    return (
        decision.selected_action is not None
        and row["event_token"] == f"ACTION:{decision.selected_action.value}"
        and row["final_expert_action"] == decision.selected_action.value
        and row["action_source"] == decision.action_source.value
        and row["basic_strategy_recommendation"]
        == (
            decision.basic_recommendation.value
            if decision.basic_recommendation is not None
            else None
        )
        and row["basic_strategy_action"]
        == (
            decision.basic_strategy_action.value
            if decision.basic_strategy_action is not None
            else None
        )
        and row["matched_deviation_id"] == decision.matched_deviation_id
        and row["applied_deviation_id"] == decision.applied_deviation_id
        and row["deviation_index_threshold"] == decision.deviation_index_threshold
        and row["deviation_threshold_met"] == decision.deviation_threshold_met
        and row["deviation_preferred_action"]
        == (
            decision.deviation_preferred_action.value
            if decision.deviation_preferred_action is not None
            else None
        )
        and row["deviation_fallback_used"] == decision.deviation_fallback_used
        and row["deviation_activated"] == decision.action_changed_from_basic
    )


def _profit_is_valid(row: Mapping[str, Any]) -> bool:
    outcome = str(row["outcome"])
    bet = int(row["bet_units"])
    wager = int(row["after_wager_units"])
    actual = int(row["profit_half_units"])
    if outcome == "PLAYER_BLACKJACK":
        expected = 3 * bet
    elif outcome == "DEALER_BLACKJACK":
        expected = -2 * bet
    elif outcome in {"PLAYER_BUST", "LOSS"}:
        expected = -2 * wager
    elif outcome in {"DEALER_BUST", "WIN"}:
        expected = 2 * wager
    elif outcome in {"BLACKJACK_PUSH", "PUSH"}:
        expected = 0
    else:
        return False
    return actual == expected


def _latent_composition_is_valid(row: Mapping[str, Any]) -> bool:
    maximums = {rank.value: (16 if rank is Rank.TEN else 4) for rank in Rank}
    for prefix in ("before", "after"):
        counts = {
            rank.value: int(row[f"latent_{prefix}_remaining_{rank.value}"])
            for rank in Rank
        }
        if any(count < 0 or count > maximums[rank] for rank, count in counts.items()):
            return False
        if sum(counts.values()) != int(row[f"{prefix}_cards_remaining"]):
            return False
    token = str(row["event_token"])
    before = {
        rank.value: int(row[f"latent_before_remaining_{rank.value}"]) for rank in Rank
    }
    after = {
        rank.value: int(row[f"latent_after_remaining_{rank.value}"]) for rank in Rank
    }
    drawn_rank: str | None = None
    if token.startswith(("P:", "D_UP:", "D_DRAW:")):
        drawn_rank = token.split(":", maxsplit=1)[1]
    elif token == "D_HOLE":
        value = row.get("latent_after_dealer_hole_rank")
        drawn_rank = str(value) if value is not None else None
    expected = dict(before)
    if drawn_rank is not None:
        if drawn_rank not in expected:
            return False
        expected[drawn_rank] -= 1
    return expected == after


def _stream_grammar_errors(
    event_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in event_rows:
        grouped[str(row["shoe_id"])].append(row)

    for shoe_id, rows in grouped.items():
        active_round = False
        active_round_id: int | None = None
        next_round_id = 1
        stage = "SHUFFLE"
        dealer_upcard: str | None = None
        revealed = False
        for index, row in enumerate(rows):
            token = str(row["event_token"])
            round_id_raw = row.get("round_id")
            if stage == "SHUFFLE":
                if token != "SHUFFLE":
                    errors.append(f"shoe {shoe_id} has invalid first token {token}")
                if round_id_raw is not None:
                    errors.append(f"shoe {shoe_id} SHUFFLE has a round ID")
                stage = "BETWEEN_ROUNDS"
                continue
            if token == "ROUND_START":
                if active_round or stage != "BETWEEN_ROUNDS":
                    errors.append(f"shoe {shoe_id} has nested round at token {index}")
                active_round = True
                try:
                    active_round_id = (
                        int(round_id_raw) if round_id_raw is not None else None
                    )
                except (TypeError, ValueError):
                    active_round_id = None
                if active_round_id != next_round_id:
                    errors.append(
                        f"shoe {shoe_id} has non-sequential round ID at token {index}"
                    )
                next_round_id += 1
                stage = "BET"
                dealer_upcard = None
                revealed = False
                continue
            if token == "SHOE_END":
                if (
                    active_round
                    or stage != "BETWEEN_ROUNDS"
                    or index != len(rows) - 1
                    or round_id_raw is not None
                ):
                    errors.append(f"shoe {shoe_id} has misplaced SHOE_END")
                stage = "DONE"
                continue
            if not active_round:
                errors.append(f"shoe {shoe_id} token {token} is outside a round")
                continue
            try:
                row_round_id = int(round_id_raw) if round_id_raw is not None else None
            except (TypeError, ValueError):
                row_round_id = None
            if row_round_id != active_round_id:
                errors.append(
                    f"shoe {shoe_id} has inconsistent round ID at token {index}"
                )
            if stage == "BET":
                stage = "PLAYER_ONE" if token.startswith("BET_") else "INVALID"
            elif stage == "PLAYER_ONE":
                stage = "DEALER_UP" if token.startswith("P:") else "INVALID"
            elif stage == "DEALER_UP":
                if token.startswith("D_UP:"):
                    dealer_upcard = token.removeprefix("D_UP:")
                    stage = "PLAYER_TWO"
                else:
                    stage = "INVALID"
            elif stage == "PLAYER_TWO":
                stage = "HIDDEN" if token.startswith("P:") else "INVALID"
            elif stage == "HIDDEN":
                if token == "D_HOLE":
                    stage = "PEEK" if dealer_upcard in {"A", "T"} else "PLAY"
                else:
                    stage = "INVALID"
            elif stage == "PEEK":
                if token == "PEEK_CLEAR":
                    stage = "PLAY"
                elif token == "PEEK_BLACKJACK":
                    stage = "POSITIVE_REVEAL"
                else:
                    stage = "INVALID"
            elif stage == "POSITIVE_REVEAL":
                if token.startswith("D_REVEAL:"):
                    revealed = True
                    stage = "POSITIVE_OUTCOME"
                else:
                    stage = "INVALID"
            elif stage == "POSITIVE_OUTCOME":
                stage = "ROUND_END" if token.startswith("OUTCOME:") else "INVALID"
            elif stage == "PLAY":
                if token.startswith("D_REVEAL:"):
                    if revealed:
                        stage = "INVALID"
                    revealed = True
                elif (token.startswith("D_DRAW:") and not revealed) or (
                    token.startswith("ACTION:") and revealed
                ):
                    stage = "INVALID"
                elif token.startswith("OUTCOME:"):
                    stage = "ROUND_END"
                elif not token.startswith(("ACTION:", "P:", "D_DRAW:")):
                    stage = "INVALID"
            elif stage == "ROUND_END":
                if token == "ROUND_END":
                    active_round = False
                    active_round_id = None
                    stage = "BETWEEN_ROUNDS"
                else:
                    stage = "INVALID"
            if stage == "INVALID":
                errors.append(
                    f"shoe {shoe_id} has invalid grammar near token {index}: {token}"
                )
                stage = "PLAY" if active_round else "BETWEEN_ROUNDS"
        if stage != "DONE":
            errors.append(f"shoe {shoe_id} does not terminate with SHOE_END")
    return errors


def _shoe_metadata_errors(
    *,
    tokens: Sequence[int],
    rows_by_shoe: Mapping[str, Sequence[Mapping[str, Any]]],
    shoe_rows: Sequence[Mapping[str, Any]],
    generation_config: GenerationConfig,
    expected_start_shoe_index: int,
) -> list[str]:
    errors: list[str] = []
    expected_start = 0
    seen_ids: set[str] = set()
    seen_indexes: set[int] = set()
    seen_seeds: set[str] = set()
    seen_rank_hashes: set[str] = set()
    seen_observable_hashes: set[str] = set()
    expected_composition = Counter(
        {rank.value: (16 if rank is Rank.TEN else 4) for rank in Rank}
    )

    for shoe_ordinal, shoe_row in enumerate(shoe_rows):
        expected_index = expected_start_shoe_index + shoe_ordinal
        shoe_id = str(shoe_row["shoe_id"])
        shoe_index = int(shoe_row["shoe_index"])
        rows = list(rows_by_shoe.get(shoe_id, ()))
        start = int(shoe_row["token_start"])
        end = int(shoe_row["token_end_exclusive"])
        transcript = [str(row["event_token"]) for row in rows]
        if (
            shoe_id in seen_ids
            or shoe_index in seen_indexes
            or shoe_index != expected_index
            or start != expected_start
            or end - start != len(rows)
            or int(shoe_row["token_count"]) != len(rows)
            or [int(row["event_index"]) for row in rows] != list(range(len(rows)))
            or [int(row["global_token_offset"]) for row in rows]
            != list(range(start, end))
            or list(tokens[start:end]) != [int(row["token_id"]) for row in rows]
        ):
            errors.append(f"shoe {shoe_id} has inconsistent offsets or identity")
        seen_ids.add(shoe_id)
        seen_indexes.add(shoe_index)
        expected_start = end

        seed = str(shoe_row["derived_seed_hex"])
        rank_hash = str(shoe_row["rank_order_sha256"])
        observable_hash = str(shoe_row["observable_token_sha256"])
        rank_order = [str(value) for value in shoe_row["deck_rank_order"]]
        instance_order = [str(value) for value in shoe_row["deck_instance_order"]]
        retry_index = int(shoe_row["retry_index"])
        expected_seed = derive_shoe_seed(
            generation_config.root_seed,
            generation_config.seed_domain_version,
            shoe_index,
            retry_index,
        )
        expected_deck = shuffled_deck(
            expected_seed.pcg_seed,
            stream=expected_seed.pcg_stream,
        )
        expected_rank_order = [card.rank.value for card in expected_deck]
        expected_instance_order = [card.instance_id for card in expected_deck]
        if (
            len(seed) != 64
            or seed in seen_seeds
            or rank_hash in seen_rank_hashes
            or observable_hash in seen_observable_hashes
            or Counter(rank_order) != expected_composition
            or len(instance_order) != 52
            or len(set(instance_order)) != 52
            or _sequence_sha256(rank_order) != rank_hash
            or _sequence_sha256(instance_order)
            != str(shoe_row["physical_order_sha256"])
            or _sequence_sha256(transcript) != observable_hash
            or retry_index != 0
            or seed != expected_seed.digest_hex
            or int(shoe_row["pcg_seed"]) != expected_seed.pcg_seed
            or int(shoe_row["pcg_stream"]) != expected_seed.pcg_stream
            or rank_order != expected_rank_order
            or instance_order != expected_instance_order
        ):
            errors.append(f"shoe {shoe_id} has invalid seed, deck, or content hash")
        seen_seeds.add(seed)
        seen_rank_hashes.add(rank_hash)
        seen_observable_hashes.add(observable_hash)

        outcome_rows = [
            row for row in rows if str(row["event_token"]).startswith("OUTCOME:")
        ]
        bet_rows = [row for row in rows if str(row["event_token"]).startswith("BET_")]
        action_rows = [
            row for row in rows if str(row["event_token"]).startswith("ACTION:")
        ]
        outcome_counts = Counter(
            str(row["event_token"]).removeprefix("OUTCOME:") for row in outcome_rows
        )
        expected_summary = {
            "round_count": len(outcome_rows),
            "initial_units_wagered": sum(
                int(row["decision_bet_units"] or 0) for row in bet_rows
            ),
            "total_units_wagered": sum(
                int(row["after_wager_units"] or 0) for row in outcome_rows
            ),
            "profit_half_units": sum(
                int(row["profit_half_units"] or 0) for row in outcome_rows
            ),
            "action_count": len(action_rows),
            "deviation_match_count": sum(
                row.get("matched_deviation_id") is not None for row in action_rows
            ),
            "deviation_activation_count": sum(
                bool(row.get("deviation_activated")) for row in action_rows
            ),
            "cards_dealt": int(rows[-1]["after_cards_dealt"]) if rows else -1,
            "cards_remaining": (int(rows[-1]["after_cards_remaining"]) if rows else -1),
        }
        if any(int(shoe_row[key]) != value for key, value in expected_summary.items()):
            errors.append(f"shoe {shoe_id} has inconsistent aggregate summary")

        cards_dealt = int(shoe_row["cards_dealt"])
        cards_remaining = int(shoe_row["cards_remaining"])
        expected_end_reason: str | None = None
        if cards_dealt >= generation_config.rules.cut_card_index:
            expected_end_reason = "CUT_CARD"
        elif cards_remaining < generation_config.rules.minimum_cards_to_start_round:
            expected_end_reason = "RESERVE_GUARD"
        if (
            shoe_row.get("schema_version") != _DATASET_SCHEMA_VERSION
            or shoe_row.get("dataset_version") != generation_config.dataset_version
            or shoe_row.get("split") != generation_config.split
            or int(shoe_row["root_seed"]) != generation_config.root_seed
            or shoe_row.get("seed_derivation_version") != SEED_DERIVATION_VERSION
            or shoe_row.get("seed_domain_version")
            != generation_config.seed_domain_version
            or not math.isclose(
                float(shoe_row["penetration"]),
                cards_dealt / 52,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or shoe_row.get("end_reason") != expected_end_reason
        ):
            errors.append(f"shoe {shoe_id} has inconsistent generation metadata")

        event_identity = {
            "schema_version": _SIDECAR_SCHEMA_VERSION,
            "split": generation_config.split,
            "shoe_id": shoe_id,
            "shoe_index": shoe_index,
            "derived_seed_hex": expected_seed.digest_hex,
            "pcg_seed": expected_seed.pcg_seed,
            "pcg_stream": expected_seed.pcg_stream,
        }
        if any(
            any(row.get(key) != value for key, value in event_identity.items())
            for row in rows
        ):
            errors.append(f"shoe {shoe_id} has inconsistent event provenance")
        try:
            stored_outcomes = json.loads(str(shoe_row["outcome_counts_json"]))
        except json.JSONDecodeError:
            stored_outcomes = None
        if stored_outcomes != dict(sorted(outcome_counts.items())):
            errors.append(f"shoe {shoe_id} has inconsistent outcome summary")

    if expected_start != len(tokens):
        errors.append("shoe token ranges do not cover the full token array")
    if len(rows_by_shoe) != len(shoe_rows):
        errors.append("event and shoe tables contain different shoe ID sets")
    return errors


def _sequence_sha256(values: Sequence[str]) -> str:
    encoded = json.dumps(
        list(values),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

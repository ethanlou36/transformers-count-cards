from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from blackjackgpt import __version__
from blackjackgpt.blackjack.cards import Rank
from blackjackgpt.provenance import sha256_file

from .generator import (
    GENERATOR_VERSION,
    GeneratedDataset,
    expert_policy_metadata,
    rules_metadata,
)
from .seeds import SEED_DERIVATION_VERSION
from .serialization import SIDECAR_SCHEMA_VERSION
from .validation import validate_dataset

MANIFEST_VERSION = "generation-manifest-v1"


def write_dataset(
    dataset: GeneratedDataset,
    output_dir: str | Path,
    *,
    debug_sample_count: int = 3,
    overwrite: bool = False,
) -> Path:
    if isinstance(debug_sample_count, bool) or not isinstance(debug_sample_count, int):
        raise TypeError("debug_sample_count must be an integer")
    if debug_sample_count < 0:
        raise ValueError("debug_sample_count must be nonnegative")

    destination = Path(output_dir).expanduser().resolve()
    _check_destination(destination, overwrite=overwrite)
    validation = validate_dataset(
        tokens=dataset.token_ids,
        event_rows=dataset.event_rows,
        shoe_rows=dataset.shoe_rows,
        generation_config=dataset.config,
        vocabulary=dataset.vocabulary,
        policy_config=dataset.policy_config,
    )
    if not validation["passed"]:
        raise ValueError(
            "refusing to write an invalid dataset: "
            + "; ".join(str(item) for item in validation["errors"])
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        _write_staged_artifacts(
            dataset,
            staging,
            validation=validation,
            debug_sample_count=debug_sample_count,
        )
        _publish_staging(staging, destination, overwrite=overwrite)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


def _write_staged_artifacts(
    dataset: GeneratedDataset,
    staging: Path,
    *,
    validation: dict[str, Any],
    debug_sample_count: int,
) -> None:
    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "artifact writing requires the declared numpy and pyarrow dependencies"
        ) from error

    if len(dataset.vocabulary) > 2**16 or any(
        token_id < 0 or token_id >= 2**16 for token_id in dataset.token_ids
    ):
        raise ValueError("version-1 token IDs must fit unsigned 16-bit storage")
    tokens = np.asarray(dataset.token_ids, dtype=np.uint16)
    offsets = [int(row["token_start"]) for row in dataset.shoe_rows]
    offsets.append(int(dataset.shoe_rows[-1]["token_end_exclusive"]))
    np.save(staging / "tokens.npy", tokens, allow_pickle=False)
    np.save(
        staging / "shoe_offsets.npy",
        np.asarray(offsets, dtype=np.int64),
        allow_pickle=False,
    )

    event_schema = _event_schema(pa)
    _assert_rows_match_schema(
        dataset.event_rows,
        event_schema.names,
        required=(
            "schema_version",
            "shoe_id",
            "event_index",
            "global_token_offset",
            "event_type",
            "event_token",
            "token_id",
        ),
    )
    event_table = pa.Table.from_pylist(list(dataset.event_rows), schema=event_schema)
    event_table = event_table.replace_schema_metadata(
        {
            b"schema_version": SIDECAR_SCHEMA_VERSION.encode(),
            b"sensitivity": b"privileged-audit-and-probe-labels",
            b"training_input": b"false",
        }
    )
    pq.write_table(
        event_table,
        staging / "events.parquet",
        compression="zstd",
        use_dictionary=True,
    )

    shoe_schema = _shoe_schema(pa)
    _assert_rows_match_schema(
        dataset.shoe_rows,
        shoe_schema.names,
        required=(
            "schema_version",
            "shoe_id",
            "shoe_index",
            "derived_seed_hex",
            "token_start",
            "token_end_exclusive",
            "round_count",
        ),
    )
    shoe_table = pa.Table.from_pylist(list(dataset.shoe_rows), schema=shoe_schema)
    shoe_table = shoe_table.replace_schema_metadata(
        {
            b"schema_version": str(dataset.config.dataset_version).encode(),
            b"sensitivity": b"privileged-reproduction-metadata",
            b"training_input": b"false",
        }
    )
    pq.write_table(
        shoe_table,
        staging / "shoes.parquet",
        compression="zstd",
        use_dictionary=True,
    )

    rules = rules_metadata(dataset.config.rules)
    policy = expert_policy_metadata(dataset.policy_config)
    _write_json(staging / "vocab.json", dataset.vocabulary.to_metadata())
    _write_json(staging / "rules.json", rules)
    _write_json(staging / "expert_policy.json", policy)
    _write_json(staging / "validation_report.json", validation)
    _write_debug_samples(
        staging / "debug_samples.jsonl",
        dataset,
        sample_count=debug_sample_count,
    )

    artifact_names = (
        "tokens.npy",
        "shoe_offsets.npy",
        "events.parquet",
        "shoes.parquet",
        "vocab.json",
        "rules.json",
        "expert_policy.json",
        "validation_report.json",
        "debug_samples.jsonl",
    )
    sensitivity = {
        "tokens.npy": "public-model-input",
        "shoe_offsets.npy": "public-sequence-boundaries",
        "events.parquet": "privileged-audit-and-probe-labels",
        "shoes.parquet": "privileged-reproduction-metadata",
        "vocab.json": "public-metadata",
        "rules.json": "public-metadata",
        "expert_policy.json": "public-metadata",
        "validation_report.json": "public-aggregate-report",
        "debug_samples.jsonl": "public-redacted-debug-data",
    }
    artifacts = {
        name: {
            "sha256": sha256_file(staging / name),
            "bytes": (staging / name).stat().st_size,
            "sensitivity": sensitivity[name],
        }
        for name in artifact_names
    }
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "generator_version": GENERATOR_VERSION,
        "blackjackgpt_version": __version__,
        "event_sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
        "vocabulary_version": dataset.vocabulary.version,
        "vocabulary_sha256": dataset.vocabulary.sha256,
        "seed_derivation_version": SEED_DERIVATION_VERSION,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
        },
        "writer": {
            "tokens_dtype": "uint16",
            "offsets_dtype": "int64",
            "parquet_compression": "zstd",
            "parquet_dictionary_encoding": True,
            "debug_sample_count": debug_sample_count,
            "atomic_staging": True,
        },
        "schema_fingerprints": {
            "events_arrow_schema_sha256": hashlib.sha256(
                event_schema.serialize().to_pybytes()
            ).hexdigest(),
            "shoes_arrow_schema_sha256": hashlib.sha256(
                shoe_schema.serialize().to_pybytes()
            ).hexdigest(),
        },
        "generation_config": dataset.config.to_metadata(),
        "source_control": _git_metadata(),
        "vocabulary": dataset.vocabulary.to_metadata(),
        "resolved_rules": rules,
        "resolved_expert_policy": policy,
        "rules_sha256": _canonical_sha256(rules),
        "expert_policy_sha256": _canonical_sha256(policy),
        "counts": {
            "shoes": validation["number_of_shoes"],
            "rounds": validation["number_of_rounds"],
            "tokens": validation["number_of_tokens"],
            "event_rows": len(dataset.event_rows),
            "shoe_rows": len(dataset.shoe_rows),
        },
        "rank_order_hashes": [
            str(row["rank_order_sha256"]) for row in dataset.shoe_rows
        ],
        "observable_token_hashes": [
            str(row["observable_token_sha256"]) for row in dataset.shoe_rows
        ],
        "validation_passed": validation["passed"],
        "artifacts": artifacts,
    }
    _write_json(staging / "generation_manifest.json", manifest)


def _event_schema(pa: Any) -> Any:
    string_fields = {
        "schema_version",
        "split",
        "shoe_id",
        "derived_seed_hex",
        "event_type",
        "event_token",
        "token_category",
        "before_dealer_upcard",
        "latent_before_dealer_hole_rank",
        "before_outcome",
        "after_dealer_upcard",
        "latent_after_dealer_hole_rank",
        "after_outcome",
        "dealer_upcard",
        "latent_dealer_hole_rank",
        "basic_strategy_recommendation",
        "basic_strategy_action",
        "matched_deviation_id",
        "applied_deviation_id",
        "deviation_preferred_action",
        "final_expert_action",
        "action_source",
        "outcome",
    }
    list_string_fields = {
        "before_exposed_cards",
        "before_player_hand",
        "before_dealer_visible_hand",
        "after_exposed_cards",
        "after_player_hand",
        "after_dealer_visible_hand",
        "exposed_cards_seen",
        "player_hand",
        "legal_actions",
    }
    bool_fields = {
        "before_dealer_hole_observable",
        "before_doubled",
        "before_peek_completed",
        "after_dealer_hole_observable",
        "after_doubled",
        "after_peek_completed",
        "dealer_hole_observable",
        "deviation_threshold_met",
        "deviation_fallback_used",
        "deviation_activated",
    }
    float_fields = {
        "before_raw_true_count",
        "after_raw_true_count",
        "deviation_index_threshold",
        "bet_ramp_minimum_raw_true_count",
        "decision_raw_true_count",
    }
    uint64_fields = {"pcg_seed", "pcg_stream"}
    integer_fields = {
        "shoe_index",
        "round_id",
        "event_index",
        "global_token_offset",
        "token_id",
        "before_cards_dealt",
        "before_cards_remaining",
        "before_running_count",
        "before_true_count_numerator",
        "before_true_count_denominator",
        "before_true_count_floor",
        "before_true_count_ceiling",
        "before_true_count_truncation",
        "before_initial_bet_units",
        "before_wager_units",
        "before_round_profit_half_units",
        "before_cumulative_profit_half_units",
        "after_cards_dealt",
        "after_cards_remaining",
        "after_running_count",
        "after_true_count_numerator",
        "after_true_count_denominator",
        "after_true_count_floor",
        "after_true_count_ceiling",
        "after_true_count_truncation",
        "after_initial_bet_units",
        "after_wager_units",
        "after_round_profit_half_units",
        "after_cumulative_profit_half_units",
        "physical_cards_remaining",
        "bet_ramp_entry_index",
        "decision_bet_units",
        "decision_running_count",
        "decision_cards_remaining",
        "decision_true_count_numerator",
        "decision_true_count_denominator",
        "decision_true_count_floor",
        "decision_true_count_ceiling",
        "decision_true_count_truncation",
        "bet_units",
        "wager_units",
        "profit_half_units",
        "cumulative_shoe_profit_half_units",
    }
    for prefix in ("before", "after"):
        integer_fields.update(
            f"latent_{prefix}_remaining_{rank.value}" for rank in Rank
        )

    ordered_names = [
        "schema_version",
        "split",
        "shoe_id",
        "shoe_index",
        "round_id",
        "event_index",
        "global_token_offset",
        "derived_seed_hex",
        "pcg_seed",
        "pcg_stream",
        "event_type",
        "event_token",
        "token_id",
        "token_category",
    ]
    for prefix in ("before", "after"):
        ordered_names.extend(
            [
                f"{prefix}_cards_dealt",
                f"{prefix}_cards_remaining",
                f"{prefix}_running_count",
                f"{prefix}_true_count_numerator",
                f"{prefix}_true_count_denominator",
                f"{prefix}_raw_true_count",
                f"{prefix}_true_count_floor",
                f"{prefix}_true_count_ceiling",
                f"{prefix}_true_count_truncation",
                f"{prefix}_exposed_cards",
                f"{prefix}_player_hand",
                f"{prefix}_dealer_visible_hand",
                f"{prefix}_dealer_upcard",
                f"latent_{prefix}_dealer_hole_rank",
                f"{prefix}_dealer_hole_observable",
                f"{prefix}_initial_bet_units",
                f"{prefix}_wager_units",
                f"{prefix}_doubled",
                f"{prefix}_peek_completed",
                f"{prefix}_outcome",
                f"{prefix}_round_profit_half_units",
                f"{prefix}_cumulative_profit_half_units",
            ]
        )
        ordered_names.extend(f"latent_{prefix}_remaining_{rank.value}" for rank in Rank)
    ordered_names.extend(
        [
            "physical_cards_remaining",
            "exposed_cards_seen",
            "player_hand",
            "dealer_upcard",
            "latent_dealer_hole_rank",
            "dealer_hole_observable",
            "legal_actions",
            "basic_strategy_recommendation",
            "basic_strategy_action",
            "matched_deviation_id",
            "applied_deviation_id",
            "deviation_index_threshold",
            "deviation_threshold_met",
            "deviation_preferred_action",
            "deviation_fallback_used",
            "deviation_activated",
            "final_expert_action",
            "action_source",
            "bet_ramp_entry_index",
            "bet_ramp_minimum_raw_true_count",
            "decision_bet_units",
            "decision_running_count",
            "decision_cards_remaining",
            "decision_true_count_numerator",
            "decision_true_count_denominator",
            "decision_raw_true_count",
            "decision_true_count_floor",
            "decision_true_count_ceiling",
            "decision_true_count_truncation",
            "bet_units",
            "wager_units",
            "outcome",
            "profit_half_units",
            "cumulative_shoe_profit_half_units",
        ]
    )

    fields = []
    for name in ordered_names:
        if name in list_string_fields:
            data_type = pa.list_(pa.string())
        elif name in bool_fields:
            data_type = pa.bool_()
        elif name in float_fields:
            data_type = pa.float64()
        elif name in uint64_fields:
            data_type = pa.uint64()
        elif name in integer_fields:
            data_type = pa.int64()
        elif name in string_fields:
            data_type = pa.string()
        else:
            raise RuntimeError(f"untyped event schema field: {name}")
        fields.append(pa.field(name, data_type, nullable=True))
    return pa.schema(fields)


def _shoe_schema(pa: Any) -> Any:
    return pa.schema(
        [
            pa.field("schema_version", pa.string()),
            pa.field("generator_version", pa.string()),
            pa.field("blackjackgpt_version", pa.string()),
            pa.field("dataset_version", pa.string()),
            pa.field("split", pa.string()),
            pa.field("shoe_id", pa.string()),
            pa.field("shoe_index", pa.int64()),
            pa.field("retry_index", pa.int64()),
            pa.field("root_seed", pa.uint64()),
            pa.field("seed_derivation_version", pa.string()),
            pa.field("seed_domain_version", pa.string()),
            pa.field("derived_seed_hex", pa.string()),
            pa.field("pcg_seed", pa.uint64()),
            pa.field("pcg_stream", pa.uint64()),
            pa.field("prng", pa.string()),
            pa.field("token_start", pa.int64()),
            pa.field("token_end_exclusive", pa.int64()),
            pa.field("token_count", pa.int64()),
            pa.field("round_count", pa.int64()),
            pa.field("cards_dealt", pa.int64()),
            pa.field("cards_remaining", pa.int64()),
            pa.field("penetration", pa.float64()),
            pa.field("end_reason", pa.string()),
            pa.field("initial_units_wagered", pa.int64()),
            pa.field("total_units_wagered", pa.int64()),
            pa.field("profit_half_units", pa.int64()),
            pa.field("action_count", pa.int64()),
            pa.field("deviation_match_count", pa.int64()),
            pa.field("deviation_activation_count", pa.int64()),
            pa.field("outcome_counts_json", pa.string()),
            pa.field("deck_rank_order", pa.list_(pa.string())),
            pa.field("deck_instance_order", pa.list_(pa.string())),
            pa.field("rank_order_sha256", pa.string()),
            pa.field("physical_order_sha256", pa.string()),
            pa.field("observable_token_sha256", pa.string()),
        ]
    )


def _write_debug_samples(
    path: Path,
    dataset: GeneratedDataset,
    *,
    sample_count: int,
) -> None:
    rows_by_shoe: dict[str, list[dict[str, Any]]] = {}
    public_keys = (
        "round_id",
        "event_index",
        "event_type",
        "event_token",
        "token_id",
        "before_cards_remaining",
        "after_cards_remaining",
        "before_running_count",
        "after_running_count",
        "before_raw_true_count",
        "after_raw_true_count",
        "player_hand",
        "dealer_upcard",
        "dealer_hole_observable",
        "legal_actions",
        "basic_strategy_action",
        "matched_deviation_id",
        "final_expert_action",
        "action_source",
        "bet_units",
        "wager_units",
        "outcome",
        "cumulative_shoe_profit_half_units",
    )
    sample_ids = {str(row["shoe_id"]) for row in dataset.shoe_rows[:sample_count]}
    for row in dataset.event_rows:
        shoe_id = str(row["shoe_id"])
        if shoe_id not in sample_ids:
            continue
        public = {key: row[key] for key in public_keys}
        if any(key.startswith("latent_") for key in public):
            raise AssertionError("debug row unexpectedly contains privileged fields")
        rows_by_shoe.setdefault(shoe_id, []).append(public)

    with path.open("w", encoding="utf-8") as handle:
        for shoe_row, tokens in zip(
            dataset.shoe_rows[:sample_count],
            dataset.transcript_tokens_by_shoe[:sample_count],
            strict=True,
        ):
            shoe_id = str(shoe_row["shoe_id"])
            payload = {
                "shoe_id": shoe_id,
                "tokens": list(tokens),
                "public_sidecar_rows": rows_by_shoe.get(shoe_id, []),
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _assert_rows_match_schema(
    rows: Any,
    schema_names: list[str],
    *,
    required: tuple[str, ...],
) -> None:
    expected = set(schema_names)
    for row_index, row in enumerate(rows):
        actual = set(row)
        if actual != expected:
            extra = sorted(actual - expected)
            missing = sorted(expected - actual)
            raise ValueError(
                f"row {row_index} does not match frozen schema; "
                f"extra={extra}, missing={missing}"
            )
        missing_required = [name for name in required if row[name] is None]
        if missing_required:
            raise ValueError(
                f"row {row_index} has null required fields: {missing_required}"
            )


def _check_destination(destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        return
    if destination.is_dir() and not any(destination.iterdir()):
        return
    if not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {destination}; pass --overwrite"
        )


def _publish_staging(staging: Path, destination: Path, *, overwrite: bool) -> None:
    backup: Path | None = None
    if destination.exists():
        if destination.is_dir() and not any(destination.iterdir()):
            destination.rmdir()
        elif overwrite:
            backup = destination.parent / (
                f".{destination.name}.backup-{uuid.uuid4().hex}"
            )
            destination.rename(backup)
        else:
            raise FileExistsError(destination)
    try:
        os.replace(staging, destination)
    except BaseException:
        if backup is not None and not destination.exists():
            backup.rename(destination)
        raise
    if backup is not None:
        if backup.is_dir():
            shutil.rmtree(backup)
        else:
            backup.unlink()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_metadata(start_path: str | Path | None = None) -> dict[str, Any]:
    command = ["git"]
    if start_path is not None:
        command.extend(("-C", str(start_path)))
    command.extend(("rev-parse", "--show-toplevel"))
    try:
        root = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"available": False, "commit": None, "dirty": None}

    try:
        commit_process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        commit = (
            commit_process.stdout.strip() if commit_process.returncode == 0 else None
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"available": True, "commit": None, "dirty": None}
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(status.strip()),
    }

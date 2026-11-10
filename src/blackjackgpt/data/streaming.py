from __future__ import annotations

import gc
import itertools
import json
import math
import platform
import resource
import shutil
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from blackjackgpt import __version__
from blackjackgpt.expert.policy import ExpertPolicy

from .generator import (
    GENERATOR_VERSION,
    GeneratedShoe,
    GenerationConfig,
    expert_policy_metadata,
    generate_one_shoe,
    iter_generated_shoes,
    rules_metadata,
)
from .seeds import SEED_DERIVATION_VERSION
from .serialization import SIDECAR_SCHEMA_VERSION
from .validation import validate_dataset
from .vocabulary import VERSION1_VOCABULARY, Vocabulary
from .writers import (
    _assert_rows_match_schema,
    _canonical_sha256,
    _check_destination,
    _event_schema,
    _file_sha256,
    _git_metadata,
    _publish_staging,
    _shoe_schema,
    _write_json,
)

STREAMING_MANIFEST_VERSION = "streaming-shard-manifest-v1"
STREAMING_VALIDATION_VERSION = "streaming-validation-v1"
MAX_SHOES_PER_SHARD = 5_000
MAX_TOKENS_PER_SHOE = 4096
MAX_EVENTS_PER_SHOE = 4096
MAX_ARROW_BYTES_PER_SHOE = 64 * 1024 * 1024
EVENT_ROW_TARGET = 8192
EVENT_BYTE_TARGET = 32 * 1024 * 1024
SHOE_ROW_TARGET = 512
SHOE_BYTE_TARGET = 8 * 1024 * 1024

_PAYLOAD_ARTIFACTS = (
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
_COMPLETE_ARTIFACTS = frozenset((*_PAYLOAD_ARTIFACTS, "generation_manifest.json"))
_PUBLIC_DEBUG_KEYS = (
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


class _BufferedParquetWriter:
    def __init__(
        self,
        *,
        pa: Any,
        pq: Any,
        path: Path,
        schema: Any,
        row_target: int,
        byte_target: int,
    ) -> None:
        self._pa = pa
        self._writer = pq.ParquetWriter(
            path,
            schema,
            compression="zstd",
            use_dictionary=True,
        )
        self._row_target = row_target
        self._byte_target = byte_target
        self._tables: list[Any] = []
        self._rows = 0
        self._bytes = 0
        self.row_group_count = 0
        self.max_buffer_rows = 0
        self.max_buffer_bytes = 0
        self._closed = False

    def append_table(self, table: Any) -> None:
        rows = int(table.num_rows)
        arrow_bytes = int(table.nbytes)
        if self._tables and (
            self._rows + rows > self._row_target
            or self._bytes + arrow_bytes > self._byte_target
        ):
            self.flush()
        self._tables.append(table)
        self._rows += rows
        self._bytes += arrow_bytes
        self.max_buffer_rows = max(self.max_buffer_rows, self._rows)
        self.max_buffer_bytes = max(self.max_buffer_bytes, self._bytes)
        if self._rows >= self._row_target or self._bytes >= self._byte_target:
            self.flush()

    def flush(self) -> None:
        if not self._tables:
            return
        table = (
            self._tables[0]
            if len(self._tables) == 1
            else self._pa.concat_tables(self._tables)
        )
        self._writer.write_table(table, row_group_size=table.num_rows)
        self.row_group_count += 1
        self._tables.clear()
        self._rows = 0
        self._bytes = 0

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._writer.close()
        self._closed = True


class _StreamingStats:
    def __init__(self, *, start_shoe_index: int) -> None:
        self.start_shoe_index = start_shoe_index
        self.shoes = 0
        self.rounds = 0
        self.tokens = 0
        self.event_rows = 0
        self.max_tokens_per_shoe = 0
        self.action_frequencies: Counter[str] = Counter()
        self.action_source_frequencies: Counter[str] = Counter()
        self.bet_frequencies: Counter[str] = Counter()
        self.outcome_frequencies: Counter[str] = Counter()
        self.true_count_histogram: Counter[float] = Counter()
        self.deviation_matches = 0
        self.deviation_activations = 0
        self.total_profit_half_units = 0
        self.total_initial_units = 0
        self.total_wagered_units = 0
        self._sum_profit_units = 0.0
        self._sum_rounds = 0
        self._sum_profit_squared = 0.0
        self._sum_profit_rounds = 0.0
        self._sum_rounds_squared = 0
        self.error_counters: Counter[str] = Counter()

    def update(
        self,
        *,
        report: Mapping[str, Any],
        shoe_row: Mapping[str, Any],
        event_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        self.shoes += 1
        self.rounds += int(report["number_of_rounds"])
        self.tokens += int(report["number_of_tokens"])
        self.event_rows += len(event_rows)
        self.max_tokens_per_shoe = max(
            self.max_tokens_per_shoe,
            int(report["number_of_tokens"]),
        )
        self.action_frequencies.update(report["action_frequencies"])
        self.action_source_frequencies.update(report["action_source_frequencies"])
        self.bet_frequencies.update(report["bet_bucket_frequencies"])
        self.outcome_frequencies.update(report["outcome_frequencies"])
        self.deviation_matches += int(report["deviation_matches"])
        self.deviation_activations += int(report["deviation_activations"])
        self.total_profit_half_units += int(report["total_profit_half_units"])

        for row in event_rows:
            token = str(row["event_token"])
            decision_true_count = row.get("decision_raw_true_count")
            if decision_true_count is not None:
                self.true_count_histogram[float(decision_true_count)] += 1
            if token.startswith("BET_"):
                self.total_initial_units += int(row["decision_bet_units"] or 0)
            elif token.startswith("OUTCOME:"):
                self.total_wagered_units += int(row["after_wager_units"] or 0)

        profit_units = int(shoe_row["profit_half_units"]) / 2
        round_count = int(shoe_row["round_count"])
        self._sum_profit_units += profit_units
        self._sum_rounds += round_count
        self._sum_profit_squared += profit_units * profit_units
        self._sum_profit_rounds += profit_units * round_count
        self._sum_rounds_squared += round_count * round_count

        for key in (
            "illegal_action_count",
            "hidden_card_leak_count",
            "count_transition_error_count",
            "physical_transition_error_count",
            "true_count_error_count",
            "policy_error_count",
            "summary_error_count",
            "vocabulary_error_count",
            "event_identity_error_count",
        ):
            self.error_counters[key] += int(report[key])

    def report(self) -> dict[str, Any]:
        errors = [
            f"{name}={count}"
            for name, count in sorted(self.error_counters.items())
            if count
        ]
        wins = sum(
            self.outcome_frequencies[name]
            for name in ("PLAYER_BLACKJACK", "DEALER_BUST", "WIN")
        )
        losses = sum(
            self.outcome_frequencies[name]
            for name in ("DEALER_BLACKJACK", "PLAYER_BUST", "LOSS")
        )
        pushes = sum(
            self.outcome_frequencies[name] for name in ("BLACKJACK_PUSH", "PUSH")
        )
        warnings: list[str] = []
        if self.shoes < 30:
            warnings.append(
                "confidence interval is based on fewer than 30 independent shoes"
            )
        return {
            "validation_version": STREAMING_VALIDATION_VERSION,
            "passed": not errors,
            "errors": errors,
            "warnings": warnings,
            "streaming": True,
            "start_shoe_index": self.start_shoe_index,
            "end_shoe_index_exclusive": self.start_shoe_index + self.shoes,
            "number_of_shoes": self.shoes,
            "number_of_rounds": self.rounds,
            "number_of_tokens": self.tokens,
            "number_of_event_rows": self.event_rows,
            "max_tokens_per_shoe": self.max_tokens_per_shoe,
            "average_rounds_per_shoe": (
                self.rounds / self.shoes if self.shoes else None
            ),
            "average_tokens_per_shoe": (
                self.tokens / self.shoes if self.shoes else None
            ),
            "action_frequencies": dict(sorted(self.action_frequencies.items())),
            "action_source_frequencies": dict(
                sorted(self.action_source_frequencies.items())
            ),
            "bet_bucket_frequencies": dict(sorted(self.bet_frequencies.items())),
            "decision_true_count_distribution": _histogram_distribution(
                self.true_count_histogram
            ),
            "deviation_matches": self.deviation_matches,
            "deviation_activations": self.deviation_activations,
            "outcome_frequencies": dict(sorted(self.outcome_frequencies.items())),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "blackjack_outcomes": {
                name: self.outcome_frequencies[name]
                for name in (
                    "PLAYER_BLACKJACK",
                    "DEALER_BLACKJACK",
                    "BLACKJACK_PUSH",
                )
            },
            "total_profit_half_units": self.total_profit_half_units,
            "profit_per_round_units": (
                (self.total_profit_half_units / 2) / self.rounds
                if self.rounds
                else None
            ),
            "profit_per_initial_unit_wagered": (
                (self.total_profit_half_units / 2) / self.total_initial_units
                if self.total_initial_units
                else None
            ),
            "profit_per_total_unit_wagered": (
                (self.total_profit_half_units / 2) / self.total_wagered_units
                if self.total_wagered_units
                else None
            ),
            "profit_per_round_95_percent_confidence_interval": (
                self._confidence_interval()
            ),
            **{
                key: self.error_counters[key]
                for key in (
                    "illegal_action_count",
                    "hidden_card_leak_count",
                    "count_transition_error_count",
                    "physical_transition_error_count",
                    "true_count_error_count",
                    "policy_error_count",
                    "summary_error_count",
                    "vocabulary_error_count",
                    "event_identity_error_count",
                )
            },
        }

    def _confidence_interval(self) -> dict[str, Any]:
        if not self.shoes or not self._sum_rounds:
            return {
                "method": "shoe_clustered_ratio_normal_95",
                "sample_size": 0,
                "mean": None,
                "standard_error": None,
                "lower": None,
                "upper": None,
            }
        mean = self._sum_profit_units / self._sum_rounds
        if self.shoes < 2:
            return {
                "method": "shoe_clustered_ratio_normal_95",
                "sample_size": 1,
                "mean": mean,
                "standard_error": None,
                "lower": None,
                "upper": None,
            }
        residual_sum_squares = (
            self._sum_profit_squared
            - 2 * mean * self._sum_profit_rounds
            + mean * mean * self._sum_rounds_squared
        )
        standard_error = math.sqrt(
            (self.shoes / (self.shoes - 1))
            * max(0.0, residual_sum_squares)
            / self._sum_rounds**2
        )
        margin = 1.96 * standard_error
        return {
            "method": "shoe_clustered_ratio_normal_95",
            "sample_size": self.shoes,
            "mean": mean,
            "standard_error": standard_error,
            "lower": mean - margin,
            "upper": mean + margin,
        }


def write_streaming_shard(
    config: GenerationConfig,
    output_dir: str | Path,
    *,
    start_shoe_index: int = 0,
    shoe_count: int | None = None,
    vocabulary: Vocabulary = VERSION1_VOCABULARY,
    debug_sample_count: int = 3,
    overwrite: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    resolved_count = config.num_shoes if shoe_count is None else shoe_count
    if resolved_count != config.num_shoes:
        raise ValueError("shoe_count must match GenerationConfig.num_shoes")
    if resolved_count > MAX_SHOES_PER_SHARD:
        raise ValueError(
            f"shoe_count exceeds the fixed shard maximum of {MAX_SHOES_PER_SHARD}"
        )

    iterator = iter_generated_shoes(
        config,
        start_shoe_index=start_shoe_index,
        shoe_count=resolved_count,
        vocabulary=vocabulary,
    )
    if isinstance(debug_sample_count, bool) or not isinstance(debug_sample_count, int):
        raise TypeError("debug_sample_count must be an integer")
    if debug_sample_count < 0:
        raise ValueError("debug_sample_count must be nonnegative")

    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("streaming writing requires numpy and pyarrow") from error

    source_control = _git_metadata(Path(__file__).resolve().parent)
    destination = Path(output_dir).expanduser().resolve()
    _check_destination(destination, overwrite=overwrite)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()

    raw_handle: BinaryIO | None = None
    debug_handle: TextIO | None = None
    offsets: Any = None
    event_sink: _BufferedParquetWriter | None = None
    shoe_sink: _BufferedParquetWriter | None = None
    started = time.perf_counter()
    try:
        event_schema = _event_schema(pa).with_metadata(
            {
                b"schema_version": SIDECAR_SCHEMA_VERSION.encode(),
                b"sensitivity": b"privileged-audit-and-probe-labels",
                b"training_input": b"false",
                b"offset_scope": b"shard-local",
            }
        )
        shoe_schema = _shoe_schema(pa).with_metadata(
            {
                b"schema_version": config.dataset_version.encode(),
                b"sensitivity": b"privileged-reproduction-metadata",
                b"training_input": b"false",
                b"offset_scope": b"shard-local",
            }
        )
        event_sink = _BufferedParquetWriter(
            pa=pa,
            pq=pq,
            path=staging / "events.parquet",
            schema=event_schema,
            row_target=EVENT_ROW_TARGET,
            byte_target=EVENT_BYTE_TARGET,
        )
        shoe_sink = _BufferedParquetWriter(
            pa=pa,
            pq=pq,
            path=staging / "shoes.parquet",
            schema=shoe_schema,
            row_target=SHOE_ROW_TARGET,
            byte_target=SHOE_BYTE_TARGET,
        )
        raw_handle = (staging / "tokens.raw").open("wb")
        debug_handle = (staging / "debug_samples.jsonl").open("w", encoding="utf-8")
        offsets = np.lib.format.open_memmap(
            staging / "shoe_offsets.npy",
            mode="w+",
            dtype=np.dtype("<i8"),
            shape=(resolved_count + 1,),
        )
        offsets[0] = 0

        stats = _StreamingStats(start_shoe_index=start_shoe_index)
        policy = ExpertPolicy()
        rank_hashes: set[str] = set()
        observable_hashes: set[str] = set()
        token_offset = 0

        for ordinal, generated in enumerate(iterator):
            _validate_generated_shoe_limits(generated)
            local_report = validate_dataset(
                tokens=generated.recorded.token_ids,
                event_rows=generated.recorded.event_rows,
                shoe_rows=(generated.shoe_row,),
                generation_config=config,
                vocabulary=vocabulary,
                policy_config=policy.config,
                expected_start_shoe_index=int(generated.shoe_row["shoe_index"]),
            )
            if not local_report["passed"]:
                raise RuntimeError(
                    "generated shoe failed validation: "
                    + "; ".join(str(item) for item in local_report["errors"])
                )

            rank_hash = str(generated.shoe_row["rank_order_sha256"])
            observable_hash = str(generated.shoe_row["observable_token_sha256"])
            if rank_hash in rank_hashes:
                raise RuntimeError("duplicate shuffled rank order within shard")
            if observable_hash in observable_hashes:
                raise RuntimeError("duplicate observable transcript within shard")
            rank_hashes.add(rank_hash)
            observable_hashes.add(observable_hash)

            event_rows, shoe_row = _rebase_generated_shoe(generated, token_offset)
            _assert_rows_match_schema(
                event_rows,
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
            _assert_rows_match_schema(
                (shoe_row,),
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
            event_table = pa.Table.from_pylist(event_rows, schema=event_schema)
            shoe_table = pa.Table.from_pylist([shoe_row], schema=shoe_schema)
            if (
                int(event_table.nbytes) + int(shoe_table.nbytes)
                > MAX_ARROW_BYTES_PER_SHOE
            ):
                raise RuntimeError("shoe exceeds max Arrow bytes before append")

            token_bytes = np.asarray(
                generated.recorded.token_ids,
                dtype=np.dtype("<u2"),
            ).tobytes()
            raw_handle.write(token_bytes)
            event_sink.append_table(event_table)
            shoe_sink.append_table(shoe_table)
            token_offset += len(generated.recorded.token_ids)
            offsets[ordinal + 1] = token_offset
            stats.update(
                report=local_report,
                shoe_row=generated.shoe_row,
                event_rows=generated.recorded.event_rows,
            )
            if ordinal < debug_sample_count:
                _write_debug_sample(debug_handle, generated)
            if progress is not None:
                progress(ordinal + 1, resolved_count)

        event_sink.close()
        shoe_sink.close()
        event_buffer_metrics = {
            "row_groups": event_sink.row_group_count,
            "max_rows": event_sink.max_buffer_rows,
            "max_arrow_bytes": event_sink.max_buffer_bytes,
        }
        shoe_buffer_metrics = {
            "row_groups": shoe_sink.row_group_count,
            "max_rows": shoe_sink.max_buffer_rows,
            "max_arrow_bytes": shoe_sink.max_buffer_bytes,
        }
        event_sink = None
        shoe_sink = None
        raw_handle.flush()
        raw_handle.close()
        raw_handle = None
        debug_handle.flush()
        debug_handle.close()
        debug_handle = None
        offsets.flush()
        del offsets
        offsets = None

        _finalize_tokens_file(
            np=np,
            raw_path=staging / "tokens.raw",
            output_path=staging / "tokens.npy",
            token_count=token_offset,
        )
        (staging / "tokens.raw").unlink()

        rules = rules_metadata(config.rules)
        policy_metadata = expert_policy_metadata(policy.config)
        validation_report = stats.report()
        if not validation_report["passed"]:
            raise RuntimeError("streaming validation counters are nonzero")
        _write_json(staging / "vocab.json", vocabulary.to_metadata())
        _write_json(staging / "rules.json", rules)
        _write_json(staging / "expert_policy.json", policy_metadata)
        _write_json(staging / "validation_report.json", validation_report)

        artifacts = {
            name: {
                "sha256": _file_sha256(staging / name),
                "bytes": (staging / name).stat().st_size,
                "sensitivity": _artifact_sensitivity(name),
            }
            for name in _PAYLOAD_ARTIFACTS
        }
        generation_elapsed = time.perf_counter() - started
        generation_max_rss = _max_rss_bytes()
        metrics: dict[str, Any] = {
            "generation_and_write_seconds": generation_elapsed,
            "generation_shoes_per_second": (
                resolved_count / generation_elapsed if generation_elapsed else None
            ),
            "generation_tokens_per_second": (
                token_offset / generation_elapsed if generation_elapsed else None
            ),
            "shoes_per_second": (
                resolved_count / generation_elapsed if generation_elapsed else None
            ),
            "tokens_per_second": (
                token_offset / generation_elapsed if generation_elapsed else None
            ),
            "generation_process_max_rss_bytes": generation_max_rss,
            "generation_write_and_staged_validation_seconds": None,
            "end_to_end_shoes_per_second": None,
            "end_to_end_tokens_per_second": None,
            "process_max_rss_bytes": generation_max_rss,
            "payload_bytes": sum(
                (staging / name).stat().st_size for name in _PAYLOAD_ARTIFACTS
            ),
            "event_row_groups": event_buffer_metrics["row_groups"],
            "shoe_row_groups": shoe_buffer_metrics["row_groups"],
        }
        manifest = {
            "manifest_version": STREAMING_MANIFEST_VERSION,
            "generator_version": GENERATOR_VERSION,
            "blackjackgpt_version": __version__,
            "event_sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
            "seed_derivation_version": SEED_DERIVATION_VERSION,
            "streaming": True,
            "offset_scope": "shard-local",
            "range": {
                "start_shoe_index": start_shoe_index,
                "end_shoe_index_exclusive": start_shoe_index + resolved_count,
                "shoe_count": resolved_count,
            },
            "generation_config": config.to_metadata(),
            "source_control": source_control,
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pyarrow": pa.__version__,
            },
            "vocabulary": vocabulary.to_metadata(),
            "vocabulary_sha256": vocabulary.sha256,
            "resolved_rules": rules,
            "rules_sha256": _canonical_sha256(rules),
            "resolved_expert_policy": policy_metadata,
            "expert_policy_sha256": _canonical_sha256(policy_metadata),
            "writer": {
                "tokens_dtype": "<u2",
                "offsets_dtype": "<i8",
                "parquet_compression": "zstd",
                "event_row_target": EVENT_ROW_TARGET,
                "event_byte_target": EVENT_BYTE_TARGET,
                "shoe_row_target": SHOE_ROW_TARGET,
                "shoe_byte_target": SHOE_BYTE_TARGET,
                "max_tokens_per_shoe": MAX_TOKENS_PER_SHOE,
                "max_events_per_shoe": MAX_EVENTS_PER_SHOE,
                "max_arrow_bytes_per_shoe": MAX_ARROW_BYTES_PER_SHOE,
                "max_shoes_per_shard": MAX_SHOES_PER_SHARD,
                "debug_sample_count": debug_sample_count,
                "event_row_groups": event_buffer_metrics["row_groups"],
                "shoe_row_groups": shoe_buffer_metrics["row_groups"],
            },
            "buffer_high_water": {
                "event_rows": event_buffer_metrics["max_rows"],
                "event_arrow_bytes": event_buffer_metrics["max_arrow_bytes"],
                "shoe_rows": shoe_buffer_metrics["max_rows"],
                "shoe_arrow_bytes": shoe_buffer_metrics["max_arrow_bytes"],
            },
            "counts": {
                "shoes": stats.shoes,
                "rounds": stats.rounds,
                "tokens": stats.tokens,
                "event_rows": stats.event_rows,
                "shoe_rows": stats.shoes,
            },
            "uniqueness": {
                "rank_orders": len(rank_hashes) == resolved_count,
                "observable_transcripts": len(observable_hashes) == resolved_count,
            },
            "metrics": metrics,
            "validation_passed": True,
            "artifacts": artifacts,
        }
        _write_json(staging / "generation_manifest.json", manifest)

        verification = verify_streaming_shard(staging, replay=True)
        if not verification["passed"]:
            raise RuntimeError(
                "staged shard failed verification: "
                + "; ".join(str(item) for item in verification["errors"])
            )
        end_to_end_elapsed = time.perf_counter() - started
        metrics.update(
            {
                "generation_write_and_staged_validation_seconds": end_to_end_elapsed,
                "end_to_end_shoes_per_second": (
                    resolved_count / end_to_end_elapsed if end_to_end_elapsed else None
                ),
                "end_to_end_tokens_per_second": (
                    token_offset / end_to_end_elapsed if end_to_end_elapsed else None
                ),
                "process_max_rss_bytes": _max_rss_bytes(),
            }
        )
        _write_json(staging / "generation_manifest.json", manifest)
        _publish_staging(staging, destination, overwrite=overwrite)
    except BaseException:
        for resource_handle in (event_sink, shoe_sink):
            if resource_handle is not None:
                with suppress(Exception):
                    resource_handle.close()
        for file_handle in (raw_handle, debug_handle):
            if file_handle is not None:
                with suppress(Exception):
                    file_handle.close()
        if offsets is not None:
            with suppress(Exception):
                offsets.flush()
        if staging.exists():
            shutil.rmtree(staging)
        _release_unused_memory()
        raise
    _release_unused_memory()
    return destination


def verify_streaming_shard(
    shard_dir: str | Path,
    *,
    replay: bool = False,
) -> dict[str, Any]:
    root = Path(shard_dir).expanduser().resolve()
    try:
        return _verify_streaming_shard(root, replay=replay)
    except Exception as error:
        return {
            "passed": False,
            "errors": [f"{type(error).__name__}: {error}"],
            "shard": str(root),
            "replay": replay,
        }
    finally:
        _release_unused_memory()


def _verify_streaming_shard(root: Path, *, replay: bool) -> dict[str, Any]:
    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "streaming verification requires numpy and pyarrow"
        ) from error

    if not root.is_dir():
        raise FileNotFoundError(f"shard directory does not exist: {root}")
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != _COMPLETE_ARTIFACTS:
        raise ValueError(
            "shard file inventory mismatch; "
            f"extra={sorted(actual_names - _COMPLETE_ARTIFACTS)}, "
            f"missing={sorted(_COMPLETE_ARTIFACTS - actual_names)}"
        )
    manifest = json.loads(
        (root / "generation_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("manifest_version") != STREAMING_MANIFEST_VERSION:
        raise ValueError("not a supported streaming shard manifest")
    for name in _PAYLOAD_ARTIFACTS:
        metadata = manifest["artifacts"].get(name)
        if metadata is None:
            raise ValueError(f"manifest is missing artifact: {name}")
        path = root / name
        if path.stat().st_size != int(metadata["bytes"]):
            raise ValueError(f"artifact size mismatch: {name}")
        if _file_sha256(path) != metadata["sha256"]:
            raise ValueError(f"artifact checksum mismatch: {name}")

    range_metadata = manifest["range"]
    start = int(range_metadata["start_shoe_index"])
    shoe_count = int(range_metadata["shoe_count"])
    end = int(range_metadata["end_shoe_index_exclusive"])
    if end != start + shoe_count or shoe_count <= 0:
        raise ValueError("invalid shard range in manifest")
    if shoe_count > MAX_SHOES_PER_SHARD:
        raise ValueError("shard range exceeds the fixed maximum shoe count")
    config_metadata = manifest["generation_config"]
    config = GenerationConfig(
        num_shoes=shoe_count,
        root_seed=int(config_metadata["root_seed"]),
        split=str(config_metadata["split"]),
        dataset_version=str(config_metadata["dataset_version"]),
    )
    if config_metadata != _json_normalize(config.to_metadata()):
        raise ValueError("generation config metadata is inconsistent")
    if (
        manifest.get("generator_version") != GENERATOR_VERSION
        or manifest.get("event_sidecar_schema_version") != SIDECAR_SCHEMA_VERSION
        or manifest.get("seed_derivation_version") != SEED_DERIVATION_VERSION
        or manifest.get("streaming") is not True
        or manifest.get("offset_scope") != "shard-local"
        or manifest.get("validation_passed") is not True
    ):
        raise ValueError("manifest version or validation provenance is inconsistent")

    vocabulary_metadata = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
    vocabulary = Vocabulary(
        version=str(vocabulary_metadata["version"]),
        tokens=tuple(str(token) for token in vocabulary_metadata["tokens"]),
    )
    if (
        vocabulary_metadata != manifest["vocabulary"]
        or vocabulary.sha256 != manifest["vocabulary_sha256"]
    ):
        raise ValueError("vocabulary hash mismatch")
    policy = ExpertPolicy()
    stored_rules = json.loads((root / "rules.json").read_text(encoding="utf-8"))
    stored_policy = json.loads(
        (root / "expert_policy.json").read_text(encoding="utf-8")
    )
    expected_rules = _json_normalize(rules_metadata(config.rules))
    expected_policy = _json_normalize(expert_policy_metadata(policy.config))
    if (
        stored_rules != expected_rules
        or stored_rules != manifest["resolved_rules"]
        or _canonical_sha256(stored_rules) != manifest["rules_sha256"]
    ):
        raise ValueError("rules metadata or hash mismatch")
    if (
        stored_policy != expected_policy
        or stored_policy != manifest["resolved_expert_policy"]
        or _canonical_sha256(stored_policy) != manifest["expert_policy_sha256"]
    ):
        raise ValueError("expert policy metadata or hash mismatch")

    tokens = np.load(root / "tokens.npy", mmap_mode="r", allow_pickle=False)
    offsets = np.load(root / "shoe_offsets.npy", mmap_mode="r", allow_pickle=False)
    if tokens.dtype != np.dtype("uint16") or tokens.ndim != 1:
        raise ValueError("tokens.npy must be one-dimensional uint16")
    if offsets.dtype != np.dtype("int64") or offsets.shape != (shoe_count + 1,):
        raise ValueError("shoe_offsets.npy has the wrong dtype or shape")
    if int(offsets[0]) != 0 or int(offsets[-1]) != len(tokens):
        raise ValueError("shoe offsets do not cover the token array")
    if any(int(right) < int(left) for left, right in itertools.pairwise(offsets)):
        raise ValueError("shoe offsets are not monotonic")

    event_file = pq.ParquetFile(root / "events.parquet")
    shoe_file = pq.ParquetFile(root / "shoes.parquet")
    if event_file.metadata.num_rows != len(tokens):
        raise ValueError("event row count does not match token count")
    if shoe_file.metadata.num_rows != shoe_count:
        raise ValueError("shoe row count does not match range")
    if event_file.schema_arrow.metadata.get(b"offset_scope") != b"shard-local":
        raise ValueError("event table is missing shard-local offset metadata")
    if shoe_file.schema_arrow.metadata.get(b"offset_scope") != b"shard-local":
        raise ValueError("shoe table is missing shard-local offset metadata")
    expected_event_schema = _event_schema(pa).with_metadata(
        {
            b"schema_version": SIDECAR_SCHEMA_VERSION.encode(),
            b"sensitivity": b"privileged-audit-and-probe-labels",
            b"training_input": b"false",
            b"offset_scope": b"shard-local",
        }
    )
    expected_shoe_schema = _shoe_schema(pa).with_metadata(
        {
            b"schema_version": config.dataset_version.encode(),
            b"sensitivity": b"privileged-reproduction-metadata",
            b"training_input": b"false",
            b"offset_scope": b"shard-local",
        }
    )
    if not event_file.schema_arrow.equals(expected_event_schema) or (
        event_file.schema_arrow.metadata != expected_event_schema.metadata
    ):
        raise ValueError("event schema does not match the expected schema")
    if not shoe_file.schema_arrow.equals(expected_shoe_schema) or (
        shoe_file.schema_arrow.metadata != expected_shoe_schema.metadata
    ):
        raise ValueError("shoe schema does not match the expected schema")
    event_iterator = _iter_parquet_rows(event_file, batch_size=1024)
    shoe_iterator = _iter_parquet_rows(shoe_file, batch_size=128)
    stats = _StreamingStats(start_shoe_index=start)
    replay_indices = _replay_indices(start, shoe_count) if replay else set()
    replay_count = 0

    for ordinal in range(shoe_count):
        shoe_row = next(shoe_iterator)
        expected_index = start + ordinal
        token_start = int(offsets[ordinal])
        token_end = int(offsets[ordinal + 1])
        token_span = token_end - token_start
        if not 0 < token_span <= MAX_TOKENS_PER_SHOE:
            raise ValueError(f"invalid token span for shoe {expected_index}")
        event_rows = [next(event_iterator) for _ in range(token_span)]
        if int(shoe_row["shoe_index"]) != expected_index:
            raise ValueError(f"unexpected shoe index at ordinal {ordinal}")
        if (
            int(shoe_row["token_start"]) != token_start
            or int(shoe_row["token_end_exclusive"]) != token_end
        ):
            raise ValueError(f"shoe offset mismatch for index {expected_index}")
        if any(
            int(row["global_token_offset"]) != token_start + event_index
            or int(row["token_id"]) != int(tokens[token_start + event_index])
            for event_index, row in enumerate(event_rows)
        ):
            raise ValueError(
                f"event/token alignment failure for index {expected_index}"
            )

        local_event_rows = [
            {
                **row,
                "global_token_offset": int(row["global_token_offset"]) - token_start,
            }
            for row in event_rows
        ]
        local_shoe_row = {
            **shoe_row,
            "token_start": 0,
            "token_end_exclusive": token_end - token_start,
        }
        local_tokens = [int(value) for value in tokens[token_start:token_end]]
        report = validate_dataset(
            tokens=local_tokens,
            event_rows=local_event_rows,
            shoe_rows=(local_shoe_row,),
            generation_config=config,
            vocabulary=vocabulary,
            policy_config=policy.config,
            expected_start_shoe_index=expected_index,
        )
        if not report["passed"]:
            raise ValueError(
                f"serialized shoe {expected_index} failed validation: "
                + "; ".join(str(item) for item in report["errors"])
            )
        stats.update(
            report=report,
            shoe_row=local_shoe_row,
            event_rows=local_event_rows,
        )

        if expected_index in replay_indices:
            expected = generate_one_shoe(
                config,
                expected_index,
                policy=policy,
                vocabulary=vocabulary,
            )
            if (
                list(expected.recorded.token_ids) != local_tokens
                or list(expected.recorded.event_rows) != local_event_rows
                or expected.shoe_row != local_shoe_row
            ):
                raise ValueError(f"deterministic replay mismatch for {expected_index}")
            replay_count += 1

    _assert_iterator_exhausted(event_iterator, "events")
    _assert_iterator_exhausted(shoe_iterator, "shoes")
    stored_report = json.loads(
        (root / "validation_report.json").read_text(encoding="utf-8")
    )
    recomputed_report = stats.report()
    if stored_report != recomputed_report:
        raise ValueError("validation report does not match streamed artifacts")
    debug_sample_rows = _verify_debug_samples(root / "debug_samples.jsonl")
    configured_debug_samples = manifest.get("writer", {}).get("debug_sample_count")
    if configured_debug_samples is not None and debug_sample_rows != min(
        int(configured_debug_samples), shoe_count
    ):
        raise ValueError("debug sample count does not match shard manifest")

    counts = manifest["counts"]
    if (
        int(counts["shoes"]) != shoe_count
        or int(counts["tokens"]) != len(tokens)
        or int(counts["event_rows"]) != event_file.metadata.num_rows
        or int(counts["shoe_rows"]) != shoe_file.metadata.num_rows
        or int(counts["rounds"]) != recomputed_report["number_of_rounds"]
    ):
        raise ValueError("manifest counts do not match artifacts")
    return {
        "passed": True,
        "errors": [],
        "shard": str(root),
        "start_shoe_index": start,
        "end_shoe_index_exclusive": end,
        "shoe_count": shoe_count,
        "token_count": len(tokens),
        "round_count": recomputed_report["number_of_rounds"],
        "replay": replay,
        "replay_count": replay_count,
    }


def _rebase_generated_shoe(
    generated: GeneratedShoe,
    token_offset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    event_rows = [
        {
            **row,
            "global_token_offset": token_offset + int(row["event_index"]),
        }
        for row in generated.recorded.event_rows
    ]
    shoe_row = {
        **generated.shoe_row,
        "token_start": token_offset,
        "token_end_exclusive": token_offset + len(generated.recorded.token_ids),
    }
    return event_rows, shoe_row


def _validate_generated_shoe_limits(generated: GeneratedShoe) -> None:
    if len(generated.recorded.token_ids) > MAX_TOKENS_PER_SHOE:
        raise RuntimeError("shoe exceeds max token count")
    if len(generated.recorded.event_rows) > MAX_EVENTS_PER_SHOE:
        raise RuntimeError("shoe exceeds max event count")
    if len(generated.recorded.token_ids) != len(generated.recorded.event_rows):
        raise RuntimeError("shoe token/event counts differ")
    if any(
        token_id < 0 or token_id >= 2**16 for token_id in generated.recorded.token_ids
    ):
        raise RuntimeError("token ID does not fit uint16")


def _finalize_tokens_file(
    *,
    np: Any,
    raw_path: Path,
    output_path: Path,
    token_count: int,
) -> None:
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.dtype("<u2"),
        shape=(token_count,),
    )
    position = 0
    with raw_path.open("rb") as source:
        while block := source.read(1024 * 1024):
            if len(block) % 2:
                raise RuntimeError("token spool has an odd byte count")
            values = np.frombuffer(block, dtype=np.dtype("<u2"))
            output[position : position + len(values)] = values
            position += len(values)
    if position != token_count:
        raise RuntimeError("token spool length does not match generated count")
    output.flush()
    del output


def _write_debug_sample(handle: TextIO, generated: GeneratedShoe) -> None:
    public_rows = [
        {key: row[key] for key in _PUBLIC_DEBUG_KEYS}
        for row in generated.recorded.event_rows
    ]
    payload = {
        "shoe_id": generated.shoe_row["shoe_id"],
        "tokens": list(generated.recorded.token_texts),
        "public_sidecar_rows": public_rows,
    }
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _verify_debug_samples(path: Path) -> int:
    forbidden = {
        "root_seed",
        "derived_seed_hex",
        "pcg_seed",
        "pcg_stream",
        "deck_rank_order",
        "deck_instance_order",
        "rank_order_sha256",
        "physical_order_sha256",
    }
    row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row_count += 1
            payload = json.loads(line)
            keys = set(_all_mapping_keys(payload))
            if any(key.startswith("latent_") for key in keys) or keys & forbidden:
                raise ValueError(
                    f"debug sample {line_number} contains privileged fields"
                )
    return row_count


def _all_mapping_keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _all_mapping_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _all_mapping_keys(nested)


def _iter_parquet_rows(
    parquet_file: Any, *, batch_size: int
) -> Iterator[dict[str, Any]]:
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def _assert_iterator_exhausted(iterator: Iterator[Any], name: str) -> None:
    try:
        next(iterator)
    except StopIteration:
        return
    raise ValueError(f"{name} table contains unexpected extra rows")


def _replay_indices(start: int, count: int) -> set[int]:
    if count <= 3:
        return set(range(start, start + count))
    return {start, start + (count - 1) // 2, start + count - 1}


def _histogram_distribution(
    histogram: Mapping[float, int],
) -> dict[str, float | int | None]:
    ordered = sorted(histogram.items())
    count = sum(weight for _, weight in ordered)
    if not count:
        return {
            "count": 0,
            "minimum": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "maximum": None,
            "mean": None,
        }
    return {
        "count": count,
        "minimum": float(ordered[0][0]),
        "p05": _weighted_quantile(ordered, count, 0.05),
        "p25": _weighted_quantile(ordered, count, 0.25),
        "median": _weighted_quantile(ordered, count, 0.50),
        "p75": _weighted_quantile(ordered, count, 0.75),
        "p95": _weighted_quantile(ordered, count, 0.95),
        "maximum": float(ordered[-1][0]),
        "mean": sum(value * weight for value, weight in ordered) / count,
    }


def _weighted_quantile(
    ordered: Sequence[tuple[float, int]],
    count: int,
    probability: float,
) -> float:
    position = probability * (count - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    lower_value = _weighted_order_statistic(ordered, lower)
    upper_value = _weighted_order_statistic(ordered, upper)
    if lower == upper:
        return lower_value
    return lower_value + (position - lower) * (upper_value - lower_value)


def _weighted_order_statistic(
    ordered: Sequence[tuple[float, int]],
    index: int,
) -> float:
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if index < cumulative:
            return float(value)
    raise IndexError(index)


def _artifact_sensitivity(name: str) -> str:
    if name == "tokens.npy":
        return "public-model-input"
    if name == "shoe_offsets.npy":
        return "public-sequence-boundaries"
    if name in {"events.parquet", "shoes.parquet"}:
        return "privileged-audit"
    if name == "debug_samples.jsonl":
        return "public-redacted-debug-data"
    return "public-metadata"


def _json_normalize(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _max_rss_bytes() -> int:
    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maximum if platform.system() == "Darwin" else maximum * 1024


def _release_unused_memory() -> None:
    gc.collect()
    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except (AttributeError, ImportError):
        pass
    gc.collect()

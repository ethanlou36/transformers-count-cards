from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sqlite3
import subprocess
import sys
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .generator import DATASET_SCHEMA_VERSION, GenerationConfig
from .streaming import (
    _release_unused_memory,
    verify_streaming_shard,
    write_streaming_shard,
)
from .writers import _file_sha256, _git_metadata

RELEASE_SPEC_VERSION = "streaming-release-spec-v1"
SPLIT_MANIFEST_VERSION = "streaming-split-manifest-v1"
DATASET_MANIFEST_VERSION = "streaming-dataset-manifest-v1"
PRODUCTION_RELEASE_ID = "blackjack-v1-100k-sharded-v1"
DEFAULT_SHARD_SHOE_COUNT = 5_000
MAX_SHARD_SHOE_COUNT = 5_000
PRODUCTION_ROOT_SEEDS = {
    "train": 0xA8F45C8C078E1645,
    "validation": 0x171706C4D84E3F7D,
    "evaluation": 0xC2CC7A38682FDF30,
}
PRODUCTION_ROOT_SEED_DERIVATION = (
    "sha256-first-u64-be:blackjackgpt/blackjack-v1-100k-sharded-v1/<split>"
)
PRODUCTION_SOURCE_REPOSITORY_ENV = "BLACKJACKGPT_PRODUCTION_SOURCE_REPOSITORY"
PRODUCTION_RUNTIME = {
    "python": "3.13.2",
    "numpy": "2.4.6",
    "pyarrow": "19.0.1",
}

_COMPONENT_PATTERN = re.compile(r"[a-z][a-z0-9_-]*\Z")
_SHARD_PATTERN = re.compile(r"shard-[0-9]{8}-[0-9]{8}\Z")
_ADDITIVE_REPORT_FIELDS = (
    "number_of_shoes",
    "number_of_rounds",
    "number_of_tokens",
    "number_of_event_rows",
    "wins",
    "losses",
    "pushes",
    "deviation_matches",
    "deviation_activations",
    "total_profit_half_units",
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
_FREQUENCY_REPORT_FIELDS = (
    "action_frequencies",
    "action_source_frequencies",
    "bet_bucket_frequencies",
    "blackjack_outcomes",
    "outcome_frequencies",
)
_IDENTITY_COLUMNS = (
    "shoe_id",
    "derived_seed_hex",
    "rank_order_sha256",
    "physical_order_sha256",
    "observable_token_sha256",
)


@dataclass(frozen=True, slots=True)
class SplitSpec:
    name: str
    root_seed: int
    start_shoe_index: int
    shoe_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _COMPONENT_PATTERN.fullmatch(
            self.name
        ):
            raise ValueError(f"invalid split name: {self.name!r}")
        _require_integer("root_seed", self.root_seed)
        _require_integer("start_shoe_index", self.start_shoe_index)
        _require_integer("shoe_count", self.shoe_count)
        if not 0 <= self.root_seed < 2**64:
            raise ValueError("root_seed must be an unsigned 64-bit integer")
        if self.start_shoe_index < 0:
            raise ValueError("start_shoe_index must be nonnegative")
        if self.shoe_count <= 0:
            raise ValueError("shoe_count must be positive")
        if self.end_shoe_index_exclusive > 2**63:
            raise ValueError("split range exceeds signed 64-bit shoe indices")

    @property
    def end_shoe_index_exclusive(self) -> int:
        return self.start_shoe_index + self.shoe_count

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root_seed": self.root_seed,
            "start_shoe_index": self.start_shoe_index,
            "end_shoe_index_exclusive": self.end_shoe_index_exclusive,
            "shoe_count": self.shoe_count,
        }

    @classmethod
    def from_metadata(cls, value: Mapping[str, Any]) -> SplitSpec:
        return cls(
            name=str(value["name"]),
            root_seed=int(value["root_seed"]),
            start_shoe_index=int(value["start_shoe_index"]),
            shoe_count=int(value["shoe_count"]),
        )


@dataclass(frozen=True, slots=True)
class ShardSpec:
    split: str
    root_seed: int
    start_shoe_index: int
    shoe_count: int

    @property
    def end_shoe_index_exclusive(self) -> int:
        return self.start_shoe_index + self.shoe_count

    @property
    def name(self) -> str:
        return shard_name(self.start_shoe_index, self.shoe_count)

    @property
    def relative_path(self) -> str:
        return f"splits/{self.split}/shards/{self.name}"


@dataclass(frozen=True, slots=True)
class ReleaseSpec:
    release_id: str
    splits: tuple[SplitSpec, ...]
    shard_shoe_count: int = DEFAULT_SHARD_SHOE_COUNT
    dataset_version: str = DATASET_SCHEMA_VERSION
    debug_sample_count: int = 3
    pilot: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.release_id, str) or not _COMPONENT_PATTERN.fullmatch(
            self.release_id
        ):
            raise ValueError(f"invalid release ID: {self.release_id!r}")
        object.__setattr__(self, "splits", tuple(self.splits))
        _require_integer("shard_shoe_count", self.shard_shoe_count)
        if not 0 < self.shard_shoe_count <= MAX_SHARD_SHOE_COUNT:
            raise ValueError(f"shard_shoe_count must be in [1, {MAX_SHARD_SHOE_COUNT}]")
        if not self.splits:
            raise ValueError("a release must contain at least one split")
        if not isinstance(self.dataset_version, str) or not self.dataset_version:
            raise ValueError("dataset_version must be a non-empty string")
        _require_integer("debug_sample_count", self.debug_sample_count)
        if self.debug_sample_count < 0:
            raise ValueError("debug_sample_count must be nonnegative")
        if not isinstance(self.pilot, bool):
            raise TypeError("pilot must be a boolean")
        names = [split.name for split in self.splits]
        seeds = [split.root_seed for split in self.splits]
        if len(names) != len(set(names)):
            raise ValueError("split names must be unique")
        if len(seeds) != len(set(seeds)):
            raise ValueError("split root seeds must be distinct")
        if any(split.shoe_count % self.shard_shoe_count for split in self.splits):
            raise ValueError("every split shoe count must be divisible by shard size")
        if self.pilot:
            if self.release_id == PRODUCTION_RELEASE_ID:
                raise ValueError("pilot cannot use the canonical production release ID")
        else:
            expected = (
                ("train", 0, 90_000),
                ("validation", 0, 5_000),
                ("evaluation", 0, 5_000),
            )
            actual = tuple(
                (split.name, split.start_shoe_index, split.shoe_count)
                for split in self.splits
            )
            if self.release_id != PRODUCTION_RELEASE_ID:
                raise ValueError(
                    f"production release ID must be {PRODUCTION_RELEASE_ID!r}"
                )
            if self.shard_shoe_count != DEFAULT_SHARD_SHOE_COUNT:
                raise ValueError("production shard size must be exactly 5000")
            if self.dataset_version != DATASET_SCHEMA_VERSION:
                raise ValueError("production dataset version is frozen")
            if self.debug_sample_count != 3:
                raise ValueError("production debug sample count must be exactly 3")
            actual_seeds = {split.name: split.root_seed for split in self.splits}
            if actual_seeds != PRODUCTION_ROOT_SEEDS:
                raise ValueError("production root seeds do not match frozen values")
            if actual != expected:
                raise ValueError("production split layout must be exactly 90k/5k/5k")

    @property
    def shoe_count(self) -> int:
        return sum(split.shoe_count for split in self.splits)

    @property
    def shard_count(self) -> int:
        return sum(split.shoe_count // self.shard_shoe_count for split in self.splits)

    def iter_shards(self) -> Iterator[ShardSpec]:
        for split in self.splits:
            for start in range(
                split.start_shoe_index,
                split.end_shoe_index_exclusive,
                self.shard_shoe_count,
            ):
                yield ShardSpec(
                    split=split.name,
                    root_seed=split.root_seed,
                    start_shoe_index=start,
                    shoe_count=self.shard_shoe_count,
                )

    def to_metadata(self) -> dict[str, Any]:
        shards_by_split: dict[str, list[dict[str, Any]]] = {
            split.name: [] for split in self.splits
        }
        for shard in self.iter_shards():
            shards_by_split[shard.split].append(
                {
                    "name": shard.name,
                    "start_shoe_index": shard.start_shoe_index,
                    "end_shoe_index_exclusive": shard.end_shoe_index_exclusive,
                    "shoe_count": shard.shoe_count,
                }
            )
        return {
            "spec_version": RELEASE_SPEC_VERSION,
            "release_id": self.release_id,
            "pilot": self.pilot,
            "dataset_version": self.dataset_version,
            "debug_sample_count": self.debug_sample_count,
            "shard_shoe_count": self.shard_shoe_count,
            "shoe_count": self.shoe_count,
            "shard_count": self.shard_count,
            "root_seed_derivation": (
                "explicit-pilot-seed" if self.pilot else PRODUCTION_ROOT_SEED_DERIVATION
            ),
            "splits": [
                {**split.to_metadata(), "shards": shards_by_split[split.name]}
                for split in self.splits
            ],
        }

    @classmethod
    def from_metadata(cls, value: Mapping[str, Any]) -> ReleaseSpec:
        if value.get("spec_version") != RELEASE_SPEC_VERSION:
            raise ValueError("unsupported release specification version")
        raw_splits = value.get("splits")
        if not isinstance(raw_splits, list):
            raise ValueError("release specification has invalid splits")
        spec = cls(
            release_id=str(value["release_id"]),
            splits=tuple(
                SplitSpec.from_metadata(_require_mapping(item, "split"))
                for item in raw_splits
            ),
            shard_shoe_count=int(value["shard_shoe_count"]),
            dataset_version=str(value["dataset_version"]),
            debug_sample_count=int(value["debug_sample_count"]),
            pilot=bool(value["pilot"]),
        )
        expected = spec.to_metadata()
        if dict(value) != expected:
            raise ValueError("release specification metadata is inconsistent")
        return spec


def production_release_spec() -> ReleaseSpec:
    return ReleaseSpec(
        release_id=PRODUCTION_RELEASE_ID,
        splits=(
            SplitSpec("train", PRODUCTION_ROOT_SEEDS["train"], 0, 90_000),
            SplitSpec("validation", PRODUCTION_ROOT_SEEDS["validation"], 0, 5_000),
            SplitSpec("evaluation", PRODUCTION_ROOT_SEEDS["evaluation"], 0, 5_000),
        ),
    )


def pilot_release_spec(
    *,
    root_seed: int,
    release_id: str = "blackjack-v1-10k-pilot-v1",
    shoe_count: int = 10_000,
    shard_shoe_count: int = DEFAULT_SHARD_SHOE_COUNT,
    debug_sample_count: int = 3,
) -> ReleaseSpec:
    return ReleaseSpec(
        release_id=release_id,
        splits=(SplitSpec("train", root_seed, 0, shoe_count),),
        shard_shoe_count=shard_shoe_count,
        debug_sample_count=debug_sample_count,
        pilot=True,
    )


def shard_name(start_shoe_index: int, shoe_count: int) -> str:
    _require_integer("start_shoe_index", start_shoe_index)
    _require_integer("shoe_count", shoe_count)
    if start_shoe_index < 0 or shoe_count <= 0:
        raise ValueError("shard range must be nonnegative and nonempty")
    end_inclusive = start_shoe_index + shoe_count - 1
    if end_inclusive >= 10**8:
        raise ValueError("shard range does not fit the fixed eight-digit name")
    return f"shard-{start_shoe_index:08d}-{end_inclusive:08d}"


def generate_release(
    spec: ReleaseSpec,
    output_dir: str | Path,
    *,
    resume: bool = False,
    isolate_shards: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    if not isinstance(spec, ReleaseSpec):
        raise TypeError("spec must be a ReleaseSpec")
    if not isinstance(resume, bool):
        raise TypeError("resume must be a boolean")
    if not isinstance(isolate_shards, bool):
        raise TypeError("isolate_shards must be a boolean")
    root = Path(output_dir).expanduser().resolve()
    existed = root.exists()
    if existed and not root.is_dir():
        raise NotADirectoryError(f"release output is not a directory: {root}")
    if existed and any(root.iterdir()) and not resume:
        raise FileExistsError(f"release directory is not empty: {root}; pass --resume")
    root.mkdir(parents=True, exist_ok=True)

    spec_payload = _resolved_spec_payload(spec)
    spec_path = root / "dataset_spec.json"
    if spec_path.exists():
        if _read_json(spec_path) != spec_payload:
            raise ValueError("existing dataset_spec.json does not match this run")
    else:
        _publish_json(spec_path, spec_payload)

    _assert_current_environment(spec_payload)
    _validate_inventory(root, spec, complete=False)
    if (root / "dataset_manifest.json").exists():
        verification = verify_release(
            root,
            replay=False,
            isolate_shards=isolate_shards,
        )
        if not verification["passed"]:
            raise ValueError(
                "completed release failed verification: "
                + "; ".join(map(str, verification["errors"]))
            )
        return {
            **verification,
            "generated_shards": 0,
            "skipped_shards": spec.shard_count,
        }

    generated = 0
    skipped = 0
    total = spec.shard_count
    for ordinal, shard in enumerate(spec.iter_shards(), start=1):
        _assert_current_environment(spec_payload)
        destination = root / shard.relative_path
        if destination.exists():
            if not resume:
                raise FileExistsError(f"shard already exists: {destination}")
            _verify_expected_shard(
                destination,
                shard,
                spec_payload,
                replay=False,
                isolate=isolate_shards,
            )
            skipped += 1
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            config = GenerationConfig(
                num_shoes=shard.shoe_count,
                root_seed=shard.root_seed,
                split=shard.split,
                dataset_version=spec.dataset_version,
            )
            _generate_expected_shard(
                config=config,
                destination=destination,
                shard=shard,
                debug_sample_count=spec.debug_sample_count,
                isolate=isolate_shards,
            )
            _verify_expected_shard(
                destination,
                shard,
                spec_payload,
                replay=False,
                isolate=isolate_shards,
            )
            generated += 1
        if progress is not None:
            progress(shard.relative_path, ordinal, total)

    aggregate = _collect_release(
        root,
        spec,
        spec_payload,
        replay=True,
        isolate_shards=isolate_shards,
    )
    for split in spec.splits:
        path = root / "splits" / split.name / "split_manifest.json"
        _publish_json(path, aggregate["split_manifests"][split.name])
    dataset_manifest = _build_dataset_manifest(
        root,
        spec,
        spec_payload,
        aggregate,
    )
    _publish_json(root / "dataset_manifest.json", dataset_manifest)
    verification = verify_release(
        root,
        replay=False,
        isolate_shards=isolate_shards,
    )
    if not verification["passed"]:
        raise RuntimeError(
            "newly completed release failed verification: "
            + "; ".join(map(str, verification["errors"]))
        )
    return {
        **verification,
        "generated_shards": generated,
        "skipped_shards": skipped,
    }


def verify_release(
    release_root: str | Path,
    *,
    replay: bool = False,
    isolate_shards: bool = True,
) -> dict[str, Any]:
    root = Path(release_root).expanduser().resolve()
    try:
        if not isinstance(isolate_shards, bool):
            raise TypeError("isolate_shards must be a boolean")
        return _verify_release(
            root,
            replay=replay,
            isolate_shards=isolate_shards,
        )
    except Exception as error:
        return {
            "passed": False,
            "errors": [f"{type(error).__name__}: {error}"],
            "release_root": str(root),
            "replay": replay,
        }


def _verify_release(
    root: Path,
    *,
    replay: bool,
    isolate_shards: bool,
) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"release directory does not exist: {root}")
    spec_payload = _read_json(root / "dataset_spec.json")
    raw_spec = _require_mapping(spec_payload.get("release"), "release")
    spec = ReleaseSpec.from_metadata(raw_spec)
    _validate_resolved_spec_payload(spec_payload, spec)
    _validate_inventory(root, spec, complete=True)
    stored_manifest = _read_json(root / "dataset_manifest.json")
    aggregate = _collect_release(
        root,
        spec,
        spec_payload,
        replay=replay,
        isolate_shards=isolate_shards,
    )
    for split in spec.splits:
        stored_split = _read_json(root / "splits" / split.name / "split_manifest.json")
        if stored_split != aggregate["split_manifests"][split.name]:
            raise ValueError(f"split manifest mismatch: {split.name}")
    expected_manifest = _build_dataset_manifest(
        root,
        spec,
        spec_payload,
        aggregate,
    )
    if stored_manifest != expected_manifest:
        raise ValueError("dataset manifest does not match verified shard inventory")
    return {
        "passed": True,
        "errors": [],
        "release_root": str(root),
        "release_id": spec.release_id,
        "complete": True,
        "shoe_count": spec.shoe_count,
        "shard_count": spec.shard_count,
        "counts": aggregate["counts"],
        "payload_bytes": aggregate["payload_bytes"],
        "replay": replay,
        "replay_count": aggregate["replay_count"],
        "dataset_manifest_sha256": _file_sha256(root / "dataset_manifest.json"),
    }


def _collect_release(
    root: Path,
    spec: ReleaseSpec,
    spec_payload: Mapping[str, Any],
    *,
    replay: bool,
    isolate_shards: bool,
) -> dict[str, Any]:
    split_records: dict[str, list[dict[str, Any]]] = {
        split.name: [] for split in spec.splits
    }
    global_counts: Counter[str] = Counter()
    global_additive: Counter[str] = Counter()
    global_frequencies: dict[str, Counter[str]] = {
        field: Counter() for field in _FREQUENCY_REPORT_FIELDS
    }
    bytes_by_sensitivity: Counter[str] = Counter()
    payload_bytes = 0
    replay_count = 0
    shared_provenance: dict[str, Any] | None = None

    scratch = root.parent / f".{root.name}.identities-{uuid.uuid4().hex}.sqlite3"
    connection = sqlite3.connect(scratch)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE identities ("
            "kind TEXT NOT NULL, value TEXT NOT NULL, split TEXT NOT NULL, "
            "shoe_index INTEGER NOT NULL, PRIMARY KEY (kind, value)) WITHOUT ROWID"
        )
        for shard in spec.iter_shards():
            path = root / shard.relative_path
            verification = _verify_expected_shard(
                path,
                shard,
                spec_payload,
                replay=replay,
                isolate=isolate_shards,
            )
            replay_count += int(verification.get("replay_count", 0))
            manifest_path = path / "generation_manifest.json"
            manifest = _read_json(manifest_path)
            provenance = _shared_shard_provenance(manifest)
            if shared_provenance is None:
                shared_provenance = provenance
            elif provenance != shared_provenance:
                raise ValueError("shards do not share identical provenance/settings")
            counts = _integer_mapping(manifest["counts"], "shard counts")
            global_counts.update(counts)
            shard_payload = int(manifest["metrics"]["payload_bytes"])
            payload_bytes += shard_payload
            for metadata in _require_mapping(
                manifest["artifacts"], "shard artifacts"
            ).values():
                artifact = _require_mapping(metadata, "artifact")
                bytes_by_sensitivity[str(artifact["sensitivity"])] += int(
                    artifact["bytes"]
                )
            report = _read_json(path / "validation_report.json")
            if report.get("passed") is not True:
                raise ValueError(
                    f"shard validation report failed: {shard.relative_path}"
                )
            for field in _ADDITIVE_REPORT_FIELDS:
                global_additive[field] += int(report[field])
            for field in _FREQUENCY_REPORT_FIELDS:
                global_frequencies[field].update(_integer_mapping(report[field], field))
            _insert_shard_identities(connection, path, shard)
            _release_unused_memory()
            split_records[shard.split].append(
                {
                    "name": shard.name,
                    "path": shard.relative_path,
                    "start_shoe_index": shard.start_shoe_index,
                    "end_shoe_index_inclusive": shard.end_shoe_index_exclusive - 1,
                    "end_shoe_index_exclusive": shard.end_shoe_index_exclusive,
                    "shoe_count": shard.shoe_count,
                    "manifest_path": f"{shard.relative_path}/generation_manifest.json",
                    "manifest_sha256": _file_sha256(manifest_path),
                    "manifest_bytes": manifest_path.stat().st_size,
                    "counts": counts,
                    "payload_bytes": shard_payload,
                }
            )
        connection.commit()
        identity_counts = {
            str(kind): int(count)
            for kind, count in connection.execute(
                "SELECT kind, COUNT(*) FROM identities GROUP BY kind ORDER BY kind"
            )
        }
    finally:
        connection.close()
        scratch.unlink(missing_ok=True)

    expected_identity_counts = dict.fromkeys(_IDENTITY_COLUMNS, spec.shoe_count)
    if identity_counts != expected_identity_counts:
        raise ValueError("global identity counts do not cover every shoe")
    if global_counts["shoes"] != spec.shoe_count:
        raise ValueError("aggregate shoe count does not match release specification")
    if global_counts["tokens"] != global_counts["event_rows"]:
        raise ValueError("aggregate token and event counts differ")
    if global_additive["number_of_shoes"] != spec.shoe_count:
        raise ValueError("validation aggregate shoe count is inconsistent")

    validation = {
        "passed": all(
            global_additive[field] == 0
            for field in (
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
        ),
        "additive": dict(sorted(global_additive.items())),
        "frequencies": {
            field: dict(sorted(counter.items()))
            for field, counter in sorted(global_frequencies.items())
        },
        "non_additive_statistics": "retained in per-shard validation reports",
    }
    if not validation["passed"]:
        raise ValueError("aggregate validation counters are nonzero")

    split_manifests: dict[str, dict[str, Any]] = {}
    for split in spec.splits:
        records = split_records[split.name]
        split_counts: Counter[str] = Counter()
        for record in records:
            split_counts.update(record["counts"])
        split_manifests[split.name] = {
            "manifest_version": SPLIT_MANIFEST_VERSION,
            "release_id": spec.release_id,
            "split": split.name,
            "root_seed": split.root_seed,
            "range": {
                "start_shoe_index": split.start_shoe_index,
                "end_shoe_index_exclusive": split.end_shoe_index_exclusive,
                "shoe_count": split.shoe_count,
            },
            "shard_shoe_count": spec.shard_shoe_count,
            "shard_count": len(records),
            "shards": records,
            "counts": dict(sorted(split_counts.items())),
            "payload_bytes": sum(int(record["payload_bytes"]) for record in records),
            "validation_passed": True,
            "complete": True,
        }
    return {
        "split_manifests": split_manifests,
        "counts": dict(sorted(global_counts.items())),
        "payload_bytes": payload_bytes,
        "bytes_by_sensitivity": dict(sorted(bytes_by_sensitivity.items())),
        "shared_provenance": shared_provenance or {},
        "validation": validation,
        "identity_counts": identity_counts,
        "replay_count": replay_count,
    }


def _build_dataset_manifest(
    root: Path,
    spec: ReleaseSpec,
    spec_payload: Mapping[str, Any],
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}

    def add_artifact(relative_path: str) -> None:
        path = root / relative_path
        artifacts[relative_path] = {
            "sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
        }

    add_artifact("dataset_spec.json")
    split_entries: dict[str, dict[str, Any]] = {}
    split_manifests = _require_mapping(aggregate["split_manifests"], "split manifests")
    for split in spec.splits:
        split_path = f"splits/{split.name}/split_manifest.json"
        add_artifact(split_path)
        split_manifest = _require_mapping(split_manifests[split.name], "split manifest")
        shards = split_manifest["shards"]
        if not isinstance(shards, list):
            raise TypeError("split shard inventory must be a list")
        for record in shards:
            shard_record = _require_mapping(record, "shard record")
            add_artifact(str(shard_record["manifest_path"]))
        split_entries[split.name] = {
            "root_seed": split.root_seed,
            "start_shoe_index": split.start_shoe_index,
            "end_shoe_index_exclusive": split.end_shoe_index_exclusive,
            "shoe_count": split.shoe_count,
            "shard_count": len(shards),
            "manifest_path": split_path,
            "manifest_sha256": artifacts[split_path]["sha256"],
            "counts": split_manifest["counts"],
            "payload_bytes": split_manifest["payload_bytes"],
            "shards": shards,
        }
    return {
        "manifest_version": DATASET_MANIFEST_VERSION,
        "release_id": spec.release_id,
        "pilot": spec.pilot,
        "dataset_version": spec.dataset_version,
        "shard_shoe_count": spec.shard_shoe_count,
        "shoe_count": spec.shoe_count,
        "shard_count": spec.shard_count,
        "dataset_spec_sha256": artifacts["dataset_spec.json"]["sha256"],
        "source_control": spec_payload["source_control"],
        "remote_provenance": spec_payload["remote_provenance"],
        "source_fingerprint_sha256": spec_payload["source_fingerprint_sha256"],
        "runtime": spec_payload["runtime"],
        "shared_shard_provenance": aggregate["shared_provenance"],
        "splits": split_entries,
        "counts": aggregate["counts"],
        "payload_bytes": aggregate["payload_bytes"],
        "bytes_by_sensitivity": aggregate["bytes_by_sensitivity"],
        "validation": aggregate["validation"],
        "global_uniqueness": {
            "passed": True,
            "counts": aggregate["identity_counts"],
            "method": "bounded-sqlite-unique-index-v1",
        },
        "artifacts": dict(sorted(artifacts.items())),
        "validation_passed": True,
        "complete": True,
    }


def _verify_expected_shard(
    path: Path,
    shard: ShardSpec,
    spec_payload: Mapping[str, Any],
    *,
    replay: bool,
    isolate: bool,
) -> dict[str, Any]:
    result = (
        _run_shard_verifier(path, replay=replay)
        if isolate
        else (verify_streaming_shard(path, replay=replay))
    )
    if not result.get("passed", False):
        raise ValueError(
            f"invalid existing shard {shard.relative_path}: "
            + "; ".join(map(str, result.get("errors", [])))
        )
    manifest = _read_json(path / "generation_manifest.json")
    expected_range = {
        "start_shoe_index": shard.start_shoe_index,
        "end_shoe_index_exclusive": shard.end_shoe_index_exclusive,
        "shoe_count": shard.shoe_count,
    }
    config = _require_mapping(manifest.get("generation_config"), "generation config")
    if manifest.get("range") != expected_range:
        raise ValueError(f"shard range does not match plan: {shard.relative_path}")
    if (
        config.get("num_shoes") != shard.shoe_count
        or config.get("root_seed") != shard.root_seed
        or config.get("split") != shard.split
        or config.get("dataset_version")
        != _require_mapping(spec_payload["release"], "release")["dataset_version"]
    ):
        raise ValueError(
            f"shard generation config does not match plan: {shard.relative_path}"
        )
    if manifest.get("source_control") != spec_payload["source_control"]:
        raise ValueError(
            f"shard source control does not match plan: {shard.relative_path}"
        )
    if manifest.get("runtime") != spec_payload["runtime"]:
        raise ValueError(f"shard runtime does not match plan: {shard.relative_path}")
    writer = _require_mapping(manifest.get("writer"), "writer")
    release_metadata = _require_mapping(spec_payload["release"], "release")
    if writer.get("debug_sample_count") != release_metadata["debug_sample_count"]:
        raise ValueError(
            f"shard debug sample count does not match plan: {shard.relative_path}"
        )
    return result


def _generate_expected_shard(
    *,
    config: GenerationConfig,
    destination: Path,
    shard: ShardSpec,
    debug_sample_count: int,
    isolate: bool,
) -> None:
    if not isolate:
        write_streaming_shard(
            config,
            destination,
            start_shoe_index=shard.start_shoe_index,
            shoe_count=shard.shoe_count,
            debug_sample_count=debug_sample_count,
            overwrite=False,
        )
        return
    command = [
        sys.executable,
        "-m",
        "blackjackgpt.data.cli",
        "generate-shard",
        "--start-shoe-index",
        str(shard.start_shoe_index),
        "--shoe-count",
        str(shard.shoe_count),
        "--seed",
        str(shard.root_seed),
        "--split",
        shard.split,
        "--dataset-version",
        config.dataset_version,
        "--output-dir",
        str(destination),
        "--debug-sample-count",
        str(debug_sample_count),
    ]
    _run_worker(command, label=f"generation failed for {shard.relative_path}")


def _run_shard_verifier(path: Path, *, replay: bool) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "blackjackgpt.data.cli",
        "verify-shard",
        "--shard",
        str(path),
    ]
    if replay:
        command.append("--replay")
    return _run_worker(command, label=f"verification failed for {path}")


def _run_worker(command: Sequence[str], *, label: str) -> dict[str, Any]:
    process = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip() or "no details"
        raise RuntimeError(f"{label}: {detail}")
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"{label}: worker returned an invalid result envelope")
    result = json.loads(lines[0])
    if not isinstance(result, dict) or result.get("passed") is not True:
        raise RuntimeError(f"{label}: worker did not report success")
    return result


def _insert_shard_identities(
    connection: sqlite3.Connection,
    path: Path,
    shard: ShardSpec,
) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("release verification requires pyarrow") from error

    parquet = pq.ParquetFile(path / "shoes.parquet")
    for batch in parquet.iter_batches(batch_size=256, columns=list(_IDENTITY_COLUMNS)):
        values = batch.to_pydict()
        row_count = batch.num_rows
        for offset in range(row_count):
            shoe_id = str(values["shoe_id"][offset])
            try:
                shoe_index = int(shoe_id.rsplit("-", 1)[1])
            except (IndexError, ValueError) as error:
                raise ValueError(f"invalid shoe ID in {shard.relative_path}") from error
            for column in _IDENTITY_COLUMNS:
                value = str(values[column][offset])
                try:
                    connection.execute(
                        "INSERT INTO identities(kind, value, split, shoe_index) "
                        "VALUES (?, ?, ?, ?)",
                        (column, value, shard.split, shoe_index),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError(
                        f"cross-shard duplicate {column}: {value}"
                    ) from error


def _shared_shard_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    writer = _require_mapping(manifest.get("writer"), "writer")
    writer_settings = {
        key: value
        for key, value in writer.items()
        if key not in {"event_row_groups", "shoe_row_groups"}
    }
    return {
        "generator_version": manifest.get("generator_version"),
        "blackjackgpt_version": manifest.get("blackjackgpt_version"),
        "event_sidecar_schema_version": manifest.get("event_sidecar_schema_version"),
        "seed_derivation_version": manifest.get("seed_derivation_version"),
        "source_control": manifest.get("source_control"),
        "runtime": manifest.get("runtime"),
        "vocabulary_sha256": manifest.get("vocabulary_sha256"),
        "rules_sha256": manifest.get("rules_sha256"),
        "expert_policy_sha256": manifest.get("expert_policy_sha256"),
        "writer": writer_settings,
    }


def _resolved_spec_payload(spec: ReleaseSpec) -> dict[str, Any]:
    try:
        import numpy as np
        import pyarrow as pa
    except ImportError as error:
        raise RuntimeError("release generation requires numpy and pyarrow") from error
    source_control = _git_metadata(Path(__file__).resolve().parent)
    if source_control.get("available") is not True or not source_control.get("commit"):
        raise RuntimeError("release generation requires source under Git control")
    if not spec.pilot and source_control.get("dirty") is not False:
        raise RuntimeError("production release generation requires a clean Git tree")
    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
    }
    if spec.pilot:
        try:
            remote_provenance = _remote_provenance(str(source_control["commit"]))
        except RuntimeError:
            remote_provenance = {
                "origin_url": None,
                "containing_remote_refs": [],
            }
    else:
        remote_provenance = _remote_provenance(str(source_control["commit"]))
        expected_source_repository = os.environ.get(PRODUCTION_SOURCE_REPOSITORY_ENV)
        if not expected_source_repository:
            raise RuntimeError(
                f"production release generation requires "
                f"{PRODUCTION_SOURCE_REPOSITORY_ENV}"
            )
        if runtime != PRODUCTION_RUNTIME:
            raise RuntimeError(
                f"production runtime must exactly match {PRODUCTION_RUNTIME}"
            )
        if remote_provenance["origin_url"] != expected_source_repository:
            raise RuntimeError(
                "production source repository does not match the configured URL; "
                "use a new release/spec version for a different source"
            )
        if not remote_provenance["containing_remote_refs"]:
            raise RuntimeError(
                "production source commit is not reachable from a fetched remote ref"
            )
    return {
        "spec_document_version": "resolved-streaming-release-spec-v1",
        "release": spec.to_metadata(),
        "source_control": source_control,
        "remote_provenance": remote_provenance,
        "source_fingerprint_sha256": _source_fingerprint(),
        "runtime": runtime,
        "fixed_boundary_storage": True,
        "audit_offset_semantics": "v1-shard-local-container-offsets",
    }


def _validate_resolved_spec_payload(
    payload: Mapping[str, Any],
    spec: ReleaseSpec,
) -> None:
    required = {
        "spec_document_version",
        "release",
        "source_control",
        "remote_provenance",
        "source_fingerprint_sha256",
        "runtime",
        "fixed_boundary_storage",
        "audit_offset_semantics",
    }
    if set(payload) != required:
        raise ValueError("dataset specification field inventory is invalid")
    if payload["spec_document_version"] != "resolved-streaming-release-spec-v1":
        raise ValueError("unsupported resolved release specification")
    if payload["release"] != spec.to_metadata():
        raise ValueError("resolved release metadata is inconsistent")
    if (
        payload["fixed_boundary_storage"] is not True
        or payload["audit_offset_semantics"] != "v1-shard-local-container-offsets"
    ):
        raise ValueError("unsupported release storage semantics")


def _assert_current_environment(spec_payload: Mapping[str, Any]) -> None:
    current = _resolved_spec_payload(
        ReleaseSpec.from_metadata(_require_mapping(spec_payload["release"], "release"))
    )
    if current != dict(spec_payload):
        raise RuntimeError(
            "source tree or runtime changed after dataset_spec.json was frozen"
        )


def _source_fingerprint() -> str:
    source_dir = Path(__file__).resolve().parents[1]
    root = source_dir.parent
    candidates = [root / "pyproject.toml"]
    candidates.extend(sorted(source_dir.rglob("*.py")))
    digest = hashlib.sha256()
    for path in candidates:
        if not path.is_file():
            raise FileNotFoundError(f"source fingerprint input is missing: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _remote_provenance(commit: str) -> dict[str, Any]:
    source_dir = Path(__file__).resolve().parent
    try:
        origin = subprocess.run(
            ["git", "-C", str(source_dir), "config", "--get", "remote.origin.url"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        refs_output = subprocess.run(
            [
                "git",
                "-C",
                str(source_dir),
                "branch",
                "-r",
                "--contains",
                commit,
                "--format=%(refname:short)",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("could not resolve Git remote provenance") from error
    return {
        "origin_url": origin,
        "containing_remote_refs": sorted(
            line.strip() for line in refs_output.splitlines() if line.strip()
        ),
    }


def _validate_inventory(root: Path, spec: ReleaseSpec, *, complete: bool) -> None:
    expected_root = {"dataset_spec.json", "splits"}
    if complete:
        expected_root.add("dataset_manifest.json")
    actual_root = {path.name for path in root.iterdir()}
    if complete:
        if actual_root != expected_root:
            raise ValueError(
                f"release inventory mismatch; expected={sorted(expected_root)}, "
                f"actual={sorted(actual_root)}"
            )
    elif not actual_root <= expected_root | {"dataset_manifest.json"}:
        raise ValueError("release contains unexpected files or directories")

    split_root = root / "splits"
    split_root.mkdir(exist_ok=True)
    expected_splits = {split.name for split in spec.splits}
    actual_splits = {path.name for path in split_root.iterdir()}
    if not actual_splits <= expected_splits:
        raise ValueError("release contains an unexpected split directory")
    shards_by_split: dict[str, set[str]] = {split.name: set() for split in spec.splits}
    for shard in spec.iter_shards():
        shards_by_split[shard.split].add(shard.name)
    for split in spec.splits:
        directory = split_root / split.name
        shards_root = directory / "shards"
        if not directory.exists():
            if complete:
                raise ValueError(f"release is missing split directory: {split.name}")
            shards_root.mkdir(parents=True)
        actual_children = {path.name for path in directory.iterdir()}
        allowed_children = {"shards", "split_manifest.json"}
        if not actual_children <= allowed_children:
            raise ValueError(f"split contains unexpected entries: {split.name}")
        if complete and actual_children != allowed_children:
            raise ValueError(f"split is incomplete: {split.name}")
        shards_root.mkdir(exist_ok=True)
        actual_shards = {path.name for path in shards_root.iterdir()}
        if not all(_SHARD_PATTERN.fullmatch(name) for name in actual_shards):
            raise ValueError(f"split contains an invalid shard name: {split.name}")
        if complete:
            if actual_shards != shards_by_split[split.name]:
                raise ValueError(f"split shard inventory mismatch: {split.name}")
        elif not actual_shards <= shards_by_split[split.name]:
            raise ValueError(f"split contains an unexpected shard: {split.name}")


def _publish_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() == encoded:
            return
        raise FileExistsError(f"refusing to replace existing control file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.staging-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _integer_mapping(value: Any, label: str) -> dict[str, int]:
    mapping = _require_mapping(value, label)
    result: dict[str, int] = {}
    for key, item in mapping.items():
        if not isinstance(key, str):
            raise TypeError(f"{label} keys must be strings")
        _require_integer(f"{label}.{key}", item)
        result[key] = int(item)
    return result


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")
    return value


def _require_integer(label: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")

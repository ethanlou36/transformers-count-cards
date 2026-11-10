from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from blackjackgpt.data.hub import HUB_REPO_ID_ENV
from blackjackgpt.provenance import sha256_file as _sha256_file

_EVENT_TOKEN_OFFSET_COLUMNS = ("global_token_offset",)
_SHOE_TOKEN_OFFSET_COLUMNS = ("token_start", "token_end_exclusive")

_SHOE_INDEX_COLUMNS = ("shoe_index",)

_COPIED_CONFIG_FILES = ("vocab.json", "rules.json", "expert_policy.json")


class ShardMergeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShardRef:
    split: str
    name: str
    directory: Path
    start_shoe_index: int
    shoe_count: int


@dataclass
class ReleasePlan:
    release_dir: Path
    spec: dict[str, Any]
    manifest: dict[str, Any]
    shards: list[ShardRef] = field(default_factory=list)

    @property
    def total_shoes(self) -> int:
        return int(self.manifest["shoe_count"])


def locate_release(input_dir: Path) -> Path:
    if (input_dir / "dataset_spec.json").exists():
        return input_dir
    candidates = sorted(input_dir.glob("**/dataset_spec.json"))
    if not candidates:
        raise ShardMergeError(
            f"no dataset_spec.json found under {input_dir}; is the release downloaded?"
        )
    if len(candidates) > 1:
        raise ShardMergeError(
            f"multiple dataset_spec.json found under {input_dir}: {candidates}"
        )
    return candidates[0].parent


def build_plan(input_dir: Path) -> ReleasePlan:
    release_dir = locate_release(input_dir)
    spec = json.loads((release_dir / "dataset_spec.json").read_text())
    manifest = json.loads((release_dir / "dataset_manifest.json").read_text())

    shards: list[ShardRef] = []
    for split in spec["release"]["splits"]:
        split_name = split["name"]
        for shard in split["shards"]:
            shard_dir = release_dir / "splits" / split_name / "shards" / shard["name"]
            if not shard_dir.is_dir():
                raise ShardMergeError(f"missing shard directory: {shard_dir}")
            shards.append(
                ShardRef(
                    split=split_name,
                    name=shard["name"],
                    directory=shard_dir,
                    start_shoe_index=int(shard["start_shoe_index"]),
                    shoe_count=int(shard["shoe_count"]),
                )
            )
    plan = ReleasePlan(release_dir=release_dir, spec=spec, manifest=manifest)
    plan.shards = shards
    return plan


def verify_manifest_checksums(plan: ReleasePlan) -> None:
    for rel_path, meta in plan.manifest.get("artifacts", {}).items():
        path = plan.release_dir / rel_path
        if not path.exists():
            raise ShardMergeError(f"manifest artifact missing: {rel_path}")
        actual = _sha256_file(path)
        if actual != meta["sha256"]:
            raise ShardMergeError(
                f"checksum mismatch for {rel_path}: expected {meta['sha256']}, "
                f"got {actual}"
            )

    for shard in plan.shards:
        gen = json.loads((shard.directory / "generation_manifest.json").read_text())
        for name, meta in gen.get("artifacts", {}).items():
            path = shard.directory / name
            if not path.exists():
                raise ShardMergeError(f"{shard.name}: payload missing: {name}")
            actual = _sha256_file(path)
            if actual != meta["sha256"]:
                raise ShardMergeError(
                    f"{shard.name}/{name} checksum mismatch: "
                    f"expected {meta['sha256']}, got {actual}"
                )


def check_shared_config(plan: ReleasePlan) -> dict[str, str]:
    keys = ("vocabulary_sha256", "rules_sha256", "expert_policy_sha256")
    shared = plan.manifest.get("shared_shard_provenance", {})
    reference: dict[str, str] | None = None
    for shard in plan.shards:
        gen = json.loads((shard.directory / "generation_manifest.json").read_text())
        got = {k: gen.get(k) for k in keys}
        if any(got[k] is None for k in keys):
            raise ShardMergeError(f"{shard.name}: missing config checksum(s)")
        if reference is None:
            reference = got
        elif got != reference:
            raise ShardMergeError(
                f"config checksum mismatch in {shard.name}: {got} != {reference}"
            )
    assert reference is not None
    for k in keys:
        if k in shared and shared[k] != reference[k]:
            raise ShardMergeError(
                f"{k} disagrees with manifest shared provenance: "
                f"{reference[k]} != {shared[k]}"
            )
    return reference


@dataclass
class MergeResult:
    output_dir: Path
    total_tokens: int
    total_shoes: int
    per_shard: list[dict[str, Any]]


def merge(
    plan: ReleasePlan,
    output_dir: Path,
    *,
    overwrite: bool = False,
    sample_size: int = 20000,
    seed: int = 0,
) -> MergeResult:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise ShardMergeError(
            f"{output_dir} is non-empty; pass --overwrite to replace it"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    verify_manifest_checksums(plan)
    shared_config = check_shared_config(plan)

    tokens_parts: list[np.ndarray] = []
    offsets_parts: list[np.ndarray] = []
    per_shard: list[dict[str, Any]] = []
    shoe_ids_seen: set[str] = set()

    token_base = 0
    shoe_base = 0
    events_writer: pq.ParquetWriter | None = None
    shoes_writer: pq.ParquetWriter | None = None
    events_path = output_dir / "events.parquet"
    shoes_path = output_dir / "shoes.parquet"

    try:
        for shard in plan.shards:
            tok = np.load(shard.directory / "tokens.npy")
            off = np.load(shard.directory / "shoe_offsets.npy").astype(np.int64)
            shard_tokens = int(off[-1])
            shard_shoes = len(off) - 1

            if len(tok) != shard_tokens:
                raise ShardMergeError(
                    f"{shard.name}: tokens ({len(tok)}) != offsets sentinel "
                    f"({shard_tokens})"
                )
            if off[0] != 0:
                raise ShardMergeError(f"{shard.name}: shoe_offsets[0] != 0")
            if shard_shoes != shard.shoe_count:
                raise ShardMergeError(
                    f"{shard.name}: shoe count {shard_shoes} != spec {shard.shoe_count}"
                )

            tokens_parts.append(tok.astype(np.uint16))
            offsets_parts.append(off[:-1] + token_base)

            events = pq.read_table(shard.directory / "events.parquet")
            events = events.sort_by("global_token_offset")
            shoes = pq.read_table(shard.directory / "shoes.parquet")

            events = _rebase_table(
                events, token_base, shoe_base, _EVENT_TOKEN_OFFSET_COLUMNS
            )
            shoes = _rebase_table(
                shoes, token_base, shoe_base, _SHOE_TOKEN_OFFSET_COLUMNS
            )

            shard_ids = events.column("shoe_id").to_pylist()
            unique_ids = set(shard_ids)
            clash = shoe_ids_seen & unique_ids
            if clash:
                raise ShardMergeError(
                    f"{shard.name}: shoe_id collision across shards: "
                    f"{sorted(clash)[:3]}..."
                )
            shoe_ids_seen |= unique_ids

            if events_writer is None or shoes_writer is None:
                events_writer = pq.ParquetWriter(events_path, events.schema)
                shoes_writer = pq.ParquetWriter(shoes_path, shoes.schema)
            events_writer.write_table(events.cast(events_writer.schema))
            shoes_writer.write_table(shoes.cast(shoes_writer.schema))

            per_shard.append(
                {
                    "split": shard.split,
                    "name": shard.name,
                    "shoes": shard_shoes,
                    "tokens": shard_tokens,
                    "token_base": token_base,
                    "shoe_base": shoe_base,
                    "tokens_sha256": _sha256_file(shard.directory / "tokens.npy"),
                    "events_sha256": _sha256_file(shard.directory / "events.parquet"),
                }
            )
            token_base += shard_tokens
            shoe_base += shard_shoes
    finally:
        if events_writer is not None:
            events_writer.close()
        if shoes_writer is not None:
            shoes_writer.close()

    tokens = np.concatenate(tokens_parts).astype(np.uint16)
    offsets = np.concatenate([*offsets_parts, np.array([token_base], dtype=np.int64)])
    offsets = offsets.astype(np.int64)

    np.save(output_dir / "tokens.npy", tokens, allow_pickle=False)
    np.save(output_dir / "shoe_offsets.npy", offsets, allow_pickle=False)

    for name in _COPIED_CONFIG_FILES:
        shutil.copy2(plan.shards[0].directory / name, output_dir / name)

    _assert_invariants(
        tokens=tokens,
        offsets=offsets,
        events_path=events_path,
        expected_shoes=plan.total_shoes,
        unique_shoe_ids=len(shoe_ids_seen),
        sample_size=sample_size,
        seed=seed,
    )

    _write_merged_manifest(plan, output_dir, tokens, offsets, per_shard, shared_config)

    return MergeResult(
        output_dir=output_dir,
        total_tokens=int(token_base),
        total_shoes=len(offsets) - 1,
        per_shard=per_shard,
    )


def _rebase_table(
    table: pa.Table, token_base: int, shoe_base: int, token_cols: tuple[str, ...]
) -> pa.Table:
    import pyarrow.compute as pc

    for col in token_cols:
        if col in table.column_names:
            rebased = pc.add(table.column(col), pa.scalar(token_base, pa.int64()))
            table = table.set_column(
                table.schema.get_field_index(col), col, rebased.cast(pa.int64())
            )
    for col in _SHOE_INDEX_COLUMNS:
        if col in table.column_names:
            local = table.column(col)
            shifted = pc.add(
                pc.subtract(local, pc.min(local)),
                pa.scalar(shoe_base, pa.int64()),
            )
            table = table.set_column(
                table.schema.get_field_index(col), col, shifted.cast(pa.int64())
            )
    return table


def _assert_invariants(
    *,
    tokens: np.ndarray,
    offsets: np.ndarray,
    events_path: Path,
    expected_shoes: int,
    unique_shoe_ids: int,
    sample_size: int,
    seed: int,
) -> None:
    if offsets[0] != 0 or offsets[-1] != len(tokens):
        raise ShardMergeError("merged shoe_offsets sentinel/base is wrong")
    if np.any(np.diff(offsets) <= 0):
        raise ShardMergeError("merged shoe_offsets not strictly increasing")
    if len(offsets) - 1 != expected_shoes:
        raise ShardMergeError(
            f"merged shoe count {len(offsets) - 1} != manifest {expected_shoes}"
        )
    from blackjackgpt.data.vocabulary import VERSION1_VOCABULARY

    shuffle_id = VERSION1_VOCABULARY.encode("SHUFFLE")
    starts = tokens[offsets[:-1]]
    if not np.all(starts == shuffle_id):
        bad = int(np.argmax(starts != shuffle_id))
        raise ShardMergeError(
            f"shoe {bad} does not begin with SHUFFLE after rebasing "
            "(offset misalignment)"
        )

    if unique_shoe_ids != expected_shoes:
        raise ShardMergeError(
            f"unique shoe_id count {unique_shoe_ids} != manifest {expected_shoes}"
        )

    cols = pq.read_table(events_path, columns=["global_token_offset", "token_id"])
    goff = cols.column("global_token_offset").to_numpy()
    tid = cols.column("token_id").to_numpy()
    if len(goff) != len(tokens):
        raise ShardMergeError(
            f"merged events rows ({len(goff)}) != tokens ({len(tokens)})"
        )
    n_sample = min(max(sample_size, 10000), len(goff))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(goff), size=n_sample, replace=False)
    if not np.array_equal(tokens[goff[idx]], tid[idx].astype(tokens.dtype)):
        raise ShardMergeError(
            "tokens[global_token_offset] != events.token_id after merge "
            "(offset rebase corrupted the sidecar alignment)"
        )


def _write_merged_manifest(
    plan: ReleasePlan,
    output_dir: Path,
    tokens: np.ndarray,
    offsets: np.ndarray,
    per_shard: list[dict[str, Any]],
    shared_config: dict[str, str],
) -> None:
    combined = hashlib.sha256(
        "\n".join(
            f"{s['name']}:{s['tokens_sha256']}:{s['events_sha256']}" for s in per_shard
        ).encode()
    ).hexdigest()
    manifest = {
        "merged_from": {
            "hf_repo": os.environ.get(HUB_REPO_ID_ENV),
            "revision": os.environ.get("BLACKJACKGPT_HUB_REVISION"),
            "release_id": plan.manifest.get("release_id"),
            "source_fingerprint_sha256": plan.manifest.get("source_fingerprint_sha256"),
            "source_commit": plan.manifest.get("source_control", {}).get("commit"),
        },
        "shared_config_sha256": shared_config,
        "counts": {
            "shoes": int(len(offsets) - 1),
            "tokens": len(tokens),
        },
        "max_shoe_length_tokens": int(np.diff(offsets).max()),
        "shard_order": per_shard,
        "combined_source_sha256": combined,
        "output_tokens_sha256": _sha256_bytes(tokens.tobytes()),
    }
    (output_dir / "merged_manifest.json").write_text(json.dumps(manifest, indent=2))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a sharded release.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    plan = build_plan(args.input)
    print(
        f"[merge] {len(plan.shards)} shards across "
        f"{len({s.split for s in plan.shards})} splits -> "
        f"{plan.total_shoes} shoes"
    )
    result = merge(
        plan,
        args.output,
        overwrite=args.overwrite,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(
        f"[merge] wrote {result.total_shoes} shoes / {result.total_tokens} tokens "
        f"to {result.output_dir}"
    )
    print("[merge] all invariants passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

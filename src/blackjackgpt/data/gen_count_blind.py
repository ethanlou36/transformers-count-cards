from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from blackjackgpt.data.generator import GenerationConfig, generate_dataset
from blackjackgpt.data.merge_shards import build_plan, merge
from blackjackgpt.data.writers import write_dataset
from blackjackgpt.expert.count_blind import count_blind_policy
from blackjackgpt.provenance import git_commit


def generate_release(
    num_shoes: int,
    seed: int,
    chunk_size: int,
    release_dir: Path,
    split: str = "lambda0",
    debug_sample_count: int = 3,
) -> Path:
    if num_shoes % chunk_size != 0:
        raise ValueError("num_shoes must be a multiple of chunk_size")
    n_chunks = num_shoes // chunk_size
    shards_root = release_dir / "splits" / split / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)

    shard_meta = []
    for c in range(n_chunks):
        start = c * chunk_size
        name = f"shard-{start:08d}-{start + chunk_size - 1:08d}"
        shard_dir = shards_root / name
        if (shard_dir / "generation_manifest.json").exists():
            print(f"[gen-l0] chunk {c + 1}/{n_chunks} {name}: SKIP (exists)")
        else:
            cfg = GenerationConfig(
                num_shoes=chunk_size,
                root_seed=seed,
                split=f"{split}-{c:04d}",
            )
            dataset = generate_dataset(cfg, policy=count_blind_policy())
            write_dataset(
                dataset,
                shard_dir,
                debug_sample_count=debug_sample_count,
                overwrite=True,
            )
            print(
                f"[gen-l0] chunk {c + 1}/{n_chunks} {name}: "
                f"{len(dataset.token_ids)} tokens"
            )
            del dataset
            gc.collect()
        shard_meta.append(
            {
                "name": name,
                "start_shoe_index": start,
                "shoe_count": chunk_size,
                "end_shoe_index_exclusive": start + chunk_size,
            }
        )

    spec = {
        "release": {
            "release_id": f"{split}-count-blind",
            "shoe_count": num_shoes,
            "splits": [
                {
                    "name": split,
                    "start_shoe_index": 0,
                    "end_shoe_index_exclusive": num_shoes,
                    "shards": shard_meta,
                }
            ],
        }
    }
    (release_dir / "dataset_spec.json").write_text(json.dumps(spec, indent=2))
    manifest = {
        "release_id": f"{split}-count-blind",
        "shoe_count": num_shoes,
        "shard_count": n_chunks,
        "artifacts": {},
        "shared_shard_provenance": {},
        "source_control": git_commit(),
        "source_fingerprint_sha256": f"{split}-count-blind",
        "generation": {"policy": "count-blind", "root_seed": seed},
    }
    (release_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    return release_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a count-blind dataset.")
    parser.add_argument("--num-shoes", type=int, default=100000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--release-dir", type=Path, default=Path("data/full_lambda0"))
    parser.add_argument("--output", type=Path, default=Path("data/merged_lambda0"))
    parser.add_argument("--split", type=str, default="lambda0")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    print(
        f"[gen-l0] generating {args.num_shoes} count-blind shoes -> {args.release_dir}"
    )
    release_dir = generate_release(
        args.num_shoes, args.seed, args.chunk_size, args.release_dir, args.split
    )
    print(f"[gen-l0] merging shards -> {args.output}")
    plan = build_plan(release_dir)
    result = merge(plan, args.output, overwrite=args.overwrite)
    print(
        f"[gen-l0] DONE: {result.total_shoes} shoes / {result.total_tokens} tokens "
        f"in {args.output}; all merge invariants passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

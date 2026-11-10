from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .generator import DATASET_SCHEMA_VERSION, GenerationConfig
from .hub import (
    HUB_REPO_ID_ENV,
    download_release_from_hub,
    upload_release_to_hub,
)
from .release import (
    generate_release,
    pilot_release_spec,
    production_release_spec,
    verify_release,
)
from .streaming import verify_streaming_shard, write_streaming_shard


def _nonnegative_int(value: str) -> int:
    parsed = int(value, 0)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value, 0)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _unsigned_u64(value: str) -> int:
    parsed = int(value, 0)
    if not 0 <= parsed < 2**64:
        raise argparse.ArgumentTypeError("value must be an unsigned 64-bit integer")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blackjackgpt-data",
        description="Generate and verify bounded-memory BlackjackGPT shards.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate-shard",
        help="generate one deterministic local shard",
    )
    generate.add_argument("--start-shoe-index", type=_nonnegative_int, required=True)
    generate.add_argument("--shoe-count", type=_positive_int, required=True)
    generate.add_argument("--seed", type=_unsigned_u64, required=True)
    generate.add_argument("--split", required=True)
    generate.add_argument("--dataset-version", default=DATASET_SCHEMA_VERSION)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--debug-sample-count", type=_nonnegative_int, default=3)
    generate.add_argument("--overwrite", action="store_true")

    verify = subparsers.add_parser(
        "verify-shard",
        help="checksum and semantically verify a completed streaming shard",
    )
    verify.add_argument("--shard", type=Path, required=True)
    verify.add_argument("--replay", action="store_true")

    pilot = subparsers.add_parser(
        "generate-pilot",
        help="generate or resume an isolated sequential streaming pilot",
    )
    pilot.add_argument("--seed", type=_unsigned_u64, required=True)
    pilot.add_argument("--shoe-count", type=_positive_int, default=10_000)
    pilot.add_argument("--shard-shoe-count", type=_positive_int, default=5_000)
    pilot.add_argument(
        "--release-id",
        default="blackjack-v1-10k-pilot-v1",
    )
    pilot.add_argument("--output-dir", type=Path, required=True)
    pilot.add_argument("--debug-sample-count", type=_nonnegative_int, default=3)
    pilot.add_argument("--resume", action="store_true")

    release = subparsers.add_parser(
        "generate-release",
        help="generate or resume the fixed sequential 90k/5k/5k release",
    )
    release.add_argument("--output-dir", type=Path, required=True)
    release.add_argument("--resume", action="store_true")

    verify_release_parser = subparsers.add_parser(
        "verify-release",
        help="verify a completed release, aggregate manifests, and all shards",
    )
    verify_release_parser.add_argument("--release-root", type=Path, required=True)
    verify_release_parser.add_argument("--replay", action="store_true")

    hub_upload = subparsers.add_parser(
        "hub-upload",
        help="explicitly upload one completed, verified release to Hugging Face",
    )
    hub_upload.add_argument("--release-root", type=Path, required=True)
    hub_upload.add_argument(
        "--repo-id", help=f"private dataset repo (or set {HUB_REPO_ID_ENV})"
    )
    hub_upload.add_argument("--revision", required=True)

    hub_download = subparsers.add_parser(
        "hub-download",
        help="download and verify a complete release, split, or shard",
    )
    hub_download.add_argument("--release-id", required=True)
    hub_download.add_argument("--destination", type=Path, required=True)
    hub_download.add_argument(
        "--repo-id", help=f"private dataset repo (or set {HUB_REPO_ID_ENV})"
    )
    hub_download.add_argument("--revision", default="main")
    hub_download.add_argument("--split")
    hub_download.add_argument("--shard")
    hub_download.add_argument("--expected-manifest-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        if args.command == "generate-shard":
            config = GenerationConfig(
                num_shoes=args.shoe_count,
                root_seed=args.seed,
                split=args.split,
                dataset_version=args.dataset_version,
            )

            def progress(completed: int, total: int) -> None:
                if completed == total or completed % 100 == 0:
                    print(f"generated {completed}/{total} shoes", file=sys.stderr)

            output = write_streaming_shard(
                config,
                args.output_dir,
                start_shoe_index=args.start_shoe_index,
                shoe_count=args.shoe_count,
                debug_sample_count=args.debug_sample_count,
                overwrite=args.overwrite,
                progress=progress,
            )
            manifest = json.loads(
                (output / "generation_manifest.json").read_text(encoding="utf-8")
            )
            result = {
                "passed": True,
                "errors": [],
                "shard": str(output),
                "start_shoe_index": manifest["range"]["start_shoe_index"],
                "end_shoe_index_exclusive": manifest["range"][
                    "end_shoe_index_exclusive"
                ],
                "shoe_count": manifest["counts"]["shoes"],
                "token_count": manifest["counts"]["tokens"],
                "round_count": manifest["counts"]["rounds"],
                "replay": True,
                "replay_count": min(3, int(manifest["counts"]["shoes"])),
            }
            print(json.dumps({"command": args.command, **result}, sort_keys=True))
            return 0

        if args.command in {"generate-pilot", "generate-release"}:
            spec = (
                pilot_release_spec(
                    root_seed=args.seed,
                    release_id=args.release_id,
                    shoe_count=args.shoe_count,
                    shard_shoe_count=args.shard_shoe_count,
                    debug_sample_count=args.debug_sample_count,
                )
                if args.command == "generate-pilot"
                else production_release_spec()
            )

            def release_progress(path: str, completed: int, total: int) -> None:
                print(
                    f"completed shard {completed}/{total}: {path}",
                    file=sys.stderr,
                )

            result = generate_release(
                spec,
                args.output_dir,
                resume=args.resume,
                progress=release_progress,
            )
        elif args.command == "verify-shard":
            result = verify_streaming_shard(args.shard, replay=args.replay)
        elif args.command == "verify-release":
            result = verify_release(args.release_root, replay=args.replay)
        elif args.command == "hub-upload":
            result = upload_release_to_hub(
                args.release_root,
                repo_id=args.repo_id,
                revision=args.revision,
            )
        else:
            result = download_release_from_hub(
                args.destination,
                release_id=args.release_id,
                repo_id=args.repo_id,
                revision=args.revision,
                split=args.split,
                shard=args.shard,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
        stream = sys.stdout if result["passed"] else sys.stderr
        print(
            json.dumps({"command": args.command, **result}, sort_keys=True), file=stream
        )
        return 0 if result["passed"] else 1
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

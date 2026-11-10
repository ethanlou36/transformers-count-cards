from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from verify_frozen_artifacts import (
    _digest,
    _symlink_components,
    frozen_inventory,
    frozen_layout_errors,
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Refresh frozen-artifact hashes.")
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument(
        "--provenance-updated",
        action="store_true",
        help="confirm PROVENANCE.txt documents the intentional artifact change",
    )
    args = parser.parse_args(argv)
    if not args.provenance_updated:
        parser.error("--provenance-updated is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    provenance = repo_root / "PROVENANCE.txt"
    if (
        not provenance.is_file()
        or provenance.is_symlink()
        or _symlink_components(repo_root, provenance)
    ):
        print(f"missing or unsafe provenance file: {provenance}")
        return 1
    layout_errors = frozen_layout_errors(repo_root)
    if layout_errors:
        print("refusing invalid frozen-artifact layout:")
        for message in layout_errors:
            print(f"- {message}")
        return 1
    inventory = sorted(frozen_inventory(repo_root))
    symlinks = [path for path in inventory if (repo_root / path).is_symlink()]
    if symlinks:
        print(f"refusing symlinked frozen artifacts: {symlinks}")
        return 1
    lines: list[str] = []
    lines.extend(f"{_digest(repo_root / path)}  {path}\n" for path in inventory)
    manifest = repo_root / "artifacts" / "frozen-artifacts.sha256"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".frozen-artifacts-", dir=manifest.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.writelines(lines)
        os.replace(temporary, manifest)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote {len(inventory)} hashes to {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

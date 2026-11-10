from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REMOVED_ROOTS = ("runs",)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _visible_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for path in root.rglob("*"):
        if path.is_symlink() or path.is_file():
            yield path


def _symlink_components(repo_root: Path, target: Path) -> list[str]:
    links: list[str] = []
    current = repo_root
    for part in target.relative_to(repo_root).parts:
        current /= part
        if current.is_symlink():
            links.append(current.relative_to(repo_root).as_posix())
    return links


def frozen_inventory(repo_root: Path) -> set[str]:
    runs = repo_root / "runs"
    inventory = (
        set()
        if _symlink_components(repo_root, runs)
        else {path.relative_to(repo_root).as_posix() for path in _visible_files(runs)}
    )
    return inventory


def frozen_layout_errors(repo_root: Path) -> list[str]:
    errors: list[str] = []
    for relative in _REMOVED_ROOTS:
        frozen_root = repo_root / relative
        links = _symlink_components(repo_root, frozen_root)
        if links:
            errors.extend(f"symlinked removed root component: {link}" for link in links)
        elif frozen_root.exists():
            errors.append(f"removed frozen root was reintroduced: {relative}")

    return errors


def read_manifest(path: Path) -> tuple[dict[str, str], list[str]]:
    entries: dict[str, str] = {}
    errors: list[str] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or _SHA256.fullmatch(parts[0]) is None:
            errors.append(f"line {line_number}: malformed manifest entry")
            continue
        relative = parts[1].lstrip("*")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            errors.append(f"line {line_number}: unsafe path: {relative}")
        elif relative in entries:
            errors.append(f"line {line_number}: duplicate path: {relative}")
        else:
            entries[relative] = parts[0]
    return entries, errors


def verify(repo_root: Path, manifest_path: Path) -> list[str]:
    repo_root = repo_root.resolve()
    manifest_path = manifest_path.expanduser()
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    errors = frozen_layout_errors(repo_root)
    entries: dict[str, str] = {}
    if manifest_path.is_symlink():
        errors.append(f"symlinked manifest is not allowed: {manifest_path}")
    elif not manifest_path.is_file():
        errors.append(f"missing or invalid manifest: {manifest_path}")
    else:
        entries, manifest_errors = read_manifest(manifest_path)
        errors.extend(manifest_errors)
    actual_inventory = frozen_inventory(repo_root)
    expected_inventory = set(entries)
    for relative in sorted(expected_inventory - actual_inventory):
        errors.append(f"missing artifact: {relative}")
    for relative in sorted(actual_inventory - expected_inventory):
        errors.append(f"unmanifested artifact: {relative}")
    for relative in sorted(expected_inventory & actual_inventory):
        artifact = repo_root / relative
        if artifact.is_symlink():
            errors.append(f"symlinked artifact is not allowed: {relative}")
        elif _digest(artifact) != entries[relative]:
            errors.append(f"checksum mismatch: {relative}")
    return errors


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Verify frozen artifacts.")
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="manifest path (default: <repo-root>/artifacts/frozen-artifacts.sha256)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    manifest = args.manifest or repo_root / "artifacts" / "frozen-artifacts.sha256"
    try:
        errors = verify(repo_root, manifest)
    except (OSError, UnicodeError) as error:
        print(f"frozen artifact verification failed: {error}")
        return 1
    if errors:
        print("frozen artifact verification failed:")
        for message in errors:
            print(f"- {message}")
        return 1
    print(f"verified {len(frozen_inventory(repo_root))} frozen artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

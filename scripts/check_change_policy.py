from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence
from pathlib import Path

_PROVENANCE = "PROVENANCE.txt"
_MANIFEST = "artifacts/frozen-artifacts.sha256"


def _is_frozen(path: str) -> bool:
    return path.startswith("runs/") or path == _MANIFEST


def provenance_error(
    changed: set[str], *, provenance_exists: bool = True
) -> str | None:
    if not provenance_exists:
        return f"missing or unsafe required provenance file: {_PROVENANCE}"
    frozen_changes = sorted(path for path in changed if _is_frozen(path))
    if frozen_changes and _PROVENANCE not in changed:
        joined = "\n  ".join(frozen_changes)
        return f"frozen evidence changed without {_PROVENANCE}:\n  {joined}"
    return None


def _is_safe_regular_file(repo_root: Path, relative: str) -> bool:
    current = repo_root
    for part in Path(relative).parts:
        current /= part
        if current.is_symlink():
            return False
    return current.is_file()


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check repository change policy.")
    parser.add_argument("--base", required=True, help="base commit SHA")
    parser.add_argument("--default-branch", default="main")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.base and set(args.base) == {"0"}:
        merge_base = _git("merge-base", "HEAD", f"origin/{args.default_branch}")
        if merge_base.returncode != 0 or not merge_base.stdout.strip():
            print(
                "cannot establish a merge base for the initial-branch push; "
                "refusing to skip change policy"
            )
            return merge_base.returncode or 1
        base = merge_base.stdout.strip()
    else:
        base = args.base
    whitespace = _git("diff", "--no-renames", "--check", base, "HEAD")
    if whitespace.returncode != 0:
        print(whitespace.stdout or whitespace.stderr, end="")
        return whitespace.returncode
    diff = _git("diff", "--no-renames", "--name-only", base, "HEAD")
    if diff.returncode != 0:
        print(diff.stderr, end="")
        return diff.returncode
    changed = {line for line in diff.stdout.splitlines() if line}
    repo_root = Path.cwd().resolve()
    error = provenance_error(
        changed,
        provenance_exists=_is_safe_regular_file(repo_root, _PROVENANCE),
    )
    if error is not None:
        print(error)
        return 1
    print(f"change policy passed for {len(changed)} paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

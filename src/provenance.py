from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def git_commit(repo_dir: str | Path | None = None) -> dict[str, object]:
    cwd = str(repo_dir) if repo_dir is not None else None
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode()
    except (subprocess.CalledProcessError, OSError):
        return {"available": False}
    return {"available": True, "commit": commit, "dirty": bool(status.strip())}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_MANIFEST_ARTIFACTS = ("tokens.npy", "shoe_offsets.npy", "vocab.json")


def dataset_manifest(dataset_dir: str | Path) -> dict[str, object]:
    root = Path(dataset_dir)
    checksums: dict[str, str] = {}
    for name in _MANIFEST_ARTIFACTS:
        path = root / name
        if path.exists():
            checksums[name] = sha256_file(path)
    combined = hashlib.sha256(
        "\n".join(f"{k}:{v}" for k, v in sorted(checksums.items())).encode("utf-8")
    ).hexdigest()
    return {
        "dataset_dir": str(root),
        "artifacts": checksums,
        "combined_sha256": combined,
    }

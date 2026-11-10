from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

from blackjackgpt.provenance import sha256_file as _sha256

HUB_REPO_ID_ENV = "BLACKJACKGPT_HUB_REPO_ID"
DATASET_MANIFEST_NAME = "dataset_manifest.json"
SHARD_MANIFEST_NAME = "generation_manifest.json"
_RELEASE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SPLIT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHARD_PATTERN = re.compile(r"shard-[0-9]{8}-[0-9]{8}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


class _RepoInfo(Protocol):
    private: bool
    sha: str


class _CommitInfo(Protocol):
    oid: str


class _HubApi(Protocol):
    def repo_info(
        self,
        repo_id: str,
        *,
        repo_type: str,
        revision: str,
    ) -> _RepoInfo: ...

    def list_repo_files(
        self,
        repo_id: str,
        *,
        repo_type: str,
        revision: str,
    ) -> list[str]: ...

    def upload_folder(
        self,
        *,
        repo_id: str,
        repo_type: str,
        folder_path: str,
        path_in_repo: str,
        revision: str,
        parent_commit: str,
        commit_message: str,
    ) -> _CommitInfo: ...


class _SnapshotDownload(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        repo_type: str,
        revision: str,
        allow_patterns: Sequence[str],
        local_dir: str,
    ) -> str: ...


def _load_huggingface_hub() -> tuple[type[_HubApi], _SnapshotDownload]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face support is not installed; install `blackjackgpt[hub]` and "
            "authenticate with `hf auth login` or the HF_TOKEN environment variable"
        ) from error
    return cast(type[_HubApi], HfApi), cast(_SnapshotDownload, snapshot_download)


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _checked_component(value: str, pattern: re.Pattern[str], label: str) -> str:
    if not pattern.fullmatch(value):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def _release_prefix(release_id: str) -> str:
    return (
        f"releases/{_checked_component(release_id, _RELEASE_ID_PATTERN, 'release ID')}"
    )


def _require_private_repo(
    api: _HubApi,
    *,
    repo_id: str,
    revision: str,
) -> str:
    info = api.repo_info(repo_id, repo_type="dataset", revision=revision)
    if not info.private:
        raise ValueError(
            f"refusing transfer because dataset repository is public: {repo_id}"
        )
    resolved = info.sha
    if not isinstance(resolved, str) or not resolved:
        raise ValueError(
            f"Hugging Face did not resolve revision {revision!r} to a commit"
        )
    return resolved


def _verify_full_release(release_root: Path) -> dict[str, Any]:
    from .release import verify_release

    result = verify_release(release_root)
    if not isinstance(result, dict):
        raise TypeError("verify_release must return a result dictionary")
    if not result.get("passed", False):
        errors = result.get("errors", ["release verification failed"])
        raise ValueError("release verification failed: " + "; ".join(map(str, errors)))
    return result


def _load_completed_manifest(release_root: Path) -> dict[str, Any]:
    path = release_root / DATASET_MANIFEST_NAME
    manifest = _read_json_object(path)
    if manifest.get("complete") is not True:
        raise ValueError("dataset manifest does not mark the release complete")
    release_id = manifest.get("release_id")
    if not isinstance(release_id, str):
        raise ValueError("dataset manifest is missing a string release_id")
    _release_prefix(release_id)
    return manifest


def _resolve_repo_id(repo_id: str | None) -> str:
    resolved = repo_id or os.environ.get(HUB_REPO_ID_ENV)
    if not resolved:
        raise ValueError(f"set {HUB_REPO_ID_ENV} or pass a Hub repository ID")
    return resolved


def upload_release_to_hub(
    release_root: str | Path,
    *,
    repo_id: str | None = None,
    revision: str,
) -> dict[str, Any]:
    repo_id = _resolve_repo_id(repo_id)
    root = Path(release_root).expanduser().resolve()
    normalized_revision = revision.strip("/")
    for prefix in ("refs/heads/", "heads/"):
        if normalized_revision.startswith(prefix):
            normalized_revision = normalized_revision.removeprefix(prefix)
            break
    if normalized_revision in {"main", "master"}:
        raise ValueError(
            "upload requires a fresh candidate branch, not the default branch"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"release directory does not exist: {root}")
    manifest = _load_completed_manifest(root)
    if manifest.get("pilot") is not False:
        raise ValueError("refusing to upload a pilot or unclassified release")
    _verify_full_release(root)

    release_id = cast(str, manifest["release_id"])
    prefix = _release_prefix(release_id)
    manifest_sha256 = _sha256(root / DATASET_MANIFEST_NAME)
    api_type, _ = _load_huggingface_hub()
    api = api_type()
    base_revision = _require_private_repo(
        api,
        repo_id=repo_id,
        revision=revision,
    )
    existing_paths = api.list_repo_files(
        repo_id,
        repo_type="dataset",
        revision=base_revision,
    )
    if any(path == prefix or path.startswith(f"{prefix}/") for path in existing_paths):
        raise FileExistsError(
            f"refusing to overwrite an existing Hugging Face release path: {prefix}"
        )
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(root),
        path_in_repo=prefix,
        revision=revision,
        parent_commit=base_revision,
        commit_message=f"Upload BlackjackGPT release {release_id}",
    )
    commit_sha = commit.oid
    if not isinstance(commit_sha, str) or not commit_sha:
        raise RuntimeError("Hugging Face upload returned no commit SHA")
    return {
        "passed": True,
        "repo_id": repo_id,
        "release_id": release_id,
        "path_in_repo": prefix,
        "requested_revision": revision,
        "base_revision": base_revision,
        "commit_sha": commit_sha,
        "dataset_manifest_sha256": manifest_sha256,
    }


def _split_records(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw_splits = manifest.get("splits")
    records: dict[str, Mapping[str, Any]] = {}
    if isinstance(raw_splits, Mapping):
        return {
            name: value
            for name, value in raw_splits.items()
            if isinstance(name, str) and isinstance(value, Mapping)
        }
    if isinstance(raw_splits, list):
        for value in raw_splits:
            if not isinstance(value, Mapping):
                continue
            name = value.get("name", value.get("split"))
            if isinstance(name, str):
                records[name] = value
        return records
    raise ValueError("dataset manifest has invalid splits metadata")


def _shard_names(split_record: Mapping[str, Any]) -> list[str]:
    raw_shards = split_record.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("split metadata has invalid shards inventory")
    names: list[str] = []
    for value in raw_shards:
        if isinstance(value, str):
            candidate = PurePosixPath(value).name
        elif isinstance(value, Mapping):
            raw_name = value.get("name", value.get("shard_id", value.get("path")))
            if not isinstance(raw_name, str):
                raise ValueError("shard metadata is missing its name")
            candidate = PurePosixPath(raw_name).name
        else:
            raise ValueError("split metadata contains an invalid shard entry")
        names.append(_checked_component(candidate, _SHARD_PATTERN, "shard name"))
    if len(names) != len(set(names)):
        raise ValueError("split metadata contains duplicate shard names")
    return names


def _artifact_entries(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = manifest.get("artifacts")
    entries: dict[str, Mapping[str, Any]] = {}
    if isinstance(raw, Mapping):
        return {
            path: metadata
            for path, metadata in raw.items()
            if isinstance(path, str) and isinstance(metadata, Mapping)
        }
    if isinstance(raw, list):
        for metadata in raw:
            if not isinstance(metadata, Mapping) or not isinstance(
                metadata.get("path"), str
            ):
                raise ValueError("dataset manifest contains an invalid artifact entry")
            entries[cast(str, metadata["path"])] = metadata
        return entries
    raise ValueError("dataset manifest has invalid artifacts metadata")


def _validate_artifact_path(path: str) -> PurePosixPath:
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        raise ValueError(f"unsafe artifact path in dataset manifest: {path!r}")
    return candidate


def _verify_manifest_artifact(
    release_root: Path,
    manifest: Mapping[str, Any],
    relative_path: str,
) -> None:
    relative = _validate_artifact_path(relative_path)
    entries = _artifact_entries(manifest)
    entry = entries.get(relative_path)
    if entry is None:
        release_id = str(manifest["release_id"])
        prefixed = f"{_release_prefix(release_id)}/{relative_path}"
        entry = entries.get(prefixed)
    if entry is None:
        raise ValueError(f"dataset manifest does not checksum {relative_path}")
    expected_sha = entry.get("sha256")
    expected_bytes = entry.get("bytes")
    if not isinstance(expected_sha, str) or not isinstance(expected_bytes, int):
        raise ValueError(f"invalid checksum metadata for {relative_path}")
    path = release_root.joinpath(*relative.parts)
    if not path.is_file():
        raise FileNotFoundError(f"download is missing {relative_path}")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"downloaded artifact size mismatch: {relative_path}")
    if _sha256(path) != expected_sha:
        raise ValueError(f"downloaded artifact checksum mismatch: {relative_path}")


def _relative_artifact_paths(manifest: Mapping[str, Any]) -> list[str]:
    prefix = f"{_release_prefix(str(manifest['release_id']))}/"
    paths: list[str] = []
    for raw_path in _artifact_entries(manifest):
        path = raw_path.removeprefix(prefix)
        _validate_artifact_path(path)
        paths.append(path)
    return paths


def _verify_downloaded_selection(
    release_root: Path,
    *,
    expected_release_id: str,
    split: str | None,
    shard: str | None,
    expected_manifest_sha256: str | None,
) -> dict[str, Any]:
    manifest = _load_completed_manifest(release_root)
    if manifest.get("release_id") != expected_release_id:
        raise ValueError("downloaded release ID does not match the requested release")
    manifest_path = release_root / DATASET_MANIFEST_NAME
    actual_manifest_sha256 = _sha256(manifest_path)
    if (
        expected_manifest_sha256 is not None
        and actual_manifest_sha256 != expected_manifest_sha256.lower()
    ):
        raise ValueError(
            "downloaded dataset manifest checksum does not match expectation"
        )

    if split is None:
        result = _verify_full_release(release_root)
        return {
            **result,
            "dataset_manifest_sha256": actual_manifest_sha256,
            "verified_shards": result.get("verified_shards", result.get("shard_count")),
        }

    split_records = _split_records(manifest)
    if split not in split_records:
        raise ValueError(f"split is not listed in dataset manifest: {split}")
    expected_names = _shard_names(split_records[split])
    selected_names = expected_names if shard is None else [shard]
    if shard is not None and shard not in expected_names:
        raise ValueError(f"shard is not listed in split {split}: {shard}")

    from .streaming import verify_streaming_shard

    if shard is None:
        shards_root = release_root / "splits" / split / "shards"
        actual_names = sorted(
            path.name for path in shards_root.iterdir() if path.is_dir()
        )
        if actual_names != sorted(expected_names):
            raise ValueError(
                f"downloaded split shard inventory mismatch: expected "
                f"{sorted(expected_names)}, found {actual_names}"
            )
        split_manifest = f"splits/{split}/split_manifest.json"
        if (release_root / split_manifest).is_file():
            _verify_manifest_artifact(release_root, manifest, split_manifest)

    for name in selected_names:
        relative_manifest = f"splits/{split}/shards/{name}/{SHARD_MANIFEST_NAME}"
        _verify_manifest_artifact(release_root, manifest, relative_manifest)
        shard_root = release_root / "splits" / split / "shards" / name
        verification = verify_streaming_shard(shard_root, replay=False)
        if not verification.get("passed", False):
            errors = verification.get("errors", ["shard verification failed"])
            raise ValueError(
                f"downloaded shard failed verification ({name}): "
                + "; ".join(map(str, errors))
            )
    selected_prefix = f"splits/{split}/"
    for relative_path in _relative_artifact_paths(manifest):
        if not relative_path.startswith(selected_prefix):
            continue
        if shard is not None and not relative_path.startswith(
            f"splits/{split}/shards/{shard}/"
        ):
            continue
        if (release_root / relative_path).is_file():
            _verify_manifest_artifact(release_root, manifest, relative_path)
    return {
        "passed": True,
        "release_id": manifest["release_id"],
        "split": split,
        "shard": shard,
        "verified_shards": len(selected_names),
        "dataset_manifest_sha256": actual_manifest_sha256,
    }


def _download_patterns(
    release_id: str,
    *,
    split: str | None,
    shard: str | None,
) -> list[str]:
    prefix = _release_prefix(release_id)
    if shard is not None and split is None:
        raise ValueError("--shard requires --split")
    if split is None:
        return [f"{prefix}/**"]
    checked_split = _checked_component(split, _SPLIT_PATTERN, "split")
    patterns = [f"{prefix}/{DATASET_MANIFEST_NAME}"]
    if shard is None:
        patterns.append(f"{prefix}/splits/{checked_split}/**")
    else:
        checked_shard = _checked_component(shard, _SHARD_PATTERN, "shard name")
        patterns.append(f"{prefix}/splits/{checked_split}/shards/{checked_shard}/**")
    return patterns


def download_release_from_hub(
    destination: str | Path,
    *,
    release_id: str,
    repo_id: str | None = None,
    revision: str = "main",
    split: str | None = None,
    shard: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    repo_id = _resolve_repo_id(repo_id)
    patterns = _download_patterns(release_id, split=split, shard=shard)
    if expected_manifest_sha256 is not None and not _SHA256_PATTERN.fullmatch(
        expected_manifest_sha256
    ):
        raise ValueError(
            "expected manifest SHA-256 must be exactly 64 hexadecimal digits"
        )
    destination_path = Path(destination).expanduser().resolve()
    if destination_path.exists():
        raise FileExistsError(
            f"download destination already exists: {destination_path}"
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    api_type, snapshot_download = _load_huggingface_hub()
    api = api_type()
    resolved_revision = _require_private_repo(
        api,
        repo_id=repo_id,
        revision=revision,
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.name}.download-",
            dir=destination_path.parent,
        )
    )
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=resolved_revision,
            allow_patterns=patterns,
            local_dir=str(staging),
        )
        release_root = staging / _release_prefix(release_id)
        verification = _verify_downloaded_selection(
            release_root,
            expected_release_id=release_id,
            split=split,
            shard=shard,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        os.replace(staging, destination_path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        **verification,
        "repo_id": repo_id,
        "destination": str(destination_path),
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "allow_patterns": patterns,
    }

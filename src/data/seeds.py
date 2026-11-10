from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

SEED_DERIVATION_VERSION = "hmac-sha256-v1"
SEED_DOMAIN_SEPARATOR = b"blackjackgpt/shoe-seed/hmac-sha256-v1\x00"

_UINT32_LIMIT = 1 << 32
_UINT63_LIMIT = 1 << 63
_UINT64_LIMIT = 1 << 64


@dataclass(frozen=True, slots=True)
class DerivedShoeSeed:
    digest: bytes
    pcg_seed: int
    pcg_stream: int

    def __post_init__(self) -> None:
        if not isinstance(self.digest, bytes) or len(self.digest) != 32:
            raise ValueError("digest must contain exactly 32 bytes")
        if isinstance(self.pcg_seed, bool) or not 0 <= self.pcg_seed < _UINT64_LIMIT:
            raise ValueError("PCG seed must be an unsigned 64-bit integer")
        if (
            isinstance(self.pcg_stream, bool)
            or not 0 <= self.pcg_stream < _UINT63_LIMIT
        ):
            raise ValueError("PCG stream must be an unsigned 63-bit integer")

    @property
    def digest_hex(self) -> str:
        return self.digest.hex()

    def to_metadata(self) -> dict[str, object]:
        return {
            "derivation_version": SEED_DERIVATION_VERSION,
            "digest_hex": self.digest_hex,
            "pcg_seed": self.pcg_seed,
            "pcg_stream": self.pcg_stream,
        }


def derive_shoe_seed(
    root_seed: int,
    dataset_version: str,
    shoe_index: int,
    retry_index: int = 0,
) -> DerivedShoeSeed:
    _validate_unsigned_integer("root_seed", root_seed, _UINT64_LIMIT)
    _validate_unsigned_integer("shoe_index", shoe_index, _UINT64_LIMIT)
    _validate_unsigned_integer("retry_index", retry_index, _UINT32_LIMIT)
    if not isinstance(dataset_version, str) or not dataset_version:
        raise ValueError("dataset_version must be a non-empty string")

    version_bytes = dataset_version.encode("utf-8")
    if len(version_bytes) >= _UINT32_LIMIT:
        raise ValueError("dataset_version UTF-8 encoding is too long")

    key = root_seed.to_bytes(8, byteorder="big", signed=False)
    message = b"".join(
        (
            SEED_DOMAIN_SEPARATOR,
            len(version_bytes).to_bytes(4, byteorder="big", signed=False),
            version_bytes,
            shoe_index.to_bytes(8, byteorder="big", signed=False),
            retry_index.to_bytes(4, byteorder="big", signed=False),
        )
    )
    digest = hmac.new(key, message, hashlib.sha256).digest()

    pcg_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    pcg_stream = int.from_bytes(digest[8:16], byteorder="big", signed=False)
    pcg_stream &= _UINT63_LIMIT - 1
    return DerivedShoeSeed(
        digest=digest,
        pcg_seed=pcg_seed,
        pcg_stream=pcg_stream,
    )


def _validate_unsigned_integer(name: str, value: int, upper_bound: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value < upper_bound:
        bit_width = upper_bound.bit_length() - 1
        raise ValueError(f"{name} must be an unsigned {bit_width}-bit integer")

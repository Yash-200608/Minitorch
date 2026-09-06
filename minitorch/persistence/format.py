from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Literal

MAGIC = b"MINITORC"  # 8-byte MiniTorch artifact identifier
FORMAT_VERSION = 1
FRAMEWORK_VERSION = "0.1.0"

ARTIFACT_MODEL = "model"
ARTIFACT_COMPRESSED_MODEL = "compressed_model"
ARTIFACT_CHECKPOINT = "checkpoint"

ArtifactType = Literal["model", "compressed_model", "checkpoint"]

_HEADER_SIZE = 32
_ARTIFACT_TYPE_CODES = {
    ARTIFACT_MODEL: 1,
    ARTIFACT_COMPRESSED_MODEL: 2,
    ARTIFACT_CHECKPOINT: 3,
}
_ARTIFACT_TYPE_NAMES = {code: name for name, code in _ARTIFACT_TYPE_CODES.items()}


@dataclass(frozen=True)
class ArtifactHeader:
    format_version: int
    artifact_type: ArtifactType
    metadata_length: int
    payload_crc32: int
    payload_length: int


class ArtifactError(Exception):
    """Raised when a MiniTorch artifact cannot be parsed safely."""


def encode_header(
    artifact_type: ArtifactType,
    metadata_length: int,
    payload_crc32: int,
    payload_length: int,
) -> bytes:
    if FORMAT_VERSION > 0xFFFF:
        raise ValueError("FORMAT_VERSION exceeds uint16 range.")
    type_code = _ARTIFACT_TYPE_CODES[artifact_type]
    return struct.pack(
        "<8sHBBIIQ4s",
        MAGIC,
        FORMAT_VERSION,
        type_code,
        0,
        metadata_length,
        payload_crc32 & 0xFFFFFFFF,
        payload_length,
        b"\0\0\0\0",
    )


def decode_header(data: bytes) -> ArtifactHeader:
    if len(data) < _HEADER_SIZE:
        raise ArtifactError("Truncated MiniTorch artifact header.")

    magic, version, type_code, _, meta_len, crc32, payload_len, _ = struct.unpack(
        "<8sHBBIIQ4s",
        data[:_HEADER_SIZE],
    )
    if magic != MAGIC:
        raise ArtifactError("Invalid MiniTorch artifact magic.")
    if version != FORMAT_VERSION:
        raise ArtifactError(f"Unsupported MiniTorch format version: {version}.")
    if type_code not in _ARTIFACT_TYPE_NAMES:
        raise ArtifactError(f"Unknown MiniTorch artifact type code: {type_code}.")

    return ArtifactHeader(
        format_version=version,
        artifact_type=_ARTIFACT_TYPE_NAMES[type_code],
        metadata_length=meta_len,
        payload_crc32=crc32,
        payload_length=payload_len,
    )


def dumps_metadata(metadata: dict[str, Any]) -> bytes:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")


def loads_metadata(data: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Invalid MiniTorch artifact metadata: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError("MiniTorch artifact metadata must be a JSON object.")
    return payload


def checksum_payload(payload: bytes) -> int:
    return zlib.crc32(payload) & 0xFFFFFFFF

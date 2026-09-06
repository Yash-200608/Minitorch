from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..compression import CompressedModel, CompressionConfig, ParameterCompressionRecord
from ..nn import Module, Parameter
from ..quantization import CodebookFitStats, QuantizedTensor
from ..rotation import build_rotation_transform
from .format import (
    ARTIFACT_COMPRESSED_MODEL,
    ARTIFACT_CHECKPOINT,
    ARTIFACT_MODEL,
    FRAMEWORK_VERSION,
    ArtifactError,
    checksum_payload,
    decode_header,
    dumps_metadata,
    encode_header,
    loads_metadata,
)
from .module_tree import build_module_from_tree, serialize_module_tree


def save_model(model: Module, path: str | Path) -> None:
    """Serialize a model's structure and parameter values to a MiniTorch artifact."""
    path = Path(path)
    metadata, payload = _serialize_model(model)
    _write_artifact(path, ARTIFACT_MODEL, metadata, payload)


def load_model(path: str | Path) -> Module:
    """Load a model from a MiniTorch artifact."""
    metadata, payload = _read_artifact(path, ARTIFACT_MODEL)
    return _deserialize_model(metadata, payload)


def save_compressed_model(compressed: CompressedModel, path: str | Path) -> None:
    """Serialize a compressed model without re-running compression."""
    path = Path(path)
    metadata, payload = _serialize_compressed_model(compressed)
    _write_artifact(path, ARTIFACT_COMPRESSED_MODEL, metadata, payload)


def load_compressed_model(path: str | Path) -> CompressedModel:
    """Load a compressed model from a MiniTorch artifact."""
    metadata, payload = _read_artifact(path, ARTIFACT_COMPRESSED_MODEL)
    return _deserialize_compressed_model(metadata, payload)


def inspect_artifact(path: str | Path) -> dict[str, Any]:
    """Inspect artifact header and metadata without reconstructing a model."""
    path = Path(path)
    raw = path.read_bytes()
    header = decode_header(raw)
    meta_start = 32
    meta_end = meta_start + header.metadata_length
    if len(raw) < meta_end:
        raise ArtifactError("Truncated MiniTorch artifact metadata.")
    metadata = loads_metadata(raw[meta_start:meta_end])
    payload = raw[meta_end:meta_end + header.payload_length]
    if len(payload) != header.payload_length:
        raise ArtifactError("Truncated MiniTorch artifact payload.")

    return {
        "path": str(path),
        "file_bytes": len(raw),
        "format_version": header.format_version,
        "artifact_type": header.artifact_type,
        "framework_version": metadata.get("framework_version"),
        "metadata": metadata,
        "payload_bytes": len(payload),
        "payload_crc32": header.payload_crc32,
    }


def format_artifact_report(path: str | Path) -> str:
    info = inspect_artifact(path)
    metadata = info["metadata"]
    lines = [
        "MiniTorch Artifact",
        "------------------",
        "",
        f"Path: {info['path']}",
        f"Type: {info['artifact_type']}",
        f"Format version: {info['format_version']}",
        f"Framework: {info['framework_version']}",
        f"File size: {info['file_bytes']} bytes",
        f"Payload size: {info['payload_bytes']} bytes",
        "",
    ]

    if info["artifact_type"] == ARTIFACT_MODEL:
        lines.append(f"Parameters: {len(metadata.get('parameters', []))}")
        lines.append(f"Module type: {metadata.get('module_tree', {}).get('type')}")
    elif info["artifact_type"] == ARTIFACT_CHECKPOINT:
        from .checkpoint import format_checkpoint_report

        return format_checkpoint_report(info["path"], metadata, info["payload_bytes"])
    else:
        config = metadata.get("compression_config", {})
        lines.extend(
            [
                f"Parameters: {len(metadata.get('parameter_names', []))}",
                "Quantization:",
                f"  {config.get('bits')}-bit",
                f"  {config.get('method')}",
                f"  packed: {config.get('packed')}",
                f"  rotation: {config.get('rotation')}",
            ]
        )

    return "\n".join(lines)


def benchmark_persistence(
    model: Module,
    compressed: CompressedModel | None = None,
    *,
    repeats: int = 3,
) -> dict[str, Any]:
    """Measure save/load timings and artifact file sizes."""
    import tempfile

    results: dict[str, Any] = {"repeats": repeats}

    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "model.mt"
        start = time.perf_counter()
        for _ in range(repeats):
            save_model(model, model_path)
        results["model_save_seconds"] = (time.perf_counter() - start) / repeats

        start = time.perf_counter()
        for _ in range(repeats):
            _ = load_model(model_path)
        results["model_load_seconds"] = (time.perf_counter() - start) / repeats
        results["model_file_bytes"] = model_path.stat().st_size

        if compressed is not None:
            compressed_path = Path(tmp) / "model.mtc"
            start = time.perf_counter()
            for _ in range(repeats):
                save_compressed_model(compressed, compressed_path)
            results["compressed_save_seconds"] = (time.perf_counter() - start) / repeats

            start = time.perf_counter()
            for _ in range(repeats):
                _ = load_compressed_model(compressed_path)
            results["compressed_load_seconds"] = (time.perf_counter() - start) / repeats
            results["compressed_file_bytes"] = compressed_path.stat().st_size
            if results["model_file_bytes"] > 0:
                results["file_size_ratio"] = (
                    results["model_file_bytes"] / results["compressed_file_bytes"]
                )

    return results


def _write_artifact(
    path: Path,
    artifact_type: str,
    metadata: dict[str, Any],
    payload: bytes,
) -> None:
    meta_bytes = dumps_metadata(metadata)
    crc = checksum_payload(payload)
    header = encode_header(artifact_type, len(meta_bytes), crc, len(payload))
    path.write_bytes(header + meta_bytes + payload)


def _read_artifact(path: Path, expected_type: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if not raw:
        raise ArtifactError("Empty MiniTorch artifact file.")

    header = decode_header(raw)
    if header.artifact_type != expected_type:
        raise ArtifactError(
            f"Expected artifact type {expected_type!r}, got {header.artifact_type!r}."
        )

    meta_start = 32
    meta_end = meta_start + header.metadata_length
    if len(raw) < meta_end:
        raise ArtifactError("Truncated MiniTorch artifact metadata.")

    metadata = loads_metadata(raw[meta_start:meta_end])
    payload = raw[meta_end:meta_end + header.payload_length]
    if len(payload) != header.payload_length:
        raise ArtifactError("Truncated MiniTorch artifact payload.")

    if checksum_payload(payload) != header.payload_crc32:
        raise ArtifactError("MiniTorch artifact payload checksum mismatch.")

    return metadata, payload


def _serialize_model(model: Module) -> tuple[dict[str, Any], bytes]:
    parameter_specs: list[dict[str, Any]] = []
    payload_parts: list[bytes] = []
    offset = 0

    for name, parameter in model.named_parameters():
        array = np.asarray(parameter.data)
        raw = array.tobytes(order="C")
        parameter_specs.append(
            {
                "name": name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "requires_grad": bool(parameter.requires_grad),
                "offset": offset,
                "length": len(raw),
            }
        )
        payload_parts.append(raw)
        offset += len(raw)

    metadata = {
        "framework_version": FRAMEWORK_VERSION,
        "module_tree": serialize_module_tree(model),
        "parameters": parameter_specs,
        "gradients_saved": False,
    }
    return metadata, b"".join(payload_parts)


def _deserialize_model(metadata: dict[str, Any], payload: bytes) -> Module:
    module_tree = metadata.get("module_tree")
    if not isinstance(module_tree, dict):
        raise ArtifactError("MiniTorch model artifact missing module_tree.")

    model = build_module_from_tree(module_tree)
    parameter_specs = metadata.get("parameters", [])
    if not isinstance(parameter_specs, list):
        raise ArtifactError("MiniTorch model artifact parameters must be a list.")

    for spec in parameter_specs:
        name = spec["name"]
        shape = tuple(int(dim) for dim in spec["shape"])
        dtype = np.dtype(spec["dtype"])
        offset = int(spec["offset"])
        length = int(spec["length"])
        end = offset + length
        if end > len(payload):
            raise ArtifactError(f"Truncated tensor payload for parameter {name!r}.")

        expected = int(np.prod(shape)) * dtype.itemsize
        if length != expected:
            raise ArtifactError(f"Invalid parameter payload length for {name!r}.")

        array = np.frombuffer(payload[offset:end], dtype=dtype).reshape(shape).copy()
        parameter = model.parameter(name)
        parameter.data = array
        parameter.requires_grad = bool(spec.get("requires_grad", True))
        parameter.grad = None

    return model


def _serialize_compressed_model(compressed: CompressedModel) -> tuple[dict[str, Any], bytes]:
    parameter_entries: list[dict[str, Any]] = []
    payload_parts: list[bytes] = []
    offset = 0

    for name in compressed.source_parameter_names:
        record = compressed.parameter_records[name]
        quantized = compressed.quantized_parameters[name]
        entry: dict[str, Any] = {
            "name": name,
            "shape": list(record.shape),
            "original_dtype": record.original_dtype,
            "bits": record.bits,
            "method": record.method,
            "packed": record.packed,
            "rotation": record.rotation,
            "minimum": float(quantized.minimum),
            "maximum": float(quantized.maximum),
            "scale": float(quantized.scale),
            "code_count": int(quantized.code_count),
        }

        if quantized.codebook_fit is not None:
            entry["codebook_fit"] = {
                "iterations": quantized.codebook_fit.iterations,
                "converged": quantized.codebook_fit.converged,
                "final_centroid_movement": quantized.codebook_fit.final_centroid_movement,
                "objective": quantized.codebook_fit.objective,
            }

        if quantized.rotation_transform is not None:
            meta = quantized.rotation_transform.metadata
            entry["rotation_metadata"] = {
                "dimension": meta.dimension,
                "permutation": list(meta.permutation),
                "signs": list(meta.signs),
                "use_hadamard": meta.use_hadamard,
                "transform_type": meta.transform_type,
            }

        if quantized.centroids is not None:
            centroid_bytes = np.asarray(quantized.centroids, dtype=np.float64).tobytes(order="C")
            entry["centroids_offset"] = offset
            entry["centroids_length"] = len(centroid_bytes)
            payload_parts.append(centroid_bytes)
            offset += len(centroid_bytes)

        if quantized.packed and quantized.packed_bytes is not None:
            packed_bytes = np.asarray(quantized.packed_bytes, dtype=np.uint8).tobytes(order="C")
            entry["packed_offset"] = offset
            entry["packed_length"] = len(packed_bytes)
            payload_parts.append(packed_bytes)
            offset += len(packed_bytes)
        elif quantized._codes is not None:
            code_bytes = np.asarray(quantized._codes, dtype=np.int32).tobytes(order="C")
            entry["codes_offset"] = offset
            entry["codes_length"] = len(code_bytes)
            payload_parts.append(code_bytes)
            offset += len(code_bytes)
        else:
            raise ArtifactError(f"Compressed parameter {name!r} has no code storage.")

        parameter_entries.append(entry)

    metadata = {
        "framework_version": FRAMEWORK_VERSION,
        "compression_config": {
            "bits": compressed.config.bits,
            "method": compressed.config.method,
            "packed": compressed.config.packed,
            "rotation": compressed.config.rotation,
            "max_iterations": compressed.config.max_iterations,
            "tolerance": compressed.config.tolerance,
        },
        "parameter_names": list(compressed.source_parameter_names),
        "parameters": parameter_entries,
    }
    return metadata, b"".join(payload_parts)


def _deserialize_compressed_model(metadata: dict[str, Any], payload: bytes) -> CompressedModel:
    config_data = metadata.get("compression_config")
    if not isinstance(config_data, dict):
        raise ArtifactError("Compressed artifact missing compression_config.")

    config = CompressionConfig(
        bits=int(config_data["bits"]),
        method=config_data["method"],
        packed=bool(config_data["packed"]),
        rotation=config_data.get("rotation"),
        max_iterations=int(config_data.get("max_iterations", 100)),
        tolerance=float(config_data.get("tolerance", 1e-6)),
    )

    names = metadata.get("parameter_names", [])
    entries = metadata.get("parameters", [])
    if not isinstance(names, list) or not isinstance(entries, list):
        raise ArtifactError("Compressed artifact parameter metadata is invalid.")

    records: dict[str, ParameterCompressionRecord] = {}
    quantized: dict[str, QuantizedTensor] = {}

    for entry in entries:
        name = entry["name"]
        shape = tuple(int(dim) for dim in entry["shape"])
        original_dtype = np.dtype(entry["original_dtype"])

        rotation_transform = None
        rotation_meta = entry.get("rotation_metadata")
        if rotation_meta is not None:
            rotation_transform = _rotation_from_metadata(rotation_meta)

        centroids = None
        if "centroids_offset" in entry:
            start = int(entry["centroids_offset"])
            end = start + int(entry["centroids_length"])
            centroids = np.frombuffer(payload[start:end], dtype=np.float64).copy()

        codebook_fit = None
        fit_data = entry.get("codebook_fit")
        if fit_data is not None:
            codebook_fit = CodebookFitStats(
                iterations=int(fit_data["iterations"]),
                converged=bool(fit_data["converged"]),
                final_centroid_movement=float(fit_data["final_centroid_movement"]),
                objective=float(fit_data["objective"]),
            )

        packed = bool(entry["packed"])
        packed_bytes = None
        codes = None
        if packed:
            start = int(entry["packed_offset"])
            end = start + int(entry["packed_length"])
            packed_bytes = np.frombuffer(payload[start:end], dtype=np.uint8).copy()
        else:
            start = int(entry["codes_offset"])
            end = start + int(entry["codes_length"])
            codes = np.frombuffer(payload[start:end], dtype=np.int32).reshape(shape).copy()

        placeholder = np.zeros(shape, dtype=np.float64)
        quantized_tensor = QuantizedTensor(
            bits=int(entry["bits"]),
            shape=shape,
            original_dtype=original_dtype,
            source_data=placeholder,
            method=entry["method"],
            minimum=float(entry["minimum"]),
            maximum=float(entry["maximum"]),
            scale=float(entry["scale"]),
            centroids=centroids,
            codebook_fit=codebook_fit,
            codes=codes,
            packed=packed,
            packed_bytes=packed_bytes,
            code_count=int(entry["code_count"]),
            rotation_transform=rotation_transform,
        )
        quantized_tensor._source_data = quantized_tensor.dequantize().data.copy()

        records[name] = ParameterCompressionRecord(
            name=name,
            shape=shape,
            original_dtype=str(original_dtype),
            original_bytes=int(placeholder.nbytes),
            compressed_bytes=int(
                quantized_tensor.report().memory.total_actual_representation_bytes
            ),
            packed_code_bytes=int(entry.get("packed_length", 0) if packed else 0),
            codebook_bytes=int(centroids.nbytes if centroids is not None else 0),
            bits=int(entry["bits"]),
            method=entry["method"],
            packed=packed,
            rotation=entry.get("rotation"),
            max_absolute_error=0.0,
            mean_absolute_error=0.0,
            rmse=0.0,
        )
        quantized[name] = quantized_tensor

    return CompressedModel(
        config=config,
        parameter_records=records,
        quantized_parameters=quantized,
        source_parameter_names=tuple(names),
    )


def _rotation_from_metadata(meta: dict[str, Any]) -> Any:
    dimension = int(meta["dimension"])
    kind = meta.get("transform_type", "combined")
    transform = build_rotation_transform(dimension, kind=kind)
    expected_perm = tuple(int(value) for value in transform.metadata.permutation)
    stored_perm = tuple(int(value) for value in meta.get("permutation", []))
    if stored_perm and stored_perm != expected_perm:
        raise ArtifactError("Rotation permutation metadata does not match deterministic transform.")
    return transform

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .bitpacking import pack_codes, unpack_codes
from .kernels.backend import DEFAULT_CODEBOOK_BACKEND, NumericalBackend, resolve_backend
from .kernels.codebook import nearest_centroid_codes
from .kernels.uniform import uniform_codes
from .parallel import ParallelConfig
from .tensor import Tensor

SUPPORTED_BIT_WIDTHS: tuple[int, ...] = (2, 4, 8)
QuantizationMethod = Literal["uniform", "codebook"]


@dataclass(frozen=True)
class QuantizationErrorMetrics:
    max_absolute_error: float
    mean_absolute_error: float
    rmse: float


@dataclass(frozen=True)
class QuantizationMemoryStats:
    original_dtype: str
    original_bits_per_value: int
    original_bytes: int
    quantized_bits_logical: int
    levels: int
    theoretical_packed_bytes: int
    actual_code_storage_bytes: int
    codebook_bytes: int
    total_theoretical_representation_bytes: int
    compression_ratio_theoretical: float
    storage_mode: str = "logical"
    actual_packed_bytes: int = 0
    logical_int32_storage_bytes: int = 0
    total_actual_representation_bytes: int = 0
    compression_ratio_actual: float = 0.0


@dataclass(frozen=True)
class CodebookFitStats:
    iterations: int
    converged: bool
    final_centroid_movement: float
    objective: float


@dataclass(frozen=True)
class QuantizationReport:
    shape: tuple[int, ...]
    bits: int
    minimum: float
    maximum: float
    memory: QuantizationMemoryStats
    error: QuantizationErrorMetrics
    method: str = "uniform"
    codebook_fit: CodebookFitStats | None = None
    centroids: tuple[float, ...] | None = None
    storage_mode: str = "logical"
    code_count: int = 0

    def format(self) -> str:
        lines = [
            "Quantization Report",
            "-------------------",
            "",
            f"Method: {self.method}",
            f"Storage mode: {self.storage_mode}",
            f"Shape: {self.shape}",
            f"Codes: {self.code_count}",
            "",
            "Original:",
            f"  dtype: {self.memory.original_dtype}",
            f"  bits/value: {self.memory.original_bits_per_value}",
            f"  memory: {self.memory.original_bytes} bytes",
            "",
            "Quantized:",
            f"  bits (logical): {self.memory.quantized_bits_logical}",
            f"  levels: {self.memory.levels}",
            f"  theoretical packed code memory: {self.memory.theoretical_packed_bytes} bytes",
            f"  logical int32 code storage: {self.memory.logical_int32_storage_bytes} bytes",
        ]

        if self.storage_mode == "packed":
            lines.append(f"  actual packed bytes: {self.memory.actual_packed_bytes} bytes")
        else:
            lines.append(f"  actual code storage: {self.memory.actual_code_storage_bytes} bytes")

        if self.method == "codebook":
            lines.extend(
                [
                    f"  codebook storage: {self.memory.codebook_bytes} bytes",
                    f"  total theoretical representation: "
                    f"{self.memory.total_theoretical_representation_bytes} bytes",
                    f"  total actual representation: "
                    f"{self.memory.total_actual_representation_bytes} bytes",
                ]
            )
        elif self.storage_mode == "packed":
            lines.append(
                f"  total actual representation: "
                f"{self.memory.total_actual_representation_bytes} bytes"
            )

        lines.extend(
            [
                "",
                "Range:",
                f"  min: {self.minimum}",
                f"  max: {self.maximum}",
                "",
                "Compression:",
                f"  theoretical (codes only): {self.memory.compression_ratio_theoretical:.2f}x",
                f"  actual representation: {self.memory.compression_ratio_actual:.2f}x",
                "",
                "Error (original vs reconstructed):",
                f"  max absolute error: {self.error.max_absolute_error}",
                f"  mean absolute error: {self.error.mean_absolute_error}",
                f"  RMSE: {self.error.rmse}",
            ]
        )

        if self.codebook_fit is not None:
            lines.extend(
                [
                    "",
                    "Codebook fit:",
                    f"  iterations: {self.codebook_fit.iterations}",
                    f"  converged: {self.codebook_fit.converged}",
                    f"  final centroid movement: {self.codebook_fit.final_centroid_movement}",
                    f"  mean squared error: {self.codebook_fit.objective}",
                ]
            )

        if self.centroids is not None:
            lines.extend(["", "Codebook", "--------"])
            for index, centroid in enumerate(self.centroids):
                lines.append(f"{index} -> {centroid}")

        lines.extend(
            [
                "",
                "Note: packed mode stores codes in uint8 bytes using bit masks and shifts.",
            ]
        )
        return "\n".join(lines)


class Codebook:
    """One-dimensional scalar codebook learned by Lloyd-style centroid iteration."""

    def __init__(
        self,
        centroids: np.ndarray,
        fit_stats: CodebookFitStats | None = None,
    ) -> None:
        self.centroids = np.asarray(centroids, dtype=np.float64)
        self.fit_stats = fit_stats

    @property
    def levels(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def dtype(self) -> np.dtype:
        return self.centroids.dtype

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        levels: int,
        *,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> Codebook:
        flat = _validate_quantization_input(values).ravel()
        if levels <= 0:
            raise ValueError("levels must be positive.")

        minimum = float(np.min(flat))
        maximum = float(np.max(flat))

        if minimum == maximum:
            centroids = np.full(levels, minimum, dtype=np.float64)
            stats = CodebookFitStats(
                iterations=0,
                converged=True,
                final_centroid_movement=0.0,
                objective=0.0,
            )
            return Codebook(centroids, stats)

        centroids = np.linspace(minimum, maximum, levels, dtype=np.float64)
        converged = False
        movement = 0.0

        for iteration in range(max_iterations):
            assignments = _nearest_centroid_codes(flat, centroids)
            new_centroids = centroids.copy()
            for index in range(levels):
                cluster_values = flat[assignments == index]
                if cluster_values.size > 0:
                    new_centroids[index] = float(np.mean(cluster_values))
                # Empty clusters keep their previous centroid.

            movement = float(np.max(np.abs(new_centroids - centroids)))
            centroids = new_centroids
            if movement < tolerance:
                converged = True
                iterations_used = iteration + 1
                break
        else:
            iterations_used = max_iterations
            converged = movement < tolerance

        assignments = _nearest_centroid_codes(flat, centroids)
        objective = float(np.mean((flat - centroids[assignments]) ** 2))
        stats = CodebookFitStats(
            iterations=iterations_used,
            converged=converged,
            final_centroid_movement=movement,
            objective=objective,
        )
        return Codebook(centroids, stats)

    def encode(
        self,
        values: np.ndarray,
        *,
        parallel_config: ParallelConfig | None = None,
        parallel: bool = False,
        workers: int | None = None,
        backend: NumericalBackend | None = None,
        encode_chunk_size: int | None = None,
    ) -> np.ndarray:
        flat = _validate_quantization_input(values).ravel()
        encode_backend = resolve_backend(backend, DEFAULT_CODEBOOK_BACKEND)
        config = _resolve_parallel_config(parallel_config, parallel, workers)
        if not config.enabled:
            codes = nearest_centroid_codes(
                flat,
                self.centroids,
                backend=encode_backend,
                chunk_size=encode_chunk_size,
            )
            return codes.astype(np.int32)

        from .parallel import execute_chunked

        chunks = execute_chunked(
            int(flat.size),
            config,
            lambda start, end: nearest_centroid_codes(
                flat[start:end],
                self.centroids,
                backend=encode_backend,
                chunk_size=encode_chunk_size,
            ),
        )
        return np.concatenate(chunks).astype(np.int32)

    def decode(self, codes: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        code_array = np.asarray(codes, dtype=np.int32).ravel()
        if np.any(code_array < 0) or np.any(code_array >= self.levels):
            raise ValueError("Code values are outside the codebook range.")
        reconstructed = self.centroids[code_array]
        return reconstructed.reshape(shape).astype(np.float64)

    def format_centroids(self) -> str:
        lines = ["Codebook", "--------"]
        for index, centroid in enumerate(self.centroids):
            lines.append(f"{index} -> {centroid}")
        return "\n".join(lines)


class QuantizedTensor:
    """
    Scalar quantization codes with metadata for reconstruction.

    Supports uniform mapping and learned codebook modes. Does not participate in
    autograd. Dequantize explicitly to continue differentiable computation.
    """

    def __init__(
        self,
        bits: int,
        shape: tuple[int, ...],
        original_dtype: np.dtype,
        source_data: np.ndarray,
        method: QuantizationMethod = "uniform",
        minimum: float = 0.0,
        maximum: float = 0.0,
        scale: float = 0.0,
        centroids: np.ndarray | None = None,
        codebook_fit: CodebookFitStats | None = None,
        codes: np.ndarray | None = None,
        packed: bool = False,
        packed_bytes: np.ndarray | None = None,
        code_count: int | None = None,
        rotation_transform: Any | None = None,
    ) -> None:
        self.bits = bits
        self.shape = shape
        self.original_dtype = original_dtype
        self._source_data = source_data
        self.method = method
        self.minimum = minimum
        self.maximum = maximum
        self.scale = scale
        self.centroids = (
            np.asarray(centroids, dtype=np.float64) if centroids is not None else None
        )
        self.codebook_fit = codebook_fit
        self.packed = packed
        self.packed_bytes = (
            np.asarray(packed_bytes, dtype=np.uint8) if packed_bytes is not None else None
        )
        self.code_count = int(code_count if code_count is not None else np.prod(shape))
        self._codes = codes
        self.rotation_transform = rotation_transform

    @property
    def rotation_applied(self) -> bool:
        return self.rotation_transform is not None

    @property
    def codes(self) -> np.ndarray:
        if self._codes is not None:
            return self._codes
        if self.packed and self.packed_bytes is not None:
            unpacked = unpack_codes(self.packed_bytes, self.bits, self.code_count)
            return unpacked.reshape(self.shape)
        raise RuntimeError("QuantizedTensor has no code storage.")

    def dequantize(self) -> Tensor:
        """Reconstruct a normal MiniTorch Tensor (not connected to autograd)."""
        code_array = self.codes
        if self.method == "codebook":
            if self.centroids is None:
                raise RuntimeError("Codebook centroids are missing.")
            codebook = Codebook(self.centroids, self.codebook_fit)
            reconstructed = codebook.decode(code_array, self.shape)
        else:
            reconstructed = _dequantize_uniform_codes(
                codes=code_array,
                minimum=self.minimum,
                maximum=self.maximum,
                scale=self.scale,
                shape=self.shape,
            )
        if self.rotation_transform is not None:
            reconstructed = self.rotation_transform.inverse(reconstructed)
        return Tensor(reconstructed, requires_grad=False)

    def report(self) -> QuantizationReport:
        """Build a report from the stored source and reconstructed values."""
        reconstructed = self.dequantize().data
        code_array = self.codes
        return build_quantization_report(
            source=self._source_data,
            codes=code_array,
            bits=self.bits,
            minimum=self.minimum,
            maximum=self.maximum,
            shape=self.shape,
            reconstructed=reconstructed,
            method=self.method,
            centroids=self.centroids,
            codebook_fit=self.codebook_fit,
            packed=self.packed,
            packed_bytes=self.packed_bytes,
            code_count=self.code_count,
        )


    @property
    def levels(self) -> int:
        return 2 ** self.bits


def quantize_array(
    data: np.ndarray,
    bits: int,
    *,
    method: QuantizationMethod = "uniform",
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    packed: bool = False,
    rotation: str | None = None,
    parallel_config: ParallelConfig | None = None,
    parallel: bool = False,
    workers: int | None = None,
) -> QuantizedTensor:
    """Quantize a NumPy array with uniform or codebook scalar quantization."""
    validate_bits(bits)
    parallel_settings = _resolve_parallel_config(parallel_config, parallel, workers)
    array = _validate_quantization_input(data)
    original = array.copy()
    rotation_transform = None
    work = array

    if rotation is not None:
        from .rotation import build_rotation_transform

        if rotation not in ("hadamard", "combined"):
            raise ValueError(f"Unsupported rotation {rotation!r}.")
        dimension = int(array.shape[-1])
        rotation_transform = build_rotation_transform(dimension, kind=rotation)
        work = rotation_transform.apply(array)

    if method == "uniform":
        quantized = _quantize_uniform(work, bits, parallel_config=parallel_settings)
    elif method == "codebook":
        quantized = _quantize_codebook(
            work,
            bits,
            max_iterations=max_iterations,
            tolerance=tolerance,
            parallel_config=parallel_settings,
        )
    else:
        raise ValueError(f"Unsupported quantization method {method!r}.")

    quantized._source_data = original
    quantized.rotation_transform = rotation_transform
    return _apply_packed_storage(
        quantized,
        packed=packed,
        parallel_config=parallel_settings,
    )


def _resolve_parallel_config(
    parallel_config: ParallelConfig | None,
    parallel: bool,
    workers: int | None,
) -> ParallelConfig:
    if parallel_config is not None:
        return parallel_config
    if parallel:
        return ParallelConfig(enabled=True, workers=workers)
    return ParallelConfig()


def _quantize_uniform(
    data: np.ndarray,
    bits: int,
    *,
    parallel_config: ParallelConfig | None = None,
) -> QuantizedTensor:
    array = _validate_quantization_input(data)
    shape = tuple(array.shape)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    levels = 2 ** bits
    max_code = levels - 1

    if minimum == maximum:
        codes = np.zeros(shape, dtype=np.int32)
        scale = 0.0
    else:
        scale = (maximum - minimum) / max_code
        flat = array.ravel()
        if parallel_config is not None and parallel_config.enabled and flat.size > 0:
            from .parallel import execute_chunked

            chunks = execute_chunked(
                int(flat.size),
                parallel_config,
                lambda start, end: uniform_codes(
                    flat[start:end],
                    minimum=minimum,
                    maximum=maximum,
                    max_code=max_code,
                    backend="vectorized",
                ),
            )
            codes = np.concatenate(chunks).reshape(shape)
        else:
            codes = uniform_codes(
                array,
                minimum=minimum,
                maximum=maximum,
                max_code=max_code,
                backend="vectorized",
            ).reshape(shape)

    return QuantizedTensor(
        bits=bits,
        shape=shape,
        original_dtype=array.dtype,
        source_data=array.copy(),
        method="uniform",
        minimum=minimum,
        maximum=maximum,
        scale=scale,
        codes=codes,
    )


def _quantize_codebook(
    data: np.ndarray,
    bits: int,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    parallel_config: ParallelConfig | None = None,
) -> QuantizedTensor:
    array = _validate_quantization_input(data)
    shape = tuple(array.shape)
    levels = 2 ** bits
    codebook = Codebook.fit(
        array,
        levels,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    codes = codebook.encode(array, parallel_config=parallel_config).reshape(shape)

    return QuantizedTensor(
        bits=bits,
        shape=shape,
        original_dtype=array.dtype,
        source_data=array.copy(),
        method="codebook",
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        centroids=codebook.centroids,
        codebook_fit=codebook.fit_stats,
        codes=codes,
    )


def _apply_packed_storage(
    quantized: QuantizedTensor,
    packed: bool,
    *,
    parallel_config: ParallelConfig | None = None,
) -> QuantizedTensor:
    if not packed:
        return quantized

    flat_codes = quantized.codes.ravel()
    packed_bytes = pack_codes(
        flat_codes,
        quantized.bits,
        parallel_config=parallel_config,
    )
    return QuantizedTensor(
        bits=quantized.bits,
        shape=quantized.shape,
        original_dtype=quantized.original_dtype,
        source_data=quantized._source_data,
        method=quantized.method,
        minimum=quantized.minimum,
        maximum=quantized.maximum,
        scale=quantized.scale,
        centroids=quantized.centroids,
        codebook_fit=quantized.codebook_fit,
        codes=None,
        packed=True,
        packed_bytes=packed_bytes,
        code_count=flat_codes.size,
        rotation_transform=quantized.rotation_transform,
    )


def _validate_quantization_input(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot quantize an empty tensor.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Quantization requires finite input values (no NaN or infinity).")
    return array


def validate_bits(bits: int) -> None:
    if bits not in SUPPORTED_BIT_WIDTHS:
        supported = ", ".join(str(value) for value in SUPPORTED_BIT_WIDTHS)
        raise ValueError(
            f"Unsupported bit width {bits}. Supported values: {supported}."
        )


def _nearest_centroid_codes(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return nearest_centroid_codes(values, centroids, backend="vectorized")


def _dequantize_uniform_codes(
    codes: np.ndarray,
    minimum: float,
    maximum: float,
    scale: float,
    shape: tuple[int, ...],
) -> np.ndarray:
    if minimum == maximum:
        return np.full(shape, minimum, dtype=np.float64)
    reconstructed = minimum + codes.astype(np.float64) * scale
    return reconstructed.reshape(shape)


def compute_error_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> QuantizationErrorMetrics:
    difference = np.asarray(original, dtype=np.float64) - np.asarray(reconstructed, dtype=np.float64)
    abs_error = np.abs(difference)
    return QuantizationErrorMetrics(
        max_absolute_error=float(np.max(abs_error)),
        mean_absolute_error=float(np.mean(abs_error)),
        rmse=float(np.sqrt(np.mean(difference ** 2))),
    )


def compute_memory_stats(
    source: np.ndarray,
    codes: np.ndarray,
    bits: int,
    centroids: np.ndarray | None = None,
    packed: bool = False,
    packed_bytes: np.ndarray | None = None,
    code_count: int | None = None,
) -> QuantizationMemoryStats:
    original_bytes = int(source.nbytes)
    original_bits = int(np.dtype(source.dtype).itemsize * 8)
    num_values = int(code_count if code_count is not None else source.size)
    theoretical_packed_bytes = int(math.ceil(num_values * bits / 8))
    logical_int32_bytes = int(codes.nbytes)
    actual_int32_bytes = logical_int32_bytes
    codebook_bytes = int(centroids.nbytes) if centroids is not None else 0
    total_theoretical = theoretical_packed_bytes + codebook_bytes
    ratio_theoretical = (
        original_bytes / theoretical_packed_bytes if theoretical_packed_bytes > 0 else 0.0
    )

    if packed and packed_bytes is not None:
        actual_packed = int(packed_bytes.nbytes)
        storage_mode = "packed"
        total_actual = actual_packed + codebook_bytes
        compression_actual = original_bytes / total_actual if total_actual > 0 else 0.0
        return QuantizationMemoryStats(
            original_dtype=str(source.dtype),
            original_bits_per_value=original_bits,
            original_bytes=original_bytes,
            quantized_bits_logical=bits,
            levels=2 ** bits,
            theoretical_packed_bytes=theoretical_packed_bytes,
            actual_code_storage_bytes=actual_int32_bytes,
            codebook_bytes=codebook_bytes,
            total_theoretical_representation_bytes=total_theoretical,
            compression_ratio_theoretical=ratio_theoretical,
            storage_mode=storage_mode,
            actual_packed_bytes=actual_packed,
            logical_int32_storage_bytes=logical_int32_bytes,
            total_actual_representation_bytes=total_actual,
            compression_ratio_actual=compression_actual,
        )

    total_actual = actual_int32_bytes + codebook_bytes
    compression_actual = original_bytes / total_actual if total_actual > 0 else 0.0
    return QuantizationMemoryStats(
        original_dtype=str(source.dtype),
        original_bits_per_value=original_bits,
        original_bytes=original_bytes,
        quantized_bits_logical=bits,
        levels=2 ** bits,
        theoretical_packed_bytes=theoretical_packed_bytes,
        actual_code_storage_bytes=actual_int32_bytes,
        codebook_bytes=codebook_bytes,
        total_theoretical_representation_bytes=total_theoretical,
        compression_ratio_theoretical=ratio_theoretical,
        storage_mode="logical",
        actual_packed_bytes=0,
        logical_int32_storage_bytes=logical_int32_bytes,
        total_actual_representation_bytes=total_actual,
        compression_ratio_actual=compression_actual,
    )


def build_quantization_report(
    source: np.ndarray,
    codes: np.ndarray,
    bits: int,
    minimum: float,
    maximum: float,
    shape: tuple[int, ...],
    reconstructed: np.ndarray,
    method: str = "uniform",
    centroids: np.ndarray | None = None,
    codebook_fit: CodebookFitStats | None = None,
    packed: bool = False,
    packed_bytes: np.ndarray | None = None,
    code_count: int | None = None,
) -> QuantizationReport:
    centroid_tuple = tuple(float(value) for value in centroids) if centroids is not None else None
    count = int(code_count if code_count is not None else codes.size)
    memory = compute_memory_stats(
        source,
        codes,
        bits,
        centroids=centroids,
        packed=packed,
        packed_bytes=packed_bytes,
        code_count=count,
    )
    return QuantizationReport(
        shape=shape,
        bits=bits,
        minimum=minimum,
        maximum=maximum,
        memory=memory,
        error=compute_error_metrics(source, reconstructed),
        method=method,
        codebook_fit=codebook_fit,
        centroids=centroid_tuple,
        storage_mode=memory.storage_mode,
        code_count=count,
    )


def benchmark_quantization_bitwidths(
    data: np.ndarray,
    *,
    repeats: int = 3,
) -> list[dict[str, Any]]:
    """
    Compare float storage vs uniform and codebook quantization on local measurements.
    """
    import time

    source = np.asarray(data, dtype=np.float64)
    results: list[dict[str, Any]] = []

    float_start = time.perf_counter()
    for _ in range(repeats):
        _ = source.copy()
    float_runtime = (time.perf_counter() - float_start) / repeats

    results.append(
        {
            "name": "float64_storage",
            "method": "float64",
            "bits": 64,
            "runtime_seconds": float_runtime,
            "original_bytes": int(source.nbytes),
            "theoretical_packed_bytes": int(source.nbytes),
            "actual_code_storage_bytes": int(source.nbytes),
            "codebook_bytes": 0,
            "max_absolute_error": 0.0,
            "rmse": 0.0,
        }
    )

    for bits in SUPPORTED_BIT_WIDTHS:
        for method in ("uniform", "codebook"):
            for packed in (False, True):
                label = "packed" if packed else "logical"
                quantize_start = time.perf_counter()
                for _ in range(repeats):
                    quantized = quantize_array(
                        source,
                        bits,
                        method=method,
                        packed=packed,
                    )
                quantize_runtime = (time.perf_counter() - quantize_start) / repeats

                dequant_start = time.perf_counter()
                for _ in range(repeats):
                    _ = quantized.dequantize().data
                dequant_runtime = (time.perf_counter() - dequant_start) / repeats

                report = quantized.report()
                results.append(
                    {
                        "name": f"{method}_{bits}bit_{label}",
                        "method": method,
                        "bits": bits,
                        "storage": label,
                        "packed": packed,
                        "runtime_seconds": quantize_runtime + dequant_runtime,
                        "quantize_seconds": quantize_runtime,
                        "dequantize_seconds": dequant_runtime,
                        "original_bytes": report.memory.original_bytes,
                        "theoretical_packed_bytes": report.memory.theoretical_packed_bytes,
                        "logical_int32_storage_bytes": report.memory.logical_int32_storage_bytes,
                        "actual_code_storage_bytes": report.memory.actual_code_storage_bytes,
                        "actual_packed_bytes": report.memory.actual_packed_bytes,
                        "codebook_bytes": report.memory.codebook_bytes,
                        "total_theoretical_representation_bytes": (
                            report.memory.total_theoretical_representation_bytes
                        ),
                        "total_actual_representation_bytes": (
                            report.memory.total_actual_representation_bytes
                        ),
                        "compression_ratio_theoretical": (
                            report.memory.compression_ratio_theoretical
                        ),
                        "compression_ratio_actual": report.memory.compression_ratio_actual,
                        "max_absolute_error": report.error.max_absolute_error,
                        "rmse": report.error.rmse,
                        "codebook_iterations": (
                            report.codebook_fit.iterations if report.codebook_fit else None
                        ),
                        "codebook_converged": (
                            report.codebook_fit.converged if report.codebook_fit else None
                        ),
                    }
                )

    return results

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .kernels.backend import DEFAULT_HADAMARD_BACKEND, NumericalBackend, resolve_backend
from .kernels.hadamard import walsh_hadamard as kernel_walsh_hadamard

RotationKind = Literal["hadamard", "combined"]


@dataclass(frozen=True)
class RotationMetadata:
    dimension: int
    permutation: tuple[int, ...]
    signs: tuple[float, ...]
    use_hadamard: bool
    transform_type: str


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _validate_last_axis(data: np.ndarray) -> int:
    if data.ndim == 0:
        raise ValueError("Rotation requires at least one dimension.")
    dimension = int(data.shape[-1])
    if not is_power_of_two(dimension):
        raise ValueError(
            f"Rotation requires the final axis to be a power of two, got {dimension}."
        )
    return dimension


def _build_sign_pattern(dimension: int) -> np.ndarray:
    signs = np.ones(dimension, dtype=np.float64)
    signs[1::2] = -1.0
    return signs


def walsh_hadamard_1d(
    values: np.ndarray,
    *,
    backend: NumericalBackend | None = None,
) -> np.ndarray:
    """Normalized Walsh-Hadamard transform for power-of-two length vectors."""
    hadamard_backend = resolve_backend(backend, DEFAULT_HADAMARD_BACKEND)
    vector = np.asarray(values, dtype=np.float64)
    length = vector.shape[0]
    if not is_power_of_two(length):
        raise ValueError(f"Hadamard length must be a power of two, got {length}.")
    return kernel_walsh_hadamard(vector, backend=hadamard_backend)


class PermutationTransform:
    def __init__(self, permutation: np.ndarray) -> None:
        permutation = np.asarray(permutation, dtype=np.int64)
        if permutation.ndim != 1:
            raise ValueError("Permutation must be one-dimensional.")
        if sorted(permutation.tolist()) != list(range(permutation.size)):
            raise ValueError("Permutation indices must be a valid reordering.")
        self.permutation = permutation
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(permutation.size)
        self.inverse_permutation = inverse

    def apply(self, data: np.ndarray) -> np.ndarray:
        return np.take(data, self.permutation, axis=-1)

    def inverse(self, data: np.ndarray) -> np.ndarray:
        return np.take(data, self.inverse_permutation, axis=-1)


class SignFlipTransform:
    def __init__(self, signs: np.ndarray) -> None:
        signs = np.asarray(signs, dtype=np.float64)
        if np.any(signs == 0):
            raise ValueError("Sign pattern must contain only +1 and -1.")
        self.signs = signs

    def apply(self, data: np.ndarray) -> np.ndarray:
        return data * self.signs

    def inverse(self, data: np.ndarray) -> np.ndarray:
        return data * self.signs


class HadamardTransform:
    """Normalized Walsh-Hadamard on the final axis."""

    def apply(self, data: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(walsh_hadamard_1d, -1, data)

    def inverse(self, data: np.ndarray) -> np.ndarray:
        return self.apply(data)


class RotationTransform:
    """
    Deterministic pipeline: permutation -> sign flip -> normalized Hadamard.

    Applies along the final tensor axis independently for each leading index.
    """

    def __init__(self, dimension: int, kind: RotationKind = "combined") -> None:
        if kind not in ("hadamard", "combined"):
            raise ValueError(f"Unsupported rotation kind {kind!r}.")
        if not is_power_of_two(dimension):
            raise ValueError(
                f"Rotation requires power-of-two dimension, got {dimension}."
            )

        permutation = np.roll(np.arange(dimension), shift=1)
        self.permutation = PermutationTransform(permutation)
        self.sign_flip = SignFlipTransform(_build_sign_pattern(dimension))
        self.hadamard = HadamardTransform()
        self.metadata = RotationMetadata(
            dimension=dimension,
            permutation=tuple(int(value) for value in permutation),
            signs=tuple(float(value) for value in self.sign_flip.signs),
            use_hadamard=True,
            transform_type=kind,
        )

    def apply(self, data: np.ndarray) -> np.ndarray:
        array = np.asarray(data, dtype=np.float64)
        _validate_last_axis(array)
        if array.shape[-1] != self.metadata.dimension:
            raise ValueError(
                f"Expected final axis {self.metadata.dimension}, got {array.shape[-1]}."
            )
        rotated = self.permutation.apply(array)
        rotated = self.sign_flip.apply(rotated)
        rotated = self.hadamard.apply(rotated)
        return rotated

    def inverse(self, data: np.ndarray) -> np.ndarray:
        array = np.asarray(data, dtype=np.float64)
        restored = self.hadamard.inverse(array)
        restored = self.sign_flip.inverse(restored)
        restored = self.permutation.inverse(restored)
        return restored

    def l2_norm_difference(self, data: np.ndarray) -> float:
        original = np.asarray(data, dtype=np.float64)
        rotated = self.apply(original)
        original_norm = float(np.linalg.norm(original.ravel()))
        rotated_norm = float(np.linalg.norm(rotated.ravel()))
        return abs(original_norm - rotated_norm)


def build_rotation_transform(dimension: int, kind: RotationKind = "combined") -> RotationTransform:
    return RotationTransform(dimension, kind=kind)


def rotate_array(data: np.ndarray, kind: RotationKind = "combined") -> tuple[np.ndarray, RotationTransform]:
    transform = build_rotation_transform(_validate_last_axis(data), kind=kind)
    return transform.apply(data), transform


def inverse_rotate_array(data: np.ndarray, transform: RotationTransform) -> np.ndarray:
    return transform.inverse(data)


@dataclass(frozen=True)
class RotationQuantizationExperiment:
    bits: int
    method: str
    packed: bool
    rotation_kind: str
    baseline_error: Any
    rotated_error: Any
    baseline_memory: Any
    rotated_memory: Any
    relative_rmse_improvement: float | None
    result_label: str

    def format(self) -> str:
        lines = [
            "ROTATION EXPERIMENT",
            "-------------------",
            "",
            f"Bits: {self.bits}",
            f"Method: {self.method}",
            f"Packed: {self.packed}",
            f"Transform: {self.rotation_kind}",
            "",
            "Baseline RMSE:",
            f"  {self.baseline_error.rmse}",
            "Rotated RMSE:",
            f"  {self.rotated_error.rmse}",
            "",
            f"Max absolute error baseline: {self.baseline_error.max_absolute_error}",
            f"Max absolute error rotated: {self.rotated_error.max_absolute_error}",
            f"Mean absolute error baseline: {self.baseline_error.mean_absolute_error}",
            f"Mean absolute error rotated: {self.rotated_error.mean_absolute_error}",
            "",
            "Difference (baseline - rotated RMSE):",
            f"  {self.baseline_error.rmse - self.rotated_error.rmse}",
            "",
            f"Relative RMSE improvement: {self.relative_rmse_improvement}",
            f"Result: {self.result_label}",
            "",
            "Baseline total actual representation bytes:",
            f"  {self.baseline_memory.total_actual_representation_bytes}",
            "Rotated total actual representation bytes:",
            f"  {self.rotated_memory.total_actual_representation_bytes}",
        ]
        return "\n".join(lines)


def compare_quantization_rotation(
    data: np.ndarray,
    bits: int,
    *,
    method: str = "uniform",
    packed: bool = False,
    rotation: RotationKind = "combined",
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> RotationQuantizationExperiment:
    from .quantization import compute_error_metrics, quantize_array

    source = np.asarray(data, dtype=np.float64)
    if source.size == 0:
        raise ValueError("Cannot run rotation experiment on empty data.")

    baseline_q = quantize_array(
        source,
        bits,
        method=method,
        max_iterations=max_iterations,
        tolerance=tolerance,
        packed=packed,
        rotation=None,
    )
    baseline_recon = baseline_q.dequantize().data
    baseline_error = compute_error_metrics(source, baseline_recon)
    baseline_report = baseline_q.report()

    rotated_q = quantize_array(
        source,
        bits,
        method=method,
        max_iterations=max_iterations,
        tolerance=tolerance,
        packed=packed,
        rotation=rotation,
    )
    rotated_recon = rotated_q.dequantize().data
    rotated_error = compute_error_metrics(source, rotated_recon)
    rotated_report = rotated_q.report()

    if baseline_error.rmse == 0.0:
        relative_improvement = None
    else:
        relative_improvement = (baseline_error.rmse - rotated_error.rmse) / baseline_error.rmse

    if rotated_error.rmse < baseline_error.rmse:
        label = "improved"
    elif rotated_error.rmse > baseline_error.rmse:
        label = "worse"
    else:
        label = "unchanged"

    return RotationQuantizationExperiment(
        bits=bits,
        method=method,
        packed=packed,
        rotation_kind=rotation,
        baseline_error=baseline_error,
        rotated_error=rotated_error,
        baseline_memory=baseline_report.memory,
        rotated_memory=rotated_report.memory,
        relative_rmse_improvement=relative_improvement,
        result_label=label,
    )


def benchmark_rotation_quantization(
    data: np.ndarray,
    *,
    bits: int = 4,
    repeats: int = 3,
) -> list[dict[str, Any]]:
    """Local measurements comparing baseline vs rotated quantization."""
    from .quantization import quantize_array

    source = np.asarray(data, dtype=np.float64)
    results: list[dict[str, Any]] = []

    for method in ("uniform", "codebook"):
        for packed in (False, True):
            for use_rotation in (False, True):
                label = "rotated" if use_rotation else "baseline"

                start = time.perf_counter()
                for _ in range(repeats):
                    quantized = quantize_array(
                        source,
                        bits,
                        method=method,
                        packed=packed,
                        rotation="combined" if use_rotation else None,
                    )
                    _ = quantized.dequantize().data
                elapsed = (time.perf_counter() - start) / repeats

                report = quantized.report()
                results.append(
                    {
                        "method": method,
                        "packed": packed,
                        "rotation": use_rotation,
                        "label": label,
                        "runtime_seconds": elapsed,
                        "rmse": report.error.rmse,
                        "actual_packed_bytes": report.memory.actual_packed_bytes,
                        "codebook_bytes": report.memory.codebook_bytes,
                        "total_actual_representation_bytes": (
                            report.memory.total_actual_representation_bytes
                        ),
                    }
                )

    return results

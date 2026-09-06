from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from minitorch.profiling.timing import benchmark_timing

from .backend import NumericalBackend


@dataclass(frozen=True)
class VectorizationBenchmarkResult:
    """Measured reference vs vectorized timing for one workload."""

    name: str
    size: int
    reference_seconds: float
    vectorized_seconds: float
    speedup: float | None
    memory_notes: str
    details: dict[str, Any]

    def report(self) -> str:
        speedup_text = f"{self.speedup:.2f}x" if self.speedup is not None else "n/a"
        return "\n".join(
            [
                f"{self.name} (n={self.size})",
                f"  reference:   {self.reference_seconds * 1000:.2f} ms",
                f"  vectorized:  {self.vectorized_seconds * 1000:.2f} ms",
                f"  speedup:     {speedup_text}",
                f"  memory:      {self.memory_notes}",
            ]
        )


def benchmark_vectorization(
    name: str,
    size: int,
    reference_fn: Callable[[], Any],
    vectorized_fn: Callable[[], Any],
    *,
    memory_notes: str = "",
    warmup: int = 2,
    repeats: int = 5,
    details: dict[str, Any] | None = None,
) -> VectorizationBenchmarkResult:
    reference = benchmark_timing(f"{name}_reference", reference_fn, warmup=warmup, repeats=repeats)
    vectorized = benchmark_timing(f"{name}_vectorized", vectorized_fn, warmup=warmup, repeats=repeats)
    speedup = None
    if vectorized.mean_seconds > 0:
        speedup = reference.mean_seconds / vectorized.mean_seconds

    return VectorizationBenchmarkResult(
        name=name,
        size=size,
        reference_seconds=reference.mean_seconds,
        vectorized_seconds=vectorized.mean_seconds,
        speedup=speedup,
        memory_notes=memory_notes,
        details=dict(details or {}),
    )


def benchmark_vectorization_matrix(
    name: str,
    sizes: dict[str, int],
    reference_builder: Callable[[int], Callable[[], Any]],
    vectorized_builder: Callable[[int], Callable[[], Any]],
    *,
    memory_notes_builder: Callable[[int], str] | None = None,
    warmup: int = 2,
    repeats: int = 5,
) -> list[VectorizationBenchmarkResult]:
    results: list[VectorizationBenchmarkResult] = []
    for label, size in sizes.items():
        notes = memory_notes_builder(size) if memory_notes_builder else ""
        results.append(
            benchmark_vectorization(
                f"{name}_{label}",
                size,
                reference_builder(size),
                vectorized_builder(size),
                memory_notes=notes,
                warmup=warmup,
                repeats=repeats,
                details={"label": label},
            )
        )
    return results


def benchmark_execution_matrix(
    name: str,
    size: int,
    modes: dict[str, Callable[[], Any]],
    *,
    warmup: int = 2,
    repeats: int = 5,
) -> dict[str, float]:
    """Compare multiple execution modes (serial/parallel × reference/vectorized)."""
    timings: dict[str, float] = {}
    for mode_name, fn in modes.items():
        result = benchmark_timing(f"{name}_{mode_name}", fn, warmup=warmup, repeats=repeats)
        timings[mode_name] = result.mean_seconds
    return timings

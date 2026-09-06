from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

"""
High-resolution timing primitives for MiniTorch performance analysis.

Warmup runs execute the benchmark function before measurements begin so the
first timed sample is less likely to reflect one-off initialization effects
(cold imports, allocator warmup, first NumPy call). Warmup does not eliminate
all measurement noise — results remain machine-dependent.
"""


@dataclass
class Timer:
    """Monotonic high-resolution timer using ``time.perf_counter()``."""

    elapsed_seconds: float = 0.0

    def start(self) -> None:
        self.elapsed_seconds = 0.0
        self._start = time.perf_counter()

    def stop(self) -> float:
        self.elapsed_seconds = time.perf_counter() - self._start
        return self.elapsed_seconds

    def __enter__(self) -> Timer:
        self.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stop()


@dataclass(frozen=True)
class TimingBenchmarkResult:
    """Repeated timing measurements with basic descriptive statistics."""

    name: str
    runs: int
    warmup: int
    mean_seconds: float
    median_seconds: float
    min_seconds: float
    max_seconds: float
    samples: tuple[float, ...]


def benchmark_timing(
    name: str,
    fn: Callable[[], Any],
    *,
    warmup: int = 3,
    repeats: int = 10,
) -> TimingBenchmarkResult:
    """
    Benchmark ``fn`` with warmup and repeated measurements.

    Returns mean, median, min, and max over ``repeats`` samples. Each sample
    is one call to ``fn`` measured with ``perf_counter``.
    """
    if warmup < 0:
        raise ValueError("warmup must be non-negative.")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    for _ in range(warmup):
        fn()

    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)

    return TimingBenchmarkResult(
        name=name,
        runs=repeats,
        warmup=warmup,
        mean_seconds=statistics.mean(samples),
        median_seconds=statistics.median(samples),
        min_seconds=min(samples),
        max_seconds=max(samples),
        samples=tuple(samples),
    )


def measure_time(fn: Callable[[], Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Run ``fn`` once and return ``(result, elapsed_seconds)``."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start

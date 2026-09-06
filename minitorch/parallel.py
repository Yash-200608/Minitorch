from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

"""
Controlled CPU parallelism for independent MiniTorch workloads.

MiniTorch defaults to serial execution. Parallel mode is explicit and opt-in.

Concurrency notes (Python on CPython):
- NumPy-heavy work often runs in native code and can release the GIL, so
  ``ThreadPoolExecutor`` may help for large independent array chunks.
- Pure Python loops (for example bit-packing) remain GIL-bound; threads may
  add overhead on small inputs.
- ``ProcessPoolExecutor`` is not used here: pickling and spawn cost dominate
  for the chunk sizes MiniTorch targets, especially on Windows.

NumPy may also use its own internal threading (OpenBLAS/MKL). Oversubscription
(worker threads × BLAS threads) can hurt performance. This module does not
force BLAS thread counts; benchmark on your machine.
"""

T = TypeVar("T")


@dataclass(frozen=True)
class ParallelConfig:
    """Explicit parallel execution settings. Disabled by default."""

    enabled: bool = False
    workers: int | None = None

    def resolve_workers(self, item_count: int) -> int:
        if not self.enabled:
            return 1
        if item_count <= 0:
            return 1
        if self.workers is not None:
            return max(1, min(self.workers, item_count))
        default = min(4, os.cpu_count() or 1)
        return max(1, min(default, item_count))

    @classmethod
    def from_flags(cls, parallel: bool = False, workers: int | None = None) -> ParallelConfig:
        return cls(enabled=parallel, workers=workers)


def chunk_ranges(count: int, workers: int) -> list[tuple[int, int]]:
    """
    Partition ``[0, count)`` into disjoint half-open ranges.

    Complexity: O(workers). Every index appears in exactly one range when
    ``count > 0``. Returns an empty list when ``count == 0``.
    """
    if count < 0:
        raise ValueError("count must be non-negative.")
    if workers <= 0:
        raise ValueError("workers must be positive.")
    if count == 0:
        return []

    workers = min(workers, count)
    base = count // workers
    remainder = count % workers

    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(workers):
        chunk_size = base + (1 if index < remainder else 0)
        end = start + chunk_size
        if start < end:
            ranges.append((start, end))
        start = end
    return ranges


def chunk_ranges_aligned(
    count: int,
    workers: int,
    align: int,
) -> list[tuple[int, int]]:
    """
    Partition ``[0, count)`` into ranges whose starts are multiples of ``align``.

    Used when each chunk produces an independent packed byte stream that is
    concatenated (for example 2-bit / 4-bit packing).
    """
    if count < 0:
        raise ValueError("count must be non-negative.")
    if workers <= 0:
        raise ValueError("workers must be positive.")
    if align <= 0:
        raise ValueError("align must be positive.")
    if count == 0:
        return []
    if align == 1:
        return chunk_ranges(count, workers)

    unit_count = (count + align - 1) // align
    unit_ranges = chunk_ranges(unit_count, min(workers, unit_count))
    ranges: list[tuple[int, int]] = []
    for unit_start, unit_end in unit_ranges:
        start = unit_start * align
        end = min(unit_end * align, count)
        if start < end:
            ranges.append((start, end))
    return ranges


def execute_chunked(
    count: int,
    config: ParallelConfig,
    worker_fn: Callable[[int, int], T],
) -> list[T]:
    """
    Run ``worker_fn(start, end)`` over disjoint chunks.

    When parallel mode is disabled, ``workers == 1``, or ``count == 0``, a
    single worker call covers ``[0, count)``.
    """
    if count < 0:
        raise ValueError("count must be non-negative.")

    workers = config.resolve_workers(count)
    ranges = chunk_ranges(count, workers)

    if not config.enabled or len(ranges) <= 1:
        if count == 0:
            return []
        return [worker_fn(0, count)]

    results: list[T] = []
    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        futures = [executor.submit(worker_fn, start, end) for start, end in ranges]
        for future in futures:
            results.append(future.result())
    return results


def merge_chunk_results(
    count: int,
    config: ParallelConfig,
    worker_fn: Callable[[int, int], T],
    merge_fn: Callable[[list[T]], T],
) -> T:
    """Chunk, execute workers, and merge with a deterministic reducer."""
    chunks = execute_chunked(count, config, worker_fn)
    if count == 0:
        return merge_fn([])
    if len(chunks) == 1:
        return chunks[0]
    return merge_fn(chunks)


@dataclass(frozen=True)
class ParallelBenchmarkResult:
    """Serial vs parallel timing comparison for one workload."""

    name: str
    workers: int
    serial_seconds: float
    parallel_seconds: float
    speedup: float | None
    serial_samples: tuple[float, ...]
    parallel_samples: tuple[float, ...]

    def report(self) -> str:
        speedup_text = f"{self.speedup:.2f}x" if self.speedup is not None else "n/a"
        lines = [
            "MiniTorch Parallel Profile",
            "--------------------------",
            "",
            f"Workload: {self.name}",
            f"Serial: {self.serial_seconds * 1000:.2f} ms",
            f"Parallel: {self.workers} workers, {self.parallel_seconds * 1000:.2f} ms",
            f"Measured speedup: {speedup_text}",
            "",
            "NOTE",
            "  Speedup applies to this workload and configuration only.",
        ]
        return "\n".join(lines)


def benchmark_parallel_comparison(
    name: str,
    serial_fn: Callable[[], Any],
    parallel_fn: Callable[[], Any],
    *,
    workers: int = 4,
    warmup: int = 3,
    repeats: int = 10,
) -> ParallelBenchmarkResult:
    """Compare serial and parallel implementations with repeated measurements."""
    from .profiling.timing import benchmark_timing

    serial = benchmark_timing(f"{name}_serial", serial_fn, warmup=warmup, repeats=repeats)
    parallel = benchmark_timing(
        f"{name}_parallel",
        parallel_fn,
        warmup=warmup,
        repeats=repeats,
    )
    speedup = None
    if parallel.mean_seconds > 0:
        speedup = serial.mean_seconds / parallel.mean_seconds

    return ParallelBenchmarkResult(
        name=name,
        workers=workers,
        serial_seconds=serial.mean_seconds,
        parallel_seconds=parallel.mean_seconds,
        speedup=speedup,
        serial_samples=serial.samples,
        parallel_samples=parallel.samples,
    )

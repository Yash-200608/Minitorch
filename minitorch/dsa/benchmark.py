from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from minitorch.cache import clear_computation_cache, get_computation_cache
from minitorch.graph import ComputationGraph
from minitorch.tensor import Tensor

from .topk import topk
from .topological_sort import topological_sort
from .traversal import depth_first_search


@dataclass(frozen=True)
class BenchmarkResult:
    """Measured benchmark output with real counters — no fabricated statistics."""

    name: str
    runtime_seconds: float
    operation_counts: dict[str, int | float]
    complexity: str
    memory_notes: str
    details: dict[str, Any] = field(default_factory=dict)


def run_timed(
    name: str,
    fn: Callable[[], Any],
    *,
    warmup: int = 1,
    repeats: int = 3,
    complexity: str,
    memory_notes: str = "",
    operation_counts: dict[str, int | float] | None = None,
    details: dict[str, Any] | None = None,
) -> BenchmarkResult:
    for _ in range(warmup):
        fn()

    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        name=name,
        runtime_seconds=elapsed / repeats,
        operation_counts=dict(operation_counts or {}),
        complexity=complexity,
        memory_notes=memory_notes,
        details=dict(details or {}),
    )


def benchmark_full_sort_vs_topk(
    values: list[float] | np.ndarray,
    k: int,
    *,
    repeats: int = 3,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """Compare O(N log N) full sort with O(N log K) heap Top-K."""

    sequence = [float(value) for value in values]
    n = len(sequence)

    def full_sort() -> list[float]:
        return sorted(sequence, reverse=True)[:k]

    sort_result = run_timed(
        name="full_sort_top_k",
        fn=full_sort,
        repeats=repeats,
        complexity="O(N log N)",
        memory_notes=f"sorted full list of {n} floats, sliced to k={k}",
        operation_counts={"n": n, "k": k},
    )

    topk_probe = topk(sequence, k=k, largest=True)

    def heap_topk() -> tuple[float, ...]:
        return topk(sequence, k=k, largest=True).values

    topk_result = run_timed(
        name="heap_topk",
        fn=heap_topk,
        repeats=repeats,
        complexity="O(N log K)",
        memory_notes=f"binary min-heap of capacity k={k} over n={n} items",
        operation_counts={
            "n": n,
            "k": k,
            "heap_pushes": topk_probe.stats.heap_pushes,
            "heap_pops": topk_probe.stats.heap_pops,
            "comparisons": topk_probe.stats.comparisons,
            "items_scanned": topk_probe.stats.items_scanned,
        },
    )

    return sort_result, topk_result


def benchmark_dfs(graph: ComputationGraph, *, repeats: int = 3) -> BenchmarkResult:
    probe = depth_first_search(graph, direction="children")

    def run_dfs() -> list[int]:
        return depth_first_search(graph, direction="children").order

    return run_timed(
        name="dfs_traversal",
        fn=run_dfs,
        repeats=repeats,
        complexity="O(V + E)",
        memory_notes=f"explicit stack over V={graph.vertex_count} nodes, E={graph.edge_count} edges",
        operation_counts={
            "vertices": graph.vertex_count,
            "edges": graph.edge_count,
            "visits": probe.stats.visits,
            "edge_inspections": probe.stats.edge_inspections,
            "stack_pushes": probe.stats.stack_pushes,
            "stack_pops": probe.stats.stack_pops,
            "max_stack_size": probe.stats.max_stack_size,
        },
    )


def benchmark_topological_sort(graph: ComputationGraph, *, repeats: int = 3) -> BenchmarkResult:
    probe = topological_sort(graph)

    def run_topo() -> list[int]:
        return topological_sort(graph).order

    return run_timed(
        name="topological_sort",
        fn=run_topo,
        repeats=repeats,
        complexity="O(V + E)",
        memory_notes=f"Kahn queue over V={graph.vertex_count} nodes, E={graph.edge_count} edges",
        operation_counts={
            "vertices": graph.vertex_count,
            "edges": graph.edge_count,
            "processed_nodes": probe.stats.processed_nodes,
            "edge_inspections": probe.stats.edge_inspections,
            "queue_pushes": probe.stats.queue_pushes,
            "queue_pops": probe.stats.queue_pops,
            "max_queue_size": probe.stats.max_queue_size,
        },
    )


def benchmark_cache_on_off(
    expression_fn: Callable[[], Tensor],
    *,
    iterations: int = 50,
    repeats: int = 1,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """
    Compare uncached vs cached tensor forward work on a deterministic expression.

    `expression_fn` must rebuild leaf tensors each call so graph ids differ while
    numeric payloads repeat — the cache keys parent data, not tensor ids.
    """

    cache = get_computation_cache()
    clear_computation_cache()
    cache.disable()

    start = time.perf_counter()
    for _ in range(repeats):
        for _ in range(iterations):
            expression_fn()
    off_runtime = (time.perf_counter() - start) / repeats

    off_result = BenchmarkResult(
        name="cache_disabled",
        runtime_seconds=off_runtime,
        operation_counts={"iterations": iterations},
        complexity="O(iterations * forward_ops)",
        memory_notes="computation cache disabled; every op runs NumPy work",
        details={"cache": cache.summary()},
    )

    clear_computation_cache()
    cache.enable()

    start = time.perf_counter()
    for _ in range(repeats):
        for _ in range(iterations):
            expression_fn()
    on_runtime = (time.perf_counter() - start) / repeats
    on_summary = cache.summary()

    on_result = BenchmarkResult(
        name="cache_enabled",
        runtime_seconds=on_runtime,
        operation_counts={
            "iterations": iterations,
            "cache_hits": on_summary["hits"],
            "cache_misses": on_summary["misses"],
            "cache_entries": on_summary["entries"],
            "hit_rate": on_summary["hit_rate"],
        },
        complexity="O(unique_keys) + O(iterations) lookups",
        memory_notes=f"HashMap memoization over {iterations} repeated forwards",
        details={"cache": on_summary},
    )

    cache.disable()
    return off_result, on_result


def build_sample_graph(depth: int = 4) -> ComputationGraph:
    """Build a simple chain graph for DFS / topological benchmarks."""
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    output = x
    for scalar in (2.0, 3.0, 0.5, 1.5)[:depth]:
        output = output * scalar + 1.0
    return ComputationGraph.from_tensor(output.mean())

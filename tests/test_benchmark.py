from minitorch import Tensor
from minitorch.dsa import (
    benchmark_cache_on_off,
    benchmark_dfs,
    benchmark_full_sort_vs_topk,
    benchmark_topological_sort,
    build_sample_graph,
)


def test_benchmark_full_sort_vs_topk_reports_real_counts():
    sort_result, topk_result = benchmark_full_sort_vs_topk(range(100), k=5, repeats=1)

    assert sort_result.name == "full_sort_top_k"
    assert topk_result.name == "heap_topk"
    assert sort_result.runtime_seconds >= 0
    assert topk_result.runtime_seconds >= 0
    assert sort_result.operation_counts["n"] == 100
    assert topk_result.operation_counts["k"] == 5
    assert topk_result.operation_counts["heap_pushes"] > 0


def test_benchmark_dfs_and_topological_sort_on_sample_graph():
    graph = build_sample_graph(depth=3)

    dfs_result = benchmark_dfs(graph, repeats=1)
    topo_result = benchmark_topological_sort(graph, repeats=1)

    assert dfs_result.operation_counts["vertices"] == graph.vertex_count
    assert dfs_result.operation_counts["visits"] == graph.vertex_count
    assert topo_result.operation_counts["processed_nodes"] == graph.vertex_count
    assert dfs_result.complexity == "O(V + E)"
    assert topo_result.complexity == "O(V + E)"


def test_benchmark_cache_on_off_reports_hit_rate():
    def expression():
        x = Tensor([1.0, 2.0, 3.0])
        y = Tensor([4.0, 5.0, 6.0])
        return (x * 2 + y).sum()

    off_result, on_result = benchmark_cache_on_off(expression, iterations=20, repeats=1)

    assert off_result.name == "cache_disabled"
    assert on_result.name == "cache_enabled"
    assert on_result.operation_counts["cache_hits"] > 0
    assert on_result.operation_counts["hit_rate"] > 0

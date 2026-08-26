from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from minitorch.graph import ComputationGraph, GraphCycleError


@dataclass(frozen=True)
class TopologicalSortStats:
    processed_nodes: int = 0
    edge_inspections: int = 0
    queue_pushes: int = 0
    queue_pops: int = 0
    max_queue_size: int = 0
    complexity: str = "O(V + E)"


@dataclass(frozen=True)
class TopologicalSortResult:
    order: list[int]
    stats: TopologicalSortStats


def topological_sort(graph: ComputationGraph) -> TopologicalSortResult:
    """
    Kahn topological sort over dependency-to-consumer edges.

    The resulting order puts dependencies before the tensors that consume them.
    Reverse-mode autodiff processes this order backwards.
    """

    if graph.vertex_count == 0:
        return TopologicalSortResult(order=[], stats=TopologicalSortStats())

    adjacency = graph.adjacency()
    indegree = {node_id: 0 for node_id in graph.nodes}
    for edge in graph.edges:
        indegree[edge.child_id] += 1

    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    queue_pushes = len(queue)
    queue_pops = 0
    edge_inspections = 0
    max_queue_size = len(queue)
    order: list[int] = []

    while queue:
        max_queue_size = max(max_queue_size, len(queue))
        node_id = queue.popleft()
        queue_pops += 1
        order.append(node_id)

        for child_id in adjacency[node_id]:
            edge_inspections += 1
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                queue.append(child_id)
                queue_pushes += 1

    if len(order) != graph.vertex_count:
        raise GraphCycleError("Cycle detected during topological sort.")

    return TopologicalSortResult(
        order=order,
        stats=TopologicalSortStats(
            processed_nodes=len(order),
            edge_inspections=edge_inspections,
            queue_pushes=queue_pushes,
            queue_pops=queue_pops,
            max_queue_size=max_queue_size,
        ),
    )

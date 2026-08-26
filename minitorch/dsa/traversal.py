from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from minitorch.graph import ComputationGraph


Direction = Literal["children", "parents"]


@dataclass(frozen=True)
class TraversalStats:
    visits: int = 0
    edge_inspections: int = 0
    stack_pushes: int = 0
    stack_pops: int = 0
    max_stack_size: int = 0
    complexity: str = "O(V + E)"


@dataclass(frozen=True)
class DFSResult:
    order: list[int]
    stats: TraversalStats


def depth_first_search(
    graph: ComputationGraph,
    start_id: int | None = None,
    direction: Direction = "children",
) -> DFSResult:
    """
    Iterative DFS over a computation graph.

    `direction="children"` follows dataflow from dependencies to consumers.
    `direction="parents"` walks backward from a result to its dependencies.
    """

    if graph.vertex_count == 0:
        return DFSResult(order=[], stats=TraversalStats())

    adjacency = graph.adjacency() if direction == "children" else graph.reverse_adjacency()
    starts = [start_id] if start_id is not None else _default_starts(graph, direction)

    for node_id in starts:
        if node_id not in graph.nodes:
            raise KeyError(f"Start node {node_id} is not present in this computation graph.")

    order: list[int] = []
    visited: set[int] = set()
    stack = list(reversed(starts))
    stack_pushes = len(stack)
    stack_pops = 0
    edge_inspections = 0
    max_stack_size = len(stack)

    while stack:
        max_stack_size = max(max_stack_size, len(stack))
        node_id = stack.pop()
        stack_pops += 1

        if node_id in visited:
            continue

        visited.add(node_id)
        order.append(node_id)

        neighbors = adjacency[node_id]
        edge_inspections += len(neighbors)
        for neighbor_id in reversed(neighbors):
            if neighbor_id not in visited:
                stack.append(neighbor_id)
                stack_pushes += 1

    return DFSResult(
        order=order,
        stats=TraversalStats(
            visits=len(order),
            edge_inspections=edge_inspections,
            stack_pushes=stack_pushes,
            stack_pops=stack_pops,
            max_stack_size=max_stack_size,
        ),
    )


def _default_starts(graph: ComputationGraph, direction: Direction) -> list[int]:
    if direction == "parents":
        return graph.leaves()
    return graph.roots()

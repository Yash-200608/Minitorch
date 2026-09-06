from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from minitorch.dsa.topological_sort import topological_sort
from minitorch.dsa.traversal import depth_first_search
from minitorch.graph import ComputationGraph, GraphEdge, GraphNode
from minitorch.tensor import Tensor


@dataclass(frozen=True)
class GraphExplorer:
    """
    Inspection interface over a real MiniTorch computation DAG.

    Reuses existing graph construction, DFS, and topological sort — no duplicate
    traversal implementations.
    """

    output: Tensor
    graph: ComputationGraph

    @classmethod
    def from_tensor(cls, output: Tensor) -> GraphExplorer:
        return cls(output=output, graph=ComputationGraph.from_tensor(output))

    def nodes(self) -> tuple[GraphNode, ...]:
        return tuple(self.graph.nodes.values())

    def edges(self) -> tuple[GraphEdge, ...]:
        return tuple(self.graph.edges)

    def parents(self, tensor: Tensor | int) -> tuple[GraphNode, ...]:
        return tuple(self.graph.parents(tensor))

    def children(self, tensor: Tensor | int) -> tuple[GraphNode, ...]:
        return tuple(self.graph.children(tensor))

    def topological_order(self) -> tuple[int, ...]:
        return tuple(topological_sort(self.graph).order)

    def paths(self, source: Tensor | int, target: Tensor | int) -> tuple[tuple[int, ...], ...]:
        """All dependency-to-consumer paths from `source` to `target`."""

        source_id = source if isinstance(source, int) else source.id
        target_id = target if isinstance(target, int) else target.id

        if source_id not in self.graph.nodes or target_id not in self.graph.nodes:
            return ()

        adjacency = self.graph.adjacency()
        found: list[tuple[int, ...]] = []

        def visit(node_id: int, path: tuple[int, ...]) -> None:
            if node_id == target_id:
                found.append(path)
                return
            for child_id in adjacency.get(node_id, []):
                if child_id in path:
                    continue
                visit(child_id, path + (child_id,))

        visit(source_id, (source_id,))
        return tuple(found)

    def depth(self, tensor: Tensor | int | None = None) -> int:
        """Longest path length from any graph root to the given node."""

        node_id = self._resolve_node_id(tensor)
        adjacency = self.graph.adjacency()
        roots = self.graph.roots()

        if node_id in roots:
            return 0

        depth_by_node = {root: 0 for root in roots}
        for ordered_id in topological_sort(self.graph).order:
            if ordered_id not in depth_by_node:
                depth_by_node[ordered_id] = 0
            for child_id in adjacency.get(ordered_id, []):
                depth_by_node[child_id] = max(
                    depth_by_node.get(child_id, 0),
                    depth_by_node[ordered_id] + 1,
                )

        return depth_by_node.get(node_id, 0)

    def dsa_profile(self) -> dict[str, Any]:
        dfs = depth_first_search(self.graph, direction="children")
        topo = topological_sort(self.graph)

        return {
            "vertices": self.graph.vertex_count,
            "edges": self.graph.edge_count,
            "dfs_visits": dfs.stats.visits,
            "dfs_edge_inspections": dfs.stats.edge_inspections,
            "dfs_stack_pushes": dfs.stats.stack_pushes,
            "dfs_stack_pops": dfs.stats.stack_pops,
            "topological_processed_nodes": topo.stats.processed_nodes,
            "topological_edge_inspections": topo.stats.edge_inspections,
            "complexity": "O(V + E)",
        }

    def describe(self) -> str:
        profile = self.dsa_profile()
        lines = [
            "GRAPH EXPLORER",
            "==============",
            f"Output id: {self.output.id}",
            f"Vertices (V): {profile['vertices']}",
            f"Edges (E): {profile['edges']}",
            f"Topological order: {list(self.topological_order())}",
            "",
            "DFS:",
            f"  visits: {profile['dfs_visits']}",
            f"  edge inspections: {profile['dfs_edge_inspections']}",
            "",
            "Complexity: O(V + E)",
        ]
        return "\n".join(lines)

    def _resolve_node_id(self, tensor: Tensor | int | None) -> int:
        if tensor is None:
            if self.graph.output_id is None:
                raise ValueError("Graph has no output id.")
            return self.graph.output_id
        return tensor if isinstance(tensor, int) else tensor.id

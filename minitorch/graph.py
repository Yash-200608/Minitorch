from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .tensor import Tensor


class GraphCycleError(ValueError):
    """Raised when a dependency cycle is found in a graph that must be a DAG."""


@dataclass(frozen=True)
class GraphNode:
    """Inspectable representation of one tensor in a computation graph."""

    id: int
    operation: str | None
    parent_ids: tuple[int, ...]
    child_ids: tuple[int, ...]
    shape: tuple[int, ...]
    requires_grad: bool
    metadata: dict[str, Any]
    tensor: Tensor | None = None


@dataclass(frozen=True)
class GraphEdge:
    """Directed edge from a dependency tensor to the tensor that consumes it."""

    parent_id: int
    child_id: int


class ComputationGraph:
    """
    A real DAG reconstructed from Tensor parent links.

    Edges point from dependency to consumer. Kahn topological sort uses that
    direction; reverse-mode autodiff then walks the same order backwards.
    """

    def __init__(
        self,
        nodes: dict[int, GraphNode] | None = None,
        edges: list[GraphEdge] | None = None,
        output_id: int | None = None,
    ):
        self.nodes = dict(nodes or {})
        self.edges = list(edges or [])
        self.output_id = output_id
        self._validate_edges_reference_known_nodes()

    @classmethod
    def from_tensor(cls, output: Tensor) -> "ComputationGraph":
        tensors: dict[int, Tensor] = {}
        edges: list[GraphEdge] = []
        visit_state: dict[int, str] = {}

        def visit(tensor: Tensor) -> None:
            state = visit_state.get(tensor.id)
            if state == "visiting":
                raise GraphCycleError(f"Cycle detected at tensor id {tensor.id}.")
            if state == "visited":
                return

            visit_state[tensor.id] = "visiting"
            tensors[tensor.id] = tensor

            for parent in tensor.parents:
                visit(parent)
                edges.append(GraphEdge(parent.id, tensor.id))

            visit_state[tensor.id] = "visited"

        visit(output)

        parent_ids_by_child: dict[int, list[int]] = {tensor_id: [] for tensor_id in tensors}
        child_ids_by_parent: dict[int, list[int]] = {tensor_id: [] for tensor_id in tensors}

        for edge in edges:
            parent_ids_by_child[edge.child_id].append(edge.parent_id)
            child_ids_by_parent[edge.parent_id].append(edge.child_id)

        nodes = {
            tensor_id: GraphNode(
                id=tensor_id,
                operation=tensor.operation,
                parent_ids=tuple(parent_ids_by_child[tensor_id]),
                child_ids=tuple(child_ids_by_parent[tensor_id]),
                shape=tensor.shape,
                requires_grad=tensor.requires_grad,
                metadata=dict(tensor.metadata),
                tensor=tensor,
            )
            for tensor_id, tensor in tensors.items()
        }

        return cls(nodes=nodes, edges=edges, output_id=output.id)

    @property
    def vertex_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def adjacency(self) -> dict[int, list[int]]:
        adjacency = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            adjacency[edge.parent_id].append(edge.child_id)
        return adjacency

    def reverse_adjacency(self) -> dict[int, list[int]]:
        reverse = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            reverse[edge.child_id].append(edge.parent_id)
        return reverse

    def roots(self) -> list[int]:
        """Return graph nodes that have no dependencies inside this graph."""
        return [node_id for node_id, node in self.nodes.items() if not node.parent_ids]

    def leaves(self) -> list[int]:
        """Return graph nodes that are not consumed by another node in this graph."""
        return [node_id for node_id, node in self.nodes.items() if not node.child_ids]

    def node_for(self, tensor_or_id: Tensor | int) -> GraphNode:
        node_id = tensor_or_id if isinstance(tensor_or_id, int) else tensor_or_id.id
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"Node {node_id} is not present in this computation graph.") from exc

    def parents(self, tensor_or_id: Tensor | int) -> list[GraphNode]:
        node = self.node_for(tensor_or_id)
        return [self.nodes[parent_id] for parent_id in node.parent_ids]

    def children(self, tensor_or_id: Tensor | int) -> list[GraphNode]:
        node = self.node_for(tensor_or_id)
        return [self.nodes[child_id] for child_id in node.child_ids]

    def summary(self) -> dict[str, Any]:
        return {
            "vertices": self.vertex_count,
            "edges": self.edge_count,
            "roots": self.roots(),
            "leaves": self.leaves(),
            "output_id": self.output_id,
            "operations": {
                node_id: node.operation
                for node_id, node in self.nodes.items()
            },
        }

    def _validate_edges_reference_known_nodes(self) -> None:
        missing = [
            edge
            for edge in self.edges
            if edge.parent_id not in self.nodes or edge.child_id not in self.nodes
        ]
        if missing:
            first = missing[0]
            raise ValueError(
                "Graph edge references unknown node: "
                f"{first.parent_id} -> {first.child_id}"
            )

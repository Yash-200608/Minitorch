from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from minitorch.autograd.engine import backward
from minitorch.autograd.explorer import GraphExplorer
from minitorch.dsa.topological_sort import topological_sort
from minitorch.graph import ComputationGraph
from minitorch.tensor import Tensor

FORWARD_TOLERANCE = 1e-9
GRADIENT_TOLERANCE = 1e-7


@dataclass(frozen=True)
class OptimizationStep:
    """One applied graph simplification."""

    rule: str
    original_node_id: int
    original_operation: str
    reason: str


@dataclass(frozen=True)
class VerificationResult:
    """Measured forward and gradient equivalence."""

    forward_passed: bool
    gradients_passed: bool
    max_forward_error: float
    max_gradient_error: float
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.forward_passed and self.gradients_passed


@dataclass(frozen=True)
class OptimizationResult:
    """Outcome of optimizing a computation graph."""

    original_output: Tensor
    optimized_output: Tensor
    accepted: bool
    steps: tuple[OptimizationStep, ...]
    verification: VerificationResult
    graph_before: ComputationGraph
    graph_after: ComputationGraph
    id_mapping: dict[int, Tensor]
    nodes_removed: int

    def before(self) -> GraphExplorer:
        return GraphExplorer(self.original_output, self.graph_before)

    def after(self) -> GraphExplorer:
        return GraphExplorer(self.optimized_output, self.graph_after)

    def verify(self) -> VerificationResult:
        return self.verification

    def dsa_profile(self) -> dict[str, Any]:
        return {
            "original_vertices": self.graph_before.vertex_count,
            "original_edges": self.graph_before.edge_count,
            "optimized_vertices": self.graph_after.vertex_count,
            "optimized_edges": self.graph_after.edge_count,
            "nodes_removed": self.nodes_removed,
            "transformations": len(self.steps),
            "complexity": "O(V + E) graph inspection + O(V) rebuild",
        }

    def report(self) -> str:
        profile = self.dsa_profile()
        lines = [
            "GRAPH OPTIMIZER",
            "===============",
            "",
            "ORIGINAL GRAPH",
            f"  Nodes: {profile['original_vertices']}",
            f"  Edges: {profile['original_edges']}",
            "",
            "TRANSFORMATIONS",
        ]

        if not self.steps:
            lines.append("  (none)")
        else:
            for index, step in enumerate(self.steps, start=1):
                lines.append(f"  {index}. {step.rule}")
                lines.append(f"     node {step.original_node_id} ({step.original_operation})")
                lines.append(f"     {step.reason}")

        lines.extend(
            [
                "",
                "OPTIMIZED GRAPH",
                f"  Nodes: {profile['optimized_vertices']}",
                f"  Edges: {profile['optimized_edges']}",
                f"  Nodes removed: {profile['nodes_removed']}",
                "",
                "VERIFICATION",
                f"  Forward: {'PASS' if self.verification.forward_passed else 'FAIL'}",
                f"  Gradients: {'PASS' if self.verification.gradients_passed else 'FAIL'}",
                f"  Max forward error: {self.verification.max_forward_error}",
                f"  Max gradient error: {self.verification.max_gradient_error}",
                "",
                f"STATUS: {'ACCEPTED' if self.accepted else 'REJECTED'}",
            ]
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.report()


class GraphOptimizer:
    """Conservative DAG rewriter with forward and gradient verification."""

    def optimize(self, output: Tensor) -> OptimizationResult:
        graph_before = ComputationGraph.from_tensor(output)
        steps: list[OptimizationStep] = []
        mapping: dict[int, Tensor] = {}

        optimized_candidate = self._rebuild(output, mapping, steps, apply_rules=True)
        verification = verify_optimization(output, optimized_candidate, mapping)
        accepted = verification.passed

        if accepted:
            optimized_output = optimized_candidate
            graph_after = ComputationGraph.from_tensor(optimized_output)
            nodes_removed = graph_before.vertex_count - graph_after.vertex_count
        else:
            optimized_output = output
            graph_after = graph_before
            nodes_removed = 0
            steps = ()

        id_mapping = {original_id: new_tensor for original_id, new_tensor in mapping.items()}

        return OptimizationResult(
            original_output=output,
            optimized_output=optimized_output,
            accepted=accepted,
            steps=tuple(steps),
            verification=verification,
            graph_before=graph_before,
            graph_after=graph_after,
            id_mapping=id_mapping,
            nodes_removed=nodes_removed,
        )

    def _rebuild(
        self,
        tensor: Tensor,
        mapping: dict[int, Tensor],
        steps: list[OptimizationStep],
        *,
        apply_rules: bool,
    ) -> Tensor:
        if tensor.id in mapping:
            return mapping[tensor.id]

        if not tensor.parents:
            cloned = Tensor(tensor.data.copy(), requires_grad=tensor.requires_grad)
            mapping[tensor.id] = cloned
            return cloned

        rebuilt_parents = tuple(
            self._rebuild(parent, mapping, steps, apply_rules=apply_rules) for parent in tensor.parents
        )

        if apply_rules:
            candidate, step = _try_apply_rules(tensor, rebuilt_parents)
            if candidate is not None and step is not None:
                if _forward_matches(tensor.data, candidate.data):
                    mapping[tensor.id] = candidate
                    steps.append(step)
                    return candidate

        replayed = _replay_operation(tensor.operation, rebuilt_parents, tensor.metadata)
        mapping[tensor.id] = replayed
        return replayed


def optimize(output: Tensor) -> OptimizationResult:
    """Optimize a tensor's computation graph with verification."""
    return GraphOptimizer().optimize(output)


def verify_optimization(
    original_output: Tensor,
    optimized_output: Tensor,
    id_mapping: dict[int, Tensor],
) -> VerificationResult:
    forward_error = _max_forward_error(original_output.data, optimized_output.data)
    forward_passed = forward_error <= FORWARD_TOLERANCE

    gradient_error = 0.0
    gradients_passed = True
    gradient_details: dict[str, Any] = {}

    if not original_output.requires_grad:
        return VerificationResult(
            forward_passed=forward_passed,
            gradients_passed=True,
            max_forward_error=forward_error,
            max_gradient_error=0.0,
            details=gradient_details,
        )

    try:
        orig_clones = _clone_graph_for_backward(original_output)
        opt_clones = _clone_graph_for_backward(optimized_output)

        orig_root = orig_clones[original_output.id]
        opt_root = opt_clones[optimized_output.id]

        _backward_for_verification(orig_root)
        _backward_for_verification(opt_root)

        original_graph = ComputationGraph.from_tensor(original_output)
        for leaf_id in original_graph.roots():
            node = original_graph.nodes[leaf_id]
            if not node.requires_grad:
                continue
            if leaf_id not in id_mapping:
                gradients_passed = False
                gradient_details[leaf_id] = "missing optimized mapping"
                continue

            opt_tensor = id_mapping[leaf_id]
            if opt_tensor.id not in opt_clones:
                gradients_passed = False
                gradient_details[leaf_id] = "missing optimized clone"
                continue

            err = _max_gradient_error(
                orig_clones[leaf_id].grad,
                opt_clones[opt_tensor.id].grad,
            )
            gradient_error = max(gradient_error, err)
            gradient_details[leaf_id] = err
            if err > GRADIENT_TOLERANCE:
                gradients_passed = False
    except Exception as exc:
        gradients_passed = False
        gradient_details["error"] = str(exc)

    return VerificationResult(
        forward_passed=forward_passed,
        gradients_passed=gradients_passed,
        max_forward_error=forward_error,
        max_gradient_error=gradient_error,
        details=gradient_details,
    )


def _clone_graph_for_backward(root: Tensor) -> dict[int, Tensor]:
    """Rebuild a fresh graph with the same structure and values for backward."""
    clones: dict[int, Tensor] = {}
    order = topological_sort(ComputationGraph.from_tensor(root)).order
    tensors = _collect_tensors(root)

    for node_id in order:
        tensor = tensors[node_id]
        if not tensor.parents:
            clones[node_id] = Tensor(tensor.data.copy(), requires_grad=tensor.requires_grad)
        else:
            parents = tuple(clones[parent.id] for parent in tensor.parents)
            clones[node_id] = _replay_operation(tensor.operation, parents, tensor.metadata)

    return clones


def _collect_tensors(root: Tensor) -> dict[int, Tensor]:
    collected: dict[int, Tensor] = {}

    def visit(tensor: Tensor) -> None:
        if tensor.id in collected:
            return
        for parent in tensor.parents:
            visit(parent)
        collected[tensor.id] = tensor

    visit(root)
    return collected


def _try_apply_rules(
    original: Tensor,
    parents: tuple[Tensor, ...],
) -> tuple[Tensor | None, OptimizationStep | None]:
    operation = original.operation
    if operation == "fake_quantize":
        return None, None

    if operation is None or not parents:
        return None, None

    if operation == "add" and len(parents) == 2:
        left, right = parents
        if _is_additive_zero(right) and not _has_nonfinite(left.data):
            return left, OptimizationStep(
                rule="add_zero",
                original_node_id=original.id,
                original_operation=operation,
                reason="x + 0 → x",
            )
        if _is_additive_zero(left) and not _has_nonfinite(right.data):
            return right, OptimizationStep(
                rule="add_zero",
                original_node_id=original.id,
                original_operation=operation,
                reason="0 + x → x",
            )

    if operation == "sub" and len(parents) == 2:
        left, right = parents
        if _is_additive_zero(right) and not _has_nonfinite(left.data):
            return left, OptimizationStep(
                rule="sub_zero",
                original_node_id=original.id,
                original_operation=operation,
                reason="x - 0 → x",
            )

    if operation == "mul" and len(parents) == 2:
        left, right = parents
        if _is_multiplicative_one(right) and not _has_nonfinite(left.data):
            return left, OptimizationStep(
                rule="mul_one",
                original_node_id=original.id,
                original_operation=operation,
                reason="x * 1 → x",
            )
        if _is_multiplicative_one(left) and not _has_nonfinite(right.data):
            return right, OptimizationStep(
                rule="mul_one",
                original_node_id=original.id,
                original_operation=operation,
                reason="1 * x → x",
            )
        if (
            (_is_multiplicative_zero(right) or _is_multiplicative_zero(left))
            and not _has_nonfinite(left.data)
            and not _has_nonfinite(right.data)
        ):
            zero = Tensor(
                np.zeros_like(original.data),
                requires_grad=original.requires_grad,
            )
            return zero, OptimizationStep(
                rule="mul_zero",
                original_node_id=original.id,
                original_operation=operation,
                reason="x * 0 → 0",
            )

    if operation == "div" and len(parents) == 2:
        left, right = parents
        if _is_multiplicative_one(right) and not _has_nonfinite(left.data):
            return left, OptimizationStep(
                rule="div_one",
                original_node_id=original.id,
                original_operation=operation,
                reason="x / 1 → x",
            )

    return None, None


def _replay_operation(
    operation: str | None,
    parents: tuple[Tensor, ...],
    metadata: dict[str, Any],
) -> Tensor:
    if operation is None:
        raise ValueError("Cannot replay a leaf tensor without an operation.")

    if operation == "add":
        return parents[0] + parents[1]
    if operation == "sub":
        return parents[0] - parents[1]
    if operation == "mul":
        return parents[0] * parents[1]
    if operation == "div":
        return parents[0] / parents[1]
    if operation == "neg":
        return -parents[0]
    if operation == "matmul":
        return parents[0] @ parents[1]
    if operation == "pow":
        return parents[0] ** parents[1]
    if operation == "sum":
        return parents[0].sum(axis=metadata.get("axis"), keepdims=metadata.get("keepdims", False))
    if operation == "mean":
        return parents[0].mean(axis=metadata.get("axis"), keepdims=metadata.get("keepdims", False))
    if operation == "reshape":
        return parents[0].reshape(*metadata["shape"])
    if operation == "transpose":
        axes = metadata.get("axes")
        return parents[0].transpose(*axes) if axes is not None else parents[0].transpose()
    if operation == "fake_quantize":
        return parents[0].fake_quantize(
            bits=int(metadata["bits"]),
            ste_mode=metadata.get("ste_mode", "identity"),
        )

    raise NotImplementedError(f"Graph optimizer cannot replay operation {operation!r}.")


def _is_additive_zero(tensor: Tensor) -> bool:
    return bool(np.all(tensor.data == 0.0))


def _is_multiplicative_one(tensor: Tensor) -> bool:
    return bool(np.all(tensor.data == 1.0))


def _is_multiplicative_zero(tensor: Tensor) -> bool:
    return bool(np.all(tensor.data == 0.0))


def _has_nonfinite(data: np.ndarray) -> bool:
    return not np.all(np.isfinite(data))


def _forward_matches(original: np.ndarray, candidate: np.ndarray) -> bool:
    return bool(
        np.allclose(original, candidate, rtol=FORWARD_TOLERANCE, atol=FORWARD_TOLERANCE, equal_nan=True)
    )


def _max_forward_error(original: np.ndarray, candidate: np.ndarray) -> float:
    if original.shape != candidate.shape:
        return float("inf")
    return float(np.max(np.abs(original - candidate)))


def _backward_for_verification(tensor: Tensor) -> None:
    if tensor.shape == ():
        backward(tensor)
    else:
        backward(tensor, np.ones_like(tensor.data))


def _max_gradient_error(original_grad: Any, optimized_grad: Any) -> float:
    if original_grad is None and optimized_grad is None:
        return 0.0
    if original_grad is None or optimized_grad is None:
        return float("inf")
    return float(np.max(np.abs(np.asarray(original_grad) - np.asarray(optimized_grad))))

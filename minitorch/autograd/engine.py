from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from minitorch.dsa.topological_sort import TopologicalSortStats, topological_sort
from minitorch.graph import ComputationGraph
from minitorch.tensor import Tensor


@dataclass(frozen=True)
class BackwardStats:
    """Real instrumentation collected during one backward pass."""

    processed_nodes: int
    gradient_accumulations: int
    topological_sort: TopologicalSortStats


@dataclass(frozen=True)
class BackwardResult:
    """Structured result returned by `backward` for inspection."""

    graph: ComputationGraph
    topological_order: list[int]
    stats: BackwardStats
    seed_gradient: np.ndarray


def backward(output: Tensor, gradient: Any | None = None) -> BackwardResult:
    """
    Run reverse-mode automatic differentiation from `output`.

    The algorithm is deliberately explicit:

    1. Reconstruct the computation graph from Tensor parent links.
    2. Topologically sort dependencies before consumers.
    3. Seed the output gradient.
    4. Walk the topological order in reverse.
    5. Apply local derivative rules and accumulate parent gradients.
    """

    if not output.requires_grad:
        raise RuntimeError("Cannot call backward on a tensor that does not require gradients.")

    graph = ComputationGraph.from_tensor(output)
    topo = topological_sort(graph)
    seed = _seed_gradient(output, gradient)

    local_grads: dict[int, np.ndarray] = {output.id: seed}
    tensors = {
        node_id: node.tensor
        for node_id, node in graph.nodes.items()
        if node.tensor is not None
    }

    processed_nodes = 0
    gradient_accumulations = 0

    for node_id in reversed(topo.order):
        tensor = tensors[node_id]
        upstream = local_grads.get(node_id)
        if upstream is None:
            continue

        processed_nodes += 1

        for parent, parent_grad in _local_gradients(tensor, upstream):
            if not parent.requires_grad:
                continue

            reduced = _unbroadcast(parent_grad, parent.shape)
            if parent.id in local_grads:
                local_grads[parent.id] = local_grads[parent.id] + reduced
            else:
                local_grads[parent.id] = reduced.copy()
            gradient_accumulations += 1

    for node_id, grad_value in local_grads.items():
        tensor = tensors[node_id]
        if not tensor.requires_grad:
            continue

        if tensor.grad is None:
            tensor.grad = grad_value.copy()
        else:
            tensor.grad = tensor.grad + grad_value

    result = BackwardResult(
        graph=graph,
        topological_order=topo.order,
        stats=BackwardStats(
            processed_nodes=processed_nodes,
            gradient_accumulations=gradient_accumulations,
            topological_sort=topo.stats,
        ),
        seed_gradient=seed.copy(),
    )
    output._last_backward_result = result
    return result


def _seed_gradient(output: Tensor, gradient: Any | None) -> np.ndarray:
    if gradient is None:
        if output.shape != ():
            raise RuntimeError(
                "Gradient must be provided when calling backward on a non-scalar tensor."
            )
        return np.ones_like(output.data, dtype=np.float64)

    grad = np.asarray(gradient.data if isinstance(gradient, Tensor) else gradient, dtype=np.float64)
    if grad.shape != output.shape:
        raise ValueError(
            f"Backward gradient shape {grad.shape} does not match output shape {output.shape}."
        )
    return grad.copy()


def _local_gradients(tensor: Tensor, upstream: np.ndarray) -> list[tuple[Tensor, np.ndarray]]:
    operation = tensor.operation
    parents = tensor.parents

    if operation is None:
        return []

    if operation == "add":
        return [(parents[0], upstream), (parents[1], upstream)]

    if operation == "sub":
        return [(parents[0], upstream), (parents[1], -upstream)]

    if operation == "mul":
        left, right = parents
        return [(left, upstream * right.data), (right, upstream * left.data)]

    if operation == "div":
        numerator, denominator = parents
        return [
            (numerator, upstream / denominator.data),
            (denominator, -upstream * numerator.data / (denominator.data**2)),
        ]

    if operation == "neg":
        return [(parents[0], -upstream)]

    if operation == "matmul":
        left, right = parents
        left_grad, right_grad = _matmul_gradients(upstream, left.data, right.data)
        return [(left, left_grad), (right, right_grad)]

    if operation == "pow":
        base, exponent = parents
        grads: list[tuple[Tensor, np.ndarray]] = []
        if base.requires_grad:
            grads.append((base, upstream * exponent.data * (base.data ** (exponent.data - 1))))
        if exponent.requires_grad:
            grads.append((exponent, upstream * tensor.data * np.log(base.data)))
        return grads

    if operation == "sum":
        parent = parents[0]
        return [(parent, _reduction_gradient(upstream, parent, tensor.metadata))]

    if operation == "mean":
        parent = parents[0]
        count = _reduction_count(parent, tensor.metadata)
        return [(parent, _reduction_gradient(upstream, parent, tensor.metadata) / count)]

    if operation == "reshape":
        parent = parents[0]
        return [(parent, upstream.reshape(parent.shape))]

    if operation == "transpose":
        parent = parents[0]
        axes = tensor.metadata.get("axes")
        if axes is None:
            return [(parent, np.transpose(upstream))]
        inverse_axes = np.argsort(axes)
        return [(parent, np.transpose(upstream, axes=inverse_axes))]

    if operation == "exp":
        parent = parents[0]
        return [(parent, upstream * tensor.data)]

    if operation == "log":
        parent = parents[0]
        return [(parent, upstream / parent.data)]

    if operation == "relu":
        parent = parents[0]
        mask = (parent.data > 0).astype(np.float64)
        return [(parent, upstream * mask)]

    if operation == "sigmoid":
        parent = parents[0]
        output = tensor.data
        return [(parent, upstream * output * (1.0 - output))]

    if operation == "tanh":
        parent = parents[0]
        output = tensor.data
        return [(parent, upstream * (1.0 - output**2))]

    if operation == "fake_quantize":
        parent = parents[0]
        ste_mode = tensor.metadata.get("ste_mode", "identity")
        if ste_mode == "identity":
            return [(parent, upstream)]
        raise NotImplementedError(f"No STE backward rule for fake_quantize mode {ste_mode!r}.")

    raise NotImplementedError(f"No backward rule has been implemented for operation {operation!r}.")


def _matmul_gradients(
    upstream: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_was_vector = left.ndim == 1
    right_was_vector = right.ndim == 1

    left_matrix = left.reshape(1, left.shape[0]) if left_was_vector else left
    right_matrix = right.reshape(right.shape[0], 1) if right_was_vector else right

    if upstream.ndim == 0:
        upstream_matrix = upstream.reshape(1, 1)
    elif left_was_vector and not right_was_vector:
        upstream_matrix = np.expand_dims(upstream, axis=-2)
    elif right_was_vector:
        upstream_matrix = np.expand_dims(upstream, axis=-1)
    else:
        upstream_matrix = upstream

    left_grad = np.matmul(upstream_matrix, np.swapaxes(right_matrix, -1, -2))
    right_grad = np.matmul(np.swapaxes(left_matrix, -1, -2), upstream_matrix)

    if left_was_vector:
        left_grad = np.squeeze(left_grad, axis=-2)
    if right_was_vector:
        right_grad = np.squeeze(right_grad, axis=-1)

    return left_grad, right_grad


def _reduction_gradient(
    upstream: np.ndarray,
    parent: Tensor,
    metadata: dict[str, Any],
) -> np.ndarray:
    axis = metadata.get("axis")
    keepdims = metadata.get("keepdims", False)
    grad = upstream

    if axis is not None and not keepdims:
        for axis_index in sorted(_normalize_reduction_axes(axis, parent.ndim)):
            grad = np.expand_dims(grad, axis=axis_index)

    return np.ones_like(parent.data, dtype=np.float64) * grad


def _reduction_count(parent: Tensor, metadata: dict[str, Any]) -> int:
    axis = metadata.get("axis")
    if axis is None:
        return parent.size

    count = 1
    for axis_index in _normalize_reduction_axes(axis, parent.ndim):
        count *= parent.shape[axis_index]
    return count


def _normalize_reduction_axes(axis: int | tuple[int, ...], ndim: int) -> tuple[int, ...]:
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    return tuple(axis_index % ndim for axis_index in axes)


def _unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    grad = np.asarray(grad, dtype=np.float64)

    if target_shape == ():
        return np.asarray(grad.sum(), dtype=np.float64)

    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    for axis, target_dim in enumerate(target_shape):
        if target_dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(target_shape)

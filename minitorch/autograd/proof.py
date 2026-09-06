from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from minitorch.tensor import Tensor

from .engine import (
    _local_gradients,
    _unbroadcast,
    backward,
)


@dataclass(frozen=True)
class ProofStep:
    """One backward contribution flowing into the proof target."""

    consumer_id: int
    consumer_operation: str | None
    parent_id: int
    upstream_gradient: np.ndarray
    local_derivative: np.ndarray
    contribution: np.ndarray
    forward_value: np.ndarray


@dataclass(frozen=True)
class GradientProof:
    """Structured explanation of how a target tensor received its gradient."""

    output_id: int
    target_id: int
    steps: tuple[ProofStep, ...]
    accumulated_gradient: np.ndarray
    path_count: int


def build_gradient_proof(output: Tensor, target: Tensor) -> GradientProof:
    """
    Explain gradient flow to `target` using the same backward rules as autograd.

    Replays the reverse topological walk with `_local_gradients` — the proof
    observes the engine's derivative rules rather than implementing a second
    autodiff system.
    """

    if output._last_backward_result is None:
        backward(output)

    result = output._last_backward_result
    if target.id not in result.graph.nodes:
        raise ValueError(f"Target tensor id {target.id} is not in the output graph.")

    tensors = {
        node_id: node.tensor
        for node_id, node in result.graph.nodes.items()
        if node.tensor is not None
    }

    local_grads: dict[int, np.ndarray] = {output.id: result.seed_gradient.copy()}
    steps: list[ProofStep] = []

    for node_id in reversed(result.topological_order):
        tensor = tensors[node_id]
        upstream = local_grads.get(node_id)
        if upstream is None:
            continue

        for parent, parent_grad in _local_gradients(tensor, upstream):
            if not parent.requires_grad:
                continue

            reduced = _unbroadcast(parent_grad, parent.shape)

            if parent.id == target.id:
                steps.append(
                    ProofStep(
                        consumer_id=node_id,
                        consumer_operation=tensor.operation,
                        parent_id=parent.id,
                        upstream_gradient=upstream.copy(),
                        local_derivative=parent_grad.copy(),
                        contribution=reduced.copy(),
                        forward_value=parent.data.copy(),
                    )
                )

            if parent.id in local_grads:
                local_grads[parent.id] = local_grads[parent.id] + reduced
            else:
                local_grads[parent.id] = reduced.copy()

    if target.grad is not None:
        accumulated = np.asarray(target.grad, dtype=np.float64).copy()
    elif target.id in local_grads:
        accumulated = local_grads[target.id].copy()
    else:
        accumulated = np.zeros_like(target.data, dtype=np.float64)

    return GradientProof(
        output_id=output.id,
        target_id=target.id,
        steps=tuple(steps),
        accumulated_gradient=accumulated,
        path_count=len(steps),
    )


def format_gradient_proof(proof: GradientProof) -> str:
    lines = [
        "GRADIENT PROOF",
        "==============",
        f"Output tensor id: {proof.output_id}",
        f"Target tensor id: {proof.target_id}",
        f"Contributing paths: {proof.path_count}",
        "",
    ]

    for index, step in enumerate(proof.steps, start=1):
        operation_note = ""
        if step.consumer_operation == "fake_quantize":
            operation_note = " (surrogate STE backward: dloss/dx ≈ upstream gradient)"
        lines.extend(
            [
                f"Step {index}",
                f"  Consumer id: {step.consumer_id}",
                f"  Operation: {step.consumer_operation!r}{operation_note}",
                f"  Parent forward value: {step.forward_value}",
                f"  Upstream gradient: {step.upstream_gradient}",
                f"  Local derivative factor: {step.local_derivative}",
                f"  Contribution to target: {step.contribution}",
                "",
            ]
        )

    lines.append(f"Accumulated gradient: {proof.accumulated_gradient}")
    return "\n".join(lines)

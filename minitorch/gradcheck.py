from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .tensor import Tensor


@dataclass(frozen=True)
class GradientCheckInputResult:
    """Gradient-check result for one input tensor."""

    input_index: int
    shape: tuple[int, ...]
    analytical: np.ndarray
    numerical: np.ndarray
    absolute_error: np.ndarray
    relative_error: np.ndarray
    max_absolute_error: float
    max_relative_error: float
    passed: bool


@dataclass(frozen=True)
class GradientCheckResult:
    """Structured finite-difference verification result."""

    passed: bool
    input_results: tuple[GradientCheckInputResult, ...]
    epsilon: float
    absolute_tolerance: float
    relative_tolerance: float

    def __bool__(self) -> bool:
        return self.passed


def gradcheck(
    function: Callable[..., Tensor],
    inputs: Tensor | Sequence[Tensor],
    *,
    epsilon: float = 1e-6,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-4,
) -> GradientCheckResult:
    """
    Compare analytical gradients against central finite differences.

    `function` must return a scalar Tensor. Each check evaluates:

        f(x + epsilon) - f(x - epsilon)
        ---------------------------------
                  2 * epsilon

    Fresh Tensor graphs are built for analytical and numerical evaluations.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be non-negative.")
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative.")

    input_tuple = _normalize_inputs(inputs)

    analytical_inputs = _clone_inputs(input_tuple, requires_grad=True)
    analytical_output = function(*analytical_inputs)
    _validate_scalar_tensor_output(analytical_output)
    analytical_output.backward()

    input_results: list[GradientCheckInputResult] = []

    for input_index, original in enumerate(input_tuple):
        analytical = _analytical_gradient(analytical_inputs[input_index])
        numerical = _numerical_gradient(function, input_tuple, input_index, epsilon)
        absolute_error = np.abs(analytical - numerical)
        relative_error = _relative_error(analytical, numerical, absolute_error)
        allowed_error = absolute_tolerance + relative_tolerance * np.maximum(
            np.abs(analytical),
            np.abs(numerical),
        )
        passed = bool(np.all(absolute_error <= allowed_error))

        input_results.append(
            GradientCheckInputResult(
                input_index=input_index,
                shape=original.shape,
                analytical=analytical,
                numerical=numerical,
                absolute_error=absolute_error,
                relative_error=relative_error,
                max_absolute_error=float(absolute_error.max()) if absolute_error.size else 0.0,
                max_relative_error=float(relative_error.max()) if relative_error.size else 0.0,
                passed=passed,
            )
        )

    return GradientCheckResult(
        passed=all(item.passed for item in input_results),
        input_results=tuple(input_results),
        epsilon=epsilon,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


def _normalize_inputs(inputs: Tensor | Sequence[Tensor]) -> tuple[Tensor, ...]:
    if isinstance(inputs, Tensor):
        return (inputs,)

    input_tuple = tuple(inputs)
    if not input_tuple:
        raise ValueError("gradcheck requires at least one input tensor.")

    for item in input_tuple:
        if not isinstance(item, Tensor):
            raise TypeError("gradcheck inputs must be Tensor instances.")

    return input_tuple


def _clone_inputs(inputs: tuple[Tensor, ...], *, requires_grad: bool) -> list[Tensor]:
    return [
        Tensor(input_tensor.data.copy(), requires_grad=requires_grad)
        for input_tensor in inputs
    ]


def _validate_scalar_tensor_output(output: Any) -> None:
    if not isinstance(output, Tensor):
        raise TypeError("gradcheck function must return a Tensor.")
    if output.shape != ():
        raise ValueError("gradcheck requires the function to return a scalar Tensor.")


def _analytical_gradient(input_tensor: Tensor) -> np.ndarray:
    if input_tensor.grad is None:
        return np.zeros_like(input_tensor.data, dtype=np.float64)
    return np.asarray(input_tensor.grad, dtype=np.float64).copy()


def _numerical_gradient(
    function: Callable[..., Tensor],
    original_inputs: tuple[Tensor, ...],
    input_index: int,
    epsilon: float,
) -> np.ndarray:
    numerical = np.zeros_like(original_inputs[input_index].data, dtype=np.float64)

    for element_index in _element_indices(original_inputs[input_index].shape):
        plus_inputs = _clone_inputs(original_inputs, requires_grad=True)
        minus_inputs = _clone_inputs(original_inputs, requires_grad=True)

        plus_inputs[input_index].data[element_index] += epsilon
        minus_inputs[input_index].data[element_index] -= epsilon

        plus_value = _scalar_value(function(*plus_inputs))
        minus_value = _scalar_value(function(*minus_inputs))
        numerical[element_index] = (plus_value - minus_value) / (2 * epsilon)

    return numerical


def _element_indices(shape: tuple[int, ...]):
    if shape == ():
        yield ()
        return

    yield from np.ndindex(shape)


def _scalar_value(output: Any) -> float:
    _validate_scalar_tensor_output(output)
    return float(output.data)


def _relative_error(
    analytical: np.ndarray,
    numerical: np.ndarray,
    absolute_error: np.ndarray,
) -> np.ndarray:
    denominator = np.maximum(np.maximum(np.abs(analytical), np.abs(numerical)), 1e-12)
    return absolute_error / denominator

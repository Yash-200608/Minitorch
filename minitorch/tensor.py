from __future__ import annotations

from itertools import count
from typing import Any

import numpy as np


_TENSOR_ID_COUNTER = count()


class Tensor:
    """
    Core numerical object used by MiniTorch.

    A Tensor stores NumPy data plus the parent/operation lineage used to rebuild
    a computation graph. `backward()` runs reverse-mode autodiff on that graph.
    """

    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        parents: tuple["Tensor", ...] | None = None,
        operation: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if isinstance(data, Tensor):
            data = data.data

        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = bool(requires_grad)

        self.grad = None
        self._last_backward_result = None

        self.id = next(_TENSOR_ID_COUNTER)
        self.parents = tuple(parents or ())
        self.operation = operation
        self.metadata = dict(metadata or {})
        self.children: list["Tensor"] = []

        for parent in self.parents:
            parent.children.append(self)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def __repr__(self) -> str:
        return (
            "Tensor("
            f"data={self.data}, "
            f"shape={self.shape}, "
            f"dtype={self.dtype}, "
            f"requires_grad={self.requires_grad}, "
            f"operation={self.operation!r}"
            ")"
        )

    def __add__(self, other: Any) -> "Tensor":
        other = self._ensure_tensor(other)
        return self._from_operation_compute(
            lambda: self.data + other.data,
            operation="add",
            parents=(self, other),
            symbol="+",
        )

    def __radd__(self, other: Any) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Tensor":
        other = self._ensure_tensor(other)
        return self._from_operation_compute(
            lambda: self.data - other.data,
            operation="sub",
            parents=(self, other),
            symbol="-",
        )

    def __rsub__(self, other: Any) -> "Tensor":
        other = self._ensure_tensor(other)
        return other.__sub__(self)

    def __mul__(self, other: Any) -> "Tensor":
        other = self._ensure_tensor(other)
        return self._from_operation_compute(
            lambda: self.data * other.data,
            operation="mul",
            parents=(self, other),
            symbol="*",
        )

    def __rmul__(self, other: Any) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "Tensor":
        other = self._ensure_tensor(other)
        return self._from_operation_compute(
            lambda: self.data / other.data,
            operation="div",
            parents=(self, other),
            symbol="/",
        )

    def __rtruediv__(self, other: Any) -> "Tensor":
        other = self._ensure_tensor(other)
        return other.__truediv__(self)

    def __neg__(self) -> "Tensor":
        return self._from_operation_compute(
            lambda: -self.data,
            operation="neg",
            parents=(self,),
            symbol="neg",
        )

    def __matmul__(self, other: Any) -> "Tensor":
        other = self._ensure_tensor(other)
        return self._from_operation_compute(
            lambda: self.data @ other.data,
            operation="matmul",
            parents=(self, other),
            symbol="@",
        )

    def __rmatmul__(self, other: Any) -> "Tensor":
        other = self._ensure_tensor(other)
        return other.__matmul__(self)

    def __pow__(self, exponent: Any) -> "Tensor":
        exponent = self._ensure_tensor(exponent)
        return self._from_operation_compute(
            lambda: self.data**exponent.data,
            operation="pow",
            parents=(self, exponent),
            symbol="**",
        )

    def __rpow__(self, base: Any) -> "Tensor":
        base = self._ensure_tensor(base)
        return base.__pow__(self)

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        return self._from_operation_compute(
            lambda: self.data.sum(axis=axis, keepdims=keepdims),
            operation="sum",
            parents=(self,),
            axis=axis,
            keepdims=keepdims,
        )

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        return self._from_operation_compute(
            lambda: self.data.mean(axis=axis, keepdims=keepdims),
            operation="mean",
            parents=(self,),
            axis=axis,
            keepdims=keepdims,
        )

    def reshape(self, *shape: int | tuple[int, ...]) -> "Tensor":
        target_shape = self._normalize_shape(shape)
        return self._from_operation_compute(
            lambda: self.data.reshape(target_shape),
            operation="reshape",
            parents=(self,),
            shape=target_shape,
        )

    def transpose(self, *axes: int | tuple[int, ...]) -> "Tensor":
        normalized_axes = self._normalize_axes(axes)
        return self._from_operation_compute(
            lambda: np.transpose(self.data, axes=normalized_axes),
            operation="transpose",
            parents=(self,),
            axes=normalized_axes,
        )

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def exp(self) -> "Tensor":
        return self._from_operation_compute(
            lambda: np.exp(self.data),
            operation="exp",
            parents=(self,),
            symbol="exp",
        )

    def log(self) -> "Tensor":
        return self._from_operation_compute(
            lambda: np.log(self.data),
            operation="log",
            parents=(self,),
            symbol="log",
        )

    def relu(self) -> "Tensor":
        return self._from_operation_compute(
            lambda: np.maximum(self.data, 0.0),
            operation="relu",
            parents=(self,),
            symbol="relu",
        )

    def sigmoid(self) -> "Tensor":
        return self._from_operation_compute(
            lambda: _stable_sigmoid(self.data),
            operation="sigmoid",
            parents=(self,),
            symbol="sigmoid",
        )

    def tanh(self) -> "Tensor":
        return self._from_operation_compute(
            lambda: np.tanh(self.data),
            operation="tanh",
            parents=(self,),
            symbol="tanh",
        )

    def fake_quantize(self, bits: int = 8, ste_mode: str = "identity") -> "Tensor":
        """
        Fake uniform quantization for quantization-aware training.

        Forward uses dequantize(quantize(x)). Backward uses the Straight-Through
        Estimator (surrogate gradient), not the true derivative of quantization.
        """
        from .qat import fake_quantize_uniform_array, validate_ste_mode

        validate_ste_mode(ste_mode)
        return self._from_operation_compute(
            lambda: fake_quantize_uniform_array(self.data, bits),
            operation="fake_quantize",
            parents=(self,),
            bits=bits,
            ste_mode=ste_mode,
            method="uniform",
            symbol="fake_quantize",
        )

    def numpy(self) -> np.ndarray:
        """Return a copy of the underlying NumPy array."""
        return self.data.copy()

    def backward(self, gradient: Any | None = None):
        """Compute gradients for tensors that contributed to this tensor."""
        from .autograd.engine import backward

        return backward(self, gradient=gradient)

    def explore(self):
        """Return a graph explorer for this tensor's computation DAG."""
        from .autograd.explorer import GraphExplorer

        return GraphExplorer.from_tensor(self)

    def proof(self, target: "Tensor"):
        """Build a structured gradient proof explaining `target`'s gradient."""
        from .autograd.proof import build_gradient_proof

        return build_gradient_proof(self, target)

    def explain(self, target: "Tensor") -> str:
        """Human-readable gradient proof for `target`."""
        from .autograd.proof import format_gradient_proof

        return format_gradient_proof(self.proof(target))

    def zero_grad(self) -> None:
        """Clear this tensor's accumulated gradient."""
        self.grad = None

    def quantize(
        self,
        bits: int = 8,
        method: str = "uniform",
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        packed: bool = False,
        rotation: str | None = None,
        parallel: bool = False,
        workers: int | None = None,
    ) -> "QuantizedTensor":
        """
        Scalar quantization into a separate QuantizedTensor.

        `method` is ``uniform`` (Phase E default) or ``codebook`` (learned centroids).
        `packed=True` stores codes in compact uint8 bit-packed bytes.
        Does not modify this tensor or attach to the autograd graph.
        """
        from .quantization import quantize_array

        return quantize_array(
            self.data,
            bits,
            method=method,
            max_iterations=max_iterations,
            tolerance=tolerance,
            packed=packed,
            rotation=rotation,
            parallel=parallel,
            workers=workers,
        )

    def quantization_report(
        self,
        bits: int = 8,
        method: str = "uniform",
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        packed: bool = False,
        rotation: str | None = None,
    ) -> "QuantizationReport":
        """Quantize, dequantize, and return a structured inspection report."""
        return self.quantize(
            bits=bits,
            method=method,
            max_iterations=max_iterations,
            tolerance=tolerance,
            packed=packed,
            rotation=rotation,
        ).report()

    def inspect(self) -> dict[str, Any]:
        """Return structured state for educational inspection."""
        return {
            "id": self.id,
            "shape": self.shape,
            "ndim": self.ndim,
            "size": self.size,
            "dtype": str(self.dtype),
            "requires_grad": self.requires_grad,
            "grad": self.grad,
            "operation": self.operation,
            "parent_ids": [parent.id for parent in self.parents],
            "child_ids": [child.id for child in self.children],
            "metadata": dict(self.metadata),
        }

    def lineage(self) -> list[dict[str, Any]]:
        """Return this tensor's dependency lineage in dependency-first order."""
        ordered: list[Tensor] = []
        visited: set[int] = set()

        def visit(tensor: Tensor) -> None:
            if tensor.id in visited:
                return
            visited.add(tensor.id)
            for parent in tensor.parents:
                visit(parent)
            ordered.append(tensor)

        visit(self)
        return [tensor.inspect() for tensor in ordered]

    @classmethod
    def _from_operation_compute(
        cls,
        compute_fn: Any,
        operation: str,
        parents: tuple["Tensor", ...],
        **metadata: Any,
    ) -> "Tensor":
        from .cache import get_computation_cache

        cache = get_computation_cache()
        if cache.enabled:
            key = cache.make_key(operation, parents, metadata)
            data = cache.get_or_compute(key, lambda: np.asarray(compute_fn(), dtype=np.float64))
        else:
            data = compute_fn()
        return cls._from_operation(data, operation, parents, **metadata)

    @classmethod
    def _from_operation(
        cls,
        data: Any,
        operation: str,
        parents: tuple["Tensor", ...],
        **metadata: Any,
    ) -> "Tensor":
        return Tensor(
            data,
            requires_grad=any(parent.requires_grad for parent in parents),
            parents=parents,
            operation=operation,
            metadata=metadata,
        )

    @staticmethod
    def _ensure_tensor(value: Any) -> "Tensor":
        if isinstance(value, Tensor):
            return value

        return Tensor(value)

    @staticmethod
    def _normalize_shape(shape: tuple[int | tuple[int, ...], ...]) -> tuple[int, ...]:
        if len(shape) == 1 and isinstance(shape[0], tuple):
            return shape[0]
        return tuple(int(dim) for dim in shape)

    @staticmethod
    def _normalize_axes(axes: tuple[int | tuple[int, ...], ...]) -> tuple[int, ...] | None:
        if not axes:
            return None
        if len(axes) == 1 and isinstance(axes[0], tuple):
            return axes[0]
        return tuple(int(axis) for axis in axes)


def _stable_sigmoid(data: np.ndarray) -> np.ndarray:
    """Sigmoid with clipping to reduce overflow in exp for large magnitudes."""
    clipped = np.clip(data, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))

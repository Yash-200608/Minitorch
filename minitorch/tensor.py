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
        return self._from_operation(
            self.data + other.data,
            operation="add",
            parents=(self, other),
            symbol="+",
        )

    def __radd__(self, other: Any) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Tensor":
        other = self._ensure_tensor(other)
        return self._from_operation(
            self.data - other.data,
            operation="sub",
            parents=(self, other),
            symbol="-",
        )

    def __rsub__(self, other: Any) -> "Tensor":
        other = self._ensure_tensor(other)
        return other.__sub__(self)

    def __mul__(self, other: Any) -> "Tensor":
        other = self._ensure_tensor(other)
        return self._from_operation(
            self.data * other.data,
            operation="mul",
            parents=(self, other),
            symbol="*",
        )

    def __rmul__(self, other: Any) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "Tensor":
        other = self._ensure_tensor(other)
        return self._from_operation(
            self.data / other.data,
            operation="div",
            parents=(self, other),
            symbol="/",
        )

    def __rtruediv__(self, other: Any) -> "Tensor":
        other = self._ensure_tensor(other)
        return other.__truediv__(self)

    def __neg__(self) -> "Tensor":
        return self._from_operation(
            -self.data,
            operation="neg",
            parents=(self,),
            symbol="neg",
        )

    def __matmul__(self, other: Any) -> "Tensor":
        other = self._ensure_tensor(other)
        return self._from_operation(
            self.data @ other.data,
            operation="matmul",
            parents=(self, other),
            symbol="@",
        )

    def __rmatmul__(self, other: Any) -> "Tensor":
        other = self._ensure_tensor(other)
        return other.__matmul__(self)

    def __pow__(self, exponent: Any) -> "Tensor":
        exponent = self._ensure_tensor(exponent)
        return self._from_operation(
            self.data**exponent.data,
            operation="pow",
            parents=(self, exponent),
            symbol="**",
        )

    def __rpow__(self, base: Any) -> "Tensor":
        base = self._ensure_tensor(base)
        return base.__pow__(self)

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        return self._from_operation(
            self.data.sum(axis=axis, keepdims=keepdims),
            operation="sum",
            parents=(self,),
            axis=axis,
            keepdims=keepdims,
        )

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        return self._from_operation(
            self.data.mean(axis=axis, keepdims=keepdims),
            operation="mean",
            parents=(self,),
            axis=axis,
            keepdims=keepdims,
        )

    def reshape(self, *shape: int | tuple[int, ...]) -> "Tensor":
        target_shape = self._normalize_shape(shape)
        return self._from_operation(
            self.data.reshape(target_shape),
            operation="reshape",
            parents=(self,),
            shape=target_shape,
        )

    def transpose(self, *axes: int | tuple[int, ...]) -> "Tensor":
        normalized_axes = self._normalize_axes(axes)
        return self._from_operation(
            np.transpose(self.data, axes=normalized_axes),
            operation="transpose",
            parents=(self,),
            axes=normalized_axes,
        )

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def numpy(self) -> np.ndarray:
        """Return a copy of the underlying NumPy array."""
        return self.data.copy()

    def backward(self, gradient: Any | None = None):
        """Compute gradients for tensors that contributed to this tensor."""
        from .autograd import backward

        return backward(self, gradient=gradient)

    def zero_grad(self) -> None:
        """Clear this tensor's accumulated gradient."""
        self.grad = None

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
    def _from_operation(
        cls,
        data: Any,
        operation: str,
        parents: tuple["Tensor", ...],
        **metadata: Any,
    ) -> "Tensor":
        return cls(
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

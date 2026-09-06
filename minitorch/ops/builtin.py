from __future__ import annotations

from .names import (
    ADD,
    DIV,
    MATMUL,
    MEAN,
    MUL,
    NEG,
    POW,
    RESHAPE,
    SUB,
    SUM,
    TRANSPOSE,
)
from .registry import OperationRegistry
from .specs import OperationSpec

_BUILTIN_SPECS: tuple[OperationSpec, ...] = (
    OperationSpec(
        name=ADD,
        category="elementwise",
        arity=2,
        differentiable=True,
        cacheable=True,
        description="Elementwise addition.",
    ),
    OperationSpec(
        name=SUB,
        category="elementwise",
        arity=2,
        differentiable=True,
        cacheable=True,
        description="Elementwise subtraction.",
    ),
    OperationSpec(
        name=MUL,
        category="elementwise",
        arity=2,
        differentiable=True,
        cacheable=True,
        description="Elementwise multiplication.",
    ),
    OperationSpec(
        name=DIV,
        category="elementwise",
        arity=2,
        differentiable=True,
        cacheable=True,
        description="Elementwise division.",
    ),
    OperationSpec(
        name=NEG,
        category="unary",
        arity=1,
        differentiable=True,
        cacheable=True,
        description="Unary negation.",
    ),
    OperationSpec(
        name=MATMUL,
        category="linear_algebra",
        arity=2,
        differentiable=True,
        cacheable=False,
        description="Matrix multiplication (@).",
    ),
    OperationSpec(
        name=POW,
        category="elementwise",
        arity=2,
        differentiable=True,
        cacheable=True,
        description="Elementwise power.",
    ),
    OperationSpec(
        name=SUM,
        category="reduction",
        arity=1,
        differentiable=True,
        cacheable=False,
        description="Sum reduction.",
    ),
    OperationSpec(
        name=MEAN,
        category="reduction",
        arity=1,
        differentiable=True,
        cacheable=False,
        description="Mean reduction.",
    ),
    OperationSpec(
        name=RESHAPE,
        category="shape",
        arity=1,
        differentiable=True,
        cacheable=False,
        description="Reshape view with new shape metadata.",
    ),
    OperationSpec(
        name=TRANSPOSE,
        category="shape",
        arity=1,
        differentiable=True,
        cacheable=False,
        description="Transpose or axis permutation.",
    ),
)


def build_builtin_registry() -> OperationRegistry:
    registry = OperationRegistry()
    for spec in _BUILTIN_SPECS:
        registry.register(spec)
    return registry


_DEFAULT_REGISTRY = build_builtin_registry()


def get_operation_registry() -> OperationRegistry:
    return _DEFAULT_REGISTRY

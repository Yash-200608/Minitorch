from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperationSpec:
    """
    Descriptive metadata for one MiniTorch tensor operation.

    Specs describe behavior; they do not execute operations.
    """

    name: str
    category: str
    arity: int
    differentiable: bool
    cacheable: bool
    description: str = ""

    def __post_init__(self) -> None:
        if self.arity < 0:
            raise ValueError("arity must be non-negative.")
        if not self.name:
            raise ValueError("name must be non-empty.")

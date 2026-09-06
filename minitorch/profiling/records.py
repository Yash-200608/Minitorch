from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExecutionRecord:
    """Immutable snapshot of one observed tensor operation execution."""

    sequence: int
    operation: str
    input_shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]
    dtype: str
    duration_ns: int
    cache_hit: bool
    cache_miss: bool
    tensor_id: int
    parent_ids: tuple[int, ...]
    parameter_name: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.duration_ns < 0:
            raise ValueError("duration_ns must be non-negative.")
        if self.cache_hit and self.cache_miss:
            raise ValueError("cache_hit and cache_miss cannot both be true.")

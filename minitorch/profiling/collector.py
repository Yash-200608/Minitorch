from __future__ import annotations

from typing import TYPE_CHECKING

from .records import ExecutionRecord

if TYPE_CHECKING:
    from minitorch.tensor import Tensor


class ExecutionCollector:
    """
    Optional execution observability layer.

    When disabled, MiniTorch tensor execution does not record operations.
    """

    def __init__(self) -> None:
        self._enabled = False
        self._records: list[ExecutionRecord] = []
        self._sequence = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def clear(self) -> None:
        self._records.clear()
        self._sequence = 0

    def records(self) -> tuple[ExecutionRecord, ...]:
        return tuple(self._records)

    def record_operation(
        self,
        operation: str,
        parents: tuple["Tensor", ...],
        output: "Tensor",
        *,
        duration_ns: int,
        cache_hit: bool = False,
        cache_miss: bool = False,
        parameter_name: str | None = None,
        error: str | None = None,
    ) -> ExecutionRecord:
        record = ExecutionRecord(
            sequence=self._sequence,
            operation=operation,
            input_shapes=tuple(parent.shape for parent in parents),
            output_shape=output.shape,
            dtype=str(output.dtype),
            duration_ns=duration_ns,
            cache_hit=cache_hit,
            cache_miss=cache_miss,
            tensor_id=output.id,
            parent_ids=tuple(parent.id for parent in parents),
            parameter_name=parameter_name,
            error=error,
        )
        self._sequence += 1
        self._records.append(record)
        return record

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class HeapStats:
    """Instrumentation collected while running heap-based Top-K."""

    heap_pushes: int = 0
    heap_pops: int = 0
    comparisons: int = 0
    items_scanned: int = 0
    complexity: str = "O(N log K)"


@dataclass(frozen=True)
class TopKResult:
    """Indices and values of the selected Top-K elements."""

    indices: tuple[int, ...]
    values: tuple[float, ...]
    stats: HeapStats


class BinaryHeap:
    """
    Explicit binary min-heap for educational Top-K selection.

    The heap stores `(priority, index)` pairs. For Top-K largest values we keep a
    min-heap of size K so the smallest candidate among the current winners sits
    at the root and can be evicted in O(log K).
    """

    def __init__(self, key: Callable[[Any], float] | None = None):
        self._items: list[tuple[float, int]] = []
        self.heap_pushes = 0
        self.heap_pops = 0
        self.comparisons = 0
        self._key = key or (lambda item: item[0])

    def __len__(self) -> int:
        return len(self._items)

    def peek(self) -> tuple[float, int]:
        if not self._items:
            raise IndexError("peek from empty heap")
        return self._items[0]

    def push(self, item: tuple[float, int]) -> None:
        self._items.append(item)
        self.heap_pushes += 1
        self._sift_up(len(self._items) - 1)

    def pop(self) -> tuple[float, int]:
        if not self._items:
            raise IndexError("pop from empty heap")
        root = self._items[0]
        last = self._items.pop()
        self.heap_pops += 1
        if self._items:
            self._items[0] = last
            self._sift_down(0)
        return root

    def replace_root(self, item: tuple[float, int]) -> tuple[float, int]:
        """Replace the root and sift down — one eviction step for Top-K."""
        if not self._items:
            self.push(item)
            return item
        root = self._items[0]
        self._items[0] = item
        self.heap_pushes += 1
        self.heap_pops += 1
        self._sift_down(0)
        return root

    def items(self) -> list[tuple[float, int]]:
        return list(self._items)

    def _sift_up(self, index: int) -> None:
        item = self._items[index]
        while index > 0:
            parent = (index - 1) // 2
            self.comparisons += 1
            if self._key(item) >= self._key(self._items[parent]):
                break
            self._items[index] = self._items[parent]
            index = parent
        self._items[index] = item

    def _sift_down(self, index: int) -> None:
        size = len(self._items)
        item = self._items[index]
        while True:
            left = 2 * index + 1
            if left >= size:
                break
            right = left + 1
            smallest = left
            self.comparisons += 1
            if right < size:
                self.comparisons += 1
                if self._key(self._items[right]) < self._key(self._items[left]):
                    smallest = right
            self.comparisons += 1
            if self._key(item) <= self._key(self._items[smallest]):
                break
            self._items[index] = self._items[smallest]
            index = smallest
        self._items[index] = item


def topk(
    values: Sequence[float] | Any,
    k: int = 10,
    largest: bool = True,
) -> TopKResult:
    """
    Return the Top-K values and their original indices using a binary heap.

    Complexity: O(N log K) heap work versus O(N log N) for sorting the full
    sequence. When K is small relative to N this is the intended win.
    """

    sequence = [float(value) for value in values]
    n = len(sequence)

    if k <= 0 or n == 0:
        return TopKResult(indices=(), values=(), stats=HeapStats(items_scanned=n))

    effective_k = min(k, n)

    if effective_k == n:
        ordered = sorted(
            range(n),
            key=lambda index: sequence[index],
            reverse=largest,
        )
        return TopKResult(
            indices=tuple(ordered),
            values=tuple(sequence[index] for index in ordered),
            stats=HeapStats(items_scanned=n, complexity="O(N log N) full sort"),
        )

    key_multiplier = 1.0 if largest else -1.0
    heap = BinaryHeap(key=lambda item: key_multiplier * item[0])
    extra_comparisons = 0

    for index, value in enumerate(sequence):
        entry = (value, index)
        if len(heap) < effective_k:
            heap.push(entry)
        else:
            root = heap.peek()
            extra_comparisons += 1
            if key_multiplier * value > key_multiplier * root[0]:
                heap.replace_root(entry)

    ordered_heap = sorted(heap.items(), key=lambda item: item[0], reverse=largest)
    indices = tuple(item[1] for item in ordered_heap)
    selected_values = tuple(item[0] for item in ordered_heap)

    stats = HeapStats(
        heap_pushes=heap.heap_pushes,
        heap_pops=heap.heap_pops,
        comparisons=heap.comparisons + extra_comparisons,
        items_scanned=n,
        complexity="O(N log K)",
    )
    return TopKResult(indices=indices, values=selected_values, stats=stats)

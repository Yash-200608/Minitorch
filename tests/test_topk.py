import pytest

from minitorch.dsa.topk import BinaryHeap, topk


def test_topk_largest_basic():
    values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]
    result = topk(values, k=3, largest=True)

    assert result.values == (9.0, 5.0, 4.0)
    assert result.indices == (5, 4, 2)
    assert result.stats.items_scanned == 7
    assert result.stats.complexity == "O(N log K)"


def test_topk_smallest_and_negative_values():
    values = [-5.0, 2.0, -1.0, 0.0, -3.0]
    result = topk(values, k=2, largest=False)

    assert result.values == (-5.0, -3.0)
    assert result.indices == (0, 4)


def test_topk_k_zero_returns_empty():
    result = topk([1.0, 2.0], k=0)

    assert result.indices == ()
    assert result.values == ()
    assert result.stats.items_scanned == 2


def test_topk_empty_input():
    result = topk([], k=5)

    assert result.indices == ()
    assert result.values == ()
    assert result.stats.items_scanned == 0


def test_topk_k_greater_than_n_returns_all_sorted():
    values = [2.0, 1.0, 3.0]
    result = topk(values, k=10, largest=True)

    assert result.values == (3.0, 2.0, 1.0)
    assert result.indices == (2, 0, 1)
    assert "full sort" in result.stats.complexity


def test_topk_duplicate_priorities_keep_distinct_indices():
    values = [5.0, 5.0, 1.0]
    result = topk(values, k=2, largest=True)

    assert result.values == (5.0, 5.0)
    assert set(result.indices) == {0, 1}


def test_binary_heap_push_pop_min_order():
    heap = BinaryHeap()
    for value in [4.0, 2.0, 7.0, 1.0]:
        heap.push((value, int(value)))

    popped = [heap.pop()[0] for _ in range(4)]
    assert popped == [1.0, 2.0, 4.0, 7.0]
    assert heap.heap_pushes == 4
    assert heap.heap_pops == 4


def test_binary_heap_empty_raises():
    heap = BinaryHeap()

    with pytest.raises(IndexError):
        heap.peek()

    with pytest.raises(IndexError):
        heap.pop()

import numpy as np

from minitorch import (
    Tensor,
    clear_computation_cache,
    disable_computation_cache,
    enable_computation_cache,
    get_computation_cache,
)


def test_cache_disabled_by_default():
    cache = get_computation_cache()
    cache.clear()

    assert cache.enabled is False


def test_cache_stores_and_hits_on_repeated_numeric_payloads():
    cache = get_computation_cache()
    clear_computation_cache()
    enable_computation_cache()

    def build():
        x = Tensor([1.0, 2.0, 3.0])
        y = Tensor([4.0, 5.0, 6.0])
        return (x * 2 + y).sum()

    first = build()
    second = build()

    np.testing.assert_allclose(first.data, second.data)
    summary = cache.summary()
    assert summary["hits"] > 0
    assert summary["misses"] > 0
    assert summary["entries"] > 0

    disable_computation_cache()
    clear_computation_cache()


def test_cache_preserves_autograd_lineage_on_hit():
    cache = get_computation_cache()
    clear_computation_cache()
    enable_computation_cache()

    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * 2
    z = (y + 3).sum()
    z.backward()

    np.testing.assert_allclose(x.grad, [2.0, 2.0])
    loss_like = x * 2 + 3
    assert loss_like.parents[0].operation == "mul"
    assert loss_like.parents[0].parents[0] is x

    disable_computation_cache()
    clear_computation_cache()


def test_cache_enable_disable_toggle():
    cache = get_computation_cache()
    clear_computation_cache()

    enable_computation_cache()
    assert cache.enabled is True

    disable_computation_cache()
    assert cache.enabled is False

    clear_computation_cache()

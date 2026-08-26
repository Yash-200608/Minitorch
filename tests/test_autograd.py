import numpy as np
import pytest

from minitorch import Tensor


def assert_grad(tensor, expected):
    assert tensor.grad is not None
    np.testing.assert_allclose(tensor.grad, np.asarray(expected, dtype=np.float64))


def test_square_gradient_prompt_case():
    x = Tensor(3.0, requires_grad=True)
    y = x * x

    result = y.backward()

    assert_grad(x, 6.0)
    assert result.topological_order[0] == x.id
    assert result.topological_order[-1] == y.id
    assert result.stats.gradient_accumulations == 2


def test_repeated_tensor_usage_accumulates_prompt_case():
    x = Tensor(3.0, requires_grad=True)
    y = x + x

    y.backward()

    assert_grad(x, 2.0)


def test_branching_and_merging_accumulate_gradients():
    x = Tensor(3.0, requires_grad=True)
    y = x * x
    z = y + x

    z.backward()

    assert_grad(x, 7.0)


def test_chained_expression_vector_gradient():
    x = Tensor([1, 2, 3], requires_grad=True)
    y = (x * 2 + 3).sum()

    y.backward()

    assert_grad(x, [2, 2, 2])


def test_sub_div_neg_and_power_gradients():
    x = Tensor([2.0, 4.0], requires_grad=True)
    y = Tensor([1.0, 2.0], requires_grad=True)
    loss = ((x - y) / y + -(x**2)).sum()

    loss.backward()

    assert_grad(x, [-3.0, -7.5])
    assert_grad(y, [-2.0, -1.0])


def test_mean_gradient():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    loss = x.mean()

    loss.backward()

    assert_grad(x, [[0.25, 0.25], [0.25, 0.25]])


def test_axis_reduction_gradient():
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    loss = x.mean(axis=1).sum()

    loss.backward()

    assert_grad(x, [[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]])


def test_broadcasting_gradients_reduce_to_parent_shape():
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    b = Tensor([10.0, 20.0, 30.0], requires_grad=True)
    loss = (x + b).sum()

    loss.backward()

    assert_grad(x, np.ones((2, 3)))
    assert_grad(b, [2.0, 2.0, 2.0])


def test_matmul_gradient_for_matrix_inputs():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    w = Tensor([[2.0], [3.0]], requires_grad=True)
    loss = (x @ w).sum()

    loss.backward()

    assert_grad(x, [[2.0, 3.0], [2.0, 3.0]])
    assert_grad(w, [[4.0], [6.0]])


def test_reshape_and_transpose_gradients():
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    loss = x.reshape(3, 2).transpose().sum()

    loss.backward()

    assert_grad(x, np.ones((2, 3)))


def test_backward_on_non_scalar_requires_seed_gradient():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2

    with pytest.raises(RuntimeError):
        y.backward()

    y.backward([1.0, 1.0, 1.0])
    assert_grad(x, [2.0, 2.0, 2.0])


def test_backward_rejects_seed_gradient_shape_mismatch():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * 2

    with pytest.raises(ValueError):
        y.backward([1.0, 1.0, 1.0])


def test_backward_rejects_tensor_without_gradient_tracking():
    x = Tensor(3.0)
    y = x * 2

    with pytest.raises(RuntimeError):
        y.backward()


def test_repeated_backward_calls_accumulate_leaf_grads_once_per_call():
    x = Tensor(3.0, requires_grad=True)
    y = x * x

    y.backward()
    y.backward()

    assert_grad(x, 12.0)
    assert_grad(y, 2.0)


def test_zero_grad_clears_accumulated_gradient():
    x = Tensor(3.0, requires_grad=True)
    y = x * x

    y.backward()
    x.zero_grad()

    assert x.grad is None

import numpy as np
import pytest

from minitorch import Tensor


def test_scalar_tensor_properties():
    x = Tensor(3.0, requires_grad=True)

    assert x.shape == ()
    assert x.ndim == 0
    assert x.size == 1
    assert x.dtype == np.float64
    assert x.requires_grad is True
    assert x.grad is None


def test_vector_matrix_and_higher_dimensional_properties():
    vector = Tensor([1, 2, 3])
    matrix = Tensor([[1, 2], [3, 4]])
    cube = Tensor(np.arange(8).reshape(2, 2, 2))

    assert vector.shape == (3,)
    assert vector.ndim == 1
    assert vector.size == 3

    assert matrix.shape == (2, 2)
    assert matrix.ndim == 2
    assert matrix.size == 4

    assert cube.shape == (2, 2, 2)
    assert cube.ndim == 3
    assert cube.size == 8


def test_arithmetic_operations_record_lineage():
    x = Tensor([1, 2, 3], requires_grad=True)
    y = Tensor([4, 5, 6])

    np.testing.assert_allclose((x + y).data, np.array([5, 7, 9]))
    np.testing.assert_allclose((y - x).data, np.array([3, 3, 3]))
    np.testing.assert_allclose((x * y).data, np.array([4, 10, 18]))
    np.testing.assert_allclose((y / x).data, np.array([4.0, 2.5, 2.0]))

    result = x * y
    assert result.requires_grad is True
    assert result.operation == "mul"
    assert result.parents == (x, y)
    assert result in x.children
    assert result in y.children


def test_scalar_interactions_and_reverse_operators():
    x = Tensor([1, 2, 4], requires_grad=True)

    np.testing.assert_allclose((x + 2).data, np.array([3, 4, 6]))
    np.testing.assert_allclose((2 + x).data, np.array([3, 4, 6]))
    np.testing.assert_allclose((x - 2).data, np.array([-1, 0, 2]))
    np.testing.assert_allclose((10 - x).data, np.array([9, 8, 6]))
    np.testing.assert_allclose((x * 2).data, np.array([2, 4, 8]))
    np.testing.assert_allclose((2 * x).data, np.array([2, 4, 8]))
    np.testing.assert_allclose((x / 2).data, np.array([0.5, 1.0, 2.0]))
    np.testing.assert_allclose((8 / x).data, np.array([8.0, 4.0, 2.0]))


def test_negation_power_and_matrix_multiplication():
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    y = Tensor([[2], [3]])

    np.testing.assert_allclose((-x).data, np.array([[-1, -2], [-3, -4]]))
    np.testing.assert_allclose((x**2).data, np.array([[1, 4], [9, 16]]))
    np.testing.assert_allclose((x @ y).data, np.array([[8], [18]]))

    assert (-x).operation == "neg"
    assert (x**2).operation == "pow"
    assert (x @ y).operation == "matmul"


def test_sum_mean_reshape_and_transpose():
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)

    assert x.sum().shape == ()
    np.testing.assert_allclose(x.sum().data, np.array(21.0))
    np.testing.assert_allclose(x.sum(axis=0).data, np.array([5, 7, 9]))

    np.testing.assert_allclose(x.mean().data, np.array(3.5))
    np.testing.assert_allclose(x.mean(axis=1).data, np.array([2.0, 5.0]))

    reshaped = x.reshape(3, 2)
    assert reshaped.shape == (3, 2)
    np.testing.assert_allclose(reshaped.data, np.array([[1, 2], [3, 4], [5, 6]]))

    transposed = x.transpose()
    assert transposed.shape == (3, 2)
    np.testing.assert_allclose(transposed.data, np.array([[1, 4], [2, 5], [3, 6]]))
    np.testing.assert_allclose(x.T.data, transposed.data)


def test_inspect_and_lineage_return_real_metadata():
    x = Tensor([1, 2], requires_grad=True)
    y = x * 2
    z = y + 3

    state = z.inspect()
    assert state["operation"] == "add"
    assert state["parent_ids"] == [y.id, z.parents[1].id]
    assert state["requires_grad"] is True

    lineage = z.lineage()
    assert [node["operation"] for node in lineage] == [None, None, "mul", None, "add"]


def test_invalid_shapes_raise_numpy_errors():
    x = Tensor([1, 2, 3])
    y = Tensor([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        x.reshape(2, 2)

    with pytest.raises(ValueError):
        x @ y

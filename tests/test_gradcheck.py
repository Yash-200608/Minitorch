import numpy as np
import pytest

from minitorch import Tensor, gradcheck


def test_gradcheck_matches_square_prompt_case():
    x = Tensor(3.0, requires_grad=True)

    result = gradcheck(lambda t: t * t, x)

    assert result.passed
    assert result.epsilon == 1e-6
    assert result.absolute_tolerance == 1e-5
    assert result.relative_tolerance == 1e-4
    assert len(result.input_results) == 1
    np.testing.assert_allclose(result.input_results[0].analytical, 6.0)
    np.testing.assert_allclose(result.input_results[0].numerical, 6.0, rtol=1e-6, atol=1e-8)


def test_gradcheck_repeated_usage_and_two_inputs():
    x = Tensor(3.0)
    y = Tensor(4.0)

    result = gradcheck(lambda a, b: (a + a) * b, (x, y))

    assert result.passed
    np.testing.assert_allclose(result.input_results[0].analytical, 8.0)
    np.testing.assert_allclose(result.input_results[1].analytical, 6.0)


def test_gradcheck_vector_and_broadcast_operations():
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = Tensor([10.0, 20.0, 30.0])

    result = gradcheck(lambda a, bias: (a + bias).sum(), (x, b))

    assert result.passed
    np.testing.assert_allclose(result.input_results[0].analytical, np.ones((2, 3)))
    np.testing.assert_allclose(result.input_results[1].analytical, [2.0, 2.0, 2.0])


def test_gradcheck_matmul_reshape_and_mean():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    w = Tensor([[2.0], [3.0]])

    result = gradcheck(lambda a, b: (a @ b).reshape(2).mean(), (x, w))

    assert result.passed
    assert result.input_results[0].shape == (2, 2)
    assert result.input_results[1].shape == (2, 1)


def test_gradcheck_reports_mismatch_when_graph_is_disconnected():
    x = Tensor([1.0, 2.0, 3.0])

    def disconnected(t):
        return Tensor(float(t.data.sum()), requires_grad=True)

    result = gradcheck(disconnected, x, absolute_tolerance=1e-8, relative_tolerance=1e-8)

    assert not result.passed
    np.testing.assert_allclose(result.input_results[0].analytical, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(result.input_results[0].numerical, [1.0, 1.0, 1.0])


def test_gradcheck_rejects_non_scalar_output():
    x = Tensor([1.0, 2.0])

    with pytest.raises(ValueError, match="scalar Tensor"):
        gradcheck(lambda t: t * 2, x)


def test_gradcheck_rejects_empty_inputs_and_non_tensors():
    with pytest.raises(ValueError, match="at least one input"):
        gradcheck(lambda: Tensor(1.0), ())

    with pytest.raises(TypeError, match="Tensor instances"):
        gradcheck(lambda t: t.sum(), [np.array([1.0, 2.0])])


def test_gradcheck_rejects_invalid_epsilon_and_tolerances():
    x = Tensor(3.0)

    with pytest.raises(ValueError, match="epsilon"):
        gradcheck(lambda t: t * t, x, epsilon=0)

    with pytest.raises(ValueError, match="absolute_tolerance"):
        gradcheck(lambda t: t * t, x, absolute_tolerance=-1)

    with pytest.raises(ValueError, match="relative_tolerance"):
        gradcheck(lambda t: t * t, x, relative_tolerance=-1)

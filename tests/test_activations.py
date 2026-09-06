import numpy as np

from minitorch import gradcheck, ReLU, Sigmoid, Tanh, Tensor


def test_relu_forward_cases():
    x = Tensor(np.array([-1.0, 0.0, 2.0]))
    y = ReLU()(x)
    np.testing.assert_allclose(y.data, [0.0, 0.0, 2.0])


def test_relu_gradcheck():
    x = Tensor(np.array([-1.0, 2.0, 3.0]))
    result = gradcheck(lambda t: t.relu().sum(), x)
    assert result.passed


def test_sigmoid_range_and_gradcheck():
    x = Tensor(np.array([-2.0, 0.0, 2.0]))
    y = x.sigmoid()
    assert np.all(y.data > 0.0)
    assert np.all(y.data < 1.0)

    result = gradcheck(lambda t: t.sigmoid().sum(), x)
    assert result.passed


def test_tanh_gradcheck():
    x = Tensor(np.array([-1.0, 0.0, 1.0]))
    y = Tanh()(x)
    np.testing.assert_allclose(y.data, np.tanh(x.data))

    result = gradcheck(lambda t: t.tanh().sum(), x)
    assert result.passed


def test_exp_and_log_gradcheck():
    x = Tensor(np.array([0.5, 1.0, 2.0]))
    assert gradcheck(lambda t: t.exp().sum(), x).passed
    assert gradcheck(lambda t: t.log().sum(), x).passed

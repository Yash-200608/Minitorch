import numpy as np

from minitorch import gradcheck, MSELoss, Tensor


def test_mse_scalar():
    criterion = MSELoss()
    pred = Tensor(3.0)
    target = Tensor(1.0)
    loss = criterion(pred, target)
    np.testing.assert_allclose(loss.data, 4.0)


def test_mse_vector():
    criterion = MSELoss()
    pred = Tensor([1.0, 3.0])
    target = Tensor([0.0, 1.0])
    loss = criterion(pred, target)
    np.testing.assert_allclose(loss.data, 2.5)


def test_mse_batch():
    criterion = MSELoss()
    pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
    target = Tensor([[0.0, 2.0], [3.0, 5.0]])
    loss = criterion(pred, target)
    np.testing.assert_allclose(loss.data, 0.5)


def test_mse_gradcheck():
    pred = Tensor([1.0, 2.0, 3.0])
    target = Tensor([0.5, 1.5, 2.5])
    criterion = MSELoss()
    result = gradcheck(lambda p: criterion(p, target), pred)
    assert result.passed

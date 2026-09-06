import numpy as np

from minitorch import Linear, Module, Parameter, ReLU, Sequential, Tensor


def test_linear_forward_shape():
    layer = Linear(3, 2)
    x = Tensor(np.ones((4, 3)))
    y = layer(x)
    assert y.shape == (4, 2)


def test_linear_output_with_controlled_weights():
    layer = Linear(2, 1)
    layer.weight.data = np.array([[1.0, 2.0]])
    layer.bias.data = np.array([3.0])

    x = Tensor(np.array([[1.0, 2.0]]))
    y = layer(x)
    np.testing.assert_allclose(y.data, [[8.0]])


def test_parameter_registration_and_lookup():
    model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
    names = [name for name, _ in model.named_parameters()]
    assert names == sorted(names)
    assert model.parameter("0.weight").shape == (4, 2)
    assert model.parameter("2.bias").shape == (1,)


def test_nested_module_parameter_count():
    model = Sequential(Linear(2, 3), Linear(3, 1))
    assert len(model.parameters()) == 4


def test_parameter_requires_grad_by_default():
    parameter = Parameter(np.array([1.0, 2.0]))
    assert parameter.requires_grad is True

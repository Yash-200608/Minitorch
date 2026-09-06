import numpy as np
import pytest

from minitorch import Linear, Module, Parameter, ParameterRegistry, Tensor


def test_parameter_registry_register_and_lookup():
    registry = ParameterRegistry()
    weight = Parameter([1.0, 2.0, 3.0])

    registry.register("layer.weight", weight)

    assert registry.parameter("layer.weight") is weight
    assert weight.name == "layer.weight"
    assert registry.names() == ("layer.weight",)


def test_parameter_registry_rejects_duplicate_names():
    registry = ParameterRegistry()
    registry.register("a", Parameter(1.0))

    with pytest.raises(KeyError, match="Duplicate"):
        registry.register("a", Parameter(2.0))


def test_parameter_registry_rejects_non_parameters():
    registry = ParameterRegistry()

    with pytest.raises(TypeError):
        registry.register("x", Tensor(1.0))


def test_module_registers_nested_parameter_paths():
    class Layer(Module):
        def __init__(self):
            super().__init__()
            self.weight = Parameter([1.0, 2.0])

    class Model(Module):
        def __init__(self):
            super().__init__()
            self.layer1 = Layer()

    model = Model()
    param = model.parameter("layer1.weight")

    assert isinstance(param, Parameter)
    np.testing.assert_allclose(param.data, [1.0, 2.0])
    assert model.registry_summary()["count"] == 1


def test_linear_forward_and_parameter_lookup():
    layer = Linear(2, 3)
    weight = layer.parameter("weight")
    bias = layer.parameter("bias")

    assert weight.shape == (3, 2)
    assert bias.shape == (3,)

    x = Tensor([[1.0, 2.0]])
    output = layer(x)
    np.testing.assert_allclose(output.data, np.zeros((1, 3)))


def test_module_zero_grad():
    layer = Linear(2, 1)
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    loss = layer(x).sum()
    loss.backward()

    assert layer.weight.grad is not None
    layer.zero_grad()
    assert layer.weight.grad is None

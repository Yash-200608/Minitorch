import math

import numpy as np
import pytest

from minitorch import Linear, Tensor
from minitorch.qat import (
    QATConfig,
    fake_quantize_uniform_array,
    is_qat_prepared,
    prepare_qat,
    validate_ste_mode,
)
from minitorch.quantization import validate_bits


def test_fake_quantize_forward_4bit():
    x = Tensor(np.array([0.0, 0.5, 1.0]), requires_grad=True)
    y = x.fake_quantize(bits=4)
    expected = fake_quantize_uniform_array(x.data, 4)
    np.testing.assert_allclose(y.data, expected)


def test_fake_quantize_ste_identity_gradient():
    x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    y = x.fake_quantize(bits=4)
    loss = y.sum()
    loss.backward()
    np.testing.assert_allclose(x.grad, np.ones(3))


def test_fake_quantize_chain_gradient():
    x = Tensor(np.array([2.0, 4.0]), requires_grad=True)
    y = x.fake_quantize(bits=4) * Tensor(np.array([3.0, 5.0]))
    loss = y.sum()
    loss.backward()
    assert x.grad is not None


def test_fake_quantize_multiple_paths_accumulation():
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    branch = x.fake_quantize(bits=4)
    loss = branch.sum() + x.sum()
    loss.backward()
    np.testing.assert_allclose(x.grad, np.array([2.0, 2.0]))


def test_fake_quantize_shapes():
    scalar = Tensor(np.array(2.5), requires_grad=True)
    vector = Tensor(np.linspace(0, 1, 5), requires_grad=True)
    matrix = Tensor(np.ones((2, 3)), requires_grad=True)
    for tensor in (scalar, vector, matrix):
        out = tensor.fake_quantize(bits=8)
        assert out.shape == tensor.shape


def test_fake_quantize_constant_tensor():
    x = Tensor(np.array([5.0, 5.0, 5.0]), requires_grad=True)
    y = x.fake_quantize(bits=4)
    np.testing.assert_allclose(y.data, 5.0)
    y.sum().backward()
    np.testing.assert_allclose(x.grad, np.ones(3))


def test_fake_quantize_rejects_non_finite():
    x = Tensor(np.array([1.0, math.nan]), requires_grad=True)
    with pytest.raises(ValueError, match="finite"):
        x.fake_quantize(bits=4)


def test_fake_quantize_bits_supported():
    data = np.linspace(-1, 1, 8)
    for bits in (2, 4, 8):
        out = fake_quantize_uniform_array(data, bits)
        assert out.shape == data.shape


def test_fake_quantize_invalid_bits():
    with pytest.raises(ValueError):
        validate_bits(3)


def test_fake_quantize_invalid_ste_mode():
    with pytest.raises(ValueError):
        validate_ste_mode("relu")


def test_prepare_qat_copies_model():
    model = Linear(2, 1)
    model.weight.data = np.ones((1, 2))
    qat_model = prepare_qat(model, QATConfig(bits=4))
    assert is_qat_prepared(qat_model)
    assert not is_qat_prepared(model)
    model.weight.data[0, 0] = 99.0
    assert qat_model.weight.data[0, 0] != 99.0


def test_linear_qat_disabled_matches_normal():
    inputs = Tensor(np.linspace(0, 1, 8).reshape(4, 2))
    normal = Linear(2, 1)
    qat = prepare_qat(Linear(2, 1), QATConfig(bits=4))
    qat.eval()
    np.testing.assert_allclose(normal(inputs).data, qat(inputs).data)


def test_linear_qat_uses_fake_quantized_weights_in_train_mode():
    model = prepare_qat(Linear(1, 1), QATConfig(bits=2))
    model.train()
    out = model(Tensor(np.array([[0.3]])))
    weight_fq = fake_quantize_uniform_array(model.weight.data, 2)
    expected = np.array([[0.3]]) @ weight_fq.T + model.bias.data
    np.testing.assert_allclose(out.data, expected)


def test_parameter_storage_remains_float():
    model = prepare_qat(Linear(2, 1), QATConfig(bits=4))
    model.train()
    _ = model(Tensor(np.ones((3, 2))))
    assert model.weight.data.dtype == np.float64


def test_graph_explorer_contains_fake_quantize():
    model = prepare_qat(Linear(1, 1), QATConfig(bits=4))
    model.train()
    out = model(Tensor(np.array([[1.0]])))
    loss = out.sum()
    explorer = loss.explore()
    ops = explorer.graph.summary().get("operations", {})
    assert "fake_quantize" in ops.values()


def test_proof_engine_fake_quantize_surrogate():
    model = prepare_qat(Linear(1, 1), QATConfig(bits=4))
    model.train()
    x = Tensor(np.array([[2.0]]))
    out = model(x)
    loss = out.sum()
    loss.backward()
    proof_text = loss.explain(model.weight)
    assert "fake_quantize" in proof_text.lower()
    assert "surrogate" in proof_text.lower()


def test_graph_optimizer_preserves_fake_quantize():
    from minitorch import optimize

    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    y = x.fake_quantize(bits=4)
    loss = y.sum()
    result = optimize(loss)
    assert result.nodes_removed == 0
    assert "fake_quantize" in result.graph_before.summary().get("operations", {}).values()


def test_model_xray_reports_qat():
    normal = Linear(1, 1)
    report_normal = normal.xray()
    assert report_normal.qat_info is not None
    assert report_normal.qat_info["enabled"] is False

    qat = prepare_qat(Linear(1, 1), QATConfig(bits=4))
    report_qat = qat.xray()
    assert report_qat.qat_info["enabled"] is True
    assert report_qat.qat_info["bits"] == 4

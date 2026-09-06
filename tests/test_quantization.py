import math

import numpy as np
import pytest

from minitorch import (
    Linear,
    MSELoss,
    QuantizedTensor,
    SGD,
    SUPPORTED_BIT_WIDTHS,
    Tensor,
    Trainer,
)
from minitorch.quantization import compute_error_metrics, compute_memory_stats, validate_bits


def test_supported_bit_widths():
    assert SUPPORTED_BIT_WIDTHS == (2, 4, 8)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_basic_quantization_round_trip(bits):
    x = Tensor(np.linspace(0.0, 1.0, 16))
    q = x.quantize(bits=bits)
    reconstructed = q.dequantize()

    assert isinstance(q, QuantizedTensor)
    assert q.bits == bits
    assert q.shape == x.shape
    assert reconstructed.shape == x.shape
    np.testing.assert_allclose(reconstructed.data, x.data, atol=1.0 / (2 ** bits))


def test_unsupported_bit_width_raises():
    x = Tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="Unsupported bit width"):
        x.quantize(bits=3)


def test_empty_tensor_raises():
    x = Tensor(np.array([]))
    with pytest.raises(ValueError, match="empty"):
        x.quantize(bits=4)


def test_non_finite_inputs_raise():
    x = Tensor([1.0, np.nan])
    with pytest.raises(ValueError, match="finite"):
        x.quantize(bits=4)

    y = Tensor([1.0, np.inf])
    with pytest.raises(ValueError, match="finite"):
        y.quantize(bits=4)


def test_constant_tensor_all_equal_nonzero():
    x = Tensor([5.0, 5.0, 5.0, 5.0])
    q = x.quantize(bits=4)
    reconstructed = q.dequantize()
    np.testing.assert_allclose(reconstructed.data, x.data)


def test_constant_tensor_all_zeros():
    x = Tensor([0.0, 0.0, 0.0])
    q = x.quantize(bits=2)
    reconstructed = q.dequantize()
    np.testing.assert_allclose(reconstructed.data, x.data)


def test_higher_dimensional_shape_preserved():
    data = np.arange(60.0).reshape(3, 4, 5)
    x = Tensor(data)
    q = x.quantize(bits=8)
    reconstructed = q.dequantize()
    assert reconstructed.shape == (3, 4, 5)


def test_deterministic_codes():
    x = Tensor(np.linspace(-2.0, 2.0, 32))
    first = x.quantize(bits=4)
    second = x.quantize(bits=4)
    np.testing.assert_array_equal(first.codes, second.codes)
    assert first.minimum == second.minimum
    assert first.maximum == second.maximum


def test_error_monotonicity_across_bit_widths():
    x = Tensor(np.linspace(0.0, 100.0, 256))
    errors = []
    for bits in [2, 4, 8]:
        report = x.quantization_report(bits=bits)
        errors.append(report.error.rmse)
    assert errors[0] >= errors[1] >= errors[2]


def test_memory_reporting_distinguishes_logical_and_actual():
    data = np.linspace(0.0, 1.0, 1000)
    x = Tensor(data)
    report = x.quantization_report(bits=4)

    assert report.memory.original_bytes == data.nbytes
    assert report.memory.theoretical_packed_bytes == math.ceil(1000 * 4 / 8)
    assert report.memory.actual_code_storage_bytes == x.quantize(4).codes.nbytes
    assert report.memory.actual_code_storage_bytes > report.memory.theoretical_packed_bytes
    assert report.memory.compression_ratio_theoretical > 1.0


def test_report_contains_real_error_metrics():
    x = Tensor([0.0, 0.5, 1.0])
    report = x.quantization_report(bits=2)
    reconstructed = x.quantize(bits=2).dequantize().data

    expected = compute_error_metrics(x.data, reconstructed)
    assert report.error.max_absolute_error == expected.max_absolute_error
    assert report.error.mean_absolute_error == expected.mean_absolute_error
    assert report.error.rmse == expected.rmse


def test_quantization_not_in_autograd_graph():
    x = Tensor([1.0, 2.0], requires_grad=True)
    q = x.quantize(bits=4)
    y = q.dequantize()
    assert y.operation is None
    assert y.parents == ()
    assert y.requires_grad is False


def test_parameter_quantize_does_not_mutate_live_parameter():
    from minitorch import Parameter

    weight = Parameter(np.array([[1.0, 2.0], [3.0, 4.0]]))
    original = weight.data.copy()
    q = weight.quantize(bits=4)
    np.testing.assert_array_equal(weight.data, original)
    assert q.codes.shape == (2, 2)


def test_training_regression_after_quantization_changes():
    inputs = Tensor(np.linspace(0.0, 1.0, 8).reshape(8, 1))
    targets = Tensor(3.0 * inputs.data + 2.0)
    model = Linear(1, 1)
    trainer = Trainer(model, SGD(model.parameters(), lr=0.1), MSELoss().forward)

    initial_loss = float(MSELoss()(model(inputs), targets).data)
    history = trainer.fit(inputs, targets, epochs=100)
    assert history.losses[-1] < initial_loss


def test_validate_bits_helper():
    validate_bits(8)
    with pytest.raises(ValueError):
        validate_bits(16)


def test_compute_memory_stats_helper():
    source = np.zeros(128, dtype=np.float64)
    codes = np.zeros(128, dtype=np.int32)
    stats = compute_memory_stats(source, codes, bits=4)
    assert stats.theoretical_packed_bytes == 64
    assert stats.actual_code_storage_bytes == codes.nbytes

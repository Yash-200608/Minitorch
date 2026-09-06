import math

import numpy as np
import pytest

from minitorch import Codebook, SUPPORTED_BIT_WIDTHS, Tensor
from minitorch.quantization import (
    CodebookFitStats,
    compute_memory_stats,
    quantize_array,
    validate_bits,
)


def test_codebook_fit_is_deterministic():
    values = np.linspace(-1.0, 1.0, 64)
    first = Codebook.fit(values, levels=16)
    second = Codebook.fit(values, levels=16)
    np.testing.assert_allclose(first.centroids, second.centroids)


def test_codebook_expected_number_of_levels():
    values = np.linspace(0.0, 10.0, 100)
    codebook = Codebook.fit(values, levels=16)
    assert codebook.levels == 16


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_codebook_quantize_round_trip(bits):
    x = Tensor(np.linspace(0.0, 1.0, 32))
    q = x.quantize(bits=bits, method="codebook")
    reconstructed = q.dequantize()
    assert q.method == "codebook"
    assert q.shape == x.shape
    assert reconstructed.shape == x.shape
    assert q.centroids is not None
    assert len(q.centroids) == 2 ** bits


def test_uniform_mode_remains_default():
    x = Tensor([0.0, 0.5, 1.0])
    q = x.quantize(bits=4)
    assert q.method == "uniform"
    report = q.report()
    assert report.method == "uniform"


def test_encode_nearest_centroid():
    codebook = Codebook(np.array([0.0, 10.0, 20.0], dtype=np.float64))
    codes = codebook.encode(np.array([0.1, 9.9, 20.0, 16.0]))
    np.testing.assert_array_equal(codes, [0, 1, 2, 2])


def test_decode_preserves_shape():
    codebook = Codebook(np.linspace(0.0, 1.0, 4))
    codes = np.array([[0, 1], [2, 3]])
    reconstructed = codebook.decode(codes, shape=(2, 2))
    assert reconstructed.shape == (2, 2)


def test_codebook_convergence_or_safe_stop():
    values = np.random.default_rng(0).normal(size=256)
    codebook = Codebook.fit(values, levels=16, max_iterations=50, tolerance=1e-6)
    assert codebook.fit_stats is not None
    assert codebook.fit_stats.iterations <= 50
    assert np.all(np.isfinite(codebook.centroids))


def test_empty_cluster_keeps_previous_centroid():
    values = np.array([0.0, 0.1, 0.9, 1.0])
    codebook = Codebook.fit(values, levels=8, max_iterations=20, tolerance=1e-12)
    assert np.all(np.isfinite(codebook.centroids))


def test_constant_tensor_codebook():
    x = Tensor([5.0, 5.0, 5.0, 5.0])
    q = x.quantize(bits=4, method="codebook")
    reconstructed = q.dequantize()
    np.testing.assert_allclose(reconstructed.data, x.data)


def test_all_zeros_codebook():
    x = Tensor([0.0, 0.0, 0.0])
    q = x.quantize(bits=2, method="codebook")
    reconstructed = q.dequantize()
    np.testing.assert_allclose(reconstructed.data, x.data)


def test_codebook_non_finite_rejected():
    x = Tensor([1.0, np.nan])
    with pytest.raises(ValueError, match="finite"):
        x.quantize(bits=4, method="codebook")


def test_codebook_empty_input_rejected():
    x = Tensor(np.array([]))
    with pytest.raises(ValueError, match="empty"):
        x.quantize(bits=4, method="codebook")


def test_levels_greater_than_unique_values():
    values = np.array([1.0, 2.0, 3.0])
    codebook = Codebook.fit(values, levels=16)
    codes = codebook.encode(values)
    reconstructed = codebook.decode(codes, shape=values.shape)
    np.testing.assert_allclose(reconstructed, values, atol=1e-6)
    assert np.all(np.isfinite(codebook.centroids))


def test_codebook_memory_reporting():
    data = np.linspace(0.0, 1.0, 128)
    x = Tensor(data)
    report = x.quantization_report(bits=4, method="codebook")
    assert report.memory.codebook_bytes > 0
    assert report.memory.total_theoretical_representation_bytes == (
        report.memory.theoretical_packed_bytes + report.memory.codebook_bytes
    )


def test_codebook_report_contains_fit_metadata():
    x = Tensor(np.linspace(-2.0, 2.0, 64))
    report = x.quantization_report(bits=4, method="codebook")
    assert report.codebook_fit is not None
    assert report.centroids is not None
    assert len(report.centroids) == 16
    assert "Codebook fit" in report.format()


def test_uniform_vs_codebook_comparison_on_clustered_data():
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=-2.0, scale=0.2, size=200)
    cluster_b = rng.normal(loc=2.0, scale=0.2, size=200)
    data = np.concatenate([cluster_a, cluster_b])
    x = Tensor(data)

    uniform_rmse = x.quantization_report(bits=4, method="uniform").error.rmse
    codebook_rmse = x.quantization_report(bits=4, method="codebook").error.rmse
    assert codebook_rmse <= uniform_rmse


def test_error_monotonicity_codebook_bit_widths():
    x = Tensor(np.linspace(0.0, 100.0, 256))
    errors = []
    for bits in [2, 4, 8]:
        report = x.quantization_report(bits=bits, method="codebook")
        errors.append(report.error.rmse)
    assert errors[0] >= errors[1] >= errors[2]


def test_codebook_not_in_autograd_graph():
    x = Tensor([1.0, 2.0], requires_grad=True)
    q = x.quantize(bits=4, method="codebook")
    y = q.dequantize()
    assert y.requires_grad is False
    assert y.operation is None


def test_compute_memory_stats_includes_codebook_bytes():
    source = np.zeros(64, dtype=np.float64)
    codes = np.zeros(64, dtype=np.int32)
    centroids = np.linspace(0.0, 1.0, 16)
    stats = compute_memory_stats(source, codes, bits=4, centroids=centroids)
    assert stats.codebook_bytes == centroids.nbytes
    assert stats.total_theoretical_representation_bytes == (
        stats.theoretical_packed_bytes + stats.codebook_bytes
    )


def test_codebook_fit_stats_objective_is_finite():
    values = np.linspace(0.0, 1.0, 50)
    codebook = Codebook.fit(values, levels=8)
    assert codebook.fit_stats is not None
    assert math.isfinite(codebook.fit_stats.objective)


def test_invalid_method_raises():
    x = Tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="Unsupported quantization method"):
        quantize_array(x.data, 4, method="invalid")


def test_validate_bits_still_rejects_invalid_bits():
    validate_bits(8)
    with pytest.raises(ValueError):
        validate_bits(3)

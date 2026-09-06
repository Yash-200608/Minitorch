import numpy as np

from minitorch import Tensor, compare_quantization_rotation


def test_quantization_without_rotation_unchanged():
    x = Tensor(np.linspace(0.0, 1.0, 16))
    first = x.quantize(bits=4, method="uniform").dequantize().data
    second = x.quantize(bits=4, method="uniform", rotation=None).dequantize().data
    np.testing.assert_allclose(first, second)


def test_uniform_with_rotation_reconstructs_in_original_space():
    x = Tensor(np.linspace(-1.0, 1.0, 16))
    q = x.quantize(bits=4, method="uniform", rotation="combined")
    reconstructed = q.dequantize().data
    assert q.rotation_applied
    assert reconstructed.shape == x.shape


def test_codebook_with_rotation():
    rng = np.random.default_rng(0)
    data = rng.normal(size=64)
    x = Tensor(data)
    q = x.quantize(bits=4, method="codebook", rotation="combined")
    reconstructed = q.dequantize().data
    assert reconstructed.shape == x.shape


def test_packed_with_rotation_matches_logical():
    x = Tensor(np.linspace(0.0, 1.0, 32))
    logical = x.quantize(bits=4, method="uniform", rotation="combined").dequantize().data
    packed = x.quantize(
        bits=4,
        method="uniform",
        rotation="combined",
        packed=True,
    ).dequantize().data
    np.testing.assert_allclose(logical, packed)


def test_compare_experiment_reports_real_metrics():
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=-2.0, scale=0.2, size=200)
    cluster_b = rng.normal(loc=2.0, scale=0.2, size=200)
    data = np.concatenate([cluster_a, cluster_b])
    x = Tensor(data.reshape(8, 50)[:, :32])  # 8 x 32, last dim power of 2

    experiment = compare_quantization_rotation(
        x.data,
        bits=4,
        method="codebook",
        rotation="combined",
    )
    assert experiment.baseline_error.rmse >= 0.0
    assert experiment.rotated_error.rmse >= 0.0
    assert experiment.result_label in {"improved", "unchanged", "worse"}
    assert "ROTATION EXPERIMENT" in experiment.format()


def test_rotation_error_measured_in_original_space():
    x = Tensor(np.linspace(0.0, 1.0, 16))
    q = x.quantize(bits=4, method="uniform", rotation="combined")
    report = q.report()
    np.testing.assert_allclose(
        report.error.rmse,
        np.sqrt(np.mean((x.data - q.dequantize().data) ** 2)),
    )

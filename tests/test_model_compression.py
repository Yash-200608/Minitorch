import numpy as np
import pytest

from minitorch import (
    CompressionConfig,
    Linear,
    MSELoss,
    ReLU,
    SGD,
    Sequential,
    Tensor,
    Trainer,
    compress_model,
    compare_model_compression,
)


def _train_linear_model(epochs: int = 150) -> Linear:
    inputs = Tensor(np.linspace(0.0, 1.0, 16).reshape(16, 1))
    targets = Tensor(3.0 * inputs.data + 2.0)
    model = Linear(1, 1)
    trainer = Trainer(model, SGD(model.parameters(), lr=0.1), MSELoss().forward)
    trainer.fit(inputs, targets, epochs=epochs)
    return model


def test_compress_model_discovers_all_parameters():
    model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
    compressed = compress_model(model)
    names = list(compressed.source_parameter_names)
    assert names == sorted(names)
    assert len(names) == 4


def test_compress_does_not_mutate_original_model():
    model = _train_linear_model()
    before = model.weight.data.copy()
    grad_before = model.weight.grad
    requires_before = model.weight.requires_grad
    compress_model(model, CompressionConfig(bits=4, method="uniform", packed=True))
    np.testing.assert_array_equal(model.weight.data, before)
    assert model.weight.requires_grad == requires_before
    assert model.weight.grad is grad_before


def test_uniform_codebook_packed_compression():
    model = Linear(8, 16)
    config = CompressionConfig(bits=4, method="codebook", packed=True)
    compressed = compress_model(model, config=config)
    report = compressed.report()
    assert report.total_compressed_bytes < report.total_original_bytes
    assert report.compression_ratio > 1.0


def test_materialize_preserves_shapes_and_forward():
    model = Sequential(Linear(4, 8), Linear(8, 2))
    compressed = compress_model(model, CompressionConfig(bits=4, packed=True))
    reconstructed = compressed.materialize(model)
    x = Tensor(np.random.default_rng(0).normal(size=(3, 4)))
    output = reconstructed(x)
    assert output.shape == (3, 2)


def test_compare_model_compression_output_and_loss():
    model = _train_linear_model()
    inputs = Tensor(np.linspace(-1.0, 1.0, 8).reshape(8, 1))
    targets = Tensor(3.0 * inputs.data + 2.0)
    report = compare_model_compression(
        model,
        inputs,
        targets,
        config=CompressionConfig(bits=4, method="uniform", packed=True),
    )
    assert report.output_rmse is not None
    assert report.original_loss is not None
    assert report.compressed_loss is not None
    assert report.loss_delta is not None


def test_rotation_configuration_when_axis_supported():
    model = Linear(8, 16)
    compressed = compress_model(
        model,
        CompressionConfig(bits=4, method="codebook", packed=True, rotation="combined"),
    )
    assert compressed.config.rotation == "combined"


def test_rotation_rejected_for_invalid_final_axis():
    model = Linear(5, 3)
    with pytest.raises(ValueError, match="power-of-two"):
        compress_model(
            model,
            CompressionConfig(bits=4, rotation="combined"),
        )


def test_codebook_bytes_included_in_total():
    model = Linear(4, 8)
    compressed = compress_model(
        model,
        CompressionConfig(bits=4, method="codebook", packed=True),
    )
    record = compressed.parameter_records["weight"]
    assert record.codebook_bytes > 0
    assert record.compressed_bytes >= record.packed_code_bytes + record.codebook_bytes


def test_deterministic_compression():
    model = Linear(4, 8)
    config = CompressionConfig(bits=4, method="uniform", packed=True)
    first = compress_model(model, config)
    second = compress_model(model, config)
    np.testing.assert_array_equal(
        first.decompress_parameter("weight"),
        second.decompress_parameter("weight"),
    )


def test_largest_parameter_errors_uses_topk():
    model = Sequential(Linear(2, 8), Linear(8, 4))
    compressed = compress_model(model, CompressionConfig(bits=2, method="uniform"))
    report = compressed.report()
    largest = report.largest_parameter_errors(k=2)
    assert len(largest) == 2
    assert largest[0].rmse >= largest[1].rmse


def test_training_then_compression_regression():
    model = _train_linear_model()
    inputs = Tensor(np.linspace(0.0, 1.0, 8).reshape(8, 1))
    targets = Tensor(3.0 * inputs.data + 2.0)
    initial_loss = float(MSELoss()(model(inputs), targets).data)
    compressed = compress_model(model, CompressionConfig(bits=4, packed=True))
    report = compressed.report(model, inputs, targets)
    assert initial_loss < 0.1
    assert report.total_compressed_bytes < report.total_original_bytes


def test_compressed_xray_returns_real_metrics():
    model = Linear(4, 8)
    compressed = compress_model(model, CompressionConfig(bits=4, packed=True))
    summary = compressed.xray()
    assert summary["parameter_count"] == 2
    assert summary["compression_ratio"] > 1.0

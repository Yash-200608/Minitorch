import numpy as np
import pytest

from minitorch import (
    CompressionConfig,
    Linear,
    MSELoss,
    MixedPrecisionConfig,
    Sequential,
    SGD,
    Tensor,
    Trainer,
    compare_mixed_precision,
    compress_model,
    compress_model_mixed_precision,
    greedy_mixed_precision_compress,
    measure_parameter_sensitivity,
    output_sensitivity_experiment,
)


def _setup():
    model = Sequential(Linear(2, 4), Linear(4, 1))
    inputs = Tensor(np.linspace(0, 1, 32).reshape(16, 2))
    targets = Tensor(3.0 * inputs.data[:, :1] + 2.0)
    trainer = Trainer(model, SGD(model.parameters(), lr=0.1), MSELoss().forward)
    trainer.fit(inputs, targets, epochs=40)
    return model, inputs, targets


def test_parameter_sensitivity_topk():
    model, _, _ = _setup()
    report = measure_parameter_sensitivity(model, bits=(2, 4, 8))
    top = report.most_sensitive_parameters(bits=4, k=2)
    assert len(top) <= 2
    assert all(isinstance(item[1], float) for item in top)


def test_output_sensitivity_experiment():
    model, inputs, targets = _setup()
    report = output_sensitivity_experiment(model, inputs, targets, bits=4)
    assert len(report.entries) == len(list(model.named_parameters()))
    assert report.entries[0].output_rmse >= 0


def test_compare_mixed_precision_table():
    model, inputs, targets = _setup()
    mixed_config = MixedPrecisionConfig(
        default_bits=8,
        parameter_bits={"0.weight": 8, "1.weight": 4},
    )
    result = compare_mixed_precision(model, inputs, targets, mixed_config=mixed_config)
    table = result.format_table()
    assert "FP32" in table
    assert "Mixed precision" in table
    assert "Uniform 4-bit" in table


def test_serial_parallel_mixed_parity():
    model, _, _ = _setup()
    config = MixedPrecisionConfig(parameter_bits={"1.weight": 4})
    serial = compress_model_mixed_precision(model, config, parallel=False)
    parallel = compress_model_mixed_precision(model, config, parallel=True, workers=2)
    assert serial.source_parameter_names == parallel.source_parameter_names
    for name in serial.source_parameter_names:
        assert (
            serial.parameter_records[name].compressed_bytes
            == parallel.parameter_records[name].compressed_bytes
        )


def test_greedy_heuristic_respects_constraint():
    model, inputs, targets = _setup()
    result = greedy_mixed_precision_compress(
        model,
        inputs,
        targets,
        max_output_rmse=1.0,
    )
    assert result.report.output_rmse is None or result.report.output_rmse <= 1.0
    for step in result.steps:
        if not step.accepted:
            assert step.reason == "constraint violated"


def test_mri_consumes_mixed_precision_report():
    from minitorch import inspect_model_mri

    model, inputs, targets = _setup()
    config = MixedPrecisionConfig(parameter_bits={"1.weight": 4})
    compressed = compress_model_mixed_precision(model, config)
    report = compressed.report(template=model, inputs=inputs, targets=targets)
    mri = inspect_model_mri(model, compression=report)
    assert mri.compression.section.data.get("mixed_precision") is True
    assert mri.compression.section.data.get("precision_map")


def test_codebook_mixed_precision():
    model, inputs, targets = _setup()
    config = MixedPrecisionConfig(
        method="codebook",
        parameter_bits={"0.weight": 4, "1.weight": 4},
    )
    compressed = compress_model_mixed_precision(model, config)
    report = compressed.report(template=model, inputs=inputs, targets=targets)
    assert report.total_compressed_bytes > 0

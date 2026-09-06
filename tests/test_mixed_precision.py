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
    compress_model,
    compress_model_mixed_precision,
    validate_mixed_precision_config,
)


def _trained_sequential():
    model = Sequential(Linear(2, 4), Linear(4, 1))
    inputs = Tensor(np.linspace(0, 1, 16).reshape(8, 2))
    targets = Tensor(3.0 * inputs.data[:, :1] + 2.0)
    trainer = Trainer(model, SGD(model.parameters(), lr=0.1), MSELoss().forward)
    trainer.fit(inputs, targets, epochs=30)
    return model, inputs, targets


def test_mixed_precision_config_effective_bits():
    model = Linear(2, 1)
    config = MixedPrecisionConfig(
        default_bits=8,
        parameter_bits={"weight": 4, "bias": 2},
    )
    validate_mixed_precision_config(model, config)
    assert config.effective_bits("weight") == 4
    assert config.effective_bits("bias") == 2
    assert config.effective_bits("missing") == 8  # not a param name


def test_mixed_precision_unknown_parameter_rejected():
    model = Linear(1, 1)
    config = MixedPrecisionConfig(parameter_bits={"not_a_param": 4})
    with pytest.raises(ValueError, match="Unknown parameter"):
        validate_mixed_precision_config(model, config)


def test_mixed_precision_invalid_bits_rejected():
    model = Linear(1, 1)
    config = MixedPrecisionConfig(default_bits=3)
    with pytest.raises(ValueError, match="Unsupported"):
        validate_mixed_precision_config(model, config)


def test_mixed_precision_compression_deterministic_order():
    model, _, _ = _trained_sequential()
    config = MixedPrecisionConfig(
        default_bits=8,
        parameter_bits={"0.weight": 4, "1.weight": 2},
    )
    a = compress_model_mixed_precision(model, config)
    b = compress_model_mixed_precision(model, config)
    assert a.source_parameter_names == b.source_parameter_names
    assert list(a.parameter_records.keys()) == list(b.parameter_records.keys())


def test_mixed_precision_model_not_mutated():
    model, _, _ = _trained_sequential()
    snapshots = {name: p.data.copy() for name, p in model.named_parameters()}
    config = MixedPrecisionConfig(parameter_bits={"0.weight": 4})
    compress_model_mixed_precision(model, config)
    for name, parameter in model.named_parameters():
        np.testing.assert_allclose(parameter.data, snapshots[name])


def test_mixed_precision_memory_sum():
    model, inputs, targets = _trained_sequential()
    config = MixedPrecisionConfig(
        default_bits=8,
        parameter_bits={"0.weight": 4, "1.weight": 4},
    )
    compressed = compress_model_mixed_precision(model, config)
    report = compressed.report(template=model, inputs=inputs, targets=targets)
    summed = sum(r.compressed_bytes for r in report.parameter_records)
    assert report.total_compressed_bytes == summed


def test_global_compress_still_works():
    model = Linear(2, 1)
    report = compress_model(model, CompressionConfig(bits=4)).report()
    assert report.total_compressed_bytes > 0
    assert report.compression_ratio > 1.0


def test_mixed_vs_uniform_materialize():
    model, inputs, targets = _trained_sequential()
    mixed = compress_model_mixed_precision(
        model,
        MixedPrecisionConfig(default_bits=8, parameter_bits={"1.weight": 4}),
    )
    eval_model = mixed.materialize(model)
    out = eval_model(inputs)
    assert out.shape == (8, 1)

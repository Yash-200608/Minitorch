"""Comparison and matrix tests for Experiment Lab."""

import pytest

from minitorch.experiments import (
    Experiment,
    ExperimentConfig,
    ExperimentTechnique,
    CompressionExperimentConfig,
    MixedPrecisionExperimentConfig,
    TrainingConfig,
    compare_experiments,
    format_experiment_comparison_table,
    run_quantization_matrix,
)


def _quick_compression(name: str, compression: CompressionExperimentConfig):
    return Experiment(
        ExperimentConfig(
            name=name,
            technique=ExperimentTechnique.POST_TRAINING_COMPRESSION,
            compression=compression,
            training=TrainingConfig(epochs=6),
            seed=5,
        )
    ).run()


def test_compare_experiments_handles_missing_metrics():
    fp32 = Experiment(
        ExperimentConfig(
            name="fp32_baseline",
            technique=ExperimentTechnique.FP32_BASELINE,
            training=TrainingConfig(epochs=4),
            seed=2,
        )
    ).run()
    uniform = _quick_compression(
        "quantization_4bit_uniform",
        CompressionExperimentConfig(bits=4, method="uniform"),
    )
    table = compare_experiments([fp32, uniform])
    assert "rows" in table
    assert len(table["rows"]) == 2
    assert table["rows"][0]["compression_ratio"] == 1.0
    assert table["rows"][1]["compressed_bytes"] is not None


def test_comparison_table_is_deterministic():
    fp32 = Experiment(
        ExperimentConfig(
            name="fp32_baseline",
            technique=ExperimentTechnique.FP32_BASELINE,
            training=TrainingConfig(epochs=3),
            seed=8,
        )
    ).run()
    uniform = _quick_compression(
        "quantization_4bit_uniform",
        CompressionExperimentConfig(bits=4, method="uniform"),
    )
    first = format_experiment_comparison_table([fp32, uniform])
    second = format_experiment_comparison_table([fp32, uniform])
    assert first == second


def test_quantization_matrix_shared_model():
    matrix = run_quantization_matrix(
        ExperimentConfig(
            name="matrix",
            technique=ExperimentTechnique.QUANTIZATION_MATRIX,
            training=TrainingConfig(epochs=8),
            seed=12,
        )
    )
    assert len(matrix.rows) >= 6
    labels = [row.technique_label for row in matrix.rows]
    assert labels[0] == "FP32"
    assert any("Mixed precision" in label for label in labels)


def test_mixed_vs_uniform_comparison():
    mixed = Experiment(
        ExperimentConfig(
            name="mixed_precision_4bit",
            technique=ExperimentTechnique.MIXED_PRECISION,
            mixed_precision=MixedPrecisionExperimentConfig(default_bits=4),
            training=TrainingConfig(epochs=6),
            seed=20,
        )
    ).run()
    uniform = _quick_compression(
        "quantization_4bit_uniform",
        CompressionExperimentConfig(bits=4, method="uniform"),
    )
    comparison = compare_experiments([uniform, mixed])
    assert comparison["rows"][0]["parameter_rmse"] is not None
    assert comparison["rows"][1]["parameter_rmse"] is not None

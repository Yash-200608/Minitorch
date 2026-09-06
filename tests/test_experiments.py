"""Tests for Experiment Lab core behavior."""

import numpy as np
import pytest

from minitorch.experiments import (
    Experiment,
    ExperimentConfig,
    ExperimentTechnique,
    CompressionExperimentConfig,
    MixedPrecisionExperimentConfig,
    TrainingConfig,
    compute_experiment_id,
    seed_everything,
    validate_experiment_result,
    clone_model_with_state,
    snapshot_model_state,
    build_baseline_model,
    build_baseline_dataset,
    list_experiments,
    CANONICAL_EXPERIMENTS,
)


def test_experiment_id_is_deterministic():
    config = ExperimentConfig(
        name="quantization_4bit_uniform",
        technique=ExperimentTechnique.POST_TRAINING_COMPRESSION,
        compression=CompressionExperimentConfig(bits=4, method="uniform"),
        seed=42,
    )
    assert compute_experiment_id(config) == compute_experiment_id(config)
    assert len(compute_experiment_id(config)) == 16


def test_experiment_config_validation():
    bad = ExperimentConfig(
        name="bad",
        technique=ExperimentTechnique.POST_TRAINING_COMPRESSION,
    )
    with pytest.raises(ValueError, match="compression config required"):
        bad.validate()


def test_fp32_baseline_experiment_runs_quickly():
    config = ExperimentConfig(
        name="fp32_baseline",
        technique=ExperimentTechnique.FP32_BASELINE,
        training=TrainingConfig(epochs=5),
        seed=7,
    )
    result = Experiment(config).run()
    validate_experiment_result(result)
    assert result.metrics.fp_final_loss is not None
    assert result.metrics.compression_ratio == 1.0
    assert result.timings.training_seconds is not None
    assert result.timings.training_seconds >= 0


def test_post_training_compression_experiment():
    config = ExperimentConfig(
        name="quantization_4bit_uniform",
        technique=ExperimentTechnique.POST_TRAINING_COMPRESSION,
        compression=CompressionExperimentConfig(bits=4, method="uniform"),
        training=TrainingConfig(epochs=8),
        seed=11,
    )
    result = Experiment(config).run()
    validate_experiment_result(result)
    assert result.metrics.compressed_bytes is not None
    assert result.metrics.parameter_rmse is not None
    assert result.metrics.output_rmse is not None
    assert result.storage_technique == "post_training_compression"


def test_mixed_precision_experiment():
    config = ExperimentConfig(
        name="mixed_precision_4bit",
        technique=ExperimentTechnique.MIXED_PRECISION,
        mixed_precision=MixedPrecisionExperimentConfig(default_bits=4),
        training=TrainingConfig(epochs=8),
        seed=3,
    )
    result = Experiment(config).run()
    validate_experiment_result(result)
    assert "precision_map" in result.extra


def test_model_snapshot_independence():
    model = build_baseline_model()
    snapshot = snapshot_model_state(model)
    clone_a = clone_model_with_state(model)
    clone_b = clone_model_with_state(model)
    clone_a.parameter(next(iter(snapshot))).data += 1.0
    clone_b_weight = clone_b.parameter(next(iter(snapshot))).data.copy()
    np.testing.assert_allclose(clone_b_weight, snapshot[next(iter(snapshot))])


def test_deterministic_metrics_repeat():
    config = ExperimentConfig(
        name="fp32_baseline",
        technique=ExperimentTechnique.FP32_BASELINE,
        training=TrainingConfig(epochs=4),
        seed=99,
    )
    first = Experiment(config).run()
    second = Experiment(config).run()
    assert first.experiment_id == second.experiment_id
    assert first.metrics.fp_final_loss == second.metrics.fp_final_loss


def test_canonical_registry_lists_experiments():
    names = list_experiments()
    assert "fp32_baseline" in names
    assert names == sorted(CANONICAL_EXPERIMENTS.keys())


def test_seed_everything_affects_numpy():
    seed_everything(0)
    a = np.random.rand()
    seed_everything(0)
    b = np.random.rand()
    assert a == b

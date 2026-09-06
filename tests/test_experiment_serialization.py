"""Serialization tests for Experiment Lab results."""

import json
import pytest

from minitorch.experiments import (
    EXPERIMENT_SCHEMA_VERSION,
    Experiment,
    ExperimentConfig,
    ExperimentTechnique,
    ExperimentResult,
    ExperimentResultError,
    TrainingConfig,
    load_experiment_result,
    save_experiment_result,
)


def _tiny_result(tmp_path):
    config = ExperimentConfig(
        name="fp32_baseline",
        technique=ExperimentTechnique.FP32_BASELINE,
        training=TrainingConfig(epochs=3),
        seed=1,
    )
    result = Experiment(config).run()
    path = tmp_path / "experiment.json"
    save_experiment_result(result, path)
    return result, path


def test_serialization_round_trip(tmp_path):
    original, path = _tiny_result(tmp_path)
    loaded = load_experiment_result(path)
    assert loaded.experiment_id == original.experiment_id
    assert loaded.metrics.fp_final_loss == original.metrics.fp_final_loss
    assert loaded.schema_version == EXPERIMENT_SCHEMA_VERSION


def test_corrupt_json_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ExperimentResultError, match="Invalid JSON"):
        load_experiment_result(path)


def test_missing_schema_rejected(tmp_path):
    path = tmp_path / "no_schema.json"
    path.write_text(json.dumps({"experiment_id": "abcd1234abcd1234"}), encoding="utf-8")
    with pytest.raises(ExperimentResultError, match="Missing schema_version"):
        load_experiment_result(path)


def test_invalid_schema_version_rejected(tmp_path):
    path = tmp_path / "old_schema.json"
    payload = {
        "schema_version": "0.1",
        "experiment_id": "abcd1234abcd1234",
        "metrics": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExperimentResultError, match="Unsupported experiment schema"):
        load_experiment_result(path)


def test_invalid_metrics_type_rejected(tmp_path):
    path = tmp_path / "bad_metrics.json"
    payload = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": "abcd1234abcd1234",
        "metrics": "bad",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExperimentResultError, match="metrics must be an object"):
        load_experiment_result(path)


def test_result_parity_structure(tmp_path):
    first, _ = _tiny_result(tmp_path)
    second = Experiment(
        ExperimentConfig.from_dict(first.configuration)
    ).run()
    assert first.to_dict().keys() == second.to_dict().keys()

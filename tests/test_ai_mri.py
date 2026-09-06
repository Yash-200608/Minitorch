import json
import math

import numpy as np
import pytest

from minitorch import (
    CompressionConfig,
    Linear,
    MSELoss,
    ModelMRI,
    ObservationSeverity,
    QATConfig,
    Sequential,
    SGD,
    Tensor,
    Trainer,
    compare_mri,
    compare_model_compression,
    inspect_model_mri,
    prepare_qat,
)
from minitorch.intelligence.ai_mri import DataStatus
from minitorch.profiling.profile import PerformanceProfile, ProfilerMode


def _tiny_model():
    return Sequential(Linear(2, 4), Linear(4, 1))


def test_mri_creation_and_structure():
    model = _tiny_model()
    mri = model.mri()
    assert isinstance(mri, ModelMRI)
    assert mri.model_summary.section.status == DataStatus.AVAILABLE
    assert mri.graph.section.status == DataStatus.UNAVAILABLE


def test_mri_model_summary_from_xray():
    model = Linear(2, 1)
    mri = inspect_model_mri(model)
    data = mri.model_summary.section.data
    assert data["parameter_count"] == 2
    assert data["trainable_numel"] == 3


def test_mri_graph_unavailable_without_output():
    mri = Linear(1, 1).mri()
    assert mri.graph.section.status == DataStatus.UNAVAILABLE
    assert "graph" in mri.graph.section.reason.lower() or "graph" in str(mri.graph.section.reason)


def test_mri_graph_with_output():
    model = Linear(1, 1)
    x = Tensor(np.array([[1.0]]))
    out = model(x)
    loss = out.sum()
    loss.backward()
    mri = model.mri(output=loss)
    assert mri.graph.section.status == DataStatus.AVAILABLE
    assert mri.graph.section.data["vertices"] > 0
    assert mri.graph_explorer() is not None


def test_mri_gradients_match_detective():
    model = Linear(1, 1)
    x = Tensor(np.array([[2.0]]))
    loss = model(x).sum()
    loss.backward()
    mri = model.mri(output=loss)
    report = mri.gradients_report()
    assert report is not None
    assert mri.gradients.section.data["near_zero_count"] == report.near_zero_count
    assert mri.gradients.section.data["non_finite_count"] == report.non_finite_count


def test_mri_non_finite_gradient_observation():
    model = Linear(1, 1)
    model.weight.grad = np.array([[math.nan]])
    mri = inspect_model_mri(model)
    critical = [o for o in mri.observations if o.severity == ObservationSeverity.CRITICAL]
    assert critical
    assert critical[0].category == "gradients"


def test_mri_performance_from_supplied_profile():
    profile = PerformanceProfile(
        name="test",
        mode=ProfilerMode.LIGHTWEIGHT.value,
        stages={"forward": 0.001, "backward": 0.005, "optimizer": 0.0002},
    )
    mri = Linear(1, 1).mri(performance=profile)
    assert mri.performance.section.status == DataStatus.AVAILABLE
    assert mri.performance.section.data["hottest_stage"] == "backward"
    perf_obs = [o for o in mri.observations if o.category == "performance"]
    assert perf_obs


def test_mri_performance_unavailable_by_default():
    mri = Linear(1, 1).mri()
    assert mri.performance.section.status == DataStatus.UNAVAILABLE


def test_mri_compression_from_report():
    model = Linear(2, 1)
    inputs = Tensor(np.linspace(0, 1, 16).reshape(8, 2))
    targets = Tensor(np.linspace(0, 1, 8).reshape(8, 1))
    report = compare_model_compression(
        model,
        inputs,
        targets,
        config=CompressionConfig(bits=4, method="uniform"),
    )
    mri = model.mri(compression=report)
    assert mri.compression.section.status == DataStatus.AVAILABLE
    assert mri.compression.section.data["compression_ratio"] == report.compression_ratio


def test_mri_qat_enabled():
    model = prepare_qat(Linear(1, 1), QATConfig(bits=4))
    mri = inspect_model_mri(model)
    assert mri.qat.section.status == DataStatus.AVAILABLE
    assert mri.qat.section.data["enabled"] is True
    assert mri.qat.section.data["bits"] == 4


def test_mri_qat_disabled():
    mri = Linear(1, 1).mri()
    assert mri.qat.section.status == DataStatus.NOT_APPLICABLE


def test_mri_to_dict_json_serializable():
    model = Linear(1, 1)
    x = Tensor(np.array([[1.0]]))
    loss = model(x).sum()
    loss.backward()
    mri = model.mri(output=loss)
    payload = mri.to_dict()
    text = mri.to_json()
    parsed = json.loads(text)
    assert parsed["model"]["status"] == "available"
    assert "observations" in parsed


def test_mri_formatted_report():
    model = Linear(1, 1)
    report = str(model.mri())
    assert "AI-MRI" in report
    assert "MODEL" in report


def test_mri_prove_delegates_to_proof_engine():
    model = Linear(1, 1)
    x = Tensor(np.array([[2.0]]))
    loss = model(x).sum()
    loss.backward()
    mri = model.mri(output=loss)
    proof = mri.prove(model.weight)
    assert proof.steps or proof.accumulated_gradient is not None


def test_mri_prove_requires_output():
    mri = Linear(1, 1).mri()
    with pytest.raises(ValueError, match="output"):
        mri.prove(Linear(1, 1).weight)


def test_mri_graph_explorer_matches_loss_explore():
    model = Linear(1, 1)
    x = Tensor(np.array([[1.0]]))
    loss = model(x).sum()
    loss.backward()
    mri = model.mri(output=loss)
    explorer = mri.graph_explorer()
    direct = loss.explore()
    assert explorer.graph.vertex_count == direct.graph.vertex_count
    assert explorer.graph.edge_count == direct.graph.edge_count


def test_compare_mri():
    before = Linear(1, 1).mri()
    model = prepare_qat(Linear(1, 1), QATConfig(bits=4))
    after = model.mri()
    diff = compare_mri(before, after)
    assert "observation_changes" in diff


def test_mri_no_health_score():
    payload = Linear(1, 1).mri().to_dict()
    assert "health_score" not in payload
    assert "health" not in payload

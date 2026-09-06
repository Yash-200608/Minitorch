import numpy as np
import pytest

from minitorch import (
    Linear,
    MSELoss,
    PerformanceProfile,
    ProfilerMode,
    Sequential,
    SGD,
    Tensor,
    Trainer,
    benchmark_timing,
    profile_cache_comparison,
    profile_compression_tradeoffs,
    profile_graph_optimizer_comparison,
    profile_model_compression,
    profile_profiling_overhead,
    profile_quantization_memory_matrix,
    profile_quantization_stages,
    profile_rotation_pipeline,
    profile_standard_tensor_operations,
    profile_tensor_operation,
    profile_training_step,
)
from minitorch.profiling.timing import Timer, measure_time


def test_timer_records_non_negative_elapsed():
    with Timer() as timer:
        _ = sum(range(1000))
    assert timer.elapsed_seconds >= 0


def test_measure_time_returns_result_and_duration():
    value, elapsed = measure_time(lambda: 2 + 2)
    assert value == 4
    assert elapsed >= 0


def test_benchmark_timing_structure_and_repeat_count():
    result = benchmark_timing("noop", lambda: None, warmup=2, repeats=5)
    assert result.name == "noop"
    assert result.warmup == 2
    assert result.runs == 5
    assert len(result.samples) == 5
    assert result.mean_seconds >= 0
    assert result.median_seconds >= 0
    assert result.min_seconds >= 0
    assert result.max_seconds >= 0
    assert result.min_seconds <= result.max_seconds


def test_benchmark_timing_warmup_does_not_count_as_sample():
    calls = 0

    def counter() -> None:
        nonlocal calls
        calls += 1

    benchmark_timing("count", counter, warmup=3, repeats=4)
    assert calls == 7


def test_profile_training_step_reports_stages_and_graph():
    inputs = Tensor(np.linspace(0, 1, 8).reshape(4, 2))
    targets = Tensor(np.linspace(0, 1, 4).reshape(4, 1))
    model = Sequential(Linear(2, 4), Linear(4, 1))
    optimizer = SGD(model.parameters(), lr=0.1)
    profile = profile_training_step(
        model,
        optimizer,
        MSELoss().forward,
        inputs,
        targets,
    )

    assert profile.stages["forward"] >= 0
    assert profile.stages["backward"] >= 0
    assert profile.stages["optimizer"] >= 0
    assert profile.graph_stats["graph_nodes"] > 0
    assert profile.dsa_stats["dfs_visits"] > 0
    assert profile.metadata["loss"] > 0
    assert profile.hottest_stage() in profile.stages


def test_profile_training_step_detailed_collects_operation_timings():
    inputs = Tensor(np.ones((4, 2)))
    targets = Tensor(np.ones((4, 1)))
    model = Linear(2, 1)
    optimizer = SGD(model.parameters(), lr=0.1)
    profile = profile_training_step(
        model,
        optimizer,
        MSELoss().forward,
        inputs,
        targets,
        mode=ProfilerMode.DETAILED,
    )
    assert profile.operation_timings
    hottest = profile.top_operations(k=3)
    assert len(hottest) <= 3
    assert all(time >= 0 for _, time in hottest)


def test_profile_tensor_operation_matmul():
    result = profile_tensor_operation("matmul", lambda left, right: left @ right, repeats=3)
    assert result.mean_seconds >= 0


def test_profile_standard_tensor_operations_cover_expected_ops():
    timings = profile_standard_tensor_operations(repeats=2)
    for op in ("add", "matmul", "relu", "mean"):
        assert op in timings
        assert timings[op] >= 0


def test_performance_profile_report_contains_hottest_stage():
    profile = PerformanceProfile(
        name="demo",
        mode="lightweight",
        stages={"forward": 0.01, "backward": 0.02},
    )
    report = profile.report()
    assert "backward" in report
    assert "HOTTEST MEASURED STAGE" in report


def test_profile_quantization_stages_reports_memory_and_timing():
    data = np.linspace(-1, 1, 16)
    profile = profile_quantization_stages(data, bits=4, method="uniform", packed=True)
    assert profile.stages["uniform_quantization"] >= 0
    assert profile.stages["packing"] >= 0
    assert profile.memory["representation_bytes"] > 0
    assert profile.memory["reconstruction_rmse"] >= 0


def test_profile_quantization_memory_matrix_has_six_entries():
    data = np.linspace(-1, 1, 32)
    matrix = profile_quantization_memory_matrix(data)
    assert len(matrix) == 6
    for entry in matrix:
        assert entry["quantization_seconds"] >= 0
        assert entry["packing_seconds"] >= 0


def test_profile_rotation_pipeline_requires_power_of_two_axis():
    data = np.linspace(-1, 1, 8)
    profile = profile_rotation_pipeline(data, bits=4)
    assert "rotation" in profile.stages
    assert profile.stages["total"] >= 0

    with pytest.raises(ValueError):
        profile_rotation_pipeline(np.linspace(-1, 1, 7), bits=4)


def test_profile_graph_optimizer_comparison_reports_both_graphs():
    inputs = Tensor(np.linspace(0, 1, 4).reshape(2, 2))
    targets = Tensor(np.linspace(0, 1, 2).reshape(2, 1))
    model = Linear(2, 1)
    profile = profile_graph_optimizer_comparison(
        model,
        inputs,
        targets,
        MSELoss().forward,
        repeats=2,
    )
    assert profile.graph_stats["original_nodes"] > 0
    assert profile.stages["original_backward"] >= 0
    assert profile.stages["optimized_backward"] >= 0


def test_profile_model_compression_reports_stages():
    model = Linear(4, 2)
    profile = profile_model_compression(model)
    assert profile.stages["compress_model"] >= 0
    assert profile.stages["materialize"] >= 0
    assert profile.memory["compression_ratio"] > 0


def test_profile_compression_tradeoffs_rows():
    model = Linear(4, 2)
    inputs = Tensor(np.linspace(0, 1, 16).reshape(4, 4))
    targets = Tensor(np.linspace(0, 1, 8).reshape(4, 2))
    rows = profile_compression_tradeoffs(model, inputs, targets)
    assert rows
    assert all(row["time_seconds"] >= 0 for row in rows)


def test_profile_cache_comparison_reports_both_modes():
    def expression() -> Tensor:
        x = Tensor([1.0, 2.0, 3.0])
        return (x * 2.0 + 1.0).sum()

    profile = profile_cache_comparison(expression, iterations=10)
    assert profile.stages["cache_disabled"] >= 0
    assert profile.stages["cache_enabled"] >= 0


def test_profile_profiling_overhead_reports_three_modes():
    profile = profile_profiling_overhead(lambda: sum(range(100)), warmup=1, repeats=3)
    assert profile.stages["none"] >= 0
    assert profile.stages["lightweight"] >= 0
    assert profile.stages["detailed"] >= 0


def test_trainer_fit_profile_flag_attaches_profile():
    inputs = Tensor(np.linspace(0, 1, 4).reshape(2, 2))
    targets = Tensor(np.linspace(0, 1, 2).reshape(2, 1))
    model = Linear(2, 1)
    trainer = Trainer(model, SGD(model.parameters(), lr=0.1), MSELoss().forward)
    history = trainer.fit(inputs, targets, epochs=2, profile=True)
    assert history.profile is not None
    assert history.profile.stages["forward"] >= 0

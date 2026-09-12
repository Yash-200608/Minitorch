"""Structural tests for Phase S benchmark tooling (no timing thresholds)."""

import json
from pathlib import Path

from benchmarks.canonical_suite import (
    CanonicalBenchmarkReport,
    BenchmarkEnvironment,
    format_benchmark_report,
    run_canonical_suite,
)
from benchmarks.test_matrix import build_test_summary, format_test_summary


def test_test_matrix_counts_match_collection():
    summary = build_test_summary(Path.cwd())
    assert summary["total_collected"] >= 422
    assert sum(summary["by_subsystem"].values()) == summary["total_collected"]
    assert "Experiment Lab" in summary["by_subsystem"]
    text = format_test_summary(summary)
    assert "MiniTorch Test Summary" in text
    assert "Total collected:" in text


def test_canonical_suite_fast_structure():
    report = run_canonical_suite(seed=7, epochs=5, fast=True)
    assert report.baseline_training["final_loss"] is not None
    assert len(report.quantization_matrix) >= 8
    assert report.qat.get("result_label")
    assert report.graph_optimizer.get("result") in {"improved", "unchanged", "worse", "not_measured"}
    assert len(report.packing_validation) >= 1
    assert report.packing_validation[0]["max_output_difference"] == 0.0
    text = format_benchmark_report(report)
    assert "QUANTIZATION" in text
    assert "FINDINGS" in text


def test_benchmark_report_serializable(tmp_path):
    report = CanonicalBenchmarkReport(
        environment=BenchmarkEnvironment.capture(),
        baseline_training={"final_loss": 0.1},
        findings=[{"category": "test", "message": "ok", "evidence": {}}],
    )
    payload = report.to_dict()
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["baseline_training"]["final_loss"] == 0.1
    assert loaded["environment"]["framework_version"]

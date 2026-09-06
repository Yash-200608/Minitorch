"""Canonical MiniTorch benchmark suite — reproducible validation measurements."""

from __future__ import annotations

import json
import platform
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from minitorch import (
    Adam,
    CompressionConfig,
    MSELoss,
    MixedPrecisionConfig,
    Sequential,
    Sigmoid,
    Tensor,
    Trainer,
    benchmark_parallel_comparison,
    compare_model_compression,
    compare_quantization_rotation,
    compress_model,
    compress_model_mixed_precision,
    profile_graph_optimizer_comparison,
    profile_training_step,
    run_qat_experiment,
    save_compressed_model,
    save_model,
    save_training_checkpoint,
)
from minitorch.experiments import (
    ExperimentConfig,
    ExperimentTechnique,
    build_baseline_dataset,
    build_baseline_model,
    clone_model_with_state,
    save_experiment_result,
    seed_everything,
)
from minitorch.experiments.runner import Experiment, _train_model
from minitorch.kernels import benchmark_vectorization, pack_codes_reference, pack_codes_vectorized
from minitorch.persistence.format import FRAMEWORK_VERSION


CANONICAL_SEED = 42
CANONICAL_EPOCHS = 35
CANONICAL_QAT_EPOCHS = 50


@dataclass
class BenchmarkEnvironment:
    python_version: str
    numpy_version: str
    platform: str
    framework_version: str
    processor: str

    @staticmethod
    def capture() -> BenchmarkEnvironment:
        return BenchmarkEnvironment(
            python_version=sys.version.split()[0],
            numpy_version=np.__version__,
            platform=platform.platform(),
            framework_version=FRAMEWORK_VERSION,
            processor=platform.processor() or "unknown",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "platform": self.platform,
            "framework_version": self.framework_version,
            "processor": self.processor,
        }


def _nonlinear_dataset(seed: int = CANONICAL_SEED) -> tuple[Tensor, Tensor]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(48, 2))
    y = np.tanh(x[:, 0] * 1.5) + 0.25 * x[:, 1]
    return Tensor(x), Tensor(y.reshape(-1, 1))


def _nonlinear_model() -> Sequential:
    from minitorch import Linear

    return Sequential(Linear(2, 8), Sigmoid(), Linear(8, 1))


@dataclass
class CanonicalBenchmarkReport:
    schema_version: str = "1.0"
    environment: BenchmarkEnvironment = field(default_factory=BenchmarkEnvironment.capture)
    seed: int = CANONICAL_SEED
    configuration: dict[str, Any] = field(default_factory=dict)
    baseline_training: dict[str, Any] = field(default_factory=dict)
    quantization_matrix: list[dict[str, Any]] = field(default_factory=list)
    packing_validation: list[dict[str, Any]] = field(default_factory=list)
    rotation_validation: list[dict[str, Any]] = field(default_factory=list)
    mixed_precision: list[dict[str, Any]] = field(default_factory=list)
    qat: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)
    graph_optimizer: dict[str, Any] = field(default_factory=dict)
    parallelism: list[dict[str, Any]] = field(default_factory=list)
    vectorization: list[dict[str, Any]] = field(default_factory=list)
    persistence_sizes: dict[str, Any] = field(default_factory=dict)
    nonlinear_workload: dict[str, Any] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "environment": self.environment.to_dict(),
            "seed": self.seed,
            "configuration": self.configuration,
            "baseline_training": self.baseline_training,
            "quantization_matrix": self.quantization_matrix,
            "packing_validation": self.packing_validation,
            "rotation_validation": self.rotation_validation,
            "mixed_precision": self.mixed_precision,
            "qat": self.qat,
            "performance": self.performance,
            "graph_optimizer": self.graph_optimizer,
            "parallelism": self.parallelism,
            "vectorization": self.vectorization,
            "persistence_sizes": self.persistence_sizes,
            "nonlinear_workload": self.nonlinear_workload,
            "findings": self.findings,
        }

    def add_finding(self, category: str, message: str, evidence: dict[str, Any]) -> None:
        self.findings.append({"category": category, "message": message, "evidence": evidence})


def run_canonical_suite(
    *,
    seed: int = CANONICAL_SEED,
    epochs: int = CANONICAL_EPOCHS,
    fast: bool = False,
) -> CanonicalBenchmarkReport:
    """Run the full canonical benchmark suite and return structured measurements."""
    report = CanonicalBenchmarkReport(seed=seed)
    report.configuration = {
        "model": "Sequential(Linear(1,8), ReLU, Linear(8,1))",
        "dataset": "deterministic regression y=3x+2",
        "optimizer": "Adam",
        "learning_rate": 0.05,
        "epochs": epochs,
        "fast_mode": fast,
    }

    seed_everything(seed)
    inputs, targets = build_baseline_dataset(seed)
    model = build_baseline_model()
    untrained_template = clone_model_with_state(model)

    loss_fn = MSELoss().forward
    trainer = Trainer(model, Adam(model.parameters(), lr=0.05), loss_fn)
    initial_loss = float(loss_fn(model(inputs), targets).data)

    train_start = time.perf_counter()
    history = trainer.fit(inputs, targets, epochs=epochs)
    train_seconds = time.perf_counter() - train_start
    final_loss = history.losses[-1]
    model_bytes = sum(p.data.nbytes for _, p in model.named_parameters())
    trained = model

    profile = profile_training_step(model, inputs, targets, loss_fn, warmup=1, repeats=2)

    report.baseline_training = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "training_seconds": train_seconds,
        "parameter_count": len(list(model.named_parameters())),
        "model_bytes": model_bytes,
        "profile_stages_seconds": dict(profile.stages),
        "hottest_stage": profile.hottest_stage(),
    }

    # Quantization matrix on trained model
    matrix_configs = [
        ("FP32", None),
        ("Uniform 8-bit", CompressionConfig(bits=8, method="uniform")),
        ("Uniform 4-bit", CompressionConfig(bits=4, method="uniform")),
        ("Uniform 2-bit", CompressionConfig(bits=2, method="uniform")),
        ("Codebook 8-bit", CompressionConfig(bits=8, method="codebook")),
        ("Codebook 4-bit", CompressionConfig(bits=4, method="codebook")),
        ("Codebook 2-bit", CompressionConfig(bits=2, method="codebook")),
    ]
    for label, comp_config in matrix_configs:
        if comp_config is None:
            fp_loss = float(loss_fn(trained(inputs), targets).data)
            report.quantization_matrix.append(
                {
                    "label": label,
                    "compressed_bytes": model_bytes,
                    "compression_ratio": 1.0,
                    "parameter_rmse": 0.0,
                    "output_rmse": 0.0,
                    "fp_loss": fp_loss,
                    "compressed_loss": fp_loss,
                    "loss_delta": 0.0,
                }
            )
            continue
        comp_report = compare_model_compression(trained, inputs, targets, config=comp_config)
        report.quantization_matrix.append(
            {
                "label": label,
                "compressed_bytes": comp_report.total_compressed_bytes,
                "compression_ratio": comp_report.compression_ratio,
                "parameter_rmse": comp_report.aggregate_parameter_rmse,
                "output_rmse": comp_report.output_rmse,
                "fp_loss": comp_report.original_loss,
                "compressed_loss": comp_report.compressed_loss,
                "loss_delta": comp_report.loss_delta,
                "codebook_bytes": sum(r.codebook_bytes for r in comp_report.parameter_records),
            }
        )

    mixed_compressed = compress_model_mixed_precision(
        trained, MixedPrecisionConfig(default_bits=8, parameter_bits={"0.weight": 8, "2.weight": 4})
    )
    mixed_report = mixed_compressed.report(template=trained, inputs=inputs, targets=targets)
    report.quantization_matrix.append(
        {
            "label": "Mixed precision",
            "compressed_bytes": mixed_report.total_compressed_bytes,
            "compression_ratio": mixed_report.compression_ratio,
            "parameter_rmse": mixed_report.aggregate_parameter_rmse,
            "output_rmse": mixed_report.output_rmse,
            "fp_loss": mixed_report.original_loss,
            "compressed_loss": mixed_report.compressed_loss,
            "loss_delta": mixed_report.loss_delta,
            "precision_map": dict(mixed_report.precision_map),
        }
    )

    # Packing validation
    for bits, method in ((4, "uniform"), (4, "codebook")):
        logical_cfg = CompressionConfig(bits=bits, method=method, packed=False)
        packed_cfg = CompressionConfig(bits=bits, method=method, packed=True)
        logical = compress_model(trained, config=logical_cfg)
        packed = compress_model(trained, config=packed_cfg)
        logical_model = logical.materialize(trained)
        packed_model = packed.materialize(trained)
        logical_out = logical_model(inputs).data
        packed_out = packed_model(inputs).data
        max_diff = float(np.max(np.abs(logical_out - packed_out)))
        logical_bytes = logical.report().total_compressed_bytes
        packed_bytes = packed.report().total_compressed_bytes
        report.packing_validation.append(
            {
                "bits": bits,
                "method": method,
                "max_output_difference": max_diff,
                "logical_bytes": logical_bytes,
                "packed_bytes": packed_bytes,
                "byte_reduction": logical_bytes - packed_bytes,
            }
        )

    # Rotation validation
    for method in ("uniform", "codebook"):
        for bits in (4, 8):
            param_data = trained.parameter("0.weight").data
            rot_exp = compare_quantization_rotation(
                param_data, bits, method=method, packed=False, rotation="combined"
            )
            report.rotation_validation.append(
                {
                    "method": method,
                    "bits": bits,
                    "baseline_rmse": rot_exp.baseline_error.rmse,
                    "rotated_rmse": rot_exp.rotated_error.rmse,
                    "result_label": rot_exp.result_label,
                    "baseline_bytes": rot_exp.baseline_memory.total_actual_representation_bytes,
                    "rotated_bytes": rot_exp.rotated_memory.total_actual_representation_bytes,
                }
            )

    # Mixed precision vs global
    for bits in (8, 4, 2):
        global_report = compare_model_compression(
            trained, inputs, targets, config=CompressionConfig(bits=bits, method="uniform")
        )
        report.mixed_precision.append(
            {
                "label": f"Global {bits}-bit",
                "compressed_bytes": global_report.total_compressed_bytes,
                "output_rmse": global_report.output_rmse,
                "loss_delta": global_report.loss_delta,
            }
        )
    report.mixed_precision.append(
        {
            "label": "Mixed (8 default, 2.weight 4-bit)",
            "compressed_bytes": mixed_report.total_compressed_bytes,
            "output_rmse": mixed_report.output_rmse,
            "loss_delta": mixed_report.loss_delta,
        }
    )

    # QAT (moderate epochs; skip in fast mode)
    qat_epochs = 25 if fast else CANONICAL_QAT_EPOCHS
    qat_template = clone_model_with_state(untrained_template)
    qat_result = run_qat_experiment(
        qat_template,
        lambda m: Adam(m.parameters(), lr=0.05),
        loss_fn,
        inputs,
        targets,
        epochs=qat_epochs,
        bits=4,
    )
    report.qat = {
        "epochs": qat_epochs,
        "normal_fp_loss": qat_result.normal_fp_loss,
        "normal_quantized_loss": qat_result.normal_quantized_loss,
        "normal_quantization_damage": qat_result.normal_quantization_damage,
        "qat_fp_loss": qat_result.qat_fp_loss,
        "qat_quantized_loss": qat_result.qat_quantized_loss,
        "qat_quantization_damage": qat_result.qat_quantization_damage,
        "normal_output_rmse": qat_result.normal_output_rmse,
        "qat_output_rmse": qat_result.qat_output_rmse,
        "normal_training_seconds": qat_result.normal_training_seconds,
        "qat_training_seconds": qat_result.qat_training_seconds,
        "forward_overhead_ratio": qat_result.qat_forward_overhead_ratio,
        "result_label": qat_result.result_label,
    }

    # Graph optimizer
    graph_model = clone_model_with_state(untrained_template)
    _train_model(graph_model, inputs, targets, TrainingConfig(epochs=epochs))
    graph_profile = profile_graph_optimizer_comparison(
        graph_model, inputs, targets, loss_fn, warmup=1, repeats=2
    )
    before_nodes = graph_profile.graph_stats.get("original_vertices")
    after_nodes = graph_profile.graph_stats.get("optimized_vertices")
    before_edges = graph_profile.graph_stats.get("original_edges")
    after_edges = graph_profile.graph_stats.get("optimized_edges")
    orig_back = graph_profile.stages.get("original_backward")
    opt_back = graph_profile.stages.get("optimized_backward")
    if orig_back is not None and opt_back is not None:
        if opt_back < orig_back - 1e-9:
            graph_result = "improved"
        elif opt_back > orig_back + 1e-9:
            graph_result = "worse"
        else:
            graph_result = "unchanged"
    else:
        graph_result = "not_measured"
    report.graph_optimizer = {
        "nodes_before": before_nodes,
        "nodes_after": after_nodes,
        "edges_before": before_edges,
        "edges_after": after_edges,
        "original_backward_seconds": orig_back,
        "optimized_backward_seconds": opt_back,
        "result": graph_result,
    }

    # Parallel compression
    comp_cfg = CompressionConfig(bits=4, method="uniform", packed=True)
    for workers in (1, 2, 4):
        if workers == 1:
            bench = benchmark_parallel_comparison(
                "compress_serial",
                lambda: compress_model(trained, config=comp_cfg),
                lambda: compress_model(trained, config=comp_cfg),
                workers=1,
                warmup=1,
                repeats=3,
            )
        else:
            bench = benchmark_parallel_comparison(
                f"compress_parallel_{workers}",
                lambda: compress_model(trained, config=comp_cfg),
                lambda: compress_model(trained, config=comp_cfg, parallel=True, workers=workers),
                workers=workers,
                warmup=1,
                repeats=3,
            )
        report.parallelism.append(
            {
                "workers": workers,
                "serial_seconds": bench.serial_seconds,
                "parallel_seconds": bench.parallel_seconds,
                "speedup": bench.speedup,
            }
        )

    # Vectorization (pack_codes)
    sizes = {"small": 64, "medium": 256, "large": 1024}
    if fast:
        sizes = {"small": 64, "medium": 256}
    for label, size in sizes.items():
        codes = (np.arange(size, dtype=np.uint8) % 16)
        vec = benchmark_vectorization(
            f"pack_codes_{label}",
            size,
            lambda: pack_codes_reference(codes, bits=4),
            lambda: pack_codes_vectorized(codes, bits=4),
            warmup=1,
            repeats=3,
        )
        report.vectorization.append(
            {
                "workload": label,
                "size": size,
                "reference_seconds": vec.reference_seconds,
                "vectorized_seconds": vec.vectorized_seconds,
                "speedup": vec.speedup,
            }
        )

    report.performance = {
        "training_profile": dict(profile.stages),
        "hottest_training_stage": profile.hottest_stage(),
    }

    # Persistence file sizes
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        model_path = tmp_path / "model.mtc"
        compressed_path = tmp_path / "compressed.mtc"
        checkpoint_path = tmp_path / "checkpoint.mtc"
        save_model(trained, str(model_path))
        compressed = compress_model(trained, config=CompressionConfig(bits=4, method="codebook"))
        save_compressed_model(compressed, str(compressed_path))
        opt = Adam(trained.parameters(), lr=0.05)
        trainer_ck = Trainer(trained, opt, loss_fn)
        trainer_ck.fit(inputs, targets, epochs=5)
        save_training_checkpoint(trained, opt, trainer_ck.resume_state(), str(checkpoint_path))
        exp_result = Experiment(
            ExperimentConfig(
                name="phase_s_export",
                technique=ExperimentTechnique.FP32_BASELINE,
                training=TrainingConfig(epochs=0),
                seed=seed,
            )
        ).run()
        exp_path = tmp_path / "experiment.json"
        save_experiment_result(exp_result, exp_path)
        report.persistence_sizes = {
            "model_artifact_bytes": model_path.stat().st_size,
            "compressed_artifact_bytes": compressed_path.stat().st_size,
            "checkpoint_artifact_bytes": checkpoint_path.stat().st_size,
            "experiment_result_bytes": exp_path.stat().st_size,
            "in_memory_model_bytes": model_bytes,
            "in_memory_compressed_bytes": compressed.report().total_compressed_bytes,
        }

    # Nonlinear workload spot check
    seed_everything(seed + 1)
    nl_inputs, nl_targets = _nonlinear_dataset(seed + 1)
    nl_model = _nonlinear_model()
    nl_trainer = Trainer(nl_model, Adam(nl_model.parameters(), lr=0.05), loss_fn)
    nl_trainer.fit(nl_inputs, nl_targets, epochs=epochs)
    nl_uniform = compare_model_compression(
        nl_model, nl_inputs, nl_targets, config=CompressionConfig(bits=4, method="uniform")
    )
    nl_codebook = compare_model_compression(
        nl_model, nl_inputs, nl_targets, config=CompressionConfig(bits=4, method="codebook")
    )
    report.nonlinear_workload = {
        "model": "Sequential(Linear(2,8), Sigmoid, Linear(8,1))",
        "final_loss": nl_trainer._loss_history[-1],
        "uniform_4bit_output_rmse": nl_uniform.output_rmse,
        "codebook_4bit_output_rmse": nl_codebook.output_rmse,
        "uniform_4bit_parameter_rmse": nl_uniform.aggregate_parameter_rmse,
        "codebook_4bit_parameter_rmse": nl_codebook.aggregate_parameter_rmse,
    }

    _derive_findings(report)
    return report


def _derive_findings(report: CanonicalBenchmarkReport) -> None:
    """Populate evidence-scoped findings from measured benchmark data."""
    matrix = report.quantization_matrix
    if len(matrix) >= 3:
        u4 = next((r for r in matrix if r["label"] == "Uniform 4-bit"), None)
        c4 = next((r for r in matrix if r["label"] == "Codebook 4-bit"), None)
        if u4 and c4:
            report.add_finding(
                "codebook",
                "Codebook 4-bit parameter RMSE compared to uniform 4-bit on canonical regression workload.",
                {
                    "uniform_parameter_rmse": u4["parameter_rmse"],
                    "codebook_parameter_rmse": c4["parameter_rmse"],
                    "uniform_output_rmse": u4["output_rmse"],
                    "codebook_output_rmse": c4["output_rmse"],
                    "codebook_bytes_overhead": c4.get("codebook_bytes"),
                },
            )

    for pack in report.packing_validation:
        if pack["max_output_difference"] == 0.0:
            report.add_finding(
                "packing",
                f"Packing preserved outputs for {pack['method']} {pack['bits']}-bit on canonical model.",
                pack,
            )

    rotations_worse = sum(1 for r in report.rotation_validation if r["result_label"] == "worse")
    rotations_improved = sum(1 for r in report.rotation_validation if r["result_label"] == "improved")
    report.add_finding(
        "rotation",
        "Rotation labels counted across parameter tensor experiments on canonical workload.",
        {
            "worse_count": rotations_worse,
            "improved_count": rotations_improved,
            "total": len(report.rotation_validation),
        },
    )

    if report.qat:
        report.add_finding(
            "qat",
            report.qat.get("result_label", "QAT experiment completed"),
            {
                "normal_quantization_damage": report.qat.get("normal_quantization_damage"),
                "qat_quantization_damage": report.qat.get("qat_quantization_damage"),
            },
        )

    go = report.graph_optimizer
    report.add_finding(
        "graph_optimizer",
        f"Graph optimizer backward result: {go.get('result')}",
        go,
    )

    parallel = report.parallelism
    if parallel:
        best = max(parallel, key=lambda x: x.get("speedup") or 0)
        report.add_finding(
            "parallelism",
            "Parallel compression speedup observed at tested worker counts.",
            {"entries": parallel, "best_speedup": best.get("speedup")},
        )


def format_benchmark_report(report: CanonicalBenchmarkReport) -> str:
    lines = [
        "MiniTorch Final Benchmark",
        "─────────────────────────",
        "",
        "Model: Sequential(Linear → ReLU → Linear)",
        "Dataset: deterministic regression (y = 3x + 2)",
        f"Seed: {report.seed}",
        "",
        "BASELINE TRAINING",
        f"  Initial loss: {report.baseline_training.get('initial_loss')}",
        f"  Final loss: {report.baseline_training.get('final_loss')}",
        f"  Training time (s): {report.baseline_training.get('training_seconds')}",
        f"  Model bytes: {report.baseline_training.get('model_bytes')}",
        "",
        "QUANTIZATION",
        "Configuration          Bytes    Ratio    Out RMSE   Loss Δ",
        "--------------------------------------------------------",
    ]
    for row in report.quantization_matrix:
        lines.append(
            f"{row['label']:<22} "
            f"{row.get('compressed_bytes', '—'):>6} "
            f"{row.get('compression_ratio', '—'):>6} "
            f"{row.get('output_rmse', '—'):>10} "
            f"{row.get('loss_delta', '—'):>8}"
        )
    lines.extend(["", "QAT",])
    if report.qat:
        lines.extend(
            [
                f"  Normal FP loss: {report.qat.get('normal_fp_loss')}",
                f"  Normal quantized loss: {report.qat.get('normal_quantized_loss')}",
                f"  QAT FP loss: {report.qat.get('qat_fp_loss')}",
                f"  QAT quantized loss: {report.qat.get('qat_quantized_loss')}",
                f"  Label: {report.qat.get('result_label')}",
            ]
        )
    lines.extend(["", "GRAPH OPTIMIZER"])
    go = report.graph_optimizer
    lines.extend(
        [
            f"  Nodes: {go.get('nodes_before')} → {go.get('nodes_after')}",
            f"  Edges: {go.get('edges_before')} → {go.get('edges_after')}",
            f"  Backward (s): {go.get('original_backward_seconds')} → {go.get('optimized_backward_seconds')}",
            f"  Result: {go.get('result')}",
        ]
    )
    lines.extend(["", "FINDINGS (scoped to this benchmark)"])
    for item in report.findings:
        lines.append(f"  [{item['category']}] {item['message']}")
    return "\n".join(lines)


def save_benchmark_report(report: CanonicalBenchmarkReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def run_and_save(output_dir: Path | None = None, fast: bool = False) -> CanonicalBenchmarkReport:
    out = output_dir or Path(__file__).resolve().parent / "results"
    report = run_canonical_suite(fast=fast)
    save_benchmark_report(report, out / "canonical_benchmark.json")
    (out / "canonical_benchmark.txt").write_text(format_benchmark_report(report), encoding="utf-8")
    return report

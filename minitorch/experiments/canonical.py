from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..compression import CompressionConfig, compare_model_compression, compress_model
from ..kernels import benchmark_vectorization
from ..losses import MSELoss
from ..mixed_precision import MixedPrecisionConfig
from ..parallel import benchmark_parallel_comparison
from ..profiling.profile import profile_graph_optimizer_comparison
from ..qat import run_qat_experiment
from ..rotation import compare_quantization_rotation
from ..tensor import Tensor
from ..training import Trainer
from .config import (
    CompressionExperimentConfig,
    ExperimentConfig,
    ExperimentTechnique,
    MixedPrecisionExperimentConfig,
    TrainingConfig,
)
from .identity import compute_experiment_id, EXPERIMENT_SCHEMA_VERSION
from .randomness import seed_everything
from .result import (
    ExperimentComparisonResult,
    ExperimentEnvironment,
    ExperimentMemory,
    ExperimentMetrics,
    ExperimentObservation,
    ExperimentResult,
    ExperimentTimings,
)
from .runner import (
    _compression_label,
    _loss_fn,
    _make_optimizer,
    _merge_repeated_runs,
    _result_from_compression_report,
    _train_model,
    build_baseline_dataset,
    build_baseline_model,
    build_dataset_from_config,
    build_model_from_config,
)
from .registry import CANONICAL_EXPERIMENTS
from .snapshot import clone_model_with_state, snapshot_model_state


def run_quantization_matrix(config: ExperimentConfig | None = None) -> ExperimentComparisonResult:
    """Canonical matrix comparing FP32 through mixed precision storage techniques."""
    base = config or ExperimentConfig(
        name="quantization_matrix",
        technique=ExperimentTechnique.QUANTIZATION_MATRIX,
        training=TrainingConfig(epochs=30),
    )
    seed_everything(base.seed)
    inputs, targets = build_dataset_from_config(base.dataset, base.seed)
    template = build_model_from_config(base.model)
    snapshot = snapshot_model_state(template)
    training_seconds, training_loss = _train_model(template, inputs, targets, base.training)
    model = clone_model_with_state(template)

    rows: list[ExperimentResult] = []

    fp_bytes = sum(p.data.nbytes for _, p in model.named_parameters())
    eval_start = time.perf_counter()
    fp_loss = float(_loss_fn()(model(inputs), targets).data)
    eval_seconds = time.perf_counter() - eval_start
    rows.append(
        ExperimentResult(
            schema_version=EXPERIMENT_SCHEMA_VERSION,
            experiment_id=compute_experiment_id(base),
            name="FP32",
            seed=base.seed,
            configuration=base.to_dict(),
            environment=ExperimentEnvironment.capture(),
            metrics=ExperimentMetrics(
                training_final_loss=training_loss,
                fp_final_loss=fp_loss,
                compressed_final_loss=fp_loss,
                loss_delta=0.0,
                parameter_rmse=0.0,
                output_rmse=0.0,
                model_bytes=fp_bytes,
                compressed_bytes=fp_bytes,
                compression_ratio=1.0,
            ),
            timings=ExperimentTimings(
                training_seconds=training_seconds,
                evaluation_seconds=eval_seconds,
            ),
            memory=ExperimentMemory(model_bytes=fp_bytes, compressed_bytes=fp_bytes),
            technique_label="FP32",
            training_technique=base.training.technique.value,
            storage_technique="fp32",
        )
    )

    matrix_configs: list[tuple[str, CompressionConfig]] = [
        ("Uniform 8-bit", CompressionConfig(bits=8, method="uniform")),
        ("Uniform 4-bit", CompressionConfig(bits=4, method="uniform")),
        ("Uniform 2-bit", CompressionConfig(bits=2, method="uniform")),
        ("Codebook 4-bit", CompressionConfig(bits=4, method="codebook")),
        (
            "Codebook + Rotation 4-bit",
            CompressionConfig(bits=4, method="codebook", rotation="combined"),
        ),
    ]

    for label, comp_config in matrix_configs:
        compress_start = time.perf_counter()
        compressed = compress_model(model, config=comp_config)
        compression_seconds = time.perf_counter() - compress_start
        materialize_start = time.perf_counter()
        _ = compressed.materialize(model)
        materialization_seconds = time.perf_counter() - materialize_start
        eval_start = time.perf_counter()
        report = compressed.report(template=model, inputs=inputs, targets=targets)
        eval_seconds = time.perf_counter() - eval_start
        rows.append(
            _result_from_compression_report(
                config=base,
                report=report,
                training_seconds=0.0,
                compression_seconds=compression_seconds,
                materialization_seconds=materialization_seconds,
                evaluation_seconds=eval_seconds,
                training_final_loss=training_loss,
                technique_label=label,
                training_technique=base.training.technique.value,
                storage_technique="post_training_compression",
                observations=[
                    ExperimentObservation(
                        category="matrix",
                        message=f"{label} measured on shared trained model.",
                    ),
                ],
            )
        )

    mixed_config = MixedPrecisionConfig(default_bits=4)
    from ..mixed_precision import compress_model_mixed_precision

    compress_start = time.perf_counter()
    mixed_compressed = compress_model_mixed_precision(model, mixed_config)
    compression_seconds = time.perf_counter() - compress_start
    materialize_start = time.perf_counter()
    _ = mixed_compressed.materialize(model)
    materialization_seconds = time.perf_counter() - materialize_start
    eval_start = time.perf_counter()
    mixed_report = mixed_compressed.report(template=model, inputs=inputs, targets=targets)
    eval_seconds = time.perf_counter() - eval_start
    rows.append(
        _result_from_compression_report(
            config=base,
            report=mixed_report,
            training_seconds=0.0,
            compression_seconds=compression_seconds,
            materialization_seconds=materialization_seconds,
            evaluation_seconds=eval_seconds,
            training_final_loss=training_loss,
            technique_label="Mixed precision 4-bit",
            training_technique=base.training.technique.value,
            storage_technique="mixed_precision",
            observations=[
                ExperimentObservation(
                    category="matrix",
                    message="Mixed precision 4-bit measured on shared trained model.",
                ),
            ],
            extra={"precision_map": dict(mixed_report.precision_map)},
        )
    )

    metric_names = tuple(
        [
            "model_bytes",
            "compressed_bytes",
            "compression_ratio",
            "parameter_rmse",
            "output_rmse",
            "loss_delta",
            "compression_seconds",
        ]
    )
    return ExperimentComparisonResult(rows=tuple(rows), metric_names=metric_names)


def run_mixed_precision_experiment(config: ExperimentConfig) -> ExperimentResult:
    from .runner import Experiment

    cfg = ExperimentConfig(
        name=config.name,
        technique=ExperimentTechnique.MIXED_PRECISION,
        seed=config.seed,
        model=config.model,
        dataset=config.dataset,
        training=config.training,
        mixed_precision=config.mixed_precision or MixedPrecisionExperimentConfig(
            default_bits=4,
            parameter_bits={"0.weight": 8, "1.weight": 4},
        ),
        repeats=config.repeats,
    )
    return Experiment(cfg).run()


def run_rotation_experiment(config: ExperimentConfig) -> ExperimentResult:
    compression_cfg = config.compression or CompressionExperimentConfig(
        bits=4, method="codebook", rotation="combined"
    )
    seed_everything(config.seed)
    inputs, targets = build_dataset_from_config(config.dataset, config.seed)
    model = build_model_from_config(config.model)
    training_seconds, training_loss = _train_model(model, inputs, targets, config.training)

    # Aggregate rotation experiment across all parameter tensors (reuse compare_quantization_rotation).
    baseline_rmse_sum = 0.0
    rotated_rmse_sum = 0.0
    count = 0
    labels: list[str] = []
    for _, parameter in model.named_parameters():
        exp = compare_quantization_rotation(
            parameter.data,
            compression_cfg.bits,
            method=compression_cfg.method,
            packed=compression_cfg.packed,
            rotation=compression_cfg.rotation or "combined",
        )
        baseline_rmse_sum += exp.baseline_error.rmse
        rotated_rmse_sum += exp.rotated_error.rmse
        count += 1
        labels.append(exp.result_label)

    avg_baseline = baseline_rmse_sum / count if count else 0.0
    avg_rotated = rotated_rmse_sum / count if count else 0.0
    if avg_rotated < avg_baseline:
        summary_label = "improved"
    elif avg_rotated > avg_baseline:
        summary_label = "worse"
    else:
        summary_label = "unchanged"

    comp_config = CompressionConfig(
        bits=compression_cfg.bits,
        method=compression_cfg.method,
        packed=compression_cfg.packed,
        rotation=compression_cfg.rotation,
    )
    compress_start = time.perf_counter()
    report = compare_model_compression(model, inputs, targets, config=comp_config)
    compression_seconds = time.perf_counter() - compress_start

    return ExperimentResult(
        schema_version=EXPERIMENT_SCHEMA_VERSION,
        experiment_id=compute_experiment_id(config),
        name=config.name,
        seed=config.seed,
        configuration=config.to_dict(),
        environment=ExperimentEnvironment.capture(),
        metrics=ExperimentMetrics(
            training_final_loss=training_loss,
            fp_final_loss=report.original_loss,
            compressed_final_loss=report.compressed_loss,
            loss_delta=report.loss_delta,
            parameter_rmse=report.aggregate_parameter_rmse,
            output_rmse=report.output_rmse,
            model_bytes=report.total_original_bytes,
            compressed_bytes=report.total_compressed_bytes,
            compression_ratio=report.compression_ratio,
        ),
        timings=ExperimentTimings(
            training_seconds=training_seconds,
            compression_seconds=compression_seconds,
        ),
        memory=ExperimentMemory(
            model_bytes=report.total_original_bytes,
            compressed_bytes=report.total_compressed_bytes,
        ),
        observations=(
            ExperimentObservation(
                category="rotation",
                message=f"Rotation experiment associated with {summary_label} parameter RMSE on this workload.",
                caution="Per-parameter rotation labels: " + ", ".join(sorted(set(labels))),
            ),
        ),
        technique_label=_compression_label(comp_config),
        training_technique=config.training.technique.value,
        storage_technique="rotation",
        extra={
            "avg_baseline_parameter_rmse": avg_baseline,
            "avg_rotated_parameter_rmse": avg_rotated,
            "rotation_summary": summary_label,
        },
    )


def run_qat_comparison_experiment(config: ExperimentConfig) -> ExperimentResult:
    compression_cfg = config.compression or CompressionExperimentConfig(bits=4, method="uniform")
    seed_everything(config.seed)
    inputs, targets = build_dataset_from_config(config.dataset, config.seed)
    template = build_model_from_config(config.model)
    snapshot = snapshot_model_state(template)

    qat_result = run_qat_experiment(
        template,
        lambda m: _make_optimizer(m, config.training),
        _loss_fn(),
        inputs,
        targets,
        epochs=config.training.epochs,
        bits=compression_cfg.bits,
        ste_mode=config.training.qat_ste_mode,
    )

    observations = [
        ExperimentObservation(
            category="qat",
            message=f"QAT comparison observed: {qat_result.result_label}.",
            caution="Training technique (QAT) is separate from post-training storage compression.",
        ),
    ]
    return ExperimentResult(
        schema_version=EXPERIMENT_SCHEMA_VERSION,
        experiment_id=compute_experiment_id(config),
        name=config.name,
        seed=config.seed,
        configuration=config.to_dict(),
        environment=ExperimentEnvironment.capture(),
        metrics=ExperimentMetrics(
            fp_final_loss=qat_result.normal_fp_loss,
            compressed_final_loss=qat_result.normal_quantized_loss,
            loss_delta=qat_result.normal_quantization_damage,
            parameter_rmse=qat_result.normal_parameter_rmse,
            output_rmse=qat_result.normal_output_rmse,
        ),
        timings=ExperimentTimings(
            training_seconds=qat_result.normal_training_seconds,
        ),
        memory=ExperimentMemory(),
        observations=tuple(observations),
        technique_label="QAT vs post-training",
        training_technique="qat_comparison",
        storage_technique="post_training_compression",
        extra={
            "qat_fp_loss": qat_result.qat_fp_loss,
            "qat_quantized_loss": qat_result.qat_quantized_loss,
            "qat_quantization_damage": qat_result.qat_quantization_damage,
            "qat_output_rmse": qat_result.qat_output_rmse,
            "qat_parameter_rmse": qat_result.qat_parameter_rmse,
            "qat_training_seconds": qat_result.qat_training_seconds,
            "result_label": qat_result.result_label,
        },
    )


def run_performance_experiment(
    config: ExperimentConfig,
    *,
    mode: Literal["parallel", "vectorized"] = "parallel",
) -> ExperimentResult:
    compression_cfg = config.compression or CompressionExperimentConfig(
        bits=4, method="uniform", packed=True
    )
    seed_everything(config.seed)
    model = build_model_from_config(config.model)
    inputs, targets = build_dataset_from_config(config.dataset, config.seed)
    _train_model(model, inputs, targets, config.training)

    comp_config = CompressionConfig(
        bits=compression_cfg.bits,
        method=compression_cfg.method,
        packed=compression_cfg.packed,
        rotation=compression_cfg.rotation,
    )

    extra: dict[str, Any] = {}
    compression_seconds: float | None = None
    if mode == "parallel":
        bench = benchmark_parallel_comparison(
            "model_compression",
            lambda: compress_model(model, config=comp_config),
            lambda: compress_model(model, config=comp_config, parallel=True),
        )
        extra = {
            "serial_seconds": bench.serial_seconds,
            "parallel_seconds": bench.parallel_seconds,
            "speedup": bench.speedup,
        }
        compression_seconds = bench.serial_seconds
        speedup_text = f"{bench.speedup:.3f}x" if bench.speedup is not None else "n/a"
        message = (
            f"Parallel compression speedup measured at {speedup_text} "
            f"(serial {bench.serial_seconds:.6f}s vs parallel {bench.parallel_seconds:.6f}s)."
        )
    else:
        from ..kernels import benchmark_vectorization, pack_codes_reference, pack_codes_vectorized

        size = 256
        codes = np.arange(size, dtype=np.uint8) % 16
        vec_result = benchmark_vectorization(
            "pack_codes",
            size,
            lambda: pack_codes_reference(codes, bits=4),
            lambda: pack_codes_vectorized(codes, bits=4),
        )
        extra = {
            "vectorization_speedup": vec_result.speedup,
            "reference_seconds": vec_result.reference_seconds,
            "vectorized_seconds": vec_result.vectorized_seconds,
        }
        message = (
            f"Vectorized pack_codes speedup measured at {vec_result.speedup:.3f}x "
            f"on size {size}."
        )
        compression_seconds = vec_result.reference_seconds

    report = compare_model_compression(model, inputs, targets, config=comp_config)
    return ExperimentResult(
        schema_version=EXPERIMENT_SCHEMA_VERSION,
        experiment_id=compute_experiment_id(config),
        name=config.name,
        seed=config.seed,
        configuration=config.to_dict(),
        environment=ExperimentEnvironment.capture(),
        metrics=ExperimentMetrics(
            fp_final_loss=report.original_loss,
            compressed_final_loss=report.compressed_loss,
            loss_delta=report.loss_delta,
            parameter_rmse=report.aggregate_parameter_rmse,
            output_rmse=report.output_rmse,
            model_bytes=report.total_original_bytes,
            compressed_bytes=report.total_compressed_bytes,
            compression_ratio=report.compression_ratio,
        ),
        timings=ExperimentTimings(compression_seconds=compression_seconds),
        memory=ExperimentMemory(
            model_bytes=report.total_original_bytes,
            compressed_bytes=report.total_compressed_bytes,
        ),
        observations=(
            ExperimentObservation(
                category="performance",
                message=message,
                caution="Benchmark setup time is excluded from training time.",
            ),
        ),
        technique_label=f"Performance {mode}",
        training_technique=config.training.technique.value,
        storage_technique="post_training_compression",
        extra=extra,
    )


def run_graph_optimizer_experiment(config: ExperimentConfig) -> ExperimentResult:
    seed_everything(config.seed)
    model = build_model_from_config(config.model)
    inputs, targets = build_dataset_from_config(config.dataset, config.seed)
    training_seconds, _ = _train_model(model, inputs, targets, config.training)

    profile_start = time.perf_counter()
    profile = profile_graph_optimizer_comparison(
        model,
        inputs,
        targets,
        _loss_fn(),
        warmup=1,
        repeats=2,
    )
    profile_seconds = time.perf_counter() - profile_start

    original = profile.stages.get("original_backward")
    optimized = profile.stages.get("optimized_backward")
    message = "Graph optimizer backward timing measured."
    if original is not None and optimized is not None:
        if optimized < original:
            message = (
                f"Optimized backward measured at {optimized:.6f}s vs original {original:.6f}s."
            )
        else:
            message = (
                f"Optimized backward measured at {optimized:.6f}s vs original {original:.6f}s "
                "(no measured speedup on this workload)."
            )

    return ExperimentResult(
        schema_version=EXPERIMENT_SCHEMA_VERSION,
        experiment_id=compute_experiment_id(config),
        name=config.name,
        seed=config.seed,
        configuration=config.to_dict(),
        environment=ExperimentEnvironment.capture(),
        metrics=ExperimentMetrics(),
        timings=ExperimentTimings(
            training_seconds=training_seconds,
            evaluation_seconds=profile_seconds,
        ),
        memory=ExperimentMemory(),
        observations=(
            ExperimentObservation(
                category="graph_optimizer",
                message=message,
            ),
        ),
        technique_label="Graph optimizer",
        training_technique=config.training.technique.value,
        storage_technique="fp32",
        extra={
            "original_backward_seconds": original,
            "optimized_backward_seconds": optimized,
            "graph_stats": dict(profile.graph_stats),
        },
    )


def run_checkpoint_resume_experiment(config: ExperimentConfig) -> ExperimentResult:
    seed_everything(config.seed)
    inputs, targets = build_dataset_from_config(config.dataset, config.seed)
    loss_fn = _loss_fn()

    template = build_model_from_config(config.model)
    continuous_model = clone_model_with_state(template)
    continuous_optimizer = _make_optimizer(continuous_model, config.training)
    continuous_trainer = Trainer(continuous_model, continuous_optimizer, loss_fn)
    half_epochs = max(1, config.training.epochs // 2)
    continuous_trainer.fit(inputs, targets, epochs=config.training.epochs)

    resumed_model = clone_model_with_state(template)
    resumed_optimizer = _make_optimizer(resumed_model, config.training)
    resumed_trainer = Trainer(resumed_model, resumed_optimizer, loss_fn)
    resumed_trainer.fit(inputs, targets, epochs=half_epochs)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.mtc"
        resumed_trainer.save_checkpoint(path)
        from ..persistence import load_training_checkpoint

        restored_model, restored_optimizer, resume_state = load_training_checkpoint(path).restore()
        restored_trainer = Trainer(restored_model, restored_optimizer, loss_fn)
        restored_trainer.apply_resume_state(resume_state)
        restored_trainer.fit(inputs, targets, epochs=config.training.epochs - half_epochs, resume=True)

    first_param_name = next(iter(snapshot_model_state(continuous_model)))
    weight_delta = float(
        np.max(
            np.abs(
                continuous_model.parameter(first_param_name).data
                - restored_model.parameter(first_param_name).data
            )
        )
    )
    loss_continuous = float(loss_fn(continuous_model(inputs), targets).data)
    loss_resumed = float(loss_fn(restored_model(inputs), targets).data)

    return ExperimentResult(
        schema_version=EXPERIMENT_SCHEMA_VERSION,
        experiment_id=compute_experiment_id(config),
        name=config.name,
        seed=config.seed,
        configuration=config.to_dict(),
        environment=ExperimentEnvironment.capture(),
        metrics=ExperimentMetrics(
            fp_final_loss=loss_continuous,
            compressed_final_loss=loss_resumed,
            loss_delta=loss_resumed - loss_continuous,
        ),
        timings=ExperimentTimings(),
        memory=ExperimentMemory(),
        observations=(
            ExperimentObservation(
                category="checkpoint",
                message=(
                    f"Checkpoint resume loss delta measured at {loss_resumed - loss_continuous:.6g}."
                ),
                caution="Weight max delta observed: " + f"{weight_delta:.6g}",
            ),
        ),
        technique_label="Checkpoint resume",
        training_technique=config.training.technique.value,
        storage_technique="fp32",
        extra={
            "continuous_loss": loss_continuous,
            "resumed_loss": loss_resumed,
            "weight_max_delta": weight_delta,
            "continuous_steps": continuous_trainer.completed_steps,
            "resumed_steps": restored_trainer.completed_steps,
        },
    )

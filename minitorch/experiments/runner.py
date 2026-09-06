from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from ..compression import (
    CompressionConfig,
    compress_model,
)
from ..losses import MSELoss
from ..mixed_precision import (
    MixedPrecisionConfig,
    compress_model_mixed_precision,
)
from ..nn import Linear, Module, ReLU, Sequential
from ..qat import QATConfig
from ..tensor import Tensor
from ..training import Adam, SGD, Trainer
from .config import (
    DatasetConfig,
    ExperimentConfig,
    ExperimentTechnique,
    LayerSpec,
    ModelConfig,
    TrainingConfig,
    TrainingTechnique,
)
from .identity import compute_experiment_id, EXPERIMENT_SCHEMA_VERSION
from .randomness import seed_everything
from .result import (
    ExperimentEnvironment,
    ExperimentMemory,
    ExperimentMetrics,
    ExperimentObservation,
    ExperimentResult,
    ExperimentTimings,
)
from .snapshot import clone_model_with_state, snapshot_model_state


def build_model_from_config(config: ModelConfig) -> Module:
    if not config.layers:
        config = ModelConfig.baseline_regression()
    modules: list[Module] = []
    for layer in config.layers:
        if layer.type == "Linear":
            if layer.in_features is None or layer.out_features is None:
                raise ValueError("Linear layer requires in_features and out_features.")
            modules.append(Linear(layer.in_features, layer.out_features))
        elif layer.type == "ReLU":
            modules.append(ReLU())
        else:
            raise ValueError(f"Unsupported layer type: {layer.type}")
    return Sequential(*modules)


def build_baseline_model() -> Module:
    return build_model_from_config(ModelConfig.baseline_regression())


def build_dataset_from_config(config: DatasetConfig, seed: int) -> tuple[Tensor, Tensor]:
    if config.kind != "deterministic_regression":
        raise ValueError(f"Unsupported dataset kind: {config.kind}")
    rng = np.random.default_rng(seed)
    if config.input_dim == 1:
        x = rng.uniform(0.0, 1.0, size=(config.n_samples, 1))
    else:
        x = rng.uniform(0.0, 1.0, size=(config.n_samples, config.input_dim))
    y = config.slope * x[:, :1] + config.intercept
    return Tensor(x), Tensor(y)


def build_baseline_dataset(seed: int = 42) -> tuple[Tensor, Tensor]:
    return build_dataset_from_config(DatasetConfig(), seed)


def _make_optimizer(model: Module, training: TrainingConfig) -> Any:
    if training.optimizer.lower() == "adam":
        return Adam(model.parameters(), lr=training.learning_rate)
    if training.optimizer.lower() == "sgd":
        return SGD(model.parameters(), lr=training.learning_rate)
    raise ValueError(f"Unsupported optimizer: {training.optimizer}")


def _loss_fn() -> Callable[[Tensor, Tensor], Tensor]:
    criterion = MSELoss()
    return criterion.forward


def _aggregate_runs(values: list[float | None]) -> tuple[float | None, float | None]:
    numeric = [float(v) for v in values if v is not None]
    if not numeric:
        return None, None
    if len(numeric) == 1:
        return numeric[0], 0.0
    return float(np.mean(numeric)), float(np.std(numeric))


def _result_from_compression_report(
    *,
    config: ExperimentConfig,
    report: Any,
    training_seconds: float,
    compression_seconds: float,
    materialization_seconds: float,
    evaluation_seconds: float,
    training_final_loss: float | None,
    technique_label: str,
    training_technique: str,
    storage_technique: str,
    observations: list[ExperimentObservation],
    extra: dict[str, Any] | None = None,
) -> ExperimentResult:
    model_bytes = report.total_original_bytes
    compressed_bytes = report.total_compressed_bytes
    ratio = report.compression_ratio if compressed_bytes > 0 else None

    packed_code = None
    codebook = None
    logical_code = None
    if hasattr(report, "parameter_records"):
        packed_code = sum(r.packed_code_bytes for r in report.parameter_records)
        codebook = sum(r.codebook_bytes for r in report.parameter_records)
        logical_code = sum(
            r.compressed_bytes - r.codebook_bytes for r in report.parameter_records
        )

    merged_extra = dict(extra or {})
    if logical_code is not None:
        merged_extra.setdefault("logical_code_bytes", logical_code)

    return ExperimentResult(
        schema_version=EXPERIMENT_SCHEMA_VERSION,
        experiment_id=compute_experiment_id(config),
        name=config.name,
        seed=config.seed,
        configuration=config.to_dict(),
        environment=ExperimentEnvironment.capture(),
        metrics=ExperimentMetrics(
            training_final_loss=training_final_loss,
            fp_final_loss=report.original_loss,
            compressed_final_loss=report.compressed_loss,
            loss_delta=report.loss_delta,
            parameter_rmse=report.aggregate_parameter_rmse,
            output_rmse=report.output_rmse,
            model_bytes=model_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=ratio,
        ),
        timings=ExperimentTimings(
            training_seconds=training_seconds,
            compression_seconds=compression_seconds,
            materialization_seconds=materialization_seconds,
            evaluation_seconds=evaluation_seconds,
        ),
        memory=ExperimentMemory(
            model_bytes=model_bytes,
            compressed_bytes=compressed_bytes,
            packed_code_bytes=packed_code,
            codebook_bytes=codebook,
            total_representation_bytes=compressed_bytes,
        ),
        observations=tuple(observations),
        repeats=config.repeats,
        technique_label=technique_label,
        training_technique=training_technique,
        storage_technique=storage_technique,
        extra=merged_extra,
    )


def _train_model(
    model: Module,
    inputs: Tensor,
    targets: Tensor,
    training: TrainingConfig,
) -> tuple[float, float]:
    loss_fn = _loss_fn()
    qat_config = None
    if training.technique == TrainingTechnique.QAT:
        qat_config = QATConfig(bits=training.qat_bits, ste_mode=training.qat_ste_mode)

    trainer = Trainer(model, _make_optimizer(model, training), loss_fn, qat_config=qat_config)
    start = time.perf_counter()
    history = trainer.fit(inputs, targets, epochs=training.epochs)
    training_seconds = time.perf_counter() - start
    final_loss = history.losses[-1] if history.losses else None
    return training_seconds, final_loss


class Experiment:
    """Run a configured MiniTorch experiment with reproducible metadata."""

    def __init__(self, config: ExperimentConfig) -> None:
        config.validate()
        self.config = config

    def run(self) -> ExperimentResult:
        technique = self.config.technique
        if technique == ExperimentTechnique.FP32_BASELINE:
            return self._run_fp32_baseline()
        if technique == ExperimentTechnique.POST_TRAINING_COMPRESSION:
            return self._run_post_training_compression()
        if technique == ExperimentTechnique.MIXED_PRECISION:
            return self._run_mixed_precision()
        if technique == ExperimentTechnique.QUANTIZATION_MATRIX:
            raise ValueError("Use run_quantization_matrix() for matrix experiments.")
        if technique == ExperimentTechnique.QAT_COMPARISON:
            return run_qat_comparison_experiment(self.config)
        if technique == ExperimentTechnique.ROTATION:
            return run_rotation_experiment(self.config)
        if technique == ExperimentTechnique.BIT_PACKING:
            return self._run_bit_packing()
        if technique == ExperimentTechnique.PERFORMANCE_PARALLEL:
            return run_performance_experiment(self.config, mode="parallel")
        if technique == ExperimentTechnique.PERFORMANCE_VECTORIZED:
            return run_performance_experiment(self.config, mode="vectorized")
        if technique == ExperimentTechnique.GRAPH_OPTIMIZER:
            return run_graph_optimizer_experiment(self.config)
        if technique == ExperimentTechnique.CHECKPOINT_RESUME:
            return run_checkpoint_resume_experiment(self.config)
        if technique == ExperimentTechnique.COMBINED:
            return self._run_combined()
        raise ValueError(f"Unsupported experiment technique: {technique}")

    def _prepare_trained_model(self) -> tuple[Module, Tensor, Tensor, float, float]:
        seed_everything(self.config.seed)
        inputs, targets = build_dataset_from_config(self.config.dataset, self.config.seed)
        template = build_model_from_config(self.config.model)
        snapshot = snapshot_model_state(template)
        training_seconds, final_loss = _train_model(
            template, inputs, targets, self.config.training
        )
        model = clone_model_with_state(template)
        restore_snapshot = snapshot_model_state(model)
        assert restore_snapshot.keys() == snapshot.keys()
        return model, inputs, targets, training_seconds, final_loss

    def _run_fp32_baseline(self) -> ExperimentResult:
        runs: list[ExperimentResult] = []
        for repeat_index in range(self.config.repeats):
            offset_config = ExperimentConfig.from_dict(self.config.to_dict())
            offset_config = ExperimentConfig(
                name=self.config.name,
                technique=self.config.technique,
                seed=self.config.seed + repeat_index,
                model=self.config.model,
                dataset=self.config.dataset,
                training=self.config.training,
                repeats=1,
            )
            seed_everything(offset_config.seed)
            inputs, targets = build_dataset_from_config(offset_config.dataset, offset_config.seed)
            model = build_model_from_config(offset_config.model)
            training_seconds, final_loss = _train_model(
                model, inputs, targets, offset_config.training
            )
            eval_start = time.perf_counter()
            fp_loss = float(_loss_fn()(model(inputs), targets).data)
            eval_seconds = time.perf_counter() - eval_start
            model_bytes = sum(p.data.nbytes for _, p in model.named_parameters())
            runs.append(
                ExperimentResult(
                    schema_version=EXPERIMENT_SCHEMA_VERSION,
                    experiment_id=compute_experiment_id(offset_config),
                    name=offset_config.name,
                    seed=offset_config.seed,
                    configuration=offset_config.to_dict(),
                    environment=ExperimentEnvironment.capture(),
                    metrics=ExperimentMetrics(
                        training_final_loss=final_loss,
                        fp_final_loss=fp_loss,
                        model_bytes=model_bytes,
                        compressed_bytes=model_bytes,
                        compression_ratio=1.0,
                        parameter_rmse=0.0,
                        output_rmse=0.0,
                        loss_delta=0.0,
                    ),
                    timings=ExperimentTimings(
                        training_seconds=training_seconds,
                        evaluation_seconds=eval_seconds,
                    ),
                    memory=ExperimentMemory(model_bytes=model_bytes, compressed_bytes=model_bytes),
                    observations=(
                        ExperimentObservation(
                            category="baseline",
                            message="FP32 baseline measured without compression.",
                        ),
                    ),
                    technique_label="FP32",
                    training_technique=offset_config.training.technique.value,
                    storage_technique="fp32",
                )
            )
        return _merge_repeated_runs(self.config, runs)

    def _run_post_training_compression(self) -> ExperimentResult:
        compression_cfg = self.config.compression
        if compression_cfg is None:
            raise ValueError("compression config required")

        runs: list[ExperimentResult] = []
        for repeat_index in range(self.config.repeats):
            offset_seed = self.config.seed + repeat_index
            seed_everything(offset_seed)
            inputs, targets = build_dataset_from_config(self.config.dataset, offset_seed)
            model = build_model_from_config(self.config.model)
            training_seconds, training_loss = _train_model(
                model, inputs, targets, self.config.training
            )

            comp_config = CompressionConfig(
                bits=compression_cfg.bits,
                method=compression_cfg.method,
                packed=compression_cfg.packed,
                rotation=compression_cfg.rotation,
            )
            compress_start = time.perf_counter()
            compressed = compress_model(model, config=comp_config)
            compression_seconds = time.perf_counter() - compress_start

            materialize_start = time.perf_counter()
            _ = compressed.materialize(model)
            materialization_seconds = time.perf_counter() - materialize_start

            eval_start = time.perf_counter()
            report = compressed.report(template=model, inputs=inputs, targets=targets)
            eval_seconds = time.perf_counter() - eval_start

            label = _compression_label(comp_config)
            runs.append(
                _result_from_compression_report(
                    config=self._repeat_config(repeat_index),
                    report=report,
                    training_seconds=training_seconds,
                    compression_seconds=compression_seconds,
                    materialization_seconds=materialization_seconds,
                    evaluation_seconds=eval_seconds,
                    training_final_loss=training_loss,
                    technique_label=label,
                    training_technique=self.config.training.technique.value,
                    storage_technique="post_training_compression",
                    observations=[
                        ExperimentObservation(
                            category="compression",
                            message=f"Post-training {label} compression measured.",
                        ),
                    ],
                )
            )
        return _merge_repeated_runs(self.config, runs)

    def _run_mixed_precision(self) -> ExperimentResult:
        mixed_cfg = self.config.mixed_precision
        if mixed_cfg is None:
            raise ValueError("mixed_precision config required")

        runs: list[ExperimentResult] = []
        for repeat_index in range(self.config.repeats):
            offset_seed = self.config.seed + repeat_index
            seed_everything(offset_seed)
            inputs, targets = build_dataset_from_config(self.config.dataset, offset_seed)
            model = build_model_from_config(self.config.model)
            training_seconds, training_loss = _train_model(
                model, inputs, targets, self.config.training
            )

            mp_config = MixedPrecisionConfig(
                default_bits=mixed_cfg.default_bits,
                parameter_bits=dict(mixed_cfg.parameter_bits),
                method=mixed_cfg.method,
                packed=mixed_cfg.packed,
                rotation=mixed_cfg.rotation,
            )
            compress_start = time.perf_counter()
            compressed = compress_model_mixed_precision(model, mp_config)
            compression_seconds = time.perf_counter() - compress_start

            materialize_start = time.perf_counter()
            _ = compressed.materialize(model)
            materialization_seconds = time.perf_counter() - materialize_start

            eval_start = time.perf_counter()
            report = compressed.report(template=model, inputs=inputs, targets=targets)
            eval_seconds = time.perf_counter() - eval_start

            runs.append(
                _result_from_compression_report(
                    config=self._repeat_config(repeat_index),
                    report=report,
                    training_seconds=training_seconds,
                    compression_seconds=compression_seconds,
                    materialization_seconds=materialization_seconds,
                    evaluation_seconds=eval_seconds,
                    training_final_loss=training_loss,
                    technique_label="Mixed precision",
                    training_technique=self.config.training.technique.value,
                    storage_technique="mixed_precision",
                    observations=[
                        ExperimentObservation(
                            category="mixed_precision",
                            message="Mixed-precision post-training compression measured.",
                            caution="Per-parameter policy is explicit; not globally optimal.",
                        ),
                    ],
                    extra={"precision_map": dict(report.precision_map)},
                )
            )
        return _merge_repeated_runs(self.config, runs)

    def _run_bit_packing(self) -> ExperimentResult:
        compression_cfg = self.config.compression or CompressionExperimentConfig(
            bits=4, method="uniform", packed=True
        )
        post_config = ExperimentConfig(
            name=self.config.name,
            technique=ExperimentTechnique.POST_TRAINING_COMPRESSION,
            seed=self.config.seed,
            model=self.config.model,
            dataset=self.config.dataset,
            training=self.config.training,
            compression=compression_cfg,
            repeats=self.config.repeats,
        )
        result = Experiment(post_config).run()
        logical = result.extra.get("logical_code_bytes")
        packed = result.memory.packed_code_bytes
        observations = list(result.observations)
        if logical is not None and packed is not None:
            observations.append(
                ExperimentObservation(
                    category="bit_packing",
                    message=(
                        f"Logical code storage {logical} bytes vs packed storage {packed} bytes observed."
                    ),
                    caution="Packing changes storage layout only; reconstruction should match unpacked codes.",
                )
            )
        result.observations = tuple(observations)
        result.technique_label = "Bit packing 4-bit"
        return result

    def _run_combined(self) -> ExperimentResult:
        """Training + optional QAT + post-training mixed precision + packing when supported."""
        mixed_cfg = self.config.mixed_precision
        if mixed_cfg is None:
            raise ValueError("combined experiment requires mixed_precision config")

        if self.config.training.technique == TrainingTechnique.QAT:
            raise ValueError("mixed-precision QAT is not implemented.")

        training = TrainingConfig(
            optimizer=self.config.training.optimizer,
            learning_rate=self.config.training.learning_rate,
            epochs=self.config.training.epochs,
            loss=self.config.training.loss,
            technique=self.config.training.technique,
            qat_bits=self.config.training.qat_bits,
            qat_ste_mode=self.config.training.qat_ste_mode,
        )
        combined_config = ExperimentConfig(
            name=self.config.name,
            technique=ExperimentTechnique.MIXED_PRECISION,
            seed=self.config.seed,
            model=self.config.model,
            dataset=self.config.dataset,
            training=training,
            mixed_precision=mixed_cfg,
            repeats=self.config.repeats,
        )
        result = Experiment(combined_config).run()
        result.technique_label = "Combined (train + mixed precision + packing)"
        result.storage_technique = "combined"
        extra_obs = list(result.observations)
        extra_obs.append(
            ExperimentObservation(
                category="combined",
                message="Combined pipeline: train then post-training mixed precision.",
            )
        )
        result.observations = tuple(extra_obs)
        return result

    def _repeat_config(self, repeat_index: int) -> ExperimentConfig:
        return ExperimentConfig(
            name=self.config.name,
            technique=self.config.technique,
            seed=self.config.seed + repeat_index,
            model=self.config.model,
            dataset=self.config.dataset,
            training=self.config.training,
            compression=self.config.compression,
            mixed_precision=self.config.mixed_precision,
            repeats=1,
        )


def _compression_label(config: CompressionConfig) -> str:
    parts = [f"Uniform {config.bits}-bit"]
    if config.method == "codebook":
        parts[0] = f"Codebook {config.bits}-bit"
    if config.rotation:
        parts.append(f"rotation={config.rotation}")
    if config.packed:
        parts.append("packed")
    return " ".join(parts)


def _merge_repeated_runs(config: ExperimentConfig, runs: list[ExperimentResult]) -> ExperimentResult:
    if not runs:
        raise ValueError("No experiment runs to merge.")
    base = runs[0]
    if len(runs) == 1:
        base.repeats = config.repeats
        base.experiment_id = compute_experiment_id(config)
        base.configuration = config.to_dict()
        return base

    metrics = ExperimentMetrics(
        training_final_loss=runs[0].metrics.training_final_loss,
        fp_final_loss=runs[0].metrics.fp_final_loss,
        compressed_final_loss=runs[0].metrics.compressed_final_loss,
        loss_delta=runs[0].metrics.loss_delta,
        parameter_rmse=runs[0].metrics.parameter_rmse,
        output_rmse=runs[0].metrics.output_rmse,
        model_bytes=runs[0].metrics.model_bytes,
        compressed_bytes=runs[0].metrics.compressed_bytes,
        compression_ratio=runs[0].metrics.compression_ratio,
        training_final_loss_mean=_aggregate_runs([r.metrics.training_final_loss for r in runs])[0],
        training_final_loss_std=_aggregate_runs([r.metrics.training_final_loss for r in runs])[1],
        fp_final_loss_mean=_aggregate_runs([r.metrics.fp_final_loss for r in runs])[0],
        fp_final_loss_std=_aggregate_runs([r.metrics.fp_final_loss for r in runs])[1],
        compressed_final_loss_mean=_aggregate_runs(
            [r.metrics.compressed_final_loss for r in runs]
        )[0],
        compressed_final_loss_std=_aggregate_runs(
            [r.metrics.compressed_final_loss for r in runs]
        )[1],
        loss_delta_mean=_aggregate_runs([r.metrics.loss_delta for r in runs])[0],
        loss_delta_std=_aggregate_runs([r.metrics.loss_delta for r in runs])[1],
        parameter_rmse_mean=_aggregate_runs([r.metrics.parameter_rmse for r in runs])[0],
        parameter_rmse_std=_aggregate_runs([r.metrics.parameter_rmse for r in runs])[1],
        output_rmse_mean=_aggregate_runs([r.metrics.output_rmse for r in runs])[0],
        output_rmse_std=_aggregate_runs([r.metrics.output_rmse for r in runs])[1],
    )
    base.metrics = metrics
    base.repeats = config.repeats
    base.experiment_id = compute_experiment_id(config)
    base.configuration = config.to_dict()
    base.seed = config.seed
    return base


def list_experiments() -> list[str]:
    from .registry import CANONICAL_EXPERIMENTS

    return sorted(CANONICAL_EXPERIMENTS.keys())

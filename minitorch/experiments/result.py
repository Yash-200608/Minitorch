from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..persistence.format import FRAMEWORK_VERSION
from .identity import EXPERIMENT_SCHEMA_VERSION


@dataclass(frozen=True)
class ExperimentEnvironment:
    python_version: str
    numpy_version: str
    platform: str
    framework_version: str
    experiment_schema_version: str

    @staticmethod
    def capture() -> ExperimentEnvironment:
        return ExperimentEnvironment(
            python_version=sys.version.split()[0],
            numpy_version=np.__version__,
            platform=platform.platform(),
            framework_version=FRAMEWORK_VERSION,
            experiment_schema_version=EXPERIMENT_SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "platform": self.platform,
            "framework_version": self.framework_version,
            "experiment_schema_version": self.experiment_schema_version,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ExperimentEnvironment:
        return ExperimentEnvironment(
            python_version=str(data.get("python_version", "")),
            numpy_version=str(data.get("numpy_version", "")),
            platform=str(data.get("platform", "")),
            framework_version=str(data.get("framework_version", FRAMEWORK_VERSION)),
            experiment_schema_version=str(
                data.get("experiment_schema_version", EXPERIMENT_SCHEMA_VERSION)
            ),
        )


@dataclass(frozen=True)
class ExperimentMetrics:
    training_final_loss: float | None = None
    fp_final_loss: float | None = None
    compressed_final_loss: float | None = None
    loss_delta: float | None = None
    parameter_rmse: float | None = None
    output_rmse: float | None = None
    model_bytes: int | None = None
    compressed_bytes: int | None = None
    compression_ratio: float | None = None
  # repeated-run aggregates
    training_final_loss_mean: float | None = None
    training_final_loss_std: float | None = None
    fp_final_loss_mean: float | None = None
    fp_final_loss_std: float | None = None
    compressed_final_loss_mean: float | None = None
    compressed_final_loss_std: float | None = None
    loss_delta_mean: float | None = None
    loss_delta_std: float | None = None
    parameter_rmse_mean: float | None = None
    parameter_rmse_std: float | None = None
    output_rmse_mean: float | None = None
    output_rmse_std: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_final_loss": self.training_final_loss,
            "fp_final_loss": self.fp_final_loss,
            "compressed_final_loss": self.compressed_final_loss,
            "loss_delta": self.loss_delta,
            "parameter_rmse": self.parameter_rmse,
            "output_rmse": self.output_rmse,
            "model_bytes": self.model_bytes,
            "compressed_bytes": self.compressed_bytes,
            "compression_ratio": self.compression_ratio,
            "training_final_loss_mean": self.training_final_loss_mean,
            "training_final_loss_std": self.training_final_loss_std,
            "fp_final_loss_mean": self.fp_final_loss_mean,
            "fp_final_loss_std": self.fp_final_loss_std,
            "compressed_final_loss_mean": self.compressed_final_loss_mean,
            "compressed_final_loss_std": self.compressed_final_loss_std,
            "loss_delta_mean": self.loss_delta_mean,
            "loss_delta_std": self.loss_delta_std,
            "parameter_rmse_mean": self.parameter_rmse_mean,
            "parameter_rmse_std": self.parameter_rmse_std,
            "output_rmse_mean": self.output_rmse_mean,
            "output_rmse_std": self.output_rmse_std,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ExperimentMetrics:
        fields = ExperimentMetrics.__dataclass_fields__
        return ExperimentMetrics(
            **{key: data.get(key) for key in fields}
        )


@dataclass(frozen=True)
class ExperimentTimings:
    training_seconds: float | None = None
    compression_seconds: float | None = None
    materialization_seconds: float | None = None
    evaluation_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_seconds": self.training_seconds,
            "compression_seconds": self.compression_seconds,
            "materialization_seconds": self.materialization_seconds,
            "evaluation_seconds": self.evaluation_seconds,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ExperimentTimings:
        return ExperimentTimings(
            training_seconds=data.get("training_seconds"),
            compression_seconds=data.get("compression_seconds"),
            materialization_seconds=data.get("materialization_seconds"),
            evaluation_seconds=data.get("evaluation_seconds"),
        )


@dataclass(frozen=True)
class ExperimentMemory:
    model_bytes: int | None = None
    compressed_bytes: int | None = None
    packed_code_bytes: int | None = None
    codebook_bytes: int | None = None
    rotation_metadata_bytes: int | None = None
    total_representation_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_bytes": self.model_bytes,
            "compressed_bytes": self.compressed_bytes,
            "packed_code_bytes": self.packed_code_bytes,
            "codebook_bytes": self.codebook_bytes,
            "rotation_metadata_bytes": self.rotation_metadata_bytes,
            "total_representation_bytes": self.total_representation_bytes,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ExperimentMemory:
        return ExperimentMemory(
            model_bytes=data.get("model_bytes"),
            compressed_bytes=data.get("compressed_bytes"),
            packed_code_bytes=data.get("packed_code_bytes"),
            codebook_bytes=data.get("codebook_bytes"),
            rotation_metadata_bytes=data.get("rotation_metadata_bytes"),
            total_representation_bytes=data.get("total_representation_bytes"),
        )


@dataclass(frozen=True)
class ExperimentObservation:
    category: str
    message: str
    caution: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "message": self.message,
            "caution": self.caution,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ExperimentObservation:
        return ExperimentObservation(
            category=str(data.get("category", "")),
            message=str(data.get("message", "")),
            caution=data.get("caution"),
        )


@dataclass
class ExperimentResult:
    schema_version: str
    experiment_id: str
    name: str
    seed: int
    configuration: dict[str, Any]
    environment: ExperimentEnvironment
    metrics: ExperimentMetrics
    timings: ExperimentTimings
    memory: ExperimentMemory
    observations: tuple[ExperimentObservation, ...] = field(default_factory=tuple)
    artifacts: dict[str, str] = field(default_factory=dict)
    repeats: int = 1
    technique_label: str = ""
    training_technique: str = "normal"
    storage_technique: str = "fp32"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "seed": self.seed,
            "configuration": self.configuration,
            "environment": self.environment.to_dict(),
            "metrics": self.metrics.to_dict(),
            "timings": self.timings.to_dict(),
            "memory": self.memory.to_dict(),
            "observations": [obs.to_dict() for obs in self.observations],
            "artifacts": dict(self.artifacts),
            "repeats": self.repeats,
            "technique_label": self.technique_label,
            "training_technique": self.training_technique,
            "storage_technique": self.storage_technique,
            "extra": dict(self.extra),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ExperimentResult:
        return ExperimentResult(
            schema_version=str(data.get("schema_version", "")),
            experiment_id=str(data.get("experiment_id", "")),
            name=str(data.get("name", "")),
            seed=int(data.get("seed", 0)),
            configuration=dict(data.get("configuration", {})),
            environment=ExperimentEnvironment.from_dict(data.get("environment", {})),
            metrics=ExperimentMetrics.from_dict(data.get("metrics", {})),
            timings=ExperimentTimings.from_dict(data.get("timings", {})),
            memory=ExperimentMemory.from_dict(data.get("memory", {})),
            observations=tuple(
                ExperimentObservation.from_dict(item)
                for item in data.get("observations", [])
            ),
            artifacts=dict(data.get("artifacts", {})),
            repeats=int(data.get("repeats", 1)),
            technique_label=str(data.get("technique_label", "")),
            training_technique=str(data.get("training_technique", "normal")),
            storage_technique=str(data.get("storage_technique", "fp32")),
            extra=dict(data.get("extra", {})),
        )

    def format_report(self) -> str:
        lines = [
            "============== MiniTorch Experiment Lab ==============",
            "",
            "EXPERIMENT",
            f"  Name: {self.name}",
            f"  Technique: {self.technique_label}",
            f"  Training: {self.training_technique}",
            f"  Storage: {self.storage_technique}",
            f"  Seed: {self.seed}",
            "",
            "METRICS",
            f"  FP loss: {self.metrics.fp_final_loss}",
            f"  Compressed loss: {self.metrics.compressed_final_loss}",
            f"  Loss delta: {self.metrics.loss_delta}",
            f"  Parameter RMSE: {self.metrics.parameter_rmse}",
            f"  Output RMSE: {self.metrics.output_rmse}",
            f"  Model bytes: {self.metrics.model_bytes}",
            f"  Compressed bytes: {self.metrics.compressed_bytes}",
            f"  Compression ratio: {self.metrics.compression_ratio}",
            "",
            "TIMINGS (seconds)",
            f"  Training: {self.timings.training_seconds}",
            f"  Compression: {self.timings.compression_seconds}",
            f"  Materialization: {self.timings.materialization_seconds}",
            f"  Evaluation: {self.timings.evaluation_seconds}",
            "",
            "REPRODUCIBILITY",
            f"  Experiment ID: {self.experiment_id}",
            f"  Schema: {self.schema_version}",
            f"  Framework: {self.environment.framework_version}",
            "",
        ]
        if self.observations:
            lines.append("OBSERVATIONS")
            for obs in self.observations:
                lines.append(f"  [{obs.category}] {obs.message}")
                if obs.caution:
                    lines.append(f"    Caution: {obs.caution}")
            lines.append("")
        lines.append("======================================================")
        return "\n".join(lines)


@dataclass(frozen=True)
class ExperimentComparisonResult:
    rows: tuple[ExperimentResult, ...]
    metric_names: tuple[str, ...]

    def format_table(self) -> str:
        from .comparison import format_experiment_comparison_table

        return format_experiment_comparison_table(self.rows)


def validate_experiment_result(result: ExperimentResult) -> None:
    if result.schema_version != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported experiment schema: {result.schema_version}")
    if not result.experiment_id or len(result.experiment_id) != 16:
        raise ValueError("Invalid experiment_id.")
    if result.metrics.compressed_bytes is not None and result.metrics.compressed_bytes < 0:
        raise ValueError("compressed_bytes must be >= 0.")
    if result.metrics.model_bytes is not None and result.metrics.model_bytes < 0:
        raise ValueError("model_bytes must be >= 0.")
    if result.metrics.parameter_rmse is not None and result.metrics.parameter_rmse < 0:
        raise ValueError("parameter_rmse must be >= 0.")
    if result.metrics.output_rmse is not None and result.metrics.output_rmse < 0:
        raise ValueError("output_rmse must be >= 0.")
    for key, value in result.timings.to_dict().items():
        if value is not None and value < 0:
            raise ValueError(f"{key} must be >= 0.")

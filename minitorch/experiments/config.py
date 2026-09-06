from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

RotationOption = Literal["combined", "hadamard"] | None


class ExperimentTechnique(str, Enum):
    FP32_BASELINE = "fp32_baseline"
    POST_TRAINING_COMPRESSION = "post_training_compression"
    MIXED_PRECISION = "mixed_precision"
    QAT_COMPARISON = "qat_comparison"
    ROTATION = "rotation"
    BIT_PACKING = "bit_packing"
    PERFORMANCE_PARALLEL = "performance_parallel"
    PERFORMANCE_VECTORIZED = "performance_vectorized"
    GRAPH_OPTIMIZER = "graph_optimizer"
    CHECKPOINT_RESUME = "checkpoint_resume"
    COMBINED = "combined"
    QUANTIZATION_MATRIX = "quantization_matrix"


class TrainingTechnique(str, Enum):
    NORMAL = "normal"
    QAT = "qat"


@dataclass(frozen=True)
class LayerSpec:
    type: str
    in_features: int | None = None
    out_features: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.type}
        if self.in_features is not None:
            data["in_features"] = self.in_features
        if self.out_features is not None:
            data["out_features"] = self.out_features
        return data

    @staticmethod
    def from_dict(data: dict[str, Any]) -> LayerSpec:
        return LayerSpec(
            type=str(data["type"]),
            in_features=data.get("in_features"),
            out_features=data.get("out_features"),
        )


@dataclass(frozen=True)
class ModelConfig:
    """Serializable model architecture specification."""

    layers: tuple[LayerSpec, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"layers": [layer.to_dict() for layer in self.layers]}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ModelConfig:
        layers = tuple(LayerSpec.from_dict(item) for item in data.get("layers", []))
        return ModelConfig(layers=layers)

    @staticmethod
    def baseline_regression() -> ModelConfig:
        return ModelConfig(
            layers=(
                LayerSpec(type="Linear", in_features=1, out_features=8),
                LayerSpec(type="ReLU"),
                LayerSpec(type="Linear", in_features=8, out_features=1),
            )
        )


@dataclass(frozen=True)
class DatasetConfig:
    kind: str = "deterministic_regression"
    n_samples: int = 32
    input_dim: int = 1
    slope: float = 3.0
    intercept: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_samples": self.n_samples,
            "input_dim": self.input_dim,
            "slope": self.slope,
            "intercept": self.intercept,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> DatasetConfig:
        return DatasetConfig(
            kind=str(data.get("kind", "deterministic_regression")),
            n_samples=int(data.get("n_samples", 32)),
            input_dim=int(data.get("input_dim", 1)),
            slope=float(data.get("slope", 3.0)),
            intercept=float(data.get("intercept", 2.0)),
        )


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str = "Adam"
    learning_rate: float = 0.05
    epochs: int = 40
    loss: str = "mse"
    technique: TrainingTechnique = TrainingTechnique.NORMAL
    qat_bits: int = 4
    qat_ste_mode: str = "identity"

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "loss": self.loss,
            "technique": self.technique.value,
            "qat_bits": self.qat_bits,
            "qat_ste_mode": self.qat_ste_mode,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> TrainingConfig:
        return TrainingConfig(
            optimizer=str(data.get("optimizer", "Adam")),
            learning_rate=float(data.get("learning_rate", 0.05)),
            epochs=int(data.get("epochs", 40)),
            loss=str(data.get("loss", "mse")),
            technique=TrainingTechnique(data.get("technique", "normal")),
            qat_bits=int(data.get("qat_bits", 4)),
            qat_ste_mode=str(data.get("qat_ste_mode", "identity")),
        )


@dataclass(frozen=True)
class CompressionExperimentConfig:
    bits: int = 4
    method: str = "uniform"
    packed: bool = False
    rotation: RotationOption = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "method": self.method,
            "packed": self.packed,
            "rotation": self.rotation,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> CompressionExperimentConfig:
        rotation = data.get("rotation")
        return CompressionExperimentConfig(
            bits=int(data.get("bits", 4)),
            method=str(data.get("method", "uniform")),
            packed=bool(data.get("packed", False)),
            rotation=rotation,
        )


@dataclass(frozen=True)
class MixedPrecisionExperimentConfig:
    default_bits: int = 8
    parameter_bits: dict[str, int] = field(default_factory=dict)
    method: str = "uniform"
    packed: bool = False
    rotation: RotationOption = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_bits": self.default_bits,
            "parameter_bits": dict(self.parameter_bits),
            "method": self.method,
            "packed": self.packed,
            "rotation": self.rotation,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> MixedPrecisionExperimentConfig:
        return MixedPrecisionExperimentConfig(
            default_bits=int(data.get("default_bits", 8)),
            parameter_bits=dict(data.get("parameter_bits", {})),
            method=str(data.get("method", "uniform")),
            packed=bool(data.get("packed", False)),
            rotation=data.get("rotation"),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    technique: ExperimentTechnique
    seed: int = 42
    model: ModelConfig = ModelConfig.baseline_regression()
    dataset: DatasetConfig = DatasetConfig()
    training: TrainingConfig = TrainingConfig()
    compression: CompressionExperimentConfig | None = None
    mixed_precision: MixedPrecisionExperimentConfig | None = None
    repeats: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "technique": self.technique.value,
            "seed": self.seed,
            "model": self.model.to_dict(),
            "dataset": self.dataset.to_dict(),
            "training": self.training.to_dict(),
            "compression": self.compression.to_dict() if self.compression else None,
            "mixed_precision": self.mixed_precision.to_dict() if self.mixed_precision else None,
            "repeats": self.repeats,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ExperimentConfig:
        compression = data.get("compression")
        mixed = data.get("mixed_precision")
        return ExperimentConfig(
            name=str(data["name"]),
            technique=ExperimentTechnique(data["technique"]),
            seed=int(data.get("seed", 42)),
            model=ModelConfig.from_dict(data.get("model", {})),
            dataset=DatasetConfig.from_dict(data.get("dataset", {})),
            training=TrainingConfig.from_dict(data.get("training", {})),
            compression=(
                CompressionExperimentConfig.from_dict(compression) if compression else None
            ),
            mixed_precision=(
                MixedPrecisionExperimentConfig.from_dict(mixed) if mixed else None
            ),
            repeats=int(data.get("repeats", 1)),
        )

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Experiment name is required.")
        if self.repeats < 1:
            raise ValueError("repeats must be >= 1.")
        if self.training.epochs < 0:
            raise ValueError("training.epochs must be >= 0.")
        if self.technique == ExperimentTechnique.POST_TRAINING_COMPRESSION and self.compression is None:
            raise ValueError("compression config required for post_training_compression.")
        if self.technique == ExperimentTechnique.MIXED_PRECISION and self.mixed_precision is None:
            raise ValueError("mixed_precision config required for mixed_precision technique.")
        if self.technique == ExperimentTechnique.COMBINED:
            if self.training.technique == TrainingTechnique.QAT and self.mixed_precision is not None:
                raise ValueError(
                    "mixed-precision QAT is not implemented; use post-training mixed precision only."
                )

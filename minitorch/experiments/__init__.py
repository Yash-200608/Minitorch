"""MiniTorch Experiment Lab — reproducible integration and comparison framework."""

from .comparison import compare_experiments, format_experiment_comparison_table
from .config import (
    CompressionExperimentConfig,
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    MixedPrecisionExperimentConfig,
    TrainingConfig,
    TrainingTechnique,
    ExperimentTechnique,
)
from .registry import CANONICAL_EXPERIMENTS
from .canonical import (
    build_baseline_model,
    build_baseline_dataset,
    run_checkpoint_resume_experiment,
    run_graph_optimizer_experiment,
    run_mixed_precision_experiment,
    run_performance_experiment,
    run_quantization_matrix,
    run_qat_comparison_experiment,
    run_rotation_experiment,
)
from .identity import compute_experiment_id
from .randomness import seed_everything
from .result import (
    ExperimentComparisonResult,
    ExperimentEnvironment,
    ExperimentMemory,
    ExperimentMetrics,
    ExperimentObservation,
    ExperimentResult,
    ExperimentTimings,
    validate_experiment_result,
)
from .runner import Experiment, list_experiments
from .serialization import (
    ExperimentResultError,
    load_experiment_result,
    save_experiment_result,
)
from .snapshot import clone_model_with_state, snapshot_model_state

EXPERIMENT_SCHEMA_VERSION = "1.0"

__all__ = [
    "EXPERIMENT_SCHEMA_VERSION",
    "CANONICAL_EXPERIMENTS",
    "CompressionExperimentConfig",
    "MixedPrecisionExperimentConfig",
    "DatasetConfig",
    "Experiment",
    "ExperimentComparisonResult",
    "ExperimentConfig",
    "ExperimentEnvironment",
    "ExperimentMemory",
    "ExperimentMetrics",
    "ExperimentObservation",
    "ExperimentResult",
    "ExperimentResultError",
    "ExperimentTechnique",
    "ExperimentTimings",
    "ModelConfig",
    "TrainingConfig",
    "TrainingTechnique",
    "build_baseline_dataset",
    "build_baseline_model",
    "clone_model_with_state",
    "compare_experiments",
    "compute_experiment_id",
    "format_experiment_comparison_table",
    "list_experiments",
    "load_experiment_result",
    "run_checkpoint_resume_experiment",
    "run_graph_optimizer_experiment",
    "run_mixed_precision_experiment",
    "run_performance_experiment",
    "run_quantization_matrix",
    "run_qat_comparison_experiment",
    "run_rotation_experiment",
    "save_experiment_result",
    "seed_everything",
    "snapshot_model_state",
    "validate_experiment_result",
]

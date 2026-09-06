from __future__ import annotations

from typing import Any

from ..nn import Module
from ..compression import CompressedModel
from .checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    TrainingCheckpoint,
    TrainingResumeState,
    benchmark_checkpoint,
    build_optimizer_from_config,
    format_checkpoint_report,
    load_training_checkpoint,
    save_training_checkpoint,
)
from .format import (
    ARTIFACT_CHECKPOINT,
    ARTIFACT_COMPRESSED_MODEL,
    ARTIFACT_MODEL,
    ArtifactError,
    FORMAT_VERSION,
    FRAMEWORK_VERSION,
    MAGIC,
)
from .io import (
    benchmark_persistence,
    format_artifact_report,
    inspect_artifact,
    load_compressed_model,
    load_model,
    save_compressed_model,
    save_model,
)

__all__ = [
    "ARTIFACT_CHECKPOINT",
    "ARTIFACT_COMPRESSED_MODEL",
    "ARTIFACT_MODEL",
    "ArtifactError",
    "CHECKPOINT_SCHEMA_VERSION",
    "FORMAT_VERSION",
    "FRAMEWORK_VERSION",
    "MAGIC",
    "TrainingCheckpoint",
    "TrainingResumeState",
    "benchmark_checkpoint",
    "benchmark_persistence",
    "build_optimizer_from_config",
    "format_artifact_report",
    "format_checkpoint_report",
    "inspect_artifact",
    "load_compressed_model",
    "load_model",
    "load_training_checkpoint",
    "save_compressed_model",
    "save_model",
    "save_training_checkpoint",
]

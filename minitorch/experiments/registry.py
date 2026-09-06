from __future__ import annotations

from .config import (
    CompressionExperimentConfig,
    ExperimentConfig,
    ExperimentTechnique,
    MixedPrecisionExperimentConfig,
    TrainingConfig,
)

CANONICAL_EXPERIMENTS: dict[str, ExperimentConfig] = {
    "fp32_baseline": ExperimentConfig(
        name="fp32_baseline",
        technique=ExperimentTechnique.FP32_BASELINE,
    ),
    "quantization_4bit_uniform": ExperimentConfig(
        name="quantization_4bit_uniform",
        technique=ExperimentTechnique.POST_TRAINING_COMPRESSION,
        compression=CompressionExperimentConfig(bits=4, method="uniform"),
    ),
    "quantization_4bit_codebook": ExperimentConfig(
        name="quantization_4bit_codebook",
        technique=ExperimentTechnique.POST_TRAINING_COMPRESSION,
        compression=CompressionExperimentConfig(bits=4, method="codebook"),
    ),
    "rotation_4bit_codebook": ExperimentConfig(
        name="rotation_4bit_codebook",
        technique=ExperimentTechnique.ROTATION,
        compression=CompressionExperimentConfig(bits=4, method="codebook", rotation="combined"),
    ),
    "mixed_precision_4bit": ExperimentConfig(
        name="mixed_precision_4bit",
        technique=ExperimentTechnique.MIXED_PRECISION,
        mixed_precision=MixedPrecisionExperimentConfig(default_bits=4),
    ),
    "qat_4bit": ExperimentConfig(
        name="qat_4bit",
        technique=ExperimentTechnique.QAT_COMPARISON,
        compression=CompressionExperimentConfig(bits=4, method="uniform"),
    ),
    "parallel_packing_4bit": ExperimentConfig(
        name="parallel_packing_4bit",
        technique=ExperimentTechnique.PERFORMANCE_PARALLEL,
        compression=CompressionExperimentConfig(bits=4, method="uniform", packed=True),
    ),
    "graph_optimizer_baseline": ExperimentConfig(
        name="graph_optimizer_baseline",
        technique=ExperimentTechnique.GRAPH_OPTIMIZER,
    ),
    "checkpoint_resume_baseline": ExperimentConfig(
        name="checkpoint_resume_baseline",
        technique=ExperimentTechnique.CHECKPOINT_RESUME,
    ),
}

"""MiniTorch Experiment Lab demonstration."""

from pathlib import Path

from minitorch.experiments import (
    Experiment,
    ExperimentConfig,
    ExperimentTechnique,
    CompressionExperimentConfig,
    MixedPrecisionExperimentConfig,
    TrainingConfig,
    format_experiment_comparison_table,
    save_experiment_result,
)
from minitorch.experiments.config import DatasetConfig, ModelConfig
from minitorch.experiments.runner import build_dataset_from_config, build_model_from_config


def main() -> None:
    training = TrainingConfig(epochs=30, learning_rate=0.05)
    seed = 42

    baseline = Experiment(
        ExperimentConfig(
            name="fp32_baseline",
            technique=ExperimentTechnique.FP32_BASELINE,
            training=training,
            seed=seed,
        )
    ).run()

    uniform_4 = Experiment(
        ExperimentConfig(
            name="quantization_4bit_uniform",
            technique=ExperimentTechnique.POST_TRAINING_COMPRESSION,
            compression=CompressionExperimentConfig(bits=4, method="uniform"),
            training=training,
            seed=seed,
        )
    ).run()

    codebook_4 = Experiment(
        ExperimentConfig(
            name="quantization_4bit_codebook",
            technique=ExperimentTechnique.POST_TRAINING_COMPRESSION,
            compression=CompressionExperimentConfig(bits=4, method="codebook"),
            training=training,
            seed=seed,
        )
    ).run()

    mixed = Experiment(
        ExperimentConfig(
            name="mixed_precision_4bit",
            technique=ExperimentTechnique.MIXED_PRECISION,
            mixed_precision=MixedPrecisionExperimentConfig(
                default_bits=8,
                parameter_bits={"0.weight": 8, "2.weight": 4},
            ),
            training=training,
            seed=seed,
        )
    ).run()

    results = [baseline, uniform_4, codebook_4, mixed]

    print("============== MiniTorch Experiment Lab ==============")
    print()
    print("EXPERIMENT")
    print("  Dataset: deterministic regression")
    print("  Model: Sequential(Linear -> ReLU -> Linear)")
    print(f"  Seed: {seed}")
    print()
    print("RESULTS")
    print()
    print(format_experiment_comparison_table(results))
    print()
    print("OBSERVATIONS")
    for result in results:
        for obs in result.observations:
            print(f"  [{obs.category}] {obs.message}")
    print()
    print("REPRODUCIBILITY")
    print(f"  Experiment ID (baseline): {baseline.experiment_id}")
    print(f"  Schema: {baseline.schema_version}")
    print("======================================================")

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    for result in results:
        save_experiment_result(result, output_dir / f"experiment_{result.experiment_id}.json")

    model = build_model_from_config(ModelConfig.from_dict(baseline.configuration["model"]))
    inputs, _ = build_dataset_from_config(
        DatasetConfig.from_dict(baseline.configuration["dataset"]), seed
    )
    mri = model.mri(output=model(inputs), experiment=uniform_4)
    print()
    print(str(mri))


if __name__ == "__main__":
    main()

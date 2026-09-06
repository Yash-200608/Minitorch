"""AI-MRI integration with Experiment Lab results."""

from minitorch.experiments import (
    Experiment,
    ExperimentConfig,
    ExperimentTechnique,
    CompressionExperimentConfig,
    TrainingConfig,
)
from minitorch.experiments.runner import build_dataset_from_config, build_model_from_config


def test_mri_consumes_experiment_result():
    config = ExperimentConfig(
        name="quantization_4bit_uniform",
        technique=ExperimentTechnique.POST_TRAINING_COMPRESSION,
        compression=CompressionExperimentConfig(bits=4, method="uniform"),
        training=TrainingConfig(epochs=6),
        seed=17,
    )
    result = Experiment(config).run()
    model = build_model_from_config(config.model)
    inputs, targets = build_dataset_from_config(config.dataset, config.seed)
    output = model(inputs)
    mri = model.mri(output=output, experiment=result)
    report = str(mri)
    assert result.experiment_id in report or result.name in report
    assert any(obs.category in ("compression", "experiment", "matrix") for obs in mri.observations)

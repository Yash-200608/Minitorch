# Phase R — Experiment Lab (Final Integration)

Phase R turns MiniTorch from a collection of techniques into a **reproducible experiment laboratory**. It composes existing systems without adding new optimizers, quantizers, or graph implementations.

## Principle

```text
Same model + same data + same seed + different technique = meaningful comparison
```

Training technique (normal vs QAT) is **separate** from storage/compression technique (uniform, codebook, packing, rotation, mixed precision).

## Core API

```python
from minitorch.experiments import (
    Experiment,
    ExperimentConfig,
    ExperimentTechnique,
    CompressionExperimentConfig,
    TrainingConfig,
    seed_everything,
    compute_experiment_id,
    save_experiment_result,
    load_experiment_result,
    compare_experiments,
    format_experiment_comparison_table,
    run_quantization_matrix,
)

config = ExperimentConfig(
    name="quantization_4bit_uniform",
    technique=ExperimentTechnique.POST_TRAINING_COMPRESSION,
    compression=CompressionExperimentConfig(bits=4, method="uniform"),
    training=TrainingConfig(epochs=40),
    seed=42,
)

result = Experiment(config).run()
print(result.format_report())
save_experiment_result(result, "results/experiment.json")
```

## Experiment identity

`experiment_id` is a 16-character SHA-256 digest of:

- normalized configuration JSON
- `EXPERIMENT_SCHEMA_VERSION` (`1.0`)
- `FRAMEWORK_VERSION`

No random UUIDs.

## Structured result

Each `ExperimentResult` records:

| Section | Contents |
| --- | --- |
| `configuration` | Serializable experiment config |
| `environment` | Python, NumPy, platform, framework versions |
| `metrics` | Loss, RMSE, bytes, compression ratio |
| `timings` | Training, compression, materialization, evaluation |
| `memory` | Model bytes, compressed bytes, codebook/packed breakdown |
| `observations` | Careful language: measured / observed / associated |

Repeated runs (`repeats=N`) aggregate mean/std for stochastic metrics.

## Evaluation pipeline

```text
train → capture final model → evaluate FP → compress/QAT evaluate → measure metrics
```

Stages are kept separate; benchmark setup is not mixed into training time without labeling.

## Canonical experiments

Registry: `list_experiments()` / `CANONICAL_EXPERIMENTS`

| Name | Focus |
| --- | --- |
| `fp32_baseline` | FP32 reference |
| `quantization_4bit_uniform` | Uniform post-training |
| `quantization_4bit_codebook` | Codebook post-training |
| `rotation_4bit_codebook` | Rotation on codebook |
| `mixed_precision_4bit` | Per-parameter precision |
| `qat_4bit` | QAT vs post-training |
| `parallel_packing_4bit` | Serial vs parallel compression |
| `graph_optimizer_baseline` | Graph optimizer timing |
| `checkpoint_resume_baseline` | Checkpoint equivalence |

`run_quantization_matrix()` trains once, then compares FP32 → uniform 2/4/8 → codebook → rotation → mixed precision on the **same** trained model.

## Comparisons

```python
table = format_experiment_comparison_table([baseline, uniform, mixed])
comparison = compare_experiments(results)  # missing metrics stay None
```

## AI-MRI integration

```python
mri = model.mri(output=output, experiment=result)
```

MRI surfaces experiment metadata and observations from Experiment Lab data — no duplicate compression math.

## Serialization

- JSON artifacts: `schema_version`, `experiment_id`, configuration, metrics, timings
- `load_experiment_result()` validates schema; corrupt files fail clearly
- Does not execute arbitrary Python

## Model snapshots

`snapshot_model_state()` / `clone_model_with_state()` preserve initial weights so experiment branches do not cross-mutate.

## Limitations

- Not full cross-hardware reproducibility (NumPy seed only where used)
- Mixed-precision QAT not implemented
- Expensive benchmarks live in `examples/` not unit tests
- Observations use cautious language; no automatic “best” claims

## Example

`examples/experiment_lab_demo.py` — baseline, 4-bit uniform/codebook, mixed precision, comparison table, MRI, JSON export.

## Boundaries

Phase R does **not** add MLflow, cloud tracking, new quantizers, or new training algorithms. Success is integration and evidence.

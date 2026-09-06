# Phase N — Training Checkpoints

Phase N extends MiniTorch persistence from **model artifacts** to **training checkpoints**.

## Model vs checkpoint

| Artifact | Contains |
|----------|----------|
| `model` | architecture + learned parameters |
| `compressed_model` | compressed evaluation representation |
| `checkpoint` | model + optimizer state + training progress (+ optional history/RNG) |

Saving a model is **not** the same as saving a training process.

## Checkpoint contents

**Persisted:**

- Model parameters (by stable name, e.g. `weight`, `bias`)
- Optimizer type and hyperparameters (`lr`, `momentum` / `betas`, `eps`)
- Optimizer state: SGD momentum buffers; Adam `m`, `v`, global step `t`
- `completed_epochs` — finished epoch iterations
- `completed_steps` — finished optimizer updates
- Optional `TrainingHistory.losses` and epoch indices
- Optional NumPy RNG state

**Not persisted:**

- `.grad` arrays (fresh gradients on next step)
- Computation cache
- Profiler / timer state
- Executor / thread-pool state
- Computational graphs

## Resume semantics

- Resume granularity: **epoch boundary** (not mid-minibatch).
- After `Trainer.from_checkpoint(...)` or `restore_into(...)`, call `fit(epochs=N, resume=True)`.
- Epoch indices and step counters continue from the checkpoint; history is not silently reset.

## API

```python
trainer.save_checkpoint("run.mtc")
checkpoint = load_training_checkpoint("run.mtc")
model, optimizer, state = checkpoint.restore()
checkpoint.restore_into(model, optimizer)

resumed_trainer = Trainer.from_checkpoint("run.mtc", loss_fn)
resumed_trainer.fit(inputs, targets, epochs=remaining, resume=True)
```

## Optimizer compatibility

- Checkpoint optimizer type must match the runtime optimizer (SGD vs Adam).
- With `strict=True`, hyperparameter mismatches raise `ValueError`.
- Parameter name/shape/dtype mismatches raise clear errors.

## RNG policy

When RNG state is included (`include_rng=True`), NumPy's global RNG is restored on load.
Exact batch ordering for shuffled DataLoaders is **not** guaranteed without additional iterator state.

## File extension

Checkpoints use the same v1 artifact envelope as Phase M. The example uses `.mtc` for convenience; distinguish by `artifact_type=checkpoint` in inspection, not only by extension.

## Glass Box inspection

```python
format_artifact_report("run.mtc")  # checkpoint-specific report
format_checkpoint_report(path, metadata, payload_bytes)
benchmark_checkpoint(model, optimizer, resume_state)  # size overhead vs model artifact
```

Adam checkpoints are typically much larger than plain model files because `m` and `v` are stored per parameter.

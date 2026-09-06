# Phase D — Training Engine

Phase D adds a complete but compact training stack on top of MiniTorch's
existing Tensor, autograd, Module, and Glass Box systems.

## Module hierarchy

`Module` supports nested submodules and deterministic parameter discovery:

```python
from minitorch import Linear, ReLU, Sequential

model = Sequential(
    Linear(2, 8),
    ReLU(),
    Linear(8, 1),
)

for name, parameter in model.named_parameters():
    print(name, parameter.shape)

weight = model.parameter("0.weight")
```

Parameters are registered in a `ParameterRegistry` (HashMap). Ordering is
sorted by dotted name for reproducibility.

`zero_grad()` sets `grad = None` (not zeros). Gradients accumulate across
multiple `backward()` calls until `zero_grad()` is called.

## Layers and activations

| Component | Formula / behavior |
|-----------|-------------------|
| `Linear` | `y = x @ W.T + b` |
| `ReLU` | `max(0, x)` |
| `Sigmoid` | stable `1 / (1 + exp(-x))` with input clipping |
| `Tanh` | `tanh(x)` |

New differentiable Tensor ops: `exp`, `log`, `relu`, `sigmoid`, `tanh`.

ReLU backward uses `x > 0` (strict). The subgradient at exactly zero is
undefined; gradcheck tests avoid zero inputs.

## Losses

`MSELoss` computes `mean((prediction - target)^2)` and returns a scalar Tensor
connected to the graph.

```python
from minitorch import MSELoss, Tensor

criterion = MSELoss()
loss = criterion(Tensor([1.0, 2.0]), Tensor([0.0, 1.0]))
loss.backward()
```

## Optimizers

Located in `minitorch/training/optimizers.py`.

### SGD

```text
parameter ← parameter - lr × gradient
```

Optional momentum stores velocity in NumPy arrays keyed by parameter id (outside
the autograd graph).

### Adam

Standard Adam with per-parameter `m`, `v`, and global step `t`. Optimizer state
is plain NumPy — it never becomes part of the computation DAG.

```python
from minitorch import Adam, SGD

optimizer = SGD(model.parameters(), lr=0.1)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Dataset and DataLoader

`TensorDataset` pairs input/target tensors by row index.

`DataLoader` supports batching, deterministic iteration, and optional shuffle
with a seed. No multiprocessing or prefetching.

## Trainer

`Trainer` exposes an explicit training step:

```text
zero_grad → forward → loss → backward → optimizer.step
```

`fit()` disables the computation cache by default during training.

```python
from minitorch import Linear, MSELoss, SGD, Tensor, Trainer

model = Linear(1, 1)
trainer = Trainer(model, SGD(model.parameters(), lr=0.1), MSELoss().forward)
history = trainer.fit(x_train, y_train, epochs=100, diagnostics=True)

print(history.losses[-1])
```

`diagnostics=True` records real `GradientHealthReport` objects from Gradient
Detective after each epoch.

`profile_step()` returns graph size, DFS visits, and topological sort counters
from real backward instrumentation.

## Glass Box during training

After training steps, use the same Phase B tools on real gradients:

```python
loss.backward()
model.gradient_report()
model.xray(output=loss)
loss.proof(model.weight)
```

No duplicate gradient analysis paths are introduced for reporting.

## Computation cache during training

Training mutates parameter `.data`. Cached forward numeric results keyed on
parent payloads must not return stale values after an optimizer step.

`Trainer.fit()` disables the cache. If you enable the cache manually, keys
include parent tensor bytes so updated weights produce new cache entries.

Regression tests verify that parameter updates do not return stale products.

## Graph optimizer during training

Graph optimization remains opt-in via `optimize(loss)`. Training does not
automatically simplify graphs.

## Example

```bash
python examples/train_demo.py
```

Trains `y = 3x + 2` with `Linear(1, 1)` and shows decreasing loss.

## Limitations

- No learning-rate schedules, validation loops, or distributed training
- No binary cross-entropy (not required for the initial toy example)
- ReLU gradient at zero is not defined
- Cache disabled in `Trainer.fit()` by design
- No SIMD, quantization, or TurboVec integration yet

## Test coverage

Phase D adds tests for layers, activations, losses, optimizers, dataloaders,
and training integration (including cache regression, non-finite diagnostics,
X-Ray, and Proof Engine after training).

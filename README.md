# MiniTorch — Glass Box AI

MiniTorch is an educational autodiff library where tensors, graphs, gradients,
and DSA algorithms stay inspectable.

## Implementation status

**Core (Milestones 1–6)**

- Tensor, DAG, DFS, topological sort, autograd, `gradcheck`

**Phase A — DSA engine**

- Heap Top-K, parameter registry, computation cache, benchmarks

**Phase B — Glass Box Intelligence**

- Autograd Proof Engine (`loss.explain(x)`)
- Graph Explorer (`loss.explore()`)
- Gradient Detective (`model.gradient_report()`)
- Model X-Ray (`model.xray()`)

**Phase C — Graph Optimizer**

- Safe DAG simplification with forward + gradient verification
- `optimize(loss)` → `OptimizationResult`

**Phase D — Training Engine**

- Layers, losses, `SGD`, `Adam`, `Trainer`, `DataLoader`

**Phase E — QuantizedTensor**

- Uniform scalar quantization (`quantize`, `dequantize`, `quantization_report`)

**Phase F — Learned Codebooks**

- `Codebook.fit`, encode/decode, uniform vs codebook comparison

**Phase G — Bit Packing**

- `pack_codes` / `unpack_codes`, `quantize(..., packed=True)`

**Phase H — Rotation Experiments**

- Deterministic rotation + `compare_quantization_rotation`

**Phase I — Model Compression**

- `compress_model`, `compare_model_compression`, post-training parameter experiments

**Phase J — Performance Profiler**

- `benchmark_timing`, `PerformanceProfile`, training/quantization/compression profiling
- `examples/performance_demo.py`, [`docs/phase_j.md`](docs/phase_j.md)

**Phase K — Parallel Execution**

- Opt-in `ParallelConfig`, chunking, parallel pack/quantize/compress
- `examples/parallel_demo.py`, [`docs/phase_k.md`](docs/phase_k.md)

**Phase L — Vectorization / SIMD-Friendly Kernels**

- Reference vs vectorized kernels (packing, codebook, uniform, Hadamard)
- `examples/vectorization_demo.py`, [`docs/phase_l.md`](docs/phase_l.md)

**Phase M — Persistence**

- Versioned `.mt` / `.mtc` artifacts, `save_model` / `load_model`, compressed save/load
- `examples/persistence_demo.py`, [`docs/phase_m.md`](docs/phase_m.md)

**Phase N — Training Checkpoints**

- Checkpoint artifacts (`artifact_type=checkpoint`): model + optimizer state + epoch/step + history
- `trainer.save_checkpoint`, `Trainer.from_checkpoint`, resume-aware `fit(..., resume=True)`
- `examples/checkpoint_resume_demo.py`, [`docs/phase_n.md`](docs/phase_n.md)

**Phase O — Quantization-Aware Training**

- Weight-only uniform fake quantization with identity STE (experimental, opt-in)
- `x.fake_quantize(bits=4)`, `prepare_qat`, `Trainer(..., qat_config=QATConfig(bits=4))`
- `examples/qat_demo.py`, [`docs/phase_o.md`](docs/phase_o.md)

**Phase P — AI-MRI**

- Unified Glass Box diagnostics: `model.mri(...)`, `inspect_model_mri`, `compare_mri`
- Composes X-Ray, gradients, graph, profiler, compression, QAT without duplicate algorithms
- `examples/ai_mri_demo.py`, [`docs/phase_p.md`](docs/phase_p.md)

**Phase Q — Mixed Precision**

- Per-parameter post-training bit widths: `MixedPrecisionConfig`, `compress_model_mixed_precision`
- Sensitivity reports, global vs mixed comparison, optional greedy heuristic
- `examples/mixed_precision_demo.py`, [`docs/phase_q.md`](docs/phase_q.md)

**Phase R — Experiment Lab**

- Reproducible experiment framework composing training, compression, QAT, profiling, checkpoints
- `Experiment(config).run()`, deterministic `experiment_id`, JSON result persistence
- `run_quantization_matrix`, `compare_experiments`, AI-MRI `model.mri(experiment=result)`
- `examples/experiment_lab_demo.py`, [`docs/phase_r.md`](docs/phase_r.md)

**Future tracks**

- See [`docs/future.md`](docs/future.md) and [`docs/phase_p.md`](docs/phase_p.md)

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Glass Box example

```python
from minitorch import Linear, Module, Tensor

class Model(Module):
    def __init__(self):
        super().__init__()
        self.layer = Linear(2, 1)

    def forward(self, x):
        return self.layer(x)

x = Tensor([[1.0, 2.0]], requires_grad=True)
model = Model()
loss = model(x).sum()

loss.backward()
print(loss.explain(x))
print(model.gradient_report())
print(model.xray(output=loss))
```

See [`examples/glass_box_demo.py`](examples/glass_box_demo.py).

## Training example

```python
from minitorch import Linear, MSELoss, SGD, Tensor, Trainer

model = Linear(1, 1)
trainer = Trainer(model, SGD(model.parameters(), lr=0.1), MSELoss().forward)
history = trainer.fit(Tensor(x_data), Tensor(y_data), epochs=200)
```

See [`examples/train_demo.py`](examples/train_demo.py).

## Quantization example

```python
from minitorch import Tensor

x = Tensor(data)
report = x.quantization_report(bits=4)
print(report.format())
```

See [`examples/quantization_demo.py`](examples/quantization_demo.py).

## Codebook example

```python
uniform = x.quantize(bits=4)
codebook = x.quantize(bits=4, method="codebook")
print(uniform.report().error.rmse, codebook.report().error.rmse)
```

See [`examples/codebook_demo.py`](examples/codebook_demo.py).

## Bit packing example

```python
q = x.quantize(bits=4, packed=True)
print(q.report().memory.actual_packed_bytes)
```

See [`examples/bitpacking_demo.py`](examples/bitpacking_demo.py).

## Rotation experiment

```python
from minitorch import compare_quantization_rotation, Tensor

experiment = compare_quantization_rotation(tensor.data, bits=4, method="codebook")
print(experiment.format())
```

See [`examples/rotation_quantization_demo.py`](examples/rotation_quantization_demo.py).

## Model compression example

```python
from minitorch import CompressionConfig, compare_model_compression

report = compare_model_compression(
    model, inputs, targets,
    config=CompressionConfig(bits=4, method="codebook", packed=True),
)
print(report.format())
```

See [`examples/model_compression_demo.py`](examples/model_compression_demo.py).

## Testing

```bash
pytest
```

**240 tests** (228 Phase A–H baseline + Phase I compression tests).

## Documentation

- [`docs/phase_i.md`](docs/phase_i.md) — model compression
- [`docs/phase_h.md`](docs/phase_h.md) — rotation experiments
- [`docs/phase_g.md`](docs/phase_g.md) — bit packing
- [`docs/phase_f.md`](docs/phase_f.md) — learned codebooks
- [`docs/phase_e.md`](docs/phase_e.md) — QuantizedTensor
- [`docs/phase_d.md`](docs/phase_d.md) — training engine
- [`docs/phase_c.md`](docs/phase_c.md) — graph optimizer + verification
- [`docs/phase_a.md`](docs/phase_a.md) — Top-K, registry, cache
- [`docs/explain.md`](docs/explain.md) — core code walkthrough
- [`docs/architecture.md`](docs/architecture.md) — package layout
- [`docs/future.md`](docs/future.md) — planned work (not built)

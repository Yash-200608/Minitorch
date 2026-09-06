# Phase K — Parallel Execution

Phase K adds **opt-in** CPU parallelism for independent MiniTorch workloads, measured against Phase J baselines.

## Philosophy

```text
Measure (Phase J) → Partition → Parallel execute → Deterministic merge → Verify → Measure again
```

Default behavior remains **serial**. Nothing is parallelized automatically based on input size.

## Python concurrency

MiniTorch uses `ThreadPoolExecutor` for chunk-level parallelism:

| Workload | Why threads |
|----------|-------------|
| Codebook encode (`np.argmin` on distances) | NumPy native work can release the GIL |
| Uniform quantization chunks | Same |
| Bit packing | Pure Python loops (GIL-bound; may not speed up small inputs) |
| Per-parameter compression | Independent parameter tasks |

`ProcessPoolExecutor` is **not** used: pickling and Windows spawn overhead dominate for typical chunk sizes.

**Oversubscription warning:** NumPy/BLAS may already use multiple threads internally. Worker threads × BLAS threads can hurt performance. Benchmark on your machine.

## Module: `minitorch/parallel.py`

| API | Role |
|-----|------|
| `ParallelConfig(enabled=False, workers=None)` | Explicit parallel settings |
| `chunk_ranges(count, workers)` | Disjoint `[start, end)` ranges, O(workers) |
| `execute_chunked(count, config, worker_fn)` | Run workers, preserve chunk order |
| `benchmark_parallel_comparison(...)` | Serial vs parallel timing + speedup |
| `ParallelBenchmarkResult` | Structured comparison report |

`workers=1` with `enabled=True` runs a single chunk (serial path).

## Supported parallel workloads

1. **Bit packing** — `pack_codes(..., parallel=True, workers=N)`
2. **Codebook encoding** — `Codebook.encode(..., parallel=True)`
3. **Uniform / codebook quantization** — `quantize_array(..., parallel=True)` and `Tensor.quantize(..., parallel=True)`
4. **Model compression** — `compress_model(..., parallel=True)`

## Not parallelized in Phase K

- Autograd graph construction / mutation
- Shared gradient accumulation
- Optimizer `step()`
- Graph Explorer / DFS / topological sort
- Graph Optimizer rewrites
- Automatic parallel mode selection

## Correctness

| Domain | Guarantee |
|--------|-----------|
| Packed bytes | **Exact** equality serial vs parallel |
| Unpack round-trip | **Exact** code equality |
| Codebook codes | **Exact** integer match |
| Float reconstruction | `allclose` at tight tolerance |
| Compression | Same parameter order, names, packed bytes |

Parallel float reductions may differ slightly if order changes; MiniTorch parallel quantize uses deterministic chunk order concatenation.

## API examples

```python
packed = pack_codes(codes, 4, parallel=True, workers=4)

q = tensor.quantize(bits=4, method="codebook", packed=True, parallel=True, workers=4)

compressed = compress_model(model, parallel=True, workers=4)

from minitorch import benchmark_parallel_comparison
result = benchmark_parallel_comparison(
    "pack",
    lambda: pack_codes(codes, 4),
    lambda: pack_codes(codes, 4, parallel=True, workers=4),
    workers=4,
)
print(result.report())
```

## Break-even behavior

Small inputs may be **slower** when parallelized due to thread overhead. This is expected and measurable with `benchmark_parallel_comparison` on small/medium/large workloads.

## Limitations

- No SIMD (Phase L)
- No GPU
- No automatic parallelization
- Speedup is workload- and machine-specific

## Example

```bash
py examples/parallel_demo.py
```

## Tests

`tests/test_parallel.py` — chunking coverage, parity, exception propagation, no runtime thresholds.

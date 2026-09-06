# Phase J — Performance Profiler

Phase J adds a measurement layer so MiniTorch can report **what was slow**, not guess why.

## Philosophy

```text
Measure → Profile → Identify bottleneck → Hypothesis → Change → Benchmark → Compare
```

This phase establishes instrumentation only. No SIMD, parallelism, GPU, or automatic optimization.

## Module layout

| Path | Role |
|------|------|
| `minitorch/profiling/timing.py` | `Timer`, `benchmark_timing`, `TimingBenchmarkResult` |
| `minitorch/profiling/profile.py` | `PerformanceProfile`, stage and workload profilers |
| `minitorch/profiling/collector.py` | Optional execution collector (future instrumentation) |

## Timing primitives

- Uses `time.perf_counter()` (monotonic, high resolution).
- `benchmark_timing(name, fn, warmup=3, repeats=10)` returns mean, median, min, max.
- **Warmup** runs execute before timed samples to reduce one-off setup effects. Warmup does not remove all noise.

## PerformanceProfile

Structured object with:

- `stages` — one-shot stage timings (forward, backward, optimizer, quantization stages, …)
- `graph_stats` / `dsa_stats` — real graph and DSA counters from existing explorers
- `operation_timings` — measured tensor op means (detailed mode)
- `memory` — deterministic nbytes / representation bytes where available
- `report()` — human-readable summary with careful “hottest measured stage” language
- `top_operations(k)` — heap-based Top-K over measured operations (reuses `dsa.topk`)

## Training integration

```python
profile = trainer.profile_step(inputs, targets)
history = trainer.fit(inputs, targets, epochs=10, profile=True)
```

Default training behavior is unchanged unless `profile=True`.

## Profilers

| Function | Measures |
|----------|----------|
| `profile_training_step` | forward / loss / backward / optimizer + graph/DSA |
| `profile_tensor_operation` | single op with warmup/repeats |
| `profile_standard_tensor_operations` | common tensor ops on small/medium workloads |
| `profile_graph_optimizer_comparison` | original vs optimized backward time + graph size |
| `profile_quantization_stages` | rotation, fit/encode/quantize, pack/unpack, dequant |
| `profile_quantization_memory_matrix` | 2/4/8-bit uniform + codebook packed matrix |
| `profile_rotation_pipeline` | rotation + quant + pack + inverse rotation |
| `profile_model_compression` | compress + materialize |
| `profile_compression_tradeoffs` | memory / RMSE / time table |
| `profile_cache_comparison` | cache disabled vs enabled |
| `profile_profiling_overhead` | none vs lightweight vs detailed instrumentation |

## Modes

- **Lightweight** — major stages only (lower overhead).
- **Detailed** — adds per-operation tensor benchmarks (can perturb runtime).

## Limitations (documented)

1. Benchmark results are **machine-dependent** (CPU, OS, Python/NumPy versions, load).
2. Profiling instrumentation adds overhead — measure with `profile_profiling_overhead`.
3. Reports identify the **hottest measured stage**, not root cause.

## Example

```bash
py examples/performance_demo.py
```

## Tests

`tests/test_profiling.py` verifies structure, repeat counts, non-negative timings, and integrations. **No exact runtime thresholds.**

## Bridge to TurboVec

Phase J is the evidence layer required before Phase K (parallelism) and Phase L (SIMD). Optimize only after measuring.

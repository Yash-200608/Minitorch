# Changelog

All notable changes to MiniTorch are documented here.
Format inspired by [Keep a Changelog](https://keepachangelog.com/).
Versioning follows the package version in `pyproject.toml`.

## [0.1.0] — 2026-09-12

Core MiniTorch is complete through **Phase S**.

### Added

- Tensor core, DAG, reverse-mode autograd, and `gradcheck`
- DSA: DFS, Kahn topological sort, binary-heap Top-K, HashMap registry
- Glass Box: Proof Engine, Graph Explorer, Gradient Detective, Model X-Ray,
  graph optimizer, AI-MRI
- Training: `Module`, `Linear`, `Sequential`, activations, `MSELoss`,
  SGD, Adam, `Dataset` / `DataLoader`, `Trainer`
- Quantization: uniform, codebook, bit packing, rotation experiments
- Post-training compression and mixed precision
- Weight-only QAT (experimental, opt-in)
- Profiling, opt-in parallelism, vectorized kernels
- Persistence for models, compressed models, and checkpoints
- Experiment Lab: deterministic IDs, comparisons, JSON results
- Canonical benchmark suite under `benchmarks/`
- Project docs: README, ARCHITECTURE, CONTRIBUTING, SECURITY, LICENSE

### Notes

- Mixed-precision QAT, activation quantization, GPU, and distributed
  training are **not implemented**.
- Graph optimizer cannot replay ReLU.
- Benchmark timings are machine-local and are not used as test thresholds.

## Phase history

| Phase | Focus |
| --- | --- |
| A | DSA engine |
| B | Glass Box intelligence |
| C | Graph optimizer |
| D | Training engine |
| E | QuantizedTensor |
| F | Learned codebooks |
| G | Bit packing |
| H | Rotation experiments |
| I | Model compression |
| J | Performance profiler |
| K | Parallel execution |
| L | Vectorized kernels |
| M | Persistence |
| N | Training checkpoints |
| O | QAT |
| P | AI-MRI |
| Q | Mixed precision |
| R | Experiment Lab |
| S | Final validation |

Optional future work is listed in [`docs/future.md`](docs/future.md).

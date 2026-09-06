# Phase L — Vectorization / SIMD-Friendly Kernels

Phase L investigates how scalar numerical code becomes **vectorized** and when that helps performance.

## Terminology

| Term | Meaning |
|------|---------|
| **Vectorization** | Expressing computation as array operations (NumPy ufuncs, BLAS) |
| **SIMD** | CPU executes one instruction on multiple data lanes (hardware) |
| **Parallelism** | Multiple workers on independent chunks (Phase K) |

NumPy vectorization often delegates to native SIMD-capable libraries, but MiniTorch does not claim a specific ISA (AVX2, NEON, etc.) unless a backend actually uses it.

## Module: `minitorch/kernels/`

| Submodule | Reference | Vectorized |
|-----------|-----------|------------|
| `uniform.py` | Python scalar loop | `np.round` + `np.clip` |
| `codebook.py` | Per-value `argmin` loop | Broadcast `|values - centroids|` |
| `packing.py` | MSB-first Python bit loop | NumPy bitwise packing |
| `hadamard.py` | Per-index butterfly loop | Slice-based butterfly stages |
| `benchmark.py` | `VectorizationBenchmarkResult` | serial vs vectorized timing |

## Default backends (stable behavior)

| Operation | Default | Opt-in |
|-----------|---------|--------|
| Bit packing | `reference` | `backend="vectorized"` |
| Codebook encode | `vectorized` (existing NumPy path) | `backend="reference"` for education |
| Walsh-Hadamard | `reference` | `backend="vectorized"` |
| Uniform quantize | `vectorized` | `backend="reference"` in kernels |

## API examples

```python
packed = pack_codes(codes, 4, backend="vectorized")

codes = codebook.encode(values, backend="reference")  # scalar reference

hadamard = walsh_hadamard_1d(vector, backend="vectorized")

from minitorch.kernels import benchmark_vectorization
result = benchmark_vectorization("pack", n, ref_fn, vec_fn)
print(result.report())
```

## Memory-aware codebook encoding

Full vectorized encode builds an `N × K` distance matrix. For large `N` and `K`, use chunked vectorization:

```python
nearest_centroid_codes_vectorized_chunked(values, centroids, chunk_size=65536)
```

## Parallel + vectorized interaction

Combining Phase K parallelism with Phase L vectorization is **not automatic**. Benchmark explicitly — oversubscription (threads × BLAS threads) can hurt.

## Correctness

- **Exact**: integer codes, packed bytes
- **allclose**: float reconstructions, Hadamard outputs

Reference implementations remain available for teaching and parity tests.

## Native SIMD gate

No C/Rust/assembly in Phase L. Native SIMD is only justified if profiling shows NumPy vectorization is insufficient for a specific kernel.

## Example

```bash
py examples/vectorization_demo.py
```

## Tests

`tests/test_vectorized_kernels.py` — parity, Hadamard properties, benchmark structure. No runtime thresholds.

## Learning objective

> Vectorization and parallelism solve different parts of the performance problem. Measure each change independently.

# Phase F — Learned Codebooks

Phase F extends Phase E uniform quantization with **data-aware scalar codebooks**
learned by Lloyd-style centroid iteration.

## Uniform vs codebook

```text
Phase E (uniform)
  float values -> fixed evenly spaced levels -> codes

Phase F (codebook)
  float values -> learn centroids from data -> nearest code -> decode via codebook
```

The educational question: at the same bit width, can learned levels reduce
reconstruction error compared with uniform spacing?

## Codebook API

```python
from minitorch import Codebook, Tensor

values = tensor.data
codebook = Codebook.fit(values, levels=16, max_iterations=100, tolerance=1e-6)
codes = codebook.encode(values)
reconstructed = codebook.decode(codes, shape=values.shape)
```

Tensor integration:

```python
q = x.quantize(bits=4, method="codebook")
uniform = x.quantize(bits=4)  # default uniform — Phase E behavior preserved
```

## Lloyd-style learning

```text
1. Initialize centroids with linspace(min, max, levels)
2. Assign each value to nearest centroid
3. Update each centroid to mean of assigned values
4. Repeat until movement < tolerance or max_iterations
```

**Empty clusters:** centroids with no assigned values keep their previous value
(no NaN centroids).

**Constant tensors:** all centroids set to the constant; objective is zero.

**Deterministic:** linspace initialization, no randomness.

## Encoding and decoding

Encoding: nearest centroid by absolute distance.

Decoding: `code -> centroids[code]` with shape metadata preserved.

`dequantize()` returns `requires_grad=False` with no graph parents.

## Memory accounting

Reports distinguish:

| Field | Meaning |
| --- | --- |
| `theoretical_packed_bytes` | `ceil(n * bits / 8)` for codes |
| `actual_code_storage_bytes` | NumPy integer array nbytes |
| `codebook_bytes` | `centroids.nbytes` |
| `total_theoretical_representation_bytes` | packed codes + codebook |

Codebook memory is not free. Compression ratios in reports still refer to
theoretical **code** packing unless total representation is explicitly cited.

## Reports

`QuantizationReport` includes method, fit statistics (iterations, converged,
movement, mean squared error), and centroid listing when using codebooks.

## Input policy (same as Phase E)

- Empty tensors -> `ValueError`
- NaN / infinity -> `ValueError`
- Supported bits: 2, 4, 8

When `levels > unique values`, fitting still succeeds; duplicate or unused
centroids may remain.

## Autograd and training boundaries

Quantization remains outside autograd. No straight-through estimators or
quantization-aware training in Phase F.

Training, graph optimizer, and cache semantics are unchanged.

## TurboVec inspiration (not replication)

Inspired by representative-value encoding, but **not** implementing:

- Beta-distribution modeling
- adaptive Simpson integration
- rotation
- SIMD layouts
- physical bit packing

## Benchmark

`benchmark_quantization_bitwidths()` compares uniform and codebook modes per
bit width with local timing and error measurements.

## Example

```bash
python examples/codebook_demo.py
```

## Future

```text
Phase F — codebooks (this phase)
Phase G — bit packing
Phase H — rotation experiments
```

## Limitations

- 1D scalar codebooks only
- Brute-force nearest-centroid search
- No differentiable quantization
- Centroids stored as float64 NumPy arrays

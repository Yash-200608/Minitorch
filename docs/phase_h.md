# Phase H — Rotation + Quantization Experiments

Phase H investigates whether **deterministic orthogonal transforms** applied
before low-bit quantization can reduce reconstruction error in original space.

This is an experiment, not a guarantee of improvement.

## Pipeline

```text
Without rotation:
  x -> quantize -> dequantize -> x_hat

With rotation:
  x -> rotate -> quantize -> dequantize -> inverse rotate -> x_hat
```

Errors are always measured as `x_hat vs x` in the **original coordinate space**.

## Transforms

Applied along the **final tensor axis** independently for each leading index.

### Permutation

Deterministic cyclic shift (invertible reordering).

### Sign flip

Alternating `+1, -1` pattern (self-inverse).

### Normalized Walsh-Hadamard

Iterative butterfly implementation with `1/sqrt(n)` scaling.

- Time: O(n log n) for length n
- Space: O(n) working array
- Self-inverse: `H(H(x)) ≈ x`

Supported final-axis sizes: powers of two (`2, 4, 8, 16, ...`).

## Combined rotation

```python
from minitorch import build_rotation_transform

transform = build_rotation_transform(16, kind="combined")
rotated = transform.apply(data)
restored = transform.inverse(rotated)
```

Order: permutation -> sign flip -> Hadamard.

Inverse: Hadamard -> sign flip -> inverse permutation.

## Quantization integration (opt-in)

Default behavior unchanged:

```python
q = x.quantize(bits=4)  # no rotation
```

Experimental:

```python
q = x.quantize(bits=4, method="codebook", rotation="combined")
```

Works with `packed=True` from Phase G.

## Experiment API

```python
from minitorch import compare_quantization_rotation, Tensor

experiment = compare_quantization_rotation(
    tensor.data,
    bits=4,
    method="uniform",
    rotation="combined",
)
print(experiment.format())
```

Reports baseline vs rotated RMSE and labels result as `improved`, `unchanged`,
or `worse` based on actual measurements.

## L2 norm preservation

Permutation and sign flips preserve norm exactly. Normalized Hadamard preserves
norm up to floating-point tolerance.

## Boundaries

- Outside autograd and training
- Does not modify graph optimizer or cache
- Not TurboVec's production two-round / ChaCha8 / SIMD pipeline

## Example

```bash
python examples/rotation_quantization_demo.py
```

## Limitations

- Final axis must be power of two for rotation APIs
- No GPU / SIMD / parallel transforms
- No quantization-aware training
- Rotation may help, hurt, or leave error unchanged depending on distribution

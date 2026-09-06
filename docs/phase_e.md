# Phase E — QuantizedTensor

Phase E introduces **uniform scalar quantization** as MiniTorch's first
TurboVec-inspired memory representation. It is educational, explicit, and outside
autograd.

## What quantization does

Normal MiniTorch tensors store `float64` values (64 bits per element in our
implementation). Quantization maps each value to one of a small number of
discrete levels:

```text
float64 values
    ↓
xmin, xmax, scale
    ↓
integer code in [0, levels - 1]
    ↓
dequantize → approximate reconstruction
```

Tradeoff:

```text
less memory (theoretically)
    vs
less numerical precision
```

## Supported bit widths

Only **2**, **4**, and **8** bits are supported initially.

```text
2 bits → 4 levels
4 bits → 16 levels
8 bits → 256 levels
```

## API

```python
from minitorch import Tensor

x = Tensor(data)
q = x.quantize(bits=4)
reconstructed = q.dequantize()
report = q.report()
print(report.format())
```

`Tensor.quantization_report(bits=4)` quantizes and returns the same structured
report without keeping the `QuantizedTensor` reference.

## Uniform scalar equations

For finite values with `xmin < xmax`:

```text
scale = (xmax - xmin) / (levels - 1)
code = round((x - xmin) / scale)
x_hat = xmin + code * scale
```

Constant tensors (`xmin == xmax`) use code `0` and reconstruct the constant
without dividing by zero.

## Memory accounting

Reports distinguish:

| Metric | Meaning |
| --- | --- |
| `original_bytes` | Actual NumPy storage of the source tensor |
| `theoretical_packed_bytes` | `ceil(n * bits / 8)` if each code used exactly `bits` bits |
| `actual_code_storage_bytes` | `codes.nbytes` for the NumPy integer array |

**4-bit logical codes are not 4-bit physical storage yet.** Codes live in NumPy
`int32` arrays. Bit packing is Phase G.

## Error metrics

Computed from original vs reconstructed data:

- maximum absolute error
- mean absolute error
- RMSE

No fabricated accuracy scores.

## Input requirements

- **Empty tensors** → `ValueError`
- **NaN / infinity** → `ValueError` (finite inputs required)
- **Unsupported bit width** → `ValueError`

## Autograd boundary

Quantization is **not differentiable** in Phase E.

```text
Tensor → quantize() → QuantizedTensor → dequantize() → Tensor
```

`dequantize()` returns `requires_grad=False` with no graph parents. To continue
training math, use the dequantized tensor explicitly — gradients do not pass
through quantization automatically.

## Parameters

`model.parameter("0.weight").quantize(bits=4)` analyzes a copy of parameter
data. Live `Parameter` storage is not modified.

## Glass Box

Use `quantization_report()` or `QuantizedTensor.report()` for inspectable
memory and error statistics. Model X-Ray does not fabricate quantization stats
unless a real quantization operation was performed.

## TurboVec inspiration (not replication)

Inspired by compact vector representations, but **not** implementing:

- Beta-distribution modeling
- Lloyd-Max codebooks
- rotation
- SIMD kernels
- physical bit packing
- nearest-neighbor search

## Future roadmap

```text
Phase E — uniform quantization (this phase)
Phase F — codebooks
Phase G — bit packing
Phase H — rotation experiments
Phase I — performance benchmarking
Phase J — parallelism
Phase K — SIMD
```

## Example

```bash
python examples/quantization_demo.py
```

## Limitations

- No bit packing, codebooks, or quantization-aware training
- Optimizer state and training loop unchanged
- Graph optimizer operates only on normal tensors
- Compression ratios are theoretical until physical packing exists

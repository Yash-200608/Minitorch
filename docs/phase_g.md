# Phase G — Bit Packing

Phase G turns logical low-bit quantization codes into **physical compact
uint8 storage** using bit masks, shifts, and deterministic byte layout.

## Before vs after

```text
Phase E/F (logical)
  4-bit code stored in int32 array

Phase G (packed)
  4-bit codes packed MSB-first into uint8 bytes
  two codes per byte
```

Bit packing is **lossless** at the code level:

```text
codes -> pack -> unpack -> codes   (exact)
```

Quantization itself remains **approximate**:

```text
float -> quantize -> dequantize -> float_hat
```

## Packing layout

Codes are packed from lowest index to highest, **MSB-first within each byte**.

### 4-bit

```text
codes [c0, c1, c2, c3]
byte0: c0 (high nibble) | c1 (low nibble)
byte2: c2 | c3
```

### 2-bit

```text
four codes per byte: c0<<6 | c1<<4 | c2<<2 | c3
```

### 8-bit

One code per byte (no nibble packing).

Unused bits in the final byte are **zero-padded**.

## API

```python
from minitorch import pack_codes, unpack_codes, Tensor

packed = pack_codes(codes, bits=4)
restored = unpack_codes(packed, bits=4, count=len(codes))

q = x.quantize(bits=4, packed=True)
reconstructed = q.dequantize()
```

`packed=False` remains the default for backward compatibility.

## Validation

Before packing:

```text
0 <= code < 2**bits
```

Invalid codes raise `ValueError` (no silent masking).

Unpacking validates that enough bytes exist for the declared `count`.

## Memory reporting

Reports now distinguish:

| Field | Meaning |
| --- | --- |
| `logical_int32_storage_bytes` | What int32 code arrays would use |
| `actual_packed_bytes` | Real uint8 packed storage |
| `theoretical_packed_bytes` | `ceil(n * bits / 8)` |
| `codebook_bytes` | Centroid storage (Phase F) |
| `total_actual_representation_bytes` | packed + codebook |
| `compression_ratio_actual` | original / total actual |

## Educational helpers

`explain_packing(codes, bits)` shows codes, binary patterns, and hex bytes.

## Autograd boundary

Packed quantization remains outside autograd, same as Phase E/F.

## Not implemented

- SIMD / parallel packing
- Rotation
- Quantization-aware training
- TurboVec low-level kernels

## Example

```bash
python examples/bitpacking_demo.py
```

## Future

```text
Phase G — bit packing (this phase)
Phase H — rotation experiments
Phase I — storage / compression experiments
```

# Phase I — Post-Training Model Compression

Phase I applies Phases E–H quantization to **trained model parameters** without
mutating the live model or changing training semantics.

## Experiment pipeline

```text
trained model
    ↓
collect parameters (ParameterRegistry / named_parameters)
    ↓
quantize each parameter (uniform / codebook / packed / rotation)
    ↓
CompressedModel (separate in-memory representation)
    ↓
materialize(template) → evaluation model
    ↓
compare memory, parameter error, outputs, task loss
```

## API

```python
from minitorch import CompressionConfig, compress_model, compare_model_compression

config = CompressionConfig(bits=4, method="codebook", packed=True)
compressed = compress_model(model, config)
report = compare_model_compression(model, inputs, targets, config=config)
print(report.format())

eval_model = compressed.materialize(model)
```

Defaults match conservative quantization settings (`bits=8`, uniform, no packing,
no rotation).

## What is measured

| Metric | Meaning |
| --- | --- |
| Parameter RMSE | Reconstructed vs original weights |
| Output RMSE | Original vs reconstructed model forward |
| Loss delta | Compressed MSE − original MSE |
| Total bytes | Sum of per-parameter actual representation sizes |
| Codebook bytes | Included in totals for codebook mode |

Parameter error, output error, and task loss are **independent** — small weight
changes can have larger output effects, and vice versa.

## Safety boundaries

- Original `Parameter.data`, `grad`, and `requires_grad` are not modified
- No quantization-aware training or automatic parameter replacement
- Compression is outside autograd and the training graph
- No on-disk persistence format in Phase I

## Inspection

- `ModelCompressionReport.parameters()` — per-parameter byte and error details
- `report.largest_parameter_errors(k=5)` — Top-K heap ranking of parameter RMSE
- `compressed.xray()` — summary dict for Glass Box inspection

## Supported models

`Sequential`, `Linear`, `ReLU`, `Sigmoid`, `Tanh`.

Rotation requires power-of-two **final axis** per parameter tensor.

## Example

```bash
python examples/model_compression_demo.py
```

## Limitations

- Per-parameter quantizers (no shared codebooks)
- No SIMD / parallel compression
- Reconstructed models use `requires_grad=False` for evaluation
- Rotation may help or hurt — report actual results

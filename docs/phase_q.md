# Phase Q — Mixed Precision Experiments

Phase Q adds **explicit per-parameter post-training precision** on top of the existing compression pipeline.

## Concept

Not every parameter may need the same bit width. Mixed precision assigns 2/4/8-bit uniformly per parameter while reusing:

- `QuantizedTensor`
- codebook / uniform quantizers
- bit packing
- rotation (where supported)
- `CompressedModel` machinery via `MixedPrecisionCompressedModel`

## API

```python
config = MixedPrecisionConfig(
    default_bits=8,
    parameter_bits={
        "0.weight": 8,
        "1.weight": 4,
    },
)
compressed = compress_model_mixed_precision(model, config)
report = compressed.report(template=model, inputs=inputs, targets=targets)

compare_mixed_precision(model, inputs, targets, mixed_config=config)
measure_parameter_sensitivity(model)
output_sensitivity_experiment(model, inputs, targets, bits=4)
greedy_mixed_precision_compress(model, inputs, targets, max_output_rmse=0.5)  # heuristic
```

`compress_model(...)` global compression is unchanged and remains the default.

## Precedence

`parameter_bits[name]` overrides `default_bits` for that parameter.

Unknown parameter names raise `ValueError`.

## Sensitivity

- **Parameter sensitivity**: reconstruction RMSE per parameter at each bit width (Top-K via heap)
- **Output sensitivity**: quantize one parameter at a time, measure output RMSE (not causal importance)

## Greedy heuristic

Optional `greedy_mixed_precision_compress` reduces precision one parameter at a time under output RMSE / loss constraints. Documented as a **heuristic**, not globally optimal.

## AI-MRI

Pass `MixedPrecisionReport` to `model.mri(compression=report)` for precision map and bytes without recomputing.

## Boundaries

- Post-training only (no mixed-precision QAT)
- No activation mixed precision
- No new quantizer implementation
- No automatic mixed precision in `compress_model`

## Complexity

Explicit configuration: O(P) parameter compressions plus evaluation cost.

Greedy search: additional compress/evaluate steps per accepted candidate.

# Phase O — Quantization-Aware Training (QAT)

Phase O adds **experimental, opt-in** weight-only fake quantization for training.

## Why QAT?

Uniform quantization maps continuous values to discrete codes. The mapping is piecewise constant, so a true derivative is zero almost everywhere and not useful for gradient-based optimization.

**Fake quantization** simulates low precision in the forward pass while parameters remain float. **Straight-Through Estimation (STE)** passes gradients through the quantizer using a **surrogate gradient** (identity STE: `dx ≈ dy`). This is **not** the true derivative of quantization.

## API

```python
# Standalone fake quantize (autograd-aware)
y = x.fake_quantize(bits=4, ste_mode="identity")

# Prepare model copy with weight-only QAT on Linear layers
qat_model = prepare_qat(model, QATConfig(bits=4))

# Trainer opt-in
trainer = Trainer(model, optimizer, loss_fn, qat_config=QATConfig(bits=4))
```

Ordinary `tensor.quantize()` is unchanged and does not participate in autograd.

## Weight-only policy

During training (in `train()` mode), Linear layers fake-quantize **weights** before matmul. Bias stays full precision. Live `Parameter.data` remains float64; optimizer state (SGD momentum, Adam m/v) stays full precision.

## Gradient checking boundary

Do not use standard `gradcheck` on fake quantization expecting numerical agreement. Use STE-specific tests instead.

| Operation | Gradient expectation |
|-----------|-------------------|
| Ordinary ops | Analytical ≈ numerical |
| `fake_quantize` | Surrogate STE rule |

## Comparison experiment

```python
run_qat_experiment(model, optimizer_factory, loss_fn, inputs, targets, bits=4)
```

Compares:

- Normal training → post-training 4-bit quantization
- QAT training → final 4-bit quantization

Metrics: FP loss, quantized loss, quantization damage (`quantized_loss - fp_loss`), output RMSE, parameter RMSE, training time.

## Checkpointing

QAT checkpoints store `qat_config` in metadata. Resume restores QAT on Linear layers via `apply_qat_config`.

## Glass Box

- Graph Explorer shows `fake_quantize` nodes when QAT is active
- Proof Engine notes surrogate STE backward for `fake_quantize`
- Model X-Ray reports QAT enabled/bits/STE
- Graph Optimizer does **not** remove fake-quant nodes

## Not implemented (future)

Codebook QAT, rotation QAT, activation QAT, packed live parameters, quantized optimizer state.

## Memory during training

Live parameters remain full precision. QAT is not memory compression during training; final quantized model storage uses existing compression reporting.

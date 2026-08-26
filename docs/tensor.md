# Tensor Foundation

`minitorch.Tensor` is the core numerical object. It provides numerical storage,
operation lineage, and the public `backward()` entry point for reverse-mode
autograd.

## Stored State

Each tensor stores:

- `data`: NumPy `float64` array storage.
- `requires_grad`: whether future autograd should track this tensor.
- `grad`: reserved for gradients; currently initialized to `None`.
- `id`: a unique tensor identity.
- `parents`: tensors used to produce this tensor.
- `children`: tensors that depend on this tensor.
- `operation`: the operation that produced this tensor, or `None` for leaves.
- `metadata`: operation-specific details such as axes, shapes, or symbols.

## Supported Operations

The tensor supports:

- `+`, `-`, `*`, `/`
- reverse scalar operations such as `2 + x` and `10 - x`
- unary negation
- matrix multiplication with `@`
- power with `**`
- `sum(axis=None, keepdims=False)`
- `mean(axis=None, keepdims=False)`
- `reshape(...)`
- `transpose(...)` and `T`

Each operation creates a new tensor and records real parent links rather than
logging strings. This is the foundation for later graph traversal and reverse
autograd.

## Inspection

`inspect()` returns structured metadata for one tensor. `lineage()` walks
parents recursively and returns dependency-first metadata for the real lineage
that produced a tensor.

```python
from minitorch import Tensor

x = Tensor([1, 2, 3], requires_grad=True)
loss_like = (x * 2).mean()

print(loss_like.inspect())
print(loss_like.lineage())
```

## Current Limitations

Numerical gradient checking, neural network modules, losses, optimizers, and
training are not complete yet.

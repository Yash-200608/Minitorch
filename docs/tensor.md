# Tensor

`minitorch.Tensor` is the only numerical type. It owns storage and the lineage
used to rebuild a computation graph.

## Stored state

| Field | Role |
| --- | --- |
| `data` | NumPy `float64` array |
| `requires_grad` | whether this tensor (or its descendants) should receive grads |
| `grad` | accumulated gradient, or `None` |
| `id` | unique integer identity |
| `parents` | tensors that produced this one |
| `children` | tensors that used this one |
| `operation` | name of the producing op, or `None` for a leaf |
| `metadata` | op details (`axis`, `keepdims`, `shape`, `axes`, `symbol`) |

Construction from another Tensor copies `.data`. Python numbers and lists are
converted with `np.asarray(..., dtype=float64)`.

## How an operation is recorded

`_from_operation` creates a new Tensor whose `requires_grad` is the OR of the
parents. It never mutates parent `.data`.

`_ensure_tensor` wraps a Python/NumPy value as a leaf Tensor (`requires_grad`
false). That is why `x * 2` has two parents.

## Supported operations

- `+`, `-`, `*`, `/` and reverse forms (`2 + x`, `10 - x`, …)
- unary `-`
- `@` (matrix multiplication)
- `**` (power)
- `sum(axis=None, keepdims=False)`
- `mean(axis=None, keepdims=False)`
- `reshape(...)`
- `transpose(...)` and `.T`

Broadcasting follows NumPy on the forward pass. Backward reduces extra
dimensions in `autograd._unbroadcast`.

## Inspection

- `inspect()` — dict for one tensor (ids, shape, op, parents, children, grad)
- `lineage()` — recursive parent visit, dependency-first list of `inspect()` dicts
- `numpy()` — copy of `data`
- `backward(gradient=None)` — delegates to `minitorch.autograd.backward`
- `zero_grad()` — sets `grad` to `None`

```python
from minitorch import Tensor

x = Tensor([1, 2, 3], requires_grad=True)
loss_like = (x * 2).mean()
print(loss_like.inspect())
print(loss_like.lineage())
```

## Current limits

- No neural-network modules on top of Tensor.
- Non-scalar `backward()` needs an explicit seed gradient.
- Power wrt the exponent uses `log(base)` in autograd; that derivative is only
  defined for positive bases.

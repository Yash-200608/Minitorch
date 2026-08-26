# Gradient checking

`minitorch.gradcheck` verifies reverse-mode rules with central finite
differences. It is Milestone 6: same math as autograd, independent estimate.

## Formula

For each element of each input:

```text
f(x + ε) − f(x − ε)
-------------------
       2ε
```

Default `ε = 1e-6`. Each perturbed evaluation builds a **new** Tensor graph so
the caller's tensors are not mutated.

## API

```python
from minitorch import Tensor, gradcheck

def f(t):
    return (t * t).sum()   # must be scalar

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
result = gradcheck(f, x)
```

`inputs` may be one `Tensor` or a sequence of tensors.

Returned `GradientCheckResult`:

- `passed`
- `input_results` — one `GradientCheckInputResult` per input
- `epsilon`, `absolute_tolerance`, `relative_tolerance`

Each input result stores analytical and numerical arrays, absolute/relative
errors, maxima, shape, and `passed`.

Pass condition (elementwise):

```text
|analytic − numeric|  <=  atol + rtol * max(|analytic|, |numeric|)
```

Defaults: `atol=1e-5`, `rtol=1e-4`. Relative error uses a `1e-12` floor in the
denominator so zeros do not explode.

If a leaf never received a gradient, the analytical array is zeros (the tensor
did not affect the scalar).

## Constraints

- `function` must return a **scalar** `Tensor` (`shape == ()`).
- `epsilon` must be positive; tolerances must be non-negative.
- There must be at least one input Tensor.

These limits are intentional: a vector output would need a Jacobian, which this
checker does not compute. Reduce with `.sum()` or `.mean()` first, as in the
autograd tests.

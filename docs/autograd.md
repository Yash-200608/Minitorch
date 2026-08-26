# Autograd

MiniTorch implements reverse-mode automatic differentiation for the current
Tensor operations.

## Algorithm

`Tensor.backward()` delegates to `minitorch.autograd.backward`:

```text
forward computation records Tensor parents
        |
        v
ComputationGraph.from_tensor(output)
        |
        v
topological_sort(graph)
        |
        v
seed output gradient
        |
        v
walk topological order in reverse
        |
        v
apply local derivative rule
        |
        v
accumulate parent gradients
```

The topological order guarantees consumers are processed before their
dependencies during the reverse pass.

## Gradient Accumulation

Each backward call uses a local dictionary keyed by tensor id. This means
multiple paths to the same tensor accumulate correctly:

```python
from minitorch import Tensor

x = Tensor(3.0, requires_grad=True)
y = x + x
y.backward()

print(x.grad)  # 2.0
```

At the end of the pass, local gradients are added into `Tensor.grad`. Repeated
calls to `backward()` therefore accumulate user-visible gradients once per call.

## Supported Local Rules

Implemented backward rules:

- addition
- subtraction
- multiplication
- division
- negation
- matrix multiplication
- power
- sum
- mean
- reshape
- transpose

Broadcasted forward operations reduce their gradients back to each parent shape.

## Current Limitations

- Backward on non-scalar tensors requires an explicit seed gradient.
- The power rule for differentiating with respect to the exponent uses
  `log(base)`, so it is mathematically valid only where that derivative is
  defined.
- Numerical gradient checking is planned next.

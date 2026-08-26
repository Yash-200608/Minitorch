# Autograd

Reverse-mode automatic differentiation for the operations defined on `Tensor`.
Entry point: `Tensor.backward()` → `minitorch.autograd.backward`.

## Algorithm

```text
forward ops record Tensor.parents
        |
        v
ComputationGraph.from_tensor(output)
        |
        v
topological_sort(graph)     # parents before children
        |
        v
seed dL/doutput             # 1.0 if scalar, else caller must pass gradient
        |
        v
for node in reversed(order):
    apply local derivative of node.operation
    unbroadcast parent grads to parent shapes
    accumulate in dict[tensor_id]
        |
        v
add local grads into Tensor.grad
```

Consumers run before producers on the reverse walk, so each node's upstream
gradient is complete before it is split to parents.

## Gradient accumulation

During one call, contributions are summed in a dict keyed by tensor id:

```python
from minitorch import Tensor

x = Tensor(3.0, requires_grad=True)
y = x + x
y.backward()
print(x.grad)  # 2.0
```

After the walk, those values are **added** into `Tensor.grad`. A second
`backward()` therefore doubles leaf grads unless you call `zero_grad()`.

Parents with `requires_grad=False` are skipped.

## Local rules

Implemented in `_local_gradients`:

| Op | Parent gradients (times upstream) |
| --- | --- |
| add | `1`, `1` |
| sub | `1`, `-1` |
| mul | `right`, `left` |
| div | `1/right`, `-left / right²` |
| neg | `-1` |
| matmul | `upstream @ right.T`, `left.T @ upstream` (with vector reshape) |
| pow | `exp * base^(exp-1)`, `out * log(base)` if exponent needs grad |
| sum | broadcast ones according to `axis` / `keepdims` |
| mean | same as sum, divided by reduced count |
| reshape | `upstream.reshape(parent.shape)` |
| transpose | inverse permutation of axes |

Broadcasted forwards are reversed by `_unbroadcast` (sum extra or size-1 axes).

## Return value

`backward()` returns `BackwardResult`: the graph, topological order, and
`BackwardStats` (`processed_nodes`, `gradient_accumulations`, nested
topological-sort stats). The same object is stored on
`output._last_backward_result`.

## Limits

- Non-scalar outputs require an explicit seed gradient of matching shape.
- Differentiating power with respect to the exponent uses `log(base)`.
- No higher-order gradients (`grad.backward()` is not a second graph).

# MiniTorch

MiniTorch is an educational deep-learning framework focused on making tensors,
computational graphs, gradient flow, and DSA algorithms inspectable.

Current implementation status:

- Tensor storage uses NumPy arrays with `float64` data.
- Tensor metadata records shape, rank, size, dtype, gradient state, operation,
  parents, children, and operation-specific metadata.
- Tensor operations currently support arithmetic, negation, matrix
  multiplication, power, reductions, reshape, and transpose.
- Computational graphs can be reconstructed from real Tensor parent links.
- DFS and topological sorting are implemented with real operation counters.
- Reverse-mode autograd supports the current tensor operations and accumulates
  gradients through real graph paths.
- Neural network modules, diagnostics, training, and visualization are planned
  later milestones.

## Installation

```bash
python -m pip install -e ".[dev]"
```

Or install the direct requirements:

```bash
python -m pip install -r requirements.txt
```

## Quickstart

```python
from minitorch import Tensor

x = Tensor([1, 2, 3], requires_grad=True)
y = x * 2
z = y + 3

print(z)
print(z.inspect())
print(z.lineage())
```

## Graph and DSA Example

```python
from minitorch import ComputationGraph, Tensor
from minitorch.dsa import depth_first_search, topological_sort

x = Tensor([1, 2, 3], requires_grad=True)
loss_like = (x * 2 + 3).mean()

graph = ComputationGraph.from_tensor(loss_like)

print(graph.summary())
print(depth_first_search(graph, start_id=loss_like.id, direction="parents"))
print(topological_sort(graph))
```

## Autograd Example

```python
from minitorch import Tensor

x = Tensor(3.0, requires_grad=True)
y = x * x

y.backward()

print(x.grad)  # 6.0
```

## Testing

```bash
pytest
```

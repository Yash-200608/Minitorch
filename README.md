# MiniTorch

MiniTorch is a small educational autodiff library. It does **not** hide the
graph. Every operation records real parent tensors, those links become a DAG,
and reverse-mode autodiff walks that DAG with topological sort.

This repository currently implements **Milestones 1–6** only:

1. Tensor storage and operations
2. Tensor tests
3. Computational graph
4. DFS and topological sort
5. Reverse-mode autograd
6. Numerical gradient checking

Neural networks, losses, optimizers, training, and Glass Box diagnostics are
**not implemented**. Empty files `nn.py`, `losses.py`, `optim.py`,
`datasets.py`, and `utils.py` are placeholders from Phase 1.

## Installation

```bash
python -m pip install -e ".[dev]"
```

Or:

```bash
python -m pip install -r requirements.txt
```

## What to open when explaining the code

Walk a listener through these files in order. Details live in `docs/`.

| Step | File | What to say |
| --- | --- | --- |
| 1 | `minitorch/tensor.py` | A Tensor is NumPy storage plus `parents`, `operation`, and `grad`. |
| 2 | `minitorch/graph.py` | `ComputationGraph.from_tensor` walks those parents and builds a DAG. |
| 3 | `minitorch/dsa/traversal.py` | Iterative DFS with a real stack and visit/edge counters. |
| 4 | `minitorch/dsa/topological_sort.py` | Kahn's algorithm: dependencies before consumers. |
| 5 | `minitorch/autograd.py` | Reverse that order, apply local derivative rules, accumulate grads. |
| 6 | `minitorch/gradcheck.py` | Central finite differences vs analytical gradients. |

Speaker notes: [`docs/explain.md`](docs/explain.md).

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

## Graph and DSA

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

## Autograd

```python
from minitorch import Tensor

x = Tensor(3.0, requires_grad=True)
y = x * x

y.backward()

print(x.grad)  # 6.0
```

`y = x + x` yields `x.grad == 2.0` because two edges accumulate.

## Gradient check

```python
from minitorch import Tensor, gradcheck

x = Tensor(3.0, requires_grad=True)
result = gradcheck(lambda t: t * t, x)

print(result.passed)
print(result.epsilon, result.absolute_tolerance, result.relative_tolerance)
print(result.input_results[0].analytical)
print(result.input_results[0].numerical)
```

## Testing

```bash
pytest
```

## Docs

- [`docs/explain.md`](docs/explain.md) — talk-through of the implemented code
- [`docs/architecture.md`](docs/architecture.md) — files and data flow
- [`docs/tensor.md`](docs/tensor.md) — Tensor design
- [`docs/dsa.md`](docs/dsa.md) — DAG, DFS, topological sort
- [`docs/autograd.md`](docs/autograd.md) — reverse-mode autodiff
- [`docs/gradcheck.md`](docs/gradcheck.md) — finite-difference verification

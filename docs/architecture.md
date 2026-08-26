# Architecture

MiniTorch is a compact package. The implemented core is Tensor + DAG + DSA +
reverse-mode autodiff + finite-difference checks. Higher-level ML pieces are
not present yet.

## Package layout

```text
minitorch/
  tensor.py                 Tensor storage, ops, inspect/lineage, backward()
  graph.py                  ComputationGraph, GraphNode, GraphEdge
  autograd.py               reverse-mode engine
  gradcheck.py              analytical vs numerical gradients
  dsa/
    traversal.py            iterative DFS
    topological_sort.py     Kahn topological sort
  nn.py, losses.py,         empty placeholders (not implemented)
  optim.py, datasets.py,
  utils.py
```

Public exports (`minitorch/__init__.py`): `Tensor`, `ComputationGraph`, graph
error/node/edge types, `backward` / `BackwardResult` / `BackwardStats`,
`gradcheck` / check result types.

## Data flow

```text
user writes  x * 2 + 3
      |
      v
Tensor ops record parents + operation names
      |
      v
ComputationGraph.from_tensor(output)
      |
      v
topological_sort  (dependencies before consumers)
      |
      v
backward walks that order in reverse
      |
      v
local derivative rules + accumulation into Tensor.grad
      |
      v
optional gradcheck via central finite differences
```

## Graph edge direction

Edges point from dependency to consumer:

```text
x -> multiply -> add -> mean
```

Topological sort needs that direction. Autograd uses `reversed(order)`.

## What is not in this codebase

Do not describe these as implemented:

- neural network `Module` / Linear / activations
- losses and optimizers
- training loop
- proof engine, gradient detective, graph explorer, graph optimizer
- Model X-Ray, AI-MRI, DSA profiler, visualization

Those names appear in the original project prompt. They are future work.

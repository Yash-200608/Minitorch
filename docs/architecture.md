# Architecture

MiniTorch is currently implemented as a compact Python package while the core
foundation is being built incrementally.

## Implemented Modules

- `minitorch.tensor`: Tensor storage, tensor metadata, operation lineage, and
  inspection helpers.
- `minitorch.graph`: Reconstruction of a computation DAG from Tensor parent
  links.
- `minitorch.dsa.traversal`: Explicit-stack DFS with measured visits, edge
  inspections, and stack operations.
- `minitorch.dsa.topological_sort`: Kahn topological sorting with measured node
  and edge processing.

## Direction Of Graph Edges

Graph edges point from dependency to consumer:

```text
x -> multiply -> add -> mean
```

This order is useful because topological sorting must process dependencies
before the tensors that consume them. Reverse-mode autodiff will use the same
topological order in reverse.

## Current Limitations

Autograd, neural network modules, losses, optimizers, diagnostics, and
visualization are not implemented yet.

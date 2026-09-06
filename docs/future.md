# Future milestones (architecture only)

These items are **not implemented**. They document planned TurboVec-inspired
direction without copying external implementations.

## Quantization

- `QuantizedTensor` storage separate from `float64` Tensor
- Codebooks mapping discrete symbols to vectors
- Bit packing for compact weight storage

## Performance layer

- SIMD-friendly layouts
- Parallel graph traversal or batched backward (experimental)
- Rotation / transform tricks for approximate search

## Integration rules when implemented

1. Correctness and inspectability before raw speed
2. Opt-in experimental flags — never silent behavior changes
3. Real benchmarks via `minitorch.dsa.benchmark` utilities
4. Each feature must name the DSA it demonstrates

## Current foundation (implemented)

| Capability | DSA |
| --- | --- |
| Top-K | Binary heap |
| Parameter registry | HashMap |
| Computation cache | HashMap memoization |
| Graph / backward | DAG + topological sort |
| DFS | Explicit stack |

Build on this layer before adding quantization or SIMD.

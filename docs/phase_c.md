# Phase C — Graph Optimizer

Phase C adds a **conservative graph optimizer** that simplifies safe algebraic
patterns on the real MiniTorch DAG and **verifies** forward and gradient
equivalence before accepting changes.

## Motivation

Glass Box AI progresses from explain → inspect → **optimize → verify**.

The optimizer does not replace autograd or invent a parallel graph system. It
rebuilds a fresh computation from the existing DAG, applies safe rules, and
checks results against the original engine.

## Supported rules

| Rule | Transformation |
| --- | --- |
| `add_zero` | `x + 0 → x`, `0 + x → x` |
| `sub_zero` | `x - 0 → x` |
| `mul_one` | `x * 1 → x`, `1 * x → x` |
| `mul_zero` | `x * 0 → 0`, `0 * x → 0` (only when verification passes) |
| `div_one` | `x / 1 → x` |

**Not implemented:** constant folding, CSE, fusion, reassociation, distributive
expansions, or division beyond `/ 1`.

## API

```python
from minitorch import Tensor
from minitorch.intelligence.graph_optimizer import optimize

x = Tensor([1.0, 2.0], requires_grad=True)
loss = (x * 1 + 0).sum()

result = optimize(loss)
print(result.report())
print(result.before().describe())
print(result.after().describe())
```

`OptimizationResult` fields:

- `accepted` — whether optimizations passed verification
- `steps` — `OptimizationStep` records (rule, node id, reason)
- `verification` — `VerificationResult` with measured errors
- `graph_before` / `graph_after` — real `ComputationGraph` objects
- `dsa_profile()` — V, E, nodes removed

If verification fails, `optimized_output` remains the **original** tensor.

## Verification strategy

For each candidate optimization:

1. Rebuild optimized graph from leaves (new tensors — no destructive mutation).
2. Compare forward outputs (`allclose`, `equal_nan=True`).
3. Clone both graphs, run `backward()` (with unit seed for non-scalars).
4. Compare gradients on differentiable **leaf** parameters only.

Rejected optimizations are not applied silently.

## Safety

- **NaN / inf:** rules skip non-finite operands; `mul_zero` is rejected when
  IEEE semantics differ (e.g. `NaN * 0`).
- **Broadcasting:** forward match required before accepting a rule.
- **Shared nodes:** memoized rebuild preserves shared subgraph references.
- **Isolated `x * 0`:** often **rejected** because disconnecting `x` breaks
  gradient paths; `x * 0 + x` can be accepted when verification passes.

## DSA involvement

- Graph inspection via existing `ComputationGraph`
- Topological rebuild order from `topological_sort`
- Complexity documented as `O(V + E)` inspection + `O(V)` rebuild

## Glass Box integration

- `result.before()` / `result.after()` use `GraphExplorer`
- `optimized_output.explore()` works on accepted graphs
- `optimized_output.proof(mapped_leaf)` uses Proof Engine on optimized graph
- `model.xray(output=optimized_output)` remains compatible

## Example

```bash
PYTHONPATH=. python examples/graph_optimizer_demo.py
```

## Limitations

- Only listed algebraic rules; no dead-code elimination across branches.
- Verification clones graphs — intended for educational scale.
- AI-MRI and training-loop integration are future work.

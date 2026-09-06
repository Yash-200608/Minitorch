# Phase B — Glass Box Intelligence

Phase B exposes MiniTorch's internal machinery so users can inspect how
gradients and graphs are produced — not just final numeric results.

## Philosophy

MiniTorch is **Glass Box AI**. After `loss.backward()`, you can ask:

- How was this gradient calculated? → **Proof Engine**
- How is the computation wired? → **Graph Explorer**
- Are gradients healthy? → **Gradient Detective**
- What does the model look like internally? → **Model X-Ray**

All outputs use **real** tensor state, graphs, and gradients. No fabricated
scores or hard-coded diagnostics.

---

## 1. Autograd Proof Engine

**Module:** `minitorch/autograd/proof.py`

Replays the reverse topological walk using the same `_local_gradients` rules as
`autograd/engine.py`. There is one autodiff source of truth.

```python
from minitorch import Tensor

x = Tensor(2.0, requires_grad=True)
w = Tensor(4.0, requires_grad=True)
z = x * w

z.backward()
proof = z.proof(x)          # GradientProof
print(z.explain(x))         # human-readable text
```

**Structures:**

- `ProofStep` — consumer op, upstream grad, local derivative, contribution
- `GradientProof` — target id, steps, `accumulated_gradient`, `path_count`

Multiple paths (e.g. `y = x*x; z = y+x`) show separate contributions that sum
to the accumulated gradient.

---

## 2. Computational Graph Explorer

**Module:** `minitorch/autograd/explorer.py`

Wraps the existing `ComputationGraph` and reuses DFS + Kahn topological sort.

```python
explorer = loss.explore()
explorer.nodes()
explorer.parents(x)
explorer.paths(x, loss)
explorer.depth(loss)
explorer.topological_order()
explorer.dsa_profile()
explorer.describe()
```

**DSA profile** reports real `V`, `E`, DFS visits, edge inspections, and
topological-sort counters from `minitorch.dsa`.

---

## 3. Gradient Detective

**Module:** `minitorch/intelligence/gradient_detective.py`

Analyzes **actual** parameter gradients after backward.

```python
report = model.gradient_report(top_k=10)
print(report.largest_gradients)
print(report.smallest_gradients)
```

**Uses `minitorch.dsa.topk`** (binary heap, O(N log K)) for largest/smallest
magnitude ranking — not full `sorted()`.

**Heuristics** (non-definitive language):

- Possible near-zero gradient
- Possible large gradient
- Missing gradients
- Non-finite gradients

---

## 4. Model X-Ray

**Module:** `minitorch/intelligence/model_xray.py`

Combines `ParameterRegistry`, optional graph/DSA stats, and Gradient Detective.

```python
loss = model(x).sum()
loss.backward()
xray = model.xray(output=loss)
print(xray.summary())
print(xray)
```

Inspects:

- module hierarchy and layer types
- parameter names, shapes, element counts, memory bytes
- graph vertices/edges/depth (when `output` provided)
- DSA statistics from Graph Explorer
- gradient health report

---

## 5. AI-MRI (not implemented)

`mri_hook_placeholder()` documents the future composition of X-Ray, Detective,
Explorer, Proofs, and DSA stats. No health score is fabricated in Phase B.

---

## Public API additions

| Entry | Description |
| --- | --- |
| `Tensor.explore()` | Graph explorer |
| `Tensor.proof(target)` | Structured gradient proof |
| `Tensor.explain(target)` | Formatted proof text |
| `Module.gradient_report()` | Gradient health report |
| `Module.xray(output=None)` | Model X-Ray report |

Also exported from `minitorch`: `GradientProof`, `GraphExplorer`,
`detect_gradients`, `build_gradient_proof`.

---

## Example

```bash
python examples/glass_box_demo.py
```

---

## Limitations

- Proof replay uses `seed_gradient` stored on `BackwardResult`; run `backward()`
  before `proof()` for consistent results.
- Graph Explorer paths follow child edges; disconnected nodes return no paths.
- Gradient Detective heuristics are thresholds, not guarantees.
- Model X-Ray graph section requires an `output` tensor from a forward pass.

---

## Turbovec boundary

Phase B strengthens Glass Box observability. **Not implemented:** quantization,
codebooks, SIMD, parallel execution, kernel generation. See `docs/future.md`.

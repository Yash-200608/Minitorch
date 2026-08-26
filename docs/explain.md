# Explain the implemented code

Use this as a talk track for **what is in the repo today**. Do not claim
modules, training, visualization, or Glass Box diagnostics exist.

Suggested order: Tensor → graph → DFS → topological sort → backward →
gradcheck. Each section names the file, the idea, and one demo you can run.

---

## 1. Tensor (`minitorch/tensor.py`)

**Idea.** MiniTorch does not wrap another autodiff library. A `Tensor` is a
NumPy `float64` array plus graph metadata:

- `data`, `shape`, `ndim`, `size`, `dtype`
- `requires_grad`, `grad`
- `id`, `parents`, `children`, `operation`, `metadata`

Every operator (`+`, `*`, `@`, `sum`, …) creates a **new** tensor and records
the real parents. `requires_grad` is true if any parent needs a gradient.
Leaves have `operation is None`.

**Demo.**

```python
from minitorch import Tensor

x = Tensor([1, 2, 3], requires_grad=True)
z = (x * 2) + 3
print(z.inspect())
print(z.lineage())
```

`inspect()` is one node. `lineage()` is a recursive parent walk in
dependency-first order.

Scalars such as `2` become tensors inside `_ensure_tensor`, so `x * 2` has two
parents even though the user wrote one Tensor.

---

## 2. Computational graph (`minitorch/graph.py`)

**Idea.** Autograd does not invent a second graph. `ComputationGraph.from_tensor`
walks `Tensor.parents` and stores:

- `GraphNode` — id, operation, shape, parent/child ids, metadata
- `GraphEdge` — parent → child (dependency → consumer)

Repeated use is preserved: `x + x` stores **two** edges from `x`. That is why
the gradient later becomes 2.

Cycle detection uses a visiting/visited DFS while building the graph.

**Demo.**

```python
from minitorch import ComputationGraph, Tensor

x = Tensor(3.0, requires_grad=True)
y = x + x
print(ComputationGraph.from_tensor(y).summary())
```

Edge direction:

```text
x  -->  add  -->  ...  -->  output
```

Dependencies come first in topological order; backward walks the reverse.

---

## 3. DFS (`minitorch/dsa/traversal.py`)

**Idea.** Educational DFS is iterative (explicit stack), not recursive.

- `direction="children"`: follow dataflow, roots first
- `direction="parents"`: walk from a result back to inputs

Counters are real: visits, edge inspections, stack pushes/pops, max stack size.
Complexity is `O(V + E)`.

---

## 4. Topological sort (`minitorch/dsa/topological_sort.py`)

**Idea.** Kahn's algorithm:

1. Count indegree of every node (how many incoming parent→child edges).
2. Queue nodes with indegree 0 (no remaining dependencies).
3. Pop, append to the order, decrement children, enqueue when indegree hits 0.
4. If not every node was processed, raise `GraphCycleError`.

Why this exists: reverse-mode autodiff must process a tensor **after** every
tensor that consumes it, so the upstream gradient is complete. That is the
topological order reversed.

---

## 5. Autograd (`minitorch/autograd.py`)

**Idea.** `Tensor.backward()` calls `backward(output)`:

1. Build `ComputationGraph.from_tensor(output)`.
2. `topological_sort(graph)`.
3. Seed `dL/doutput` (1 for scalars; required for non-scalars).
4. Walk `reversed(order)`.
5. Apply the local rule for `tensor.operation`.
6. Unbroadcast the parent gradient to the parent shape.
7. Accumulate into a dict keyed by tensor id, then add into `Tensor.grad`.

Classic checks:

```python
x = Tensor(3.0, requires_grad=True)
(x * x).backward()   # x.grad == 6
```

```python
x = Tensor(3.0, requires_grad=True)
(x + x).backward()   # x.grad == 2
```

A second `backward()` **adds** again. `zero_grad()` clears `Tensor.grad`.

Local rules live in `_local_gradients`. Broadcasting is undone in `_unbroadcast`.

---

## 6. Gradcheck (`minitorch/gradcheck.py`)

**Idea.** Do not trust the analytic rules blindly. For each input element:

```text
(f(x + ε) − f(x − ε)) / (2ε)
```

Compare that number to `tensor.grad`. Tolerances are returned, not hidden:

`allowed = atol + rtol * max(|analytic|, |numeric|)`.

The function must return a **scalar** Tensor. Fresh tensors are cloned so the
check does not mutate the caller's graph.

---

## File map (implemented vs empty)

| Path | Status |
| --- | --- |
| `minitorch/tensor.py` | Implemented |
| `minitorch/graph.py` | Implemented |
| `minitorch/dsa/traversal.py` | Implemented |
| `minitorch/dsa/topological_sort.py` | Implemented |
| `minitorch/autograd.py` | Implemented |
| `minitorch/gradcheck.py` | Implemented |
| `minitorch/nn.py` | Empty placeholder |
| `minitorch/losses.py` | Empty placeholder |
| `minitorch/optim.py` | Empty placeholder |
| `minitorch/datasets.py` | Empty placeholder |
| `minitorch/utils.py` | Empty placeholder |

Tests: `tests/test_tensor.py`, `test_graph.py`, `test_dsa.py`,
`test_autograd.py`, `test_gradcheck.py`.

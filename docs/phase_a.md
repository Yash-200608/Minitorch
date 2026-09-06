# Phase A — DSA Engine Extensions

MiniTorch Phase A adds reusable DSA utilities on top of the existing Tensor,
graph, and autograd core.

## Heap Top-K (`minitorch/dsa/topk.py`)

**Problem.** Diagnostics (gradient ranking, MRI summaries) need the largest or
smallest values without sorting an entire array.

**DSA.** Binary min-heap of capacity `K`. For Top-K largest values, the heap root
holds the smallest among the current winners so a better value evicts it in
`O(log K)`.

**Complexity.** `O(N log K)` vs `O(N log N)` full sort.

```python
from minitorch.dsa import topk

result = topk([3, 1, 4, 1, 5, 9, 2], k=3, largest=True)
print(result.values)   # (9, 5, 4)
print(result.indices)  # original positions
print(result.stats.heap_pushes, result.stats.comparisons)
```

Edge cases: `k=0`, empty input, `k > n`, duplicate priorities, negative values.

---

## Parameter registry (`minitorch/registry.py`, `minitorch/nn.py`)

**Problem.** Diagnostics need stable names (`layer1.weight`) separate from
internal tensor ids.

**DSA.** HashMap (`dict`) from dotted string paths to `Parameter` tensors.

```python
from minitorch import Linear, Module, Parameter

class Layer(Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter([1.0, 2.0])

class Model(Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Layer()

model = Model()
w = model.parameter("layer1.weight")
```

`Module` also provides `parameters()`, `named_parameters()`, `zero_grad()`,
`train()`, and `eval()`. `Linear` is a minimal fully-connected layer for tests
and demos.

Operation results on parameters return plain `Tensor` objects (not `Parameter`).

---

## Computation cache (`minitorch/cache.py`)

**Problem.** Repeated deterministic forwards with identical numeric payloads
redo NumPy work even when the graph structure is rebuilt.

**DSA.** HashMap memoization keyed by `(operation, parent data fingerprints,
metadata)`. Values are NumPy arrays; each forward still creates new `Tensor`
nodes with correct parent links for autograd.

**Complexity.** `O(1)` expected lookup per op when enabled; memory grows with
unique keys.

```python
from minitorch import Tensor, enable_computation_cache, disable_computation_cache

enable_computation_cache()
x = Tensor([1, 2, 3])
y = (x * 2).sum()  # second identical build hits the cache

disable_computation_cache()
```

Toggle: `enable_computation_cache()`, `disable_computation_cache()`,
`clear_computation_cache()`, `get_computation_cache().summary()`.

---

## DSA benchmarks (`minitorch/dsa/benchmark.py`)

**Problem.** Compare algorithmic choices with measured runtime and real counters.

```python
from minitorch.dsa import (
    benchmark_full_sort_vs_topk,
    benchmark_dfs,
    benchmark_topological_sort,
    benchmark_cache_on_off,
    build_sample_graph,
)

sort_r, topk_r = benchmark_full_sort_vs_topk(range(1000), k=10)
dfs_r = benchmark_dfs(build_sample_graph())
topo_r = benchmark_topological_sort(build_sample_graph())

def expr():
    from minitorch import Tensor
    return (Tensor([1, 2, 3]) * 2 + Tensor([4, 5, 6])).sum()

off, on = benchmark_cache_on_off(expr, iterations=50)
```

Each `BenchmarkResult` reports `runtime_seconds`, `operation_counts`,
`complexity`, and `memory_notes`. No fabricated statistics.

---

## Future milestones (not implemented)

Architecture placeholders only — do not implement yet:

- `QuantizedTensor`
- Codebooks and bit packing
- Rotation transforms
- SIMD / parallel execution
- TurboVec-style compression

See `docs/future.md`.

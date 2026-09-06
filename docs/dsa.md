# DSA

MiniTorch uses a real DAG and textbook graph algorithms. Counts come from
instrumentation in the functions, not from hardcoded numbers.

## Computation graph

`ComputationGraph.from_tensor(output)` walks `Tensor.parents`.

- `GraphNode`: id, operation, parent/child ids, shape, `requires_grad`, metadata
- `GraphEdge`: `parent_id -> child_id`

`x + x` keeps two edges. Gradient accumulation later depends on that.

Helpers: `adjacency()`, `reverse_adjacency()`, `roots()`, `leaves()`,
`parents()`, `children()`, `summary()`, `vertex_count`, `edge_count`.

While building, a visiting/visited map raises `GraphCycleError` on a cycle.

## DFS

`depth_first_search(graph, start_id=None, direction="children"|"parents")`

Iterative stack in `minitorch/dsa/traversal.py`.

| Metric | Meaning |
| --- | --- |
| visits | nodes appended to the order |
| edge inspections | neighbors looked at |
| stack pushes / pops | explicit stack traffic |
| max stack size | high-water mark |
| complexity | always reported as `O(V + E)` |

`direction="parents"` uses `reverse_adjacency()` so you can start at the loss
and walk toward inputs.

## Topological sort

`topological_sort(graph)` is Kahn's algorithm in
`minitorch/dsa/topological_sort.py`.

```text
indegree[v] = number of incoming edges
queue <- nodes with indegree 0
while queue:
    v = popleft
    append v to order
    for each child u of v:
        indegree[u] -= 1
        if indegree[u] == 0: enqueue u
if |order| != |V|: cycle
```

Result: dependencies before consumers.

Why autograd needs it: the reverse pass must visit a node only after every
consumer has already contributed its upstream gradient. Reversing Kahn order
gives that schedule.

Tracked metrics: processed nodes, edge inspections, queue pushes/pops, max
queue size. Complexity `O(V + E)`.

Tests in `tests/test_dsa.py` cover empty, single node, chain, branch/merge,
diamond, and cycle.

## Top-K (`minitorch/dsa/topk.py`)

`topk(values, k=10, largest=True)` uses a binary min-heap.

- Complexity: `O(N log K)` vs `O(N log N)` full sort
- Returns indices, values, and `HeapStats`
- See [`phase_a.md`](phase_a.md)

## Benchmarks (`minitorch/dsa/benchmark.py`)

Timed utilities compare full sort vs Top-K, DFS, topological sort, and cache
on/off. Each `BenchmarkResult` reports runtime, operation counts, complexity,
and memory notes from real instrumentation.

## Glass Box integration (Phase B)

`GraphExplorer.dsa_profile()` and Model X-Ray reuse these counters for real
`V`, `E`, DFS visits, and topological-sort statistics. Gradient Detective uses
`topk` for magnitude ranking. See [`phase_b.md`](phase_b.md).

## Training integration (Phase D)

During training, the same structures support inspectable learning:

| Role | Structure |
| --- | --- |
| Parameter lookup | `ParameterRegistry` HashMap |
| Optimizer `m`, `v` state | dict keyed by `id(parameter)` |
| Forward/backward graph | `ComputationGraph` DAG |
| Backward schedule | reverse Kahn topological order |
| Gradient ranking in diagnostics | `topk` binary heap |
| `Trainer.profile_step` | graph size + DFS + topo counters |

`Trainer.fit()` disables the computation cache by default because parameters
mutate each step. See [`phase_d.md`](phase_d.md).

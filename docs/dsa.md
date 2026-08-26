# DSA Foundation

MiniTorch uses real data structures and graph algorithms to expose how tensor
computations are connected.

## Computation Graph

`ComputationGraph.from_tensor(output)` walks the actual `Tensor.parents` links
and builds:

- `GraphNode` records containing tensor id, operation, shape, parent ids, child
  ids, gradient requirement, and operation metadata.
- `GraphEdge` records from dependency to consumer.

Repeated dependencies are preserved. For example, `x + x` records two edges
from `x` to the add result, which matters later for gradient accumulation.

## DFS

`depth_first_search(graph)` uses an explicit stack.

Tracked metrics:

- visits
- edge inspections
- stack pushes
- stack pops
- maximum stack size

Complexity: `O(V + E)`.

## Topological Sort

`topological_sort(graph)` uses Kahn's algorithm:

```text
compute indegree for every node
enqueue nodes with indegree 0
while queue is not empty:
    pop a node
    append it to the ordering
    decrement indegree for each child
    enqueue children whose indegree becomes 0
```

The result places dependencies before consumers. If not every node is processed,
the graph contains a cycle and `GraphCycleError` is raised.

Tracked metrics:

- processed nodes
- edge inspections
- queue pushes
- queue pops
- maximum queue size

Complexity: `O(V + E)`.

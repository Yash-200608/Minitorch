import pytest

from minitorch import ComputationGraph, GraphCycleError, GraphEdge, GraphNode
from minitorch.dsa import depth_first_search, topological_sort


def node(
    node_id,
    *,
    parents=(),
    children=(),
    operation=None,
):
    return GraphNode(
        id=node_id,
        operation=operation,
        parent_ids=tuple(parents),
        child_ids=tuple(children),
        shape=(),
        requires_grad=False,
        metadata={},
    )


def graph(nodes, edges, output_id=None):
    return ComputationGraph(
        nodes={item.id: item for item in nodes},
        edges=[GraphEdge(parent, child) for parent, child in edges],
        output_id=output_id,
    )


def test_empty_graph_traversal_and_topological_sort():
    empty = ComputationGraph()

    dfs = depth_first_search(empty)
    topo = topological_sort(empty)

    assert dfs.order == []
    assert dfs.stats.visits == 0
    assert topo.order == []
    assert topo.stats.processed_nodes == 0


def test_single_node_graph():
    single = graph([node(1)], [], output_id=1)

    assert depth_first_search(single).order == [1]
    assert topological_sort(single).order == [1]


def test_chain_graph_order_and_counts():
    chain = graph(
        [
            node(1, children=(2,)),
            node(2, parents=(1,), children=(3,), operation="mul"),
            node(3, parents=(2,), operation="sum"),
        ],
        [(1, 2), (2, 3)],
        output_id=3,
    )

    dfs = depth_first_search(chain)
    topo = topological_sort(chain)

    assert dfs.order == [1, 2, 3]
    assert dfs.stats.visits == 3
    assert dfs.stats.edge_inspections == 2
    assert topo.order == [1, 2, 3]
    assert topo.stats.processed_nodes == 3
    assert topo.stats.edge_inspections == 2


def test_branch_merge_graph_topological_order():
    merge = graph(
        [
            node(1, children=(3,)),
            node(2, children=(3,)),
            node(3, parents=(1, 2), operation="add"),
        ],
        [(1, 3), (2, 3)],
        output_id=3,
    )

    topo = topological_sort(merge)

    assert topo.order[:2] == [1, 2]
    assert topo.order[-1] == 3
    assert topo.stats.queue_pushes == 3


def test_diamond_graph_parent_direction_dfs():
    diamond = graph(
        [
            node(1, children=(2, 3)),
            node(2, parents=(1,), children=(4,), operation="mul"),
            node(3, parents=(1,), children=(4,), operation="add"),
            node(4, parents=(2, 3), operation="mean"),
        ],
        [(1, 2), (1, 3), (2, 4), (3, 4)],
        output_id=4,
    )

    dfs = depth_first_search(diamond, start_id=4, direction="parents")
    topo = topological_sort(diamond)

    assert dfs.order == [4, 2, 1, 3]
    assert dfs.stats.edge_inspections == 4
    assert topo.order[0] == 1
    assert topo.order[-1] == 4
    assert topo.stats.edge_inspections == 4


def test_topological_sort_detects_cycle():
    cyclic = graph(
        [
            node(1, parents=(3,), children=(2,)),
            node(2, parents=(1,), children=(3,)),
            node(3, parents=(2,), children=(1,)),
        ],
        [(1, 2), (2, 3), (3, 1)],
        output_id=3,
    )

    with pytest.raises(GraphCycleError):
        topological_sort(cyclic)

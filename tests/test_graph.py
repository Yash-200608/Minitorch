import pytest

from minitorch import ComputationGraph, GraphCycleError, Tensor


def test_graph_from_single_tensor():
    x = Tensor(3.0, requires_grad=True)
    graph = ComputationGraph.from_tensor(x)

    assert graph.vertex_count == 1
    assert graph.edge_count == 0
    assert graph.output_id == x.id
    assert graph.roots() == [x.id]
    assert graph.leaves() == [x.id]

    node = graph.node_for(x)
    assert node.operation is None
    assert node.shape == ()
    assert node.requires_grad is True


def test_graph_from_chain_records_real_dependencies():
    x = Tensor([1, 2, 3], requires_grad=True)
    y = x * 2
    z = y + 3

    graph = ComputationGraph.from_tensor(z)

    assert graph.vertex_count == 5
    assert graph.edge_count == 4
    assert graph.output_id == z.id
    assert graph.node_for(z).operation == "add"
    assert [node.id for node in graph.parents(z)] == [y.id, z.parents[1].id]
    assert y.id in [node.id for node in graph.children(x)]

    summary = graph.summary()
    assert summary["vertices"] == 5
    assert summary["edges"] == 4
    assert summary["output_id"] == z.id


def test_graph_preserves_branching_merging_and_repeated_edges():
    x = Tensor(3.0, requires_grad=True)
    y = x + x
    z = y * x

    graph = ComputationGraph.from_tensor(z)

    assert graph.vertex_count == 3
    assert graph.edge_count == 4
    assert graph.node_for(y).parent_ids == (x.id, x.id)
    assert graph.node_for(x).child_ids.count(y.id) == 2
    assert graph.node_for(x).child_ids.count(z.id) == 1


def test_graph_from_tensor_detects_dependency_cycle():
    x = Tensor(1.0)
    y = Tensor(2.0, parents=(x,), operation="manual")
    x.parents = (y,)

    with pytest.raises(GraphCycleError):
        ComputationGraph.from_tensor(x)


def test_node_lookup_rejects_unknown_node():
    x = Tensor(1.0)
    graph = ComputationGraph.from_tensor(x)

    with pytest.raises(KeyError):
        graph.node_for(999999)

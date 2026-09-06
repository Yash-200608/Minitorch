import pytest

from minitorch import Tensor
from minitorch.autograd.explorer import GraphExplorer


def test_explorer_node_and_edge_counts():
    x = Tensor([1, 2, 3], requires_grad=True)
    loss = (x * 2 + 3).mean()

    explorer = loss.explore()

    assert explorer.graph.vertex_count >= 3
    assert len(explorer.nodes()) == explorer.graph.vertex_count
    assert len(explorer.edges()) == explorer.graph.edge_count


def test_explorer_parents_and_children():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * 2
    z = y + 3

    explorer = z.explore()
    y_node = explorer.graph.node_for(y)

    parent_ids = [node.id for node in explorer.parents(y)]
    assert x.id in parent_ids
    assert y.id in [node.id for node in explorer.children(x)]


def test_explorer_topological_order():
    x = Tensor([1, 2, 3], requires_grad=True)
    loss = (x * 2).mean()

    explorer = loss.explore()
    order = explorer.topological_order()

    assert order[0] == x.id
    assert order[-1] == loss.id


def test_explorer_paths_from_source_to_loss():
    x = Tensor(2.0, requires_grad=True)
    y = x * 3
    loss = y + x

    explorer = loss.explore()
    paths = explorer.paths(x, loss)

    assert len(paths) == 2
    assert all(path[-1] == loss.id for path in paths)
    assert all(path[0] == x.id for path in paths)


def test_explorer_depth():
    x = Tensor([1, 2], requires_grad=True)
    y = x * 2
    z = y + 1
    loss = z.mean()

    explorer = loss.explore()
    assert explorer.depth(x) == 0
    assert explorer.depth(loss) >= 3


def test_explorer_dsa_profile_real_counters():
    x = Tensor([1, 2, 3], requires_grad=True)
    loss = (x * 2 + 3).mean()

    profile = loss.explore().dsa_profile()

    assert profile["vertices"] == loss.explore().graph.vertex_count
    assert profile["edges"] == loss.explore().graph.edge_count
    assert profile["dfs_visits"] == profile["vertices"]
    assert profile["topological_processed_nodes"] == profile["vertices"]
    assert profile["complexity"] == "O(V + E)"


def test_explorer_paths_empty_for_disconnected():
    x = Tensor(1.0)
    y = Tensor(2.0)

    explorer = GraphExplorer.from_tensor(x + y)
    paths = explorer.paths(x, y)
    assert paths == ()

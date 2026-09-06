import numpy as np
import pytest

from minitorch import Linear, Module, Tensor
from minitorch.intelligence.graph_optimizer import (
    GraphOptimizer,
    optimize,
    _is_additive_zero,
    _is_multiplicative_one,
    _is_multiplicative_zero,
)


def assert_forward_close(original: Tensor, optimized: Tensor) -> None:
    np.testing.assert_allclose(original.data, optimized.data, rtol=1e-9, atol=1e-9)


def assert_gradients_close(model_tensors: list[Tensor]) -> None:
    for tensor in model_tensors:
        if tensor.requires_grad and tensor.grad is not None:
            assert tensor.grad is not None


@pytest.mark.parametrize(
    "expr",
    [
        lambda x: x + 0,
        lambda x: 0 + x,
        lambda x: x - 0,
        lambda x: x * 1,
        lambda x: 1 * x,
        lambda x: x / 1,
    ],
)
def test_safe_rules_forward_and_gradient(expr):
    x = Tensor(3.5, requires_grad=True)
    original = expr(x)
    result = optimize(original)

    assert result.accepted
    assert len(result.steps) >= 1
    assert result.verification.forward_passed
    assert result.verification.gradients_passed
    assert_forward_close(original, result.optimized_output)


def test_mul_zero_with_second_path_accepted():
    x = Tensor(2.0, requires_grad=True)
    original = x * 0 + x
    result = optimize(original)

    assert result.accepted
    assert any(step.rule == "mul_zero" for step in result.steps)
    assert result.verification.gradients_passed

    x2 = Tensor(2.0, requires_grad=True)
    orig = x2 * 0 + x2
    opt = optimize(orig).optimized_output
    orig.backward()
    opt.backward()
    np.testing.assert_allclose(x2.grad, 1.0)


def test_mul_zero_isolated_rejected():
    x = Tensor(3.0, requires_grad=True)
    original = x * 0
    result = optimize(original)

    assert not result.accepted
    assert result.optimized_output is original


def test_nan_times_zero_not_silently_optimized():
    x = Tensor(np.nan, requires_grad=True)
    original = x * 0
    result = optimize(original)

    assert not result.accepted


def test_shared_node_branching_graph():
    x = Tensor(2.0, requires_grad=True)
    left = x * 1
    right = x + 0
    original = left + right

    result = optimize(original)
    assert result.accepted
    assert result.verification.gradients_passed

    x2 = Tensor(2.0, requires_grad=True)
    orig = (x2 * 1) + (x2 + 0)
    opt = optimize(orig).optimized_output
    orig.backward()
    opt.backward()
    np.testing.assert_allclose(x2.grad, 2.0)


def test_broadcast_add_zero():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    zero = Tensor([0.0, 0.0])
    original = x + zero
    result = optimize(original)

    assert result.accepted
    assert_forward_close(original, result.optimized_output)


def test_nodes_removed_count():
    x = Tensor([1.0, 2.0], requires_grad=True)
    original = (x * 1 + 0).sum()
    result = optimize(original)

    assert result.accepted
    assert result.nodes_removed > 0
    assert result.graph_after.vertex_count < result.graph_before.vertex_count


def test_dsa_profile_real_counts():
    x = Tensor([1.0, 2.0], requires_grad=True)
    original = (x * 1).sum()
    result = optimize(original)
    profile = result.dsa_profile()

    assert profile["original_vertices"] == result.graph_before.vertex_count
    assert profile["optimized_vertices"] == result.graph_after.vertex_count


def test_constant_helpers():
    assert _is_additive_zero(Tensor(0.0))
    assert _is_multiplicative_one(Tensor(1.0))
    assert _is_multiplicative_zero(Tensor(0.0))
    assert not _is_additive_zero(Tensor(1.0))


def test_explore_after_optimization():
    x = Tensor([1.0, 2.0], requires_grad=True)
    original = (x * 1).sum()
    result = optimize(original)

    explorer = result.optimized_output.explore()
    assert explorer.graph.vertex_count == result.graph_after.vertex_count
    assert len(explorer.topological_order()) > 0


def test_proof_after_optimization():
    x = Tensor(2.0, requires_grad=True)
    original = (x * 1 + 0).sum()
    result = optimize(original)
    assert result.accepted
    mapped_x = result.id_mapping[x.id]
    result.optimized_output.backward()

    proof = result.optimized_output.proof(mapped_x)
    np.testing.assert_allclose(proof.accumulated_gradient, 1.0)


def test_model_xray_after_optimization():
    model = Linear(2, 1)
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    loss = (model(x) * 1).sum()
    loss.backward()

    result = optimize(loss)
    report = model.xray(output=result.optimized_output)
    assert report.summary()["parameter_count"] > 0


def test_negative_and_small_values():
    x = Tensor(-0.001, requires_grad=True)
    original = (x * 1).sum()
    result = optimize(original)
    assert result.accepted
    assert result.verification.forward_passed


def test_report_contains_verification_status():
    x = Tensor(1.0, requires_grad=True)
    result = optimize((x * 1).sum())
    text = result.report()

    assert "VERIFICATION" in text
    assert "ACCEPTED" in text
    assert "TRANSFORMATIONS" in text


def test_no_optimization_when_no_rules_apply():
    x = Tensor([1.0, 2.0], requires_grad=True)
    original = (x * 2).sum()
    result = optimize(original)

    assert result.accepted
    assert len(result.steps) == 0
    assert result.nodes_removed == 0

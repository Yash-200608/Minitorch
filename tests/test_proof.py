import numpy as np
import pytest

from minitorch import Tensor, gradcheck
from minitorch.autograd.proof import build_gradient_proof, format_gradient_proof


def test_proof_simple_multiplication():
    x = Tensor(2.0, requires_grad=True)
    w = Tensor(4.0, requires_grad=True)
    z = x * w

    z.backward()
    proof = build_gradient_proof(z, x)

    assert proof.target_id == x.id
    assert proof.path_count == 1
    assert proof.steps[0].consumer_operation == "mul"
    np.testing.assert_allclose(proof.accumulated_gradient, 4.0)
    np.testing.assert_allclose(proof.steps[0].contribution, 4.0)


def test_proof_addition_chain():
    x = Tensor(3.0, requires_grad=True)
    y = x + x

    y.backward()
    proof = build_gradient_proof(y, x)

    assert proof.path_count == 2
    np.testing.assert_allclose(proof.accumulated_gradient, 2.0)


def test_proof_multiple_paths_accumulation():
    x = Tensor(3.0, requires_grad=True)
    y = x * x
    z = y + x

    z.backward()
    proof = build_gradient_proof(z, x)

    assert proof.path_count == 3  # mul records two edges from x, add records one
    np.testing.assert_allclose(proof.accumulated_gradient, 7.0)


def test_proof_matches_gradcheck():
    x = Tensor(2.0, requires_grad=True)
    loss = (x * x).sum()

    loss.backward()
    proof = build_gradient_proof(loss, x)

    check = gradcheck(lambda t: (t * t).sum(), x)
    np.testing.assert_allclose(proof.accumulated_gradient, check.input_results[0].analytical)


def test_tensor_explain_returns_text():
    x = Tensor(2.0, requires_grad=True)
    z = x * 4.0

    z.backward()
    text = z.explain(x)

    assert "GRADIENT PROOF" in text
    assert "Accumulated gradient" in text


def test_proof_requires_target_in_graph():
    x = Tensor(1.0, requires_grad=True)
    y = Tensor(2.0, requires_grad=True)
    z = x * y

    z.backward()

    with pytest.raises(ValueError):
        build_gradient_proof(z, Tensor(99.0))


def test_proof_leaf_target_no_steps():
    x = Tensor(3.0, requires_grad=True)
    x.backward()
    proof = build_gradient_proof(x, x)

    assert proof.path_count == 0
    np.testing.assert_allclose(proof.accumulated_gradient, 1.0)


def test_format_gradient_proof():
    x = Tensor(2.0, requires_grad=True)
    y = x * 3.0
    y.backward()
    text = format_gradient_proof(build_gradient_proof(y, x))
    assert "Operation: 'mul'" in text

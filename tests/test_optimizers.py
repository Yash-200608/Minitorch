import numpy as np

from minitorch import Adam, Parameter, SGD, Tensor


def test_sgd_update_without_momentum():
    parameter = Parameter(np.array([1.0, 2.0]))
    parameter.grad = np.array([0.5, 1.0])
    optimizer = SGD([parameter], lr=0.1)

    optimizer.step()
    np.testing.assert_allclose(parameter.data, [0.95, 1.9])


def test_sgd_momentum_update():
    parameter = Parameter(np.array([1.0]))
    parameter.grad = np.array([1.0])
    optimizer = SGD([parameter], lr=0.1, momentum=0.5)

    optimizer.step()
    np.testing.assert_allclose(parameter.data, [0.9])

    parameter.grad = np.array([1.0])
    optimizer.step()
    np.testing.assert_allclose(parameter.data, [0.75])


def test_sgd_zero_grad_sets_none():
    parameter = Parameter(np.array([1.0]))
    parameter.grad = np.array([1.0])
    optimizer = SGD([parameter])
    optimizer.zero_grad()
    assert parameter.grad is None


def test_adam_first_update_direction():
    parameter = Parameter(np.array([1.0]))
    parameter.grad = np.array([1.0])
    optimizer = Adam([parameter], lr=0.1)

    optimizer.step()
    assert parameter.data[0] < 1.0


def test_adam_state_progression_isolated_per_parameter():
    left = Parameter(np.array([1.0]))
    right = Parameter(np.array([2.0]))
    left.grad = np.array([1.0])
    right.grad = np.array([2.0])

    optimizer = Adam([left, right], lr=0.1)
    optimizer.step()

    left.grad = np.array([1.0])
    right.grad = np.array([2.0])
    optimizer.step()

    assert left.data[0] != right.data[0]


def test_optimizer_step_does_not_attach_graph_nodes():
    parameter = Parameter(np.array([1.0]))
    x = Tensor(2.0, requires_grad=True)
    loss = (x * parameter).sum()
    loss.backward()

    optimizer = SGD([parameter], lr=0.1)
    optimizer.step()

    assert parameter.operation is None
    assert parameter.parents == ()

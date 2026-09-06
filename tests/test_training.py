import math

import numpy as np

from minitorch import (
    Linear,
    MSELoss,
    Parameter,
    SGD,
    Sequential,
    Tensor,
    Trainer,
    enable_computation_cache,
    clear_computation_cache,
    disable_computation_cache,
    get_computation_cache,
)


def _make_regression_dataset():
    x_values = np.linspace(0.0, 1.0, 8).reshape(8, 1)
    y_values = 3.0 * x_values + 2.0
    return Tensor(x_values), Tensor(y_values)


def test_training_reduces_loss_over_epochs():
    inputs, targets = _make_regression_dataset()
    model = Linear(1, 1)
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer, criterion.forward)

    initial_loss = float(criterion(model(inputs), targets).data)
    history = trainer.fit(inputs, targets, epochs=200)
    final_loss = history.losses[-1]

    assert final_loss < initial_loss
    assert final_loss < 0.05


def test_gradient_accumulation_and_zero_grad():
    parameter = Parameter(np.array([1.0]))
    x = Tensor(np.array([2.0]))
    loss = (parameter * x).sum()

    loss.backward()
    loss.backward()
    np.testing.assert_allclose(parameter.grad, [4.0])

    parameter.zero_grad()
    assert parameter.grad is None


def test_cache_does_not_return_stale_results_after_parameter_update():
    cache = get_computation_cache()
    clear_computation_cache()
    enable_computation_cache()

    parameter = Parameter(np.array([1.0]))
    x = Tensor(np.array([2.0]))
    first = (parameter * x).sum()
    np.testing.assert_allclose(first.data, 2.0)

    parameter.data = np.array([3.0])
    second = (parameter * x).sum()
    np.testing.assert_allclose(second.data, 6.0)

    disable_computation_cache()
    clear_computation_cache()


def test_non_finite_gradient_diagnostics():
    model = Linear(1, 1)
    model.weight.grad = np.array([[math.nan]])
    model.bias.grad = np.array([math.inf])

    report = model.gradient_report()
    assert report.non_finite_count >= 2


def test_model_xray_after_training_step():
    inputs, targets = _make_regression_dataset()
    model = Linear(1, 1)
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer, criterion.forward)

    trainer.train_step(inputs, targets)
    loss = criterion(model(inputs), targets)
    loss.backward()

    report = model.xray(output=loss)
    assert report.parameter_infos
    assert report.graph_summary is not None
    assert report.gradient_report is not None
    assert report.gradient_report.parameters_inspected > 0


def test_proof_engine_after_training_step():
    inputs, targets = _make_regression_dataset()
    model = Linear(1, 1)
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer, criterion.forward)

    trainer.train_step(inputs, targets)
    loss = criterion(model(inputs), targets)
    loss.backward()

    proof = loss.proof(model.weight)
    assert proof.steps


def test_trainer_diagnostics_mode():
    inputs, targets = _make_regression_dataset()
    model = Linear(1, 1)
    trainer = Trainer(model, SGD(model.parameters(), lr=0.1), MSELoss().forward)
    history = trainer.fit(inputs, targets, epochs=3, diagnostics=True)

    assert history.diagnostics is not None
    assert len(history.diagnostics) == 3


def test_trainer_profile_step_reports_counters():
    inputs, targets = _make_regression_dataset()
    model = Linear(1, 1)
    trainer = Trainer(model, SGD(model.parameters(), lr=0.1), MSELoss().forward)
    profile = trainer.profile_step(inputs, targets)
    payload = profile.as_dict()

    assert payload["graph_nodes"] > 0
    assert payload["dfs_visits"] > 0
    assert payload["loss"] > 0
    assert profile.stages["forward"] >= 0
    assert profile.stages["backward"] >= 0

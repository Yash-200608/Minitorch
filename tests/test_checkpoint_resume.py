import numpy as np

from minitorch import (
    Adam,
    Linear,
    MSELoss,
    SGD,
    Tensor,
    Trainer,
    benchmark_checkpoint,
    load_training_checkpoint,
)


def _dataset(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=(16, 1))
    y = 2.5 * x + 1.0
    return Tensor(x), Tensor(y)


def _copy_adam_state(optimizer, model):
    return {
        "step": optimizer._step,
        "m1": {id(model.weight): optimizer._moment1[id(model.weight)].copy()},
        "m2": {id(model.weight): optimizer._moment2[id(model.weight)].copy()},
    }


def _adam_state_equal(a_state, b_state, a_model, b_model):
    assert a_state["step"] == b_state["step"]
    np.testing.assert_allclose(
        a_state["m1"][id(a_model.weight)],
        b_state["m1"][id(b_model.weight)],
    )
    np.testing.assert_allclose(
        a_state["m2"][id(a_model.weight)],
        b_state["m2"][id(b_model.weight)],
    )


def test_adam_checkpoint_step_parity_after_four_updates():
    """Continuous steps 1-4 must match steps 1-2 + checkpoint + steps 3-4."""
    inputs, targets = _dataset()
    loss_fn = MSELoss().forward

    continuous_model = Linear(1, 1)
    continuous_optimizer = Adam(continuous_model.parameters(), lr=0.05)
    continuous_trainer = Trainer(continuous_model, continuous_optimizer, loss_fn)
    for _ in range(4):
        continuous_trainer.train_step(inputs, targets)
    continuous_state = _copy_adam_state(continuous_optimizer, continuous_model)

    resumed_model = Linear(1, 1)
    resumed_optimizer = Adam(resumed_model.parameters(), lr=0.05)
    resumed_trainer = Trainer(resumed_model, resumed_optimizer, loss_fn)
    for _ in range(2):
        resumed_trainer.train_step(inputs, targets)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "adam.mtc"
        resumed_trainer.save_checkpoint(path)
        restored_model, restored_optimizer, resume_state = load_training_checkpoint(path).restore()
        assert resume_state.completed_steps == 2

        restored_trainer = Trainer(restored_model, restored_optimizer, loss_fn)
        restored_trainer.apply_resume_state(resume_state)
        for _ in range(2):
            restored_trainer.train_step(inputs, targets)

    restored_state = _copy_adam_state(restored_optimizer, restored_model)
    _adam_state_equal(continuous_state, restored_state, continuous_model, restored_model)
    np.testing.assert_allclose(continuous_model.weight.data, restored_model.weight.data)
    assert continuous_trainer.completed_steps == restored_trainer.completed_steps == 4


def test_continuous_vs_checkpoint_resume_training():
    inputs, targets = _dataset(seed=7)
    epochs = 10
    half = epochs // 2
    loss_fn = MSELoss().forward

    continuous_model = Linear(1, 1)
    continuous_trainer = Trainer(
        continuous_model,
        Adam(continuous_model.parameters(), lr=0.05),
        loss_fn,
    )
    continuous_history = continuous_trainer.fit(inputs, targets, epochs=epochs)

    resumed_model = Linear(1, 1)
    resumed_trainer = Trainer(
        resumed_model,
        Adam(resumed_model.parameters(), lr=0.05),
        loss_fn,
    )
    resumed_trainer.fit(inputs, targets, epochs=half)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "resume.mtc"
        resumed_trainer.save_checkpoint(path)
        restored_trainer = Trainer.from_checkpoint(path, loss_fn)
        restored_history = restored_trainer.fit(inputs, targets, epochs=half, resume=True)

    np.testing.assert_allclose(
        continuous_model.weight.data,
        restored_trainer.model.weight.data,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        continuous_history.losses[-1],
        restored_history.losses[-1],
        rtol=1e-10,
        atol=1e-10,
    )
    assert restored_trainer.completed_epochs == epochs
    assert restored_trainer.completed_steps == epochs


def test_sgd_momentum_resume_equivalence():
    inputs, targets = _dataset(seed=3)
    loss_fn = MSELoss().forward
    epochs = 8
    half = epochs // 2

    continuous_model = Linear(1, 1)
    continuous_optimizer = SGD(continuous_model.parameters(), lr=0.1, momentum=0.9)
    continuous_trainer = Trainer(continuous_model, continuous_optimizer, loss_fn)
    continuous_trainer.fit(inputs, targets, epochs=epochs)

    split_model = Linear(1, 1)
    split_optimizer = SGD(split_model.parameters(), lr=0.1, momentum=0.9)
    split_trainer = Trainer(split_model, split_optimizer, loss_fn)
    split_trainer.fit(inputs, targets, epochs=half)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sgd.mtc"
        split_trainer.save_checkpoint(path)
        restored_trainer = Trainer.from_checkpoint(path, loss_fn)
        restored_trainer.fit(inputs, targets, epochs=half, resume=True)

    np.testing.assert_allclose(continuous_model.weight.data, restored_trainer.model.weight.data)
    np.testing.assert_allclose(
        continuous_optimizer._velocities[id(continuous_model.weight)],
        restored_trainer.optimizer._velocities[id(restored_trainer.model.weight)],
    )


def test_checkpoint_size_overhead_report():
    inputs, targets = _dataset()
    model = Linear(1, 1)
    optimizer = Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, optimizer, MSELoss().forward)
    trainer.fit(inputs, targets, epochs=5)

    results = benchmark_checkpoint(model, optimizer, trainer.resume_state())
    assert results["checkpoint_file_bytes"] > results["model_file_bytes"]
    assert results["checkpoint_overhead_ratio"] > 1.0

    param_bytes = int(model.weight.data.nbytes + model.bias.data.nbytes)
    optimizer_bytes = (
        optimizer._moment1[id(model.weight)].nbytes
        + optimizer._moment2[id(model.weight)].nbytes
        + optimizer._moment1[id(model.bias)].nbytes
        + optimizer._moment2[id(model.bias)].nbytes
    )
    assert optimizer_bytes >= param_bytes

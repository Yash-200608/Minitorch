import numpy as np

from minitorch import Adam, Linear, MSELoss, QATConfig, Tensor, Trainer


def _dataset():
    x = np.linspace(0.0, 1.0, 8).reshape(8, 1)
    y = 2.0 * x + 1.0
    return Tensor(x), Tensor(y)


def test_qat_checkpoint_resume(tmp_path):
    inputs, targets = _dataset()
    loss_fn = MSELoss().forward
    epochs = 8
    half = epochs // 2

    base = Linear(1, 1)
    continuous = Trainer(
        base,
        Adam(base.parameters(), lr=0.05),
        loss_fn,
        qat_config=QATConfig(bits=4),
    )
    continuous.fit(inputs, targets, epochs=epochs)

    split_base = Linear(1, 1)
    split = Trainer(
        split_base,
        Adam(split_base.parameters(), lr=0.05),
        loss_fn,
        qat_config=QATConfig(bits=4),
    )
    split.fit(inputs, targets, epochs=half)

    path = tmp_path / "qat.mtc"
    split.save_checkpoint(path)
    restored = Trainer.from_checkpoint(path, loss_fn)
    from minitorch.qat import is_qat_prepared

    assert is_qat_prepared(restored.model)
    restored.fit(inputs, targets, epochs=half, resume=True)

    np.testing.assert_allclose(
        continuous.model.weight.data,
        restored.model.weight.data,
        rtol=1e-9,
        atol=1e-9,
    )
    assert continuous.completed_steps == restored.completed_steps == epochs


def test_qat_checkpoint_metadata(tmp_path):
    base = Linear(1, 1)
    trainer = Trainer(
        base,
        Adam(base.parameters(), lr=0.05),
        MSELoss().forward,
        qat_config=QATConfig(bits=4, ste_mode="identity"),
    )
    trainer.fit(*_dataset(), epochs=2)
    path = tmp_path / "qat.mtc"
    trainer.save_checkpoint(path)

    from minitorch import inspect_artifact

    info = inspect_artifact(path)
    assert info["metadata"]["qat_config"]["bits"] == 4
    assert info["metadata"]["qat_config"]["ste_mode"] == "identity"

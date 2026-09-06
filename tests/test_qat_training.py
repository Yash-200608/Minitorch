import numpy as np

from minitorch import Linear, MSELoss, QATConfig, SGD, Tensor, Trainer


def _dataset():
    x = np.linspace(0.0, 1.0, 16).reshape(16, 1)
    y = 3.0 * x + 2.0
    return Tensor(x), Tensor(y)


def test_qat_training_reduces_loss():
    inputs, targets = _dataset()
    model = Linear(1, 1)
    trainer = Trainer(
        model,
        SGD(model.parameters(), lr=0.1),
        MSELoss().forward,
        qat_config=QATConfig(bits=4),
    )
    history = trainer.fit(inputs, targets, epochs=150)
    assert history.losses[-1] < history.losses[0]


def test_normal_training_regression_unchanged():
    inputs, targets = _dataset()
    model = Linear(1, 1)
    trainer = Trainer(model, SGD(model.parameters(), lr=0.1), MSELoss().forward)
    history = trainer.fit(inputs, targets, epochs=5)
    assert len(history.losses) == 5


def test_qat_vs_post_training_comparison_runs():
    from minitorch.qat import run_qat_experiment

    model = Linear(1, 1)
    inputs, targets = _dataset()
    result = run_qat_experiment(
        model,
        lambda m: SGD(m.parameters(), lr=0.1),
        MSELoss().forward,
        inputs,
        targets,
        epochs=80,
        bits=4,
    )
    assert result.normal_fp_loss >= 0
    assert result.qat_fp_loss >= 0
    assert "QAT" in result.result_label
    report = result.format()
    assert "QUANTIZATION DAMAGE" in report

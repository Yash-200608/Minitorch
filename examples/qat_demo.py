"""Quantization-aware training demo: normal vs QAT vs post-training quantization."""

import numpy as np

from minitorch import (
    Adam,
    Linear,
    MSELoss,
    QATConfig,
    SGD,
    Tensor,
    Trainer,
    run_qat_experiment,
)


def main() -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 32).reshape(32, 1)
    y = 3.0 * x + 2.0
    inputs, targets = Tensor(x), Tensor(y)

    result = run_qat_experiment(
        Linear(1, 1),
        lambda model: Adam(model.parameters(), lr=0.05),
        MSELoss().forward,
        inputs,
        targets,
        epochs=120,
        bits=4,
    )
    print(result.format())
    print()

    qat_model = Trainer(
        Linear(1, 1),
        SGD(Linear(1, 1).parameters(), lr=0.1),
        MSELoss().forward,
        qat_config=QATConfig(bits=4),
    ).model
    print(qat_model.xray())


if __name__ == "__main__":
    main()

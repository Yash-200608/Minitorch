"""Mixed-precision post-training compression experiment."""

import numpy as np

from minitorch import (
    Linear,
    MSELoss,
    MixedPrecisionConfig,
    Sequential,
    SGD,
    Tensor,
    Trainer,
    benchmark_mixed_precision,
    compare_mixed_precision,
)


def main() -> None:
    model = Sequential(Linear(2, 4), Linear(4, 1))
    inputs = Tensor(np.linspace(0, 1, 32).reshape(16, 2))
    targets = Tensor(3.0 * inputs.data[:, :1] + 2.0)

    trainer = Trainer(model, SGD(model.parameters(), lr=0.1), MSELoss().forward)
    trainer.fit(inputs, targets, epochs=60)

    uniform_4 = MixedPrecisionConfig(default_bits=4)
    mixed = MixedPrecisionConfig(
        default_bits=8,
        parameter_bits={"0.weight": 8, "1.weight": 4},
    )

    comparison = compare_mixed_precision(
        model,
        inputs,
        targets,
        mixed_config=mixed,
        global_bits=(8, 4, 2),
    )
    print(comparison.format_table())
    print()

    bench = benchmark_mixed_precision(model, mixed, inputs=inputs, targets=targets)
    print("Benchmark:")
    for key, value in bench.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

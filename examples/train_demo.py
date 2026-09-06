"""Train a tiny linear regression model: y = 3x + 2."""

import numpy as np

from minitorch import Linear, MSELoss, SGD, Tensor, Trainer


def main() -> None:
    rng = np.random.default_rng(0)
    x_values = rng.uniform(0.0, 1.0, size=(16, 1))
    y_values = 3.0 * x_values + 2.0

    inputs = Tensor(x_values)
    targets = Tensor(y_values)

    model = Linear(1, 1)
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer, criterion.forward)

    initial_loss = float(criterion(model(inputs), targets).data)
    history = trainer.fit(inputs, targets, epochs=300, diagnostics=True)
    final_loss = history.losses[-1]

    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print(f"Learned weight: {model.weight.data[0, 0]:.4f}")
    print(f"Learned bias:   {model.bias.data[0]:.4f}")

    gradient_report = model.gradient_report()
    print(f"Gradient health entries: {gradient_report.parameters_inspected}")


if __name__ == "__main__":
    main()

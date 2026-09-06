"""Demonstrate AI-MRI unified Glass Box diagnostics."""

import numpy as np

from minitorch import (
    CompressionConfig,
    Linear,
    MSELoss,
    QATConfig,
    SGD,
    Tensor,
    Trainer,
    compare_model_compression,
)


def main() -> None:
    model = Linear(2, 1)
    values = np.linspace(0, 1, 32)
    inputs = Tensor(values.reshape(16, 2))
    targets = Tensor(3.0 * inputs.data[:, :1] + 2.0)

    trainer = Trainer(model, SGD(model.parameters(), lr=0.1), MSELoss().forward)
    trainer.train_step(inputs, targets)
    loss = MSELoss().forward(model(inputs), targets)
    loss.backward()

    profile = trainer.profile_step(inputs, targets)
    compression = compare_model_compression(
        model,
        inputs,
        targets,
        config=CompressionConfig(bits=4, method="uniform"),
    )

    mri = model.mri(
        output=loss,
        performance=profile,
        compression=compression,
        training_state={
            "completed_epochs": trainer.completed_epochs,
            "completed_steps": trainer.completed_steps,
        },
    )

    print(mri)
    print()
    print("JSON keys:", sorted(mri.to_dict().keys()))
    print()
    proof = mri.prove(model.weight)
    print(f"Proof paths to weight: {proof.path_count}")


if __name__ == "__main__":
    main()

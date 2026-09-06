"""Demonstrate training checkpoint save, load, and resume equivalence."""

from pathlib import Path

import numpy as np

from minitorch import (
    Adam,
    Linear,
    MSELoss,
    Tensor,
    Trainer,
    format_checkpoint_report,
    load_training_checkpoint,
)


def main() -> None:
    rng = np.random.default_rng(42)
    x = rng.uniform(0.0, 1.0, size=(32, 1))
    y = 2.0 * x + 0.5
    inputs, targets = Tensor(x), Tensor(y)
    loss_fn = MSELoss().forward
    total_epochs = 20
    checkpoint_epoch = total_epochs // 2

    continuous = Linear(1, 1)
    continuous_trainer = Trainer(continuous, Adam(continuous.parameters(), lr=0.05), loss_fn)
    continuous_history = continuous_trainer.fit(inputs, targets, epochs=total_epochs)

    split = Linear(1, 1)
    split_trainer = Trainer(split, Adam(split.parameters(), lr=0.05), loss_fn)
    split_trainer.fit(inputs, targets, epochs=checkpoint_epoch)
    pre_resume_loss = split_trainer._loss_history[-1]

    checkpoint_path = Path("checkpoint_resume_demo.mtc")
    split_trainer.save_checkpoint(checkpoint_path)
    checkpoint = load_training_checkpoint(checkpoint_path)
    print(format_checkpoint_report(str(checkpoint_path), checkpoint.metadata, len(checkpoint.payload)))
    print()

    print(f"Checkpoint epoch: {checkpoint.resume_state.completed_epochs}")
    print(f"Checkpoint step: {checkpoint.resume_state.completed_steps}")
    print(f"Pre-resume loss: {pre_resume_loss:.6f}")

    resumed_trainer = Trainer.from_checkpoint(str(checkpoint_path), loss_fn)
    resumed_history = resumed_trainer.fit(inputs, targets, epochs=checkpoint_epoch, resume=True)
    print(f"Post-resume loss: {resumed_history.losses[-1]:.6f}")
    print(f"Continuous final loss: {continuous_history.losses[-1]:.6f}")

    np.testing.assert_allclose(
        continuous.weight.data,
        resumed_trainer.model.weight.data,
        rtol=1e-9,
        atol=1e-9,
    )
    print("Final parameters match uninterrupted training.")


if __name__ == "__main__":
    main()

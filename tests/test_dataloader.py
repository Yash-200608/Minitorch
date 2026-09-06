import numpy as np

from minitorch import DataLoader, Linear, MSELoss, Parameter, SGD, Tensor, TensorDataset, Trainer


def test_dataloader_batches_deterministic():
    inputs = Tensor(np.arange(12.0).reshape(6, 2))
    targets = Tensor(np.arange(6.0))
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    batches = list(loader)
    assert len(batches) == 3
    batch_inputs, batch_targets = batches[0]
    assert batch_inputs.shape == (2, 2)
    assert batch_targets.shape == (2,)


def test_dataloader_shuffle_with_seed():
    inputs = Tensor(np.arange(10.0).reshape(5, 2))
    targets = Tensor(np.arange(5.0))
    dataset = TensorDataset(inputs, targets)

    first = [batch[0].data[0, 0] for batch in DataLoader(dataset, batch_size=1, shuffle=True, seed=7)]
    second = [batch[0].data[0, 0] for batch in DataLoader(dataset, batch_size=1, shuffle=True, seed=7)]
    np.testing.assert_allclose(first, second)

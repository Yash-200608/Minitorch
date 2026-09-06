from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

from .tensor import Tensor


class Dataset:
    """Simple indexable dataset interface."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class TensorDataset(Dataset):
    """Pair tensors by row index."""

    def __init__(self, inputs: Tensor, targets: Tensor) -> None:
        self.inputs = inputs
        self.targets = targets
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("Input and target batch dimensions must match.")

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return (
            Tensor(self.inputs.data[index], requires_grad=self.inputs.requires_grad),
            Tensor(self.targets.data[index], requires_grad=self.targets.requires_grad),
        )


class DataLoader:
    """Deterministic batch iterator with optional shuffle."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            inputs: list[Tensor] = []
            targets: list[Tensor] = []
            for index in batch_indices:
                input_row, target_row = self.dataset[index]
                inputs.append(input_row)
                targets.append(target_row)
            yield _stack_rows(inputs), _stack_rows(targets)


def _stack_rows(rows: Sequence[Tensor]) -> Tensor:
    data = np.stack([row.data for row in rows], axis=0)
    return Tensor(data, requires_grad=any(row.requires_grad for row in rows))

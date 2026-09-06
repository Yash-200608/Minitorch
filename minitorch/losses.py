from __future__ import annotations

from .nn import Module
from .tensor import Tensor


class MSELoss(Module):
    """Mean squared error: mean((prediction - target)^2)."""

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        target_tensor = Tensor._ensure_tensor(target)
        diff = prediction - target_tensor
        return (diff * diff).mean()

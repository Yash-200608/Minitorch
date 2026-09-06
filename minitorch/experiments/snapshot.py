from __future__ import annotations

import numpy as np

from ..compression import clone_module_structure
from ..nn import Module


def snapshot_model_state(model: Module) -> dict[str, np.ndarray]:
    return {name: parameter.data.copy() for name, parameter in model.named_parameters()}


def restore_model_state(model: Module, snapshot: dict[str, np.ndarray]) -> None:
    for name, parameter in model.named_parameters():
        if name not in snapshot:
            raise KeyError(f"Snapshot missing parameter {name!r}.")
        parameter.data = snapshot[name].copy()


def clone_model_with_state(model: Module) -> Module:
    clone = clone_module_structure(model)
    restore_model_state(clone, snapshot_model_state(model))
    return clone

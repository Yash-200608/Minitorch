from __future__ import annotations

from typing import Any

from ..nn import Linear, Module, ReLU, Sequential, Sigmoid, Tanh
from .format import ArtifactError


def serialize_module_tree(module: Module) -> dict[str, Any]:
    if isinstance(module, Sequential):
        return {
            "type": "Sequential",
            "layers": [
                serialize_module_tree(module._modules[name])
                for name in module._layer_order
            ],
        }

    if isinstance(module, Linear):
        weight_shape = module.weight.shape
        return {
            "type": "Linear",
            "in_features": int(weight_shape[1]),
            "out_features": int(weight_shape[0]),
        }

    if isinstance(module, ReLU):
        return {"type": "ReLU"}
    if isinstance(module, Sigmoid):
        return {"type": "Sigmoid"}
    if isinstance(module, Tanh):
        return {"type": "Tanh"}

    raise TypeError(
        f"Cannot serialize module type {type(module).__name__}. "
        "Supported: Sequential, Linear, ReLU, Sigmoid, Tanh."
    )


def build_module_from_tree(spec: dict[str, Any]) -> Module:
    module_type = spec.get("type")
    if module_type == "Sequential":
        children = [build_module_from_tree(child) for child in spec.get("layers", [])]
        return Sequential(*children)

    if module_type == "Linear":
        return Linear(int(spec["in_features"]), int(spec["out_features"]))

    if module_type == "ReLU":
        return ReLU()
    if module_type == "Sigmoid":
        return Sigmoid()
    if module_type == "Tanh":
        return Tanh()

    raise ArtifactError(f"Unknown module type in artifact: {module_type!r}")

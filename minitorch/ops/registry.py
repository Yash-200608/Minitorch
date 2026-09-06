from __future__ import annotations

from .specs import OperationSpec


class OperationRegistry:
    """HashMap registry of canonical operation specifications."""

    def __init__(self) -> None:
        self._specs: dict[str, OperationSpec] = {}

    def register(self, spec: OperationSpec, *, allow_replace: bool = False) -> None:
        if spec.name in self._specs and not allow_replace:
            raise KeyError(f"Operation {spec.name!r} is already registered.")
        self._specs[spec.name] = spec

    def get(self, name: str) -> OperationSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown operation {name!r}.") from exc

    def contains(self, name: str) -> bool:
        return name in self._specs

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs.keys())

    def all_specs(self) -> tuple[OperationSpec, ...]:
        return tuple(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

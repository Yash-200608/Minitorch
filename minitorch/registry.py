from __future__ import annotations

from typing import Any, Iterator

from .tensor import Tensor

_PARAMETER_MARKER = "_is_minitorch_parameter"


def _is_parameter(tensor: Tensor) -> bool:
    return getattr(tensor, _PARAMETER_MARKER, False)


class ParameterRegistry:
    """
    HashMap registry mapping human-readable names to Parameter tensors.

    This separates stable diagnostic names from internal tensor identities and is
    the lookup layer for `model.parameter("layer1.weight")`.
    """

    def __init__(self) -> None:
        self._name_to_param: dict[str, Tensor] = {}

    def register(self, name: str, parameter: Tensor) -> None:
        if not _is_parameter(parameter):
            raise TypeError("ParameterRegistry only stores Parameter instances.")
        if name in self._name_to_param:
            raise KeyError(f"Duplicate parameter name: {name!r}.")
        self._name_to_param[name] = parameter
        parameter._registry_name = name

    def get(self, name: str) -> Tensor:
        try:
            return self._name_to_param[name]
        except KeyError as exc:
            raise KeyError(f"Parameter {name!r} is not registered.") from exc

    def parameter(self, name: str) -> Tensor:
        """Return the Parameter registered under `name`."""
        return self.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(self._name_to_param.keys())

    def as_dict(self) -> dict[str, Tensor]:
        return dict(self._name_to_param)

    def __len__(self) -> int:
        return len(self._name_to_param)

    def __contains__(self, name: str) -> bool:
        return name in self._name_to_param

    def items(self) -> Iterator[tuple[str, Tensor]]:
        return iter(self._name_to_param.items())

    def clear(self) -> None:
        for parameter in self._name_to_param.values():
            parameter._registry_name = None
        self._name_to_param.clear()

    def summary(self) -> dict[str, Any]:
        return {
            "count": len(self._name_to_param),
            "names": self.names(),
        }

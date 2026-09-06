from __future__ import annotations

from typing import Literal

NumericalBackend = Literal["reference", "vectorized"]

DEFAULT_PACKING_BACKEND: NumericalBackend = "reference"
DEFAULT_CODEBOOK_BACKEND: NumericalBackend = "vectorized"
DEFAULT_HADAMARD_BACKEND: NumericalBackend = "reference"


def resolve_backend(
    backend: NumericalBackend | None,
    default: NumericalBackend,
) -> NumericalBackend:
    return backend if backend is not None else default

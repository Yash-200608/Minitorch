from __future__ import annotations

import math

import numpy as np

from .backend import NumericalBackend


def uniform_codes_reference(
    values: np.ndarray,
    *,
    minimum: float,
    maximum: float,
    max_code: int,
) -> np.ndarray:
    """Scalar reference uniform quantization mapping."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    if minimum == maximum:
        return np.zeros(flat.size, dtype=np.int32)

    scale = (maximum - minimum) / max_code
    codes = np.empty(flat.size, dtype=np.int32)
    for index, value in enumerate(flat):
        rounded = round((float(value) - minimum) / scale)
        if rounded < 0:
            rounded = 0
        elif rounded > max_code:
            rounded = max_code
        codes[index] = int(rounded)
    return codes


def uniform_codes_vectorized(
    values: np.ndarray,
    *,
    minimum: float,
    maximum: float,
    max_code: int,
) -> np.ndarray:
    """NumPy vectorized uniform quantization mapping."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    if minimum == maximum:
        return np.zeros(flat.size, dtype=np.int32)

    scale = (maximum - minimum) / max_code
    codes = np.round((flat - minimum) / scale)
    return np.clip(codes, 0, max_code).astype(np.int32)


def uniform_codes(
    values: np.ndarray,
    *,
    minimum: float,
    maximum: float,
    max_code: int,
    backend: NumericalBackend = "vectorized",
) -> np.ndarray:
    if backend == "reference":
        return uniform_codes_reference(
            values,
            minimum=minimum,
            maximum=maximum,
            max_code=max_code,
        )
    return uniform_codes_vectorized(
        values,
        minimum=minimum,
        maximum=maximum,
        max_code=max_code,
    )

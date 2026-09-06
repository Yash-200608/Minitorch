from __future__ import annotations

import math

import numpy as np

from .backend import NumericalBackend


def walsh_hadamard_reference(values: np.ndarray) -> np.ndarray:
    """Educational O(n log n) Walsh-Hadamard with per-element Python butterflies."""
    vector = np.asarray(values, dtype=np.float64)
    length = vector.shape[0]
    if length == 0 or (length & (length - 1)) != 0:
        raise ValueError(f"Hadamard length must be a positive power of two, got {length}.")

    result = vector.copy()
    stride = 1
    while stride < length:
        for start in range(0, length, stride * 2):
            for index in range(start, start + stride):
                left = result[index]
                right = result[index + stride]
                result[index] = left + right
                result[index + stride] = left - right
        stride *= 2
    return result / math.sqrt(length)


def walsh_hadamard_vectorized(values: np.ndarray) -> np.ndarray:
    """
    Vectorized Walsh-Hadamard: butterfly stages use NumPy slice operations.

    Algorithm remains O(n log n) per vector; inner index loops are replaced
  with vectorized slice updates over each butterfly block.
    """
    vector = np.asarray(values, dtype=np.float64)
    length = vector.shape[0]
    if length == 0 or (length & (length - 1)) != 0:
        raise ValueError(f"Hadamard length must be a positive power of two, got {length}.")

    result = vector.copy()
    stride = 1
    while stride < length:
        for start in range(0, length, stride * 2):
            left = result[start:start + stride]
            right = result[start + stride:start + 2 * stride]
            new_left = left + right
            new_right = left - right
            result[start:start + stride] = new_left
            result[start + stride:start + 2 * stride] = new_right
        stride *= 2
    return result / math.sqrt(length)


def walsh_hadamard(
    values: np.ndarray,
    *,
    backend: NumericalBackend = "reference",
) -> np.ndarray:
    if backend == "vectorized":
        return walsh_hadamard_vectorized(values)
    return walsh_hadamard_reference(values)

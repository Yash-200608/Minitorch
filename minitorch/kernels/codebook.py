from __future__ import annotations

import numpy as np

from .backend import NumericalBackend

DEFAULT_ENCODE_CHUNK_SIZE = 65536


def nearest_centroid_codes_reference(
    values: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Scalar nearest-centroid assignment (educational reference)."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    centroid_array = np.asarray(centroids, dtype=np.float64)
    codes = np.empty(flat.size, dtype=np.int64)
    for index, value in enumerate(flat):
        distances = np.abs(centroid_array - float(value))
        codes[index] = int(np.argmin(distances))
    return codes


def nearest_centroid_codes_vectorized(
    values: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Vectorized nearest-centroid assignment via broadcast distance matrix."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    centroid_array = np.asarray(centroids, dtype=np.float64)
    distances = np.abs(flat[:, None] - centroid_array[None, :])
    return np.argmin(distances, axis=1)


def nearest_centroid_codes_vectorized_chunked(
    values: np.ndarray,
    centroids: np.ndarray,
    *,
    chunk_size: int = DEFAULT_ENCODE_CHUNK_SIZE,
) -> np.ndarray:
    """Memory-aware chunked vectorized nearest-centroid assignment."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    centroid_array = np.asarray(centroids, dtype=np.float64)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if flat.size == 0:
        return np.array([], dtype=np.int64)

    chunks: list[np.ndarray] = []
    for start in range(0, flat.size, chunk_size):
        end = min(start + chunk_size, flat.size)
        chunk_codes = nearest_centroid_codes_vectorized(flat[start:end], centroid_array)
        chunks.append(chunk_codes)
    return np.concatenate(chunks)


def nearest_centroid_codes(
    values: np.ndarray,
    centroids: np.ndarray,
    *,
    backend: NumericalBackend = "vectorized",
    chunk_size: int | None = None,
) -> np.ndarray:
    if backend == "reference":
        return nearest_centroid_codes_reference(values, centroids)
    if chunk_size is not None:
        return nearest_centroid_codes_vectorized_chunked(
            values,
            centroids,
            chunk_size=chunk_size,
        )
    return nearest_centroid_codes_vectorized(values, centroids)

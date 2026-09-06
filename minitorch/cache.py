from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .tensor import Tensor


@dataclass
class CacheStats:
    """Instrumentation for one cache-enabled execution window."""

    hits: int = 0
    misses: int = 0
    stores: int = 0
    lookups: int = 0

    @property
    def size(self) -> int:
        return self.hits + self.misses

    def hit_rate(self) -> float:
        if self.lookups == 0:
            return 0.0
        return self.hits / self.lookups


@dataclass
class ComputationCache:
    """
    HashMap memoization for deterministic tensor forward numpy work.

  Keys hash operation name, parent tensor payloads, and metadata. Cached values
    are NumPy arrays; each Tensor still records its own graph parents on lookup so
    autograd lineage stays correct while redundant numeric work is skipped.
    """

    enabled: bool = False
    _store: dict[tuple[Any, ...], np.ndarray] = field(default_factory=dict)
    stats: CacheStats = field(default_factory=CacheStats)

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def clear(self) -> None:
        self._store.clear()
        self.stats = CacheStats()

    def make_key(
        self,
        operation: str,
        parents: tuple[Tensor, ...],
        metadata: dict[str, Any],
    ) -> tuple[Any, ...]:
        parent_fingerprints = tuple(
            (parent.shape, parent.data.tobytes()) for parent in parents
        )
        metadata_items = tuple(sorted(_hashable_metadata(metadata).items()))
        return (operation, parent_fingerprints, metadata_items)

    def get(self, key: tuple[Any, ...]) -> np.ndarray | None:
        self.stats.lookups += 1
        cached = self._store.get(key)
        if cached is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return cached

    def store(self, key: tuple[Any, ...], value: np.ndarray) -> None:
        self._store[key] = np.asarray(value, dtype=np.float64).copy()
        self.stats.stores += 1

    def get_or_compute(
        self,
        key: tuple[Any, ...],
        compute_fn: Callable[[], np.ndarray],
    ) -> np.ndarray:
        if not self.enabled:
            return np.asarray(compute_fn(), dtype=np.float64)

        cached = self.get(key)
        if cached is not None:
            return cached.copy()

        result = np.asarray(compute_fn(), dtype=np.float64)
        self.store(key, result)
        return result.copy()

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "entries": len(self._store),
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "lookups": self.stats.lookups,
            "hit_rate": self.stats.hit_rate(),
        }


_GLOBAL_CACHE = ComputationCache()


def get_computation_cache() -> ComputationCache:
    return _GLOBAL_CACHE


def enable_computation_cache() -> ComputationCache:
    cache = get_computation_cache()
    cache.enable()
    return cache


def disable_computation_cache() -> ComputationCache:
    cache = get_computation_cache()
    cache.disable()
    return cache


def clear_computation_cache() -> None:
    get_computation_cache().clear()


def _hashable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    hashed: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, tuple):
            hashed[key] = value
        elif isinstance(value, (int, float, str, bool)) or value is None:
            hashed[key] = value
        else:
            hashed[key] = repr(value)
    return hashed

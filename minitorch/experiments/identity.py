from __future__ import annotations

import hashlib
import json
from typing import Any

from ..persistence.format import FRAMEWORK_VERSION
from .config import ExperimentConfig

EXPERIMENT_SCHEMA_VERSION = "1.0"


def _normalize_for_identity(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: _normalize_for_identity(data[key]) for key in sorted(data)}
    if isinstance(data, list):
        return [_normalize_for_identity(item) for item in data]
    return data


def compute_experiment_id(config: ExperimentConfig) -> str:
    """Deterministic experiment identity from normalized configuration."""
    payload = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "framework_version": FRAMEWORK_VERSION,
        "configuration": _normalize_for_identity(config.to_dict()),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:16]

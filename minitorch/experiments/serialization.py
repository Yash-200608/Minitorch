from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .identity import EXPERIMENT_SCHEMA_VERSION
from .result import ExperimentResult, validate_experiment_result


class ExperimentResultError(Exception):
    """Raised when an experiment result file cannot be parsed safely."""


def save_experiment_result(result: ExperimentResult, path: str | Path) -> None:
    validate_experiment_result(result)
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    target.write_text(payload, encoding="utf-8")


def load_experiment_result(path: str | Path) -> ExperimentResult:
    target = Path(path)
    if not target.exists():
        raise ExperimentResultError(f"Experiment result not found: {target}")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentResultError(f"Invalid JSON in experiment result: {exc}") from exc

    if not isinstance(data, dict):
        raise ExperimentResultError("Experiment result root must be a JSON object.")

    schema = data.get("schema_version")
    if schema is None:
        raise ExperimentResultError("Missing schema_version in experiment result.")
    if schema != EXPERIMENT_SCHEMA_VERSION:
        raise ExperimentResultError(f"Unsupported experiment schema version: {schema}")

    experiment_id = data.get("experiment_id")
    if not experiment_id or not isinstance(experiment_id, str):
        raise ExperimentResultError("Missing or invalid experiment_id.")

    metrics = data.get("metrics")
    if metrics is not None and not isinstance(metrics, dict):
        raise ExperimentResultError("metrics must be an object when present.")

    configuration = data.get("configuration")
    if configuration is not None and not isinstance(configuration, dict):
        raise ExperimentResultError("configuration must be an object when present.")

    result = ExperimentResult.from_dict(data)
    validate_experiment_result(result)
    return result

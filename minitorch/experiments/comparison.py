from __future__ import annotations

from typing import Any

from .result import ExperimentResult

_COMPARISON_COLUMNS: tuple[tuple[str, str], ...] = (
    ("configuration", "technique_label"),
    ("model_bytes", "metrics.model_bytes"),
    ("compressed_bytes", "metrics.compressed_bytes"),
    ("compression_ratio", "metrics.compression_ratio"),
    ("parameter_rmse", "metrics.parameter_rmse"),
    ("output_rmse", "metrics.output_rmse"),
    ("fp_loss", "metrics.fp_final_loss"),
    ("loss_delta", "metrics.loss_delta"),
    ("training_time", "timings.training_seconds"),
    ("compression_time", "timings.compression_seconds"),
)


def _resolve_metric(result: ExperimentResult, path: str) -> Any:
    root, attr = path.split(".", 1)
    if root == "metrics":
        return getattr(result.metrics, attr, None)
    if root == "timings":
        return getattr(result.timings, attr, None)
    if root == "memory":
        return getattr(result.memory, attr, None)
    return getattr(result, attr, None)


def compare_experiments(results: tuple[ExperimentResult, ...] | list[ExperimentResult]) -> dict[str, Any]:
    """Align experiment results by metric names; missing values remain None."""
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "experiment_id": result.experiment_id,
            "name": result.name,
            "technique_label": result.technique_label,
        }
        for column_name, path in _COMPARISON_COLUMNS:
            if column_name == "configuration":
                continue
            row[column_name] = _resolve_metric(result, path)
        rows.append(row)
    return {
        "columns": [name for name, _ in _COMPARISON_COLUMNS],
        "rows": rows,
    }


def format_experiment_comparison_table(
    results: tuple[ExperimentResult, ...] | list[ExperimentResult],
) -> str:
    comparison = compare_experiments(results)
    columns = comparison["columns"]
    header = "  ".join(columns)
    lines = [header, "-" * len(header)]
    for row in comparison["rows"]:
        values: list[str] = []
        for column in columns:
            if column == "configuration":
                values.append(str(row.get("technique_label", "")))
            else:
                value = row.get(column)
                if value is None:
                    values.append("—")
                elif isinstance(value, float):
                    values.append(f"{value:.6g}")
                else:
                    values.append(str(value))
        lines.append("  ".join(values))
    return "\n".join(lines)

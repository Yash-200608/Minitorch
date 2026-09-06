from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from minitorch.dsa.topk import topk
from minitorch.nn import Module, Parameter


DEFAULT_NEAR_ZERO_THRESHOLD = 1e-7
DEFAULT_LARGE_THRESHOLD = 1e3


@dataclass(frozen=True)
class GradientMagnitude:
    name: str
    magnitude: float
    parameter: Parameter


@dataclass(frozen=True)
class GradientFinding:
    parameter_name: str
    kind: str
    message: str
    magnitude: float | None = None


@dataclass(frozen=True)
class GradientHealthReport:
    parameters_inspected: int
    largest: GradientMagnitude | None
    smallest_nonzero: GradientMagnitude | None
    near_zero_count: int
    missing_count: int
    non_finite_count: int
    findings: tuple[GradientFinding, ...]
    largest_gradients: tuple[GradientMagnitude, ...]
    smallest_gradients: tuple[GradientMagnitude, ...]


def _safe_magnitude(grad: np.ndarray) -> float:
    flat = np.asarray(grad, dtype=np.float64).ravel()
    if flat.size == 0:
        return 0.0
    return float(np.max(np.abs(flat)))


def _is_finite_magnitude(magnitude: float) -> bool:
    return np.isfinite(magnitude)


def detect_gradients(
    model: Module,
    *,
    near_zero_threshold: float = DEFAULT_NEAR_ZERO_THRESHOLD,
    large_threshold: float = DEFAULT_LARGE_THRESHOLD,
    top_k: int = 10,
) -> GradientHealthReport:
    """Analyze real parameter gradients and rank magnitudes with heap Top-K."""

    findings: list[GradientFinding] = []
    finite_entries: list[tuple[str, float, Parameter]] = []
    missing_count = 0
    non_finite_count = 0
    near_zero_count = 0

    largest: GradientMagnitude | None = None
    smallest_nonzero: GradientMagnitude | None = None

    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            missing_count += 1
            findings.append(
                GradientFinding(
                    parameter_name=name,
                    kind="missing",
                    message="Parameter has no gradient after backward.",
                )
            )
            continue

        magnitude = _safe_magnitude(grad)
        if not _is_finite_magnitude(magnitude):
            non_finite_count += 1
            findings.append(
                GradientFinding(
                    parameter_name=name,
                    kind="non_finite",
                    message="Parameter gradient contains non-finite values.",
                    magnitude=magnitude,
                )
            )
            continue

        finite_entries.append((name, magnitude, parameter))

        if largest is None or magnitude > largest.magnitude:
            largest = GradientMagnitude(name=name, magnitude=magnitude, parameter=parameter)

        if magnitude > 0 and (
            smallest_nonzero is None or magnitude < smallest_nonzero.magnitude
        ):
            smallest_nonzero = GradientMagnitude(name=name, magnitude=magnitude, parameter=parameter)

        if magnitude <= near_zero_threshold:
            near_zero_count += 1
            findings.append(
                GradientFinding(
                    parameter_name=name,
                    kind="near_zero",
                    message="Possible near-zero gradient detected.",
                    magnitude=magnitude,
                )
            )
        elif magnitude >= large_threshold:
            findings.append(
                GradientFinding(
                    parameter_name=name,
                    kind="large",
                    message="Possible large gradient detected.",
                    magnitude=magnitude,
                )
            )

    largest_topk = _topk_magnitudes(finite_entries, k=top_k, largest=True)
    smallest_topk = _topk_magnitudes(
        [entry for entry in finite_entries if entry[1] > 0],
        k=top_k,
        largest=False,
    )

    return GradientHealthReport(
        parameters_inspected=len(list(model.named_parameters())),
        largest=largest,
        smallest_nonzero=smallest_nonzero,
        near_zero_count=near_zero_count,
        missing_count=missing_count,
        non_finite_count=non_finite_count,
        findings=tuple(findings),
        largest_gradients=largest_topk,
        smallest_gradients=smallest_topk,
    )


def _topk_magnitudes(
    entries: list[tuple[str, float, Parameter]],
    *,
    k: int,
    largest: bool,
) -> tuple[GradientMagnitude, ...]:
    if not entries or k <= 0:
        return ()

    magnitudes = [entry[1] for entry in entries]
    effective_k = min(k, len(magnitudes))
    result = topk(magnitudes, k=effective_k, largest=largest)

    ranked: list[GradientMagnitude] = []
    for heap_index in result.indices:
        name, magnitude, parameter = entries[heap_index]
        ranked.append(GradientMagnitude(name=name, magnitude=magnitude, parameter=parameter))

    if largest:
        ranked.sort(key=lambda item: item.magnitude, reverse=True)
    else:
        ranked.sort(key=lambda item: item.magnitude)

    return tuple(ranked)


def format_gradient_report(report: GradientHealthReport) -> str:
    lines = [
        "Gradient Health",
        "---------------",
        f"Parameters inspected: {report.parameters_inspected}",
        "",
    ]

    if report.largest is not None:
        lines.append(f"Largest gradient: {report.largest.name} = {report.largest.magnitude}")
    if report.smallest_nonzero is not None:
        lines.append(
            f"Smallest non-zero: {report.smallest_nonzero.name} = "
            f"{report.smallest_nonzero.magnitude}"
        )

    lines.extend(
        [
            "",
            f"Near-zero parameters: {report.near_zero_count}",
            f"Missing gradients: {report.missing_count}",
            f"Non-finite gradients: {report.non_finite_count}",
            "",
            "Possible issues:",
        ]
    )

    issue_findings = [f for f in report.findings if f.kind != "missing"]
    if not issue_findings:
        lines.append("No heuristic issues detected.")
    else:
        for finding in issue_findings:
            prefix = "WARNING"
            lines.append(f"{prefix} {finding.message} ({finding.parameter_name})")

    if report.largest_gradients:
        lines.append("")
        lines.append("Top largest gradient magnitudes:")
        for item in report.largest_gradients:
            lines.append(f"  {item.name}: {item.magnitude}")

    return "\n".join(lines)

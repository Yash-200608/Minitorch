from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

from minitorch.autograd.explorer import GraphExplorer
from minitorch.autograd.proof import GradientProof, build_gradient_proof
from minitorch.nn import Module, Parameter
from minitorch.tensor import Tensor

if TYPE_CHECKING:
    from minitorch.compression import ModelCompressionReport
    from minitorch.experiments.result import ExperimentResult
    from minitorch.profiling.profile import PerformanceProfile
    from minitorch.qat import QATExperimentResult

from .gradient_detective import GradientHealthReport
from .model_xray import build_model_xray, ModelXRayReport


class DataStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


class ObservationSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class Observation:
    severity: ObservationSeverity
    category: str
    message: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class Recommendation:
    message: str
    category: str
    related_observation_categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiagnosticSection:
    status: DataStatus
    reason: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSummaryDiagnostics:
    section: DiagnosticSection


@dataclass(frozen=True)
class GraphDiagnostics:
    section: DiagnosticSection
    explorer: GraphExplorer | None = None


@dataclass(frozen=True)
class GradientDiagnostics:
    section: DiagnosticSection
    report: GradientHealthReport | None = None


@dataclass(frozen=True)
class PerformanceDiagnostics:
    section: DiagnosticSection
    profile: Any | None = None


@dataclass(frozen=True)
class CompressionDiagnostics:
    section: DiagnosticSection


@dataclass(frozen=True)
class QATDiagnostics:
    section: DiagnosticSection


@dataclass(frozen=True)
class TrainingStateDiagnostics:
    section: DiagnosticSection


@dataclass(frozen=True)
class MemoryDiagnostics:
    section: DiagnosticSection


@dataclass
class ModelMRI:
    """Unified Glass Box diagnostic composition over existing MiniTorch evidence."""

    model: Module
    output: Tensor | None
    xray: ModelXRayReport
    model_summary: ModelSummaryDiagnostics
    graph: GraphDiagnostics
    gradients: GradientDiagnostics
    performance: PerformanceDiagnostics
    compression: CompressionDiagnostics
    qat: QATDiagnostics
    training_state: TrainingStateDiagnostics
    memory: MemoryDiagnostics
    observations: tuple[Observation, ...]
    recommendations: tuple[Recommendation, ...]

    def graph_explorer(self) -> GraphExplorer | None:
        return self.graph.explorer

    def gradients_report(self) -> GradientHealthReport | None:
        return self.gradients.report

    def performance_profile(self) -> Any | None:
        return self.performance.profile

    def prove(self, parameter: Parameter | Tensor) -> GradientProof:
        if self.output is None:
            raise ValueError(
                "Proof requires a representative output tensor (forward/backward graph)."
            )
        target = parameter if isinstance(parameter, Parameter) else parameter
        return build_gradient_proof(self.output, target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": _serialize_section(self.model_summary.section),
            "graph": _serialize_section(self.graph.section),
            "gradients": _serialize_section(self.gradients.section),
            "performance": _serialize_section(self.performance.section),
            "compression": _serialize_section(self.compression.section),
            "qat": _serialize_section(self.qat.section),
            "training_state": _serialize_section(self.training_state.section),
            "memory": _serialize_section(self.memory.section),
            "observations": [
                {
                    "severity": obs.severity.value,
                    "category": obs.category,
                    "message": obs.message,
                    "evidence": _json_safe(obs.evidence),
                }
                for obs in self.observations
            ],
            "recommendations": [
                {
                    "message": rec.message,
                    "category": rec.category,
                    "related_observation_categories": list(rec.related_observation_categories),
                }
                for rec in self.recommendations
            ],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def __str__(self) -> str:
        return format_mri_report(self)


def inspect_model_mri(
    model: Module,
    output: Tensor | None = None,
    *,
    performance: Any | None = None,
    compression: Any | None = None,
    qat_experiment: Any | None = None,
    training_state: dict[str, Any] | None = None,
    experiment: Any | None = None,
) -> ModelMRI:
    """
    Build an AI-MRI snapshot from existing structured MiniTorch diagnostics.

    Does not train, compress, quantize, or benchmark unless the caller supplies
    precomputed profile/compression/QAT experiment results.
    """
    xray = build_model_xray(model, output=output)

    model_summary = _build_model_summary(xray)
    graph_diag = _build_graph_diagnostics(output)
    gradient_diag = _build_gradient_diagnostics(xray.gradient_report)
    performance_diag = _build_performance_diagnostics(performance)
    compression_diag = _build_compression_diagnostics(compression)
    qat_diag = _build_qat_diagnostics(xray.qat_info, qat_experiment)
    merged_training_state = _merge_experiment_training_state(training_state, experiment)
    training_diag = _build_training_state_diagnostics(merged_training_state)
    memory_diag = _build_memory_diagnostics(xray, compression)

    observations = _build_observations(
        gradient_diag,
        performance_diag,
        compression_diag,
        qat_diag,
        training_diag,
        experiment=experiment,
    )
    recommendations = _build_recommendations(observations)

    return ModelMRI(
        model=model,
        output=output,
        xray=xray,
        model_summary=model_summary,
        graph=graph_diag,
        gradients=gradient_diag,
        performance=performance_diag,
        compression=compression_diag,
        qat=qat_diag,
        training_state=training_diag,
        memory=memory_diag,
        observations=observations,
        recommendations=recommendations,
    )


def compare_mri(before: ModelMRI, after: ModelMRI) -> dict[str, Any]:
    """Compare two MRI snapshots (observations and key measurements)."""
    before_obs = {obs.category: obs for obs in before.observations}
    after_obs = {obs.category: obs for obs in after.observations}
    categories = sorted(set(before_obs) | set(after_obs))
    observation_changes = []
    for category in categories:
        b = before_obs.get(category)
        a = after_obs.get(category)
        if b is None and a is not None:
            observation_changes.append({"category": category, "change": "added", "after": a.message})
        elif a is None and b is not None:
            observation_changes.append({"category": category, "change": "removed", "before": b.message})
        elif b is not None and a is not None and b.message != a.message:
            observation_changes.append(
                {
                    "category": category,
                    "change": "changed",
                    "before": b.message,
                    "after": a.message,
                }
            )

    def _stage_delta(key: str) -> dict[str, Any] | None:
        b = before.performance.section.data.get(key)
        a = after.performance.section.data.get(key)
        if b is None and a is None:
            return None
        return {"before": b, "after": a}

    return {
        "observation_changes": observation_changes,
        "gradient_near_zero": {
            "before": before.gradients.section.data.get("near_zero_count"),
            "after": after.gradients.section.data.get("near_zero_count"),
        },
        "performance_hottest_stage": _stage_delta("hottest_stage"),
        "qat_enabled": {
            "before": before.qat.section.data.get("enabled"),
            "after": after.qat.section.data.get("enabled"),
        },
    }


def format_mri_report(mri: ModelMRI) -> str:
    lines = [
        "============== MiniTorch AI-MRI ==============",
        "",
        "MODEL",
        f"  Type: {mri.model_summary.section.data.get('module_type')}",
        f"  Parameters: {mri.model_summary.section.data.get('parameter_count')}",
        f"  Trainable values: {mri.model_summary.section.data.get('trainable_numel')}",
        "",
    ]

    if mri.graph.section.status == DataStatus.AVAILABLE:
        g = mri.graph.section.data
        lines.extend(
            [
                "GRAPH",
                f"  Nodes: {g.get('vertices')}",
                f"  Edges: {g.get('edges')}",
                f"  Depth: {g.get('depth')}",
                f"  DFS visits: {g.get('dfs_visits')}",
                "",
            ]
        )
    else:
        lines.extend(["GRAPH", f"  {mri.graph.section.status.value}", f"  {mri.graph.section.reason}", "",])

    if mri.gradients.section.status == DataStatus.AVAILABLE:
        gr = mri.gradients.section.data
        lines.extend(
            [
                "GRADIENTS",
                f"  Parameters inspected: {gr.get('parameters_inspected')}",
                f"  With gradients: {gr.get('with_gradients')}",
                f"  Near-zero: {gr.get('near_zero_count')}",
                f"  Missing: {gr.get('missing_count')}",
                f"  Non-finite: {gr.get('non_finite_count')}",
                "",
            ]
        )
    else:
        lines.extend(["GRADIENTS", f"  {mri.gradients.section.status.value}", "",])

    if mri.performance.section.status == DataStatus.AVAILABLE:
        perf = mri.performance.section.data
        lines.append("PERFORMANCE")
        for stage in ("forward", "backward", "optimizer"):
            if stage in perf:
                lines.append(f"  {stage}: {perf[stage] * 1000:.2f} ms")
        if perf.get("hottest_stage"):
            lines.append(f"  Hottest measured stage: {perf.get('hottest_stage')}")
        lines.append("")
    else:
        lines.extend(["PERFORMANCE", "  Not profiled", "",])

    if mri.compression.section.status == DataStatus.AVAILABLE:
        c = mri.compression.section.data
        if c.get("mixed_precision"):
            lines.extend(["COMPRESSION (mixed precision)", f"  Precision map: {c.get('precision_map')}",])
        else:
            lines.append("COMPRESSION")
        lines.extend(
            [
                f"  Original bytes: {c.get('total_original_bytes')}",
                f"  Compressed bytes: {c.get('total_compressed_bytes')}",
                f"  Ratio: {c.get('compression_ratio')}",
                f"  Output RMSE: {c.get('output_rmse')}",
                "",
            ]
        )
    else:
        lines.extend(["COMPRESSION", "  unavailable", "",])

    if mri.qat.section.status == DataStatus.AVAILABLE:
        q = mri.qat.section.data
        lines.extend(
            [
                "QAT",
                f"  Enabled: {q.get('enabled')}",
                f"  Bits: {q.get('bits')}",
                f"  STE: {q.get('ste_mode')}",
                "",
            ]
        )
    else:
        lines.extend(["QAT", "  not applicable", "",])

    if mri.training_state.section.status == DataStatus.AVAILABLE:
        ts = mri.training_state.section.data
        if ts.get("experiment_id"):
            lines.extend(
                [
                    "EXPERIMENT",
                    f"  ID: {ts.get('experiment_id')}",
                    f"  Name: {ts.get('experiment_name')}",
                    f"  Technique: {ts.get('technique_label')}",
                    f"  Seed: {ts.get('seed')}",
                    "",
                ]
            )

    if mri.observations:
        lines.append("OBSERVATIONS")
        for obs in mri.observations:
            icon = _severity_icon(obs.severity)
            lines.append(f"  {icon} {obs.message}")
        lines.append("")

    if mri.recommendations:
        lines.append("RECOMMENDATIONS")
        for rec in mri.recommendations:
            lines.append(f"  {rec.message}")
        lines.append("")

    lines.append("==============================================")
    return "\n".join(lines)


def _build_model_summary(xray: ModelXRayReport) -> ModelSummaryDiagnostics:
    trainable_numel = sum(p.numel for p in xray.parameter_infos if p.trainable)
    parameters = [
        {
            "name": p.name,
            "shape": list(p.shape),
            "dtype": p.dtype,
            "trainable": p.trainable,
            "memory_bytes": p.memory_bytes,
        }
        for p in xray.parameter_infos
    ]
    data = {
        "module_type": type(xray.model).__name__,
        "architecture": [{"name": n, "type": t} for n, t in xray.architecture],
        "parameter_count": len(xray.parameter_infos),
        "trainable_parameter_count": sum(1 for p in xray.parameter_infos if p.trainable),
        "trainable_numel": trainable_numel,
        "parameters": parameters,
    }
    return ModelSummaryDiagnostics(
        section=DiagnosticSection(status=DataStatus.AVAILABLE, data=data)
    )


def _build_graph_diagnostics(output: Tensor | None) -> GraphDiagnostics:
    if output is None:
        return GraphDiagnostics(
            section=DiagnosticSection(
                status=DataStatus.UNAVAILABLE,
                reason="No representative computational graph supplied.",
            ),
            explorer=None,
        )
    explorer = GraphExplorer.from_tensor(output)
    profile = explorer.dsa_profile()
    summary = explorer.graph.summary()
    operations = summary.get("operations", {})
    op_names = sorted({op for op in operations.values() if op is not None})
    data = {
        "vertices": profile["vertices"],
        "edges": profile["edges"],
        "depth": explorer.depth(),
        "dfs_visits": profile["dfs_visits"],
        "dfs_edge_inspections": profile["dfs_edge_inspections"],
        "topological_processed_nodes": profile["topological_processed_nodes"],
        "topological_order_length": len(explorer.topological_order()),
        "operations": op_names,
        "output_id": output.id,
    }
    return GraphDiagnostics(
        section=DiagnosticSection(status=DataStatus.AVAILABLE, data=data),
        explorer=explorer,
    )


def _build_gradient_diagnostics(report: GradientHealthReport | None) -> GradientDiagnostics:
    if report is None:
        return GradientDiagnostics(
            section=DiagnosticSection(
                status=DataStatus.UNAVAILABLE,
                reason="Gradient report unavailable.",
            ),
            report=None,
        )
    with_gradients = report.parameters_inspected - report.missing_count
    data = {
        "parameters_inspected": report.parameters_inspected,
        "with_gradients": with_gradients,
        "near_zero_count": report.near_zero_count,
        "missing_count": report.missing_count,
        "non_finite_count": report.non_finite_count,
        "largest": _magnitude_entry(report.largest),
        "smallest_nonzero": _magnitude_entry(report.smallest_nonzero),
        "top_largest": [_magnitude_entry(m) for m in report.largest_gradients],
        "top_smallest": [_magnitude_entry(m) for m in report.smallest_gradients],
    }
    return GradientDiagnostics(
        section=DiagnosticSection(status=DataStatus.AVAILABLE, data=data),
        report=report,
    )


def _build_performance_diagnostics(profile: Any | None) -> PerformanceDiagnostics:
    if profile is None:
        return PerformanceDiagnostics(
            section=DiagnosticSection(
                status=DataStatus.UNAVAILABLE,
                reason="No performance profile supplied.",
            ),
            profile=None,
        )
    data: dict[str, Any] = {
        "name": profile.name,
        "mode": profile.mode,
        "stages": dict(profile.stages),
        "hottest_stage": profile.hottest_stage(),
        "top_operations": [
            {"name": name, "seconds": seconds} for name, seconds in profile.top_operations()
        ],
        "graph_stats": dict(profile.graph_stats),
        "dsa_stats": dict(profile.dsa_stats),
    }
    for stage, seconds in profile.stages.items():
        data[stage] = seconds
    return PerformanceDiagnostics(
        section=DiagnosticSection(status=DataStatus.AVAILABLE, data=data),
        profile=profile,
    )


def _build_compression_diagnostics(report: Any | None) -> CompressionDiagnostics:
    if report is None:
        return CompressionDiagnostics(
            section=DiagnosticSection(
                status=DataStatus.UNAVAILABLE,
                reason="No compression experiment supplied.",
            )
        )
    data = {
        "parameter_count": report.parameter_count,
        "total_original_bytes": report.total_original_bytes,
        "total_compressed_bytes": report.total_compressed_bytes,
        "compression_ratio": report.compression_ratio,
        "aggregate_parameter_rmse": report.aggregate_parameter_rmse,
        "output_rmse": report.output_rmse,
        "output_max_absolute_error": getattr(report, "output_max_absolute_error", None),
        "original_loss": report.original_loss,
        "compressed_loss": report.compressed_loss,
        "loss_delta": report.loss_delta,
        "top_parameter_errors": [
            {
                "name": rec.name,
                "rmse": rec.rmse,
                "max_absolute_error": rec.max_absolute_error,
            }
            for rec in report.largest_parameter_errors(k=5)
        ],
    }
    mixed_config = getattr(report, "mixed_config", None)
    if mixed_config is not None:
        data["mixed_precision"] = True
        data["precision_map"] = dict(getattr(report, "precision_map", {}))
        data["configuration"] = mixed_config.to_dict()
    else:
        config = report.config
        data["configuration"] = {
            "bits": config.bits,
            "method": config.method,
            "packed": config.packed,
            "rotation": config.rotation,
        }
    return CompressionDiagnostics(section=DiagnosticSection(status=DataStatus.AVAILABLE, data=data))


def _build_qat_diagnostics(
    qat_info: dict[str, Any] | None,
    experiment: Any | None,
) -> QATDiagnostics:
    if qat_info is None and experiment is None:
        return QATDiagnostics(
            section=DiagnosticSection(
                status=DataStatus.NOT_APPLICABLE,
                reason="No QAT configuration or experiment supplied.",
            )
        )
    data: dict[str, Any] = dict(qat_info or {})
    if experiment is not None:
        data.update(
            {
                "experiment_bits": experiment.bits,
                "experiment_ste_mode": experiment.ste_mode,
                "normal_fp_loss": experiment.normal_fp_loss,
                "qat_fp_loss": experiment.qat_fp_loss,
                "normal_quantized_loss": experiment.normal_quantized_loss,
                "qat_quantized_loss": experiment.qat_quantized_loss,
                "normal_quantization_damage": experiment.normal_quantization_damage,
                "qat_quantization_damage": experiment.qat_quantization_damage,
                "qat_forward_overhead_ratio": experiment.qat_forward_overhead_ratio,
                "experiment_result": experiment.result_label,
            }
        )
    status = DataStatus.AVAILABLE if data.get("enabled") or experiment is not None else DataStatus.NOT_APPLICABLE
    reason = None if status == DataStatus.AVAILABLE else "QAT is not enabled on this model."
    return QATDiagnostics(section=DiagnosticSection(status=status, reason=reason, data=data))


def _build_training_state_diagnostics(state: dict[str, Any] | None) -> TrainingStateDiagnostics:
    if state is None:
        return TrainingStateDiagnostics(
            section=DiagnosticSection(
                status=DataStatus.UNAVAILABLE,
                reason="No training state supplied.",
            )
        )
    return TrainingStateDiagnostics(
        section=DiagnosticSection(status=DataStatus.AVAILABLE, data=dict(state))
    )


def _build_memory_diagnostics(
    xray: ModelXRayReport,
    compression: Any | None,
) -> MemoryDiagnostics:
    parameter_bytes = sum(p.memory_bytes for p in xray.parameter_infos)
    data: dict[str, Any] = {
        "parameter_storage_bytes": parameter_bytes,
    }
    if compression is not None:
        data["compressed_representation_bytes"] = compression.total_compressed_bytes
        data["compression_ratio"] = compression.compression_ratio
    return MemoryDiagnostics(section=DiagnosticSection(status=DataStatus.AVAILABLE, data=data))


def _merge_experiment_training_state(
    training_state: dict[str, Any] | None,
    experiment: Any | None,
) -> dict[str, Any] | None:
    if experiment is None:
        return training_state
    summary = {
        "experiment_id": getattr(experiment, "experiment_id", None),
        "experiment_name": getattr(experiment, "name", None),
        "technique_label": getattr(experiment, "technique_label", None),
        "training_technique": getattr(experiment, "training_technique", None),
        "storage_technique": getattr(experiment, "storage_technique", None),
        "seed": getattr(experiment, "seed", None),
        "metrics": getattr(experiment.metrics, "to_dict", lambda: {})(),
        "timings": getattr(experiment.timings, "to_dict", lambda: {})(),
    }
    if training_state is None:
        return summary
    merged = dict(training_state)
    merged.update(summary)
    merged["experiment"] = summary
    return merged


def _build_observations(
    gradients: GradientDiagnostics,
    performance: PerformanceDiagnostics,
    compression: CompressionDiagnostics,
    qat: QATDiagnostics,
    training: TrainingStateDiagnostics,
    experiment: Any | None = None,
) -> tuple[Observation, ...]:
    observations: list[Observation] = []

    if gradients.section.status == DataStatus.AVAILABLE:
        g = gradients.section.data
        inspected = int(g.get("parameters_inspected", 0))
        near_zero = int(g.get("near_zero_count", 0))
        non_finite = int(g.get("non_finite_count", 0))
        missing = int(g.get("missing_count", 0))

        if non_finite > 0:
            observations.append(
                Observation(
                    severity=ObservationSeverity.CRITICAL,
                    category="gradients",
                    message="Non-finite gradients detected.",
                    evidence={"non_finite_count": non_finite, "parameters_inspected": inspected},
                )
            )
        if near_zero > 0:
            observations.append(
                Observation(
                    severity=ObservationSeverity.WARNING,
                    category="gradients",
                    message="Possible near-zero gradient concentration.",
                    evidence={
                        "near_zero_count": near_zero,
                        "parameters_inspected": inspected,
                    },
                )
            )
        if missing > 0 and inspected > 0:
            observations.append(
                Observation(
                    severity=ObservationSeverity.WARNING,
                    category="gradients",
                    message="Some parameters have missing gradients.",
                    evidence={"missing_count": missing, "parameters_inspected": inspected},
                )
            )

    if performance.section.status == DataStatus.AVAILABLE:
        hottest = performance.section.data.get("hottest_stage")
        if hottest:
            observations.append(
                Observation(
                    severity=ObservationSeverity.INFO,
                    category="performance",
                    message=f"{hottest} is the hottest measured stage.",
                    evidence={
                        "hottest_stage": hottest,
                        "stage_seconds": performance.section.data.get("stages", {}).get(hottest),
                    },
                )
            )

    if compression.section.status == DataStatus.AVAILABLE:
        c = compression.section.data
        ratio = c.get("compression_ratio")
        if ratio is not None:
            observations.append(
                Observation(
                    severity=ObservationSeverity.INFO,
                    category="compression",
                    message=f"Compression ratio measured at {ratio:.2f}x.",
                    evidence={
                        "compression_ratio": ratio,
                        "total_original_bytes": c.get("total_original_bytes"),
                        "total_compressed_bytes": c.get("total_compressed_bytes"),
                    },
                )
            )
        if c.get("loss_delta") is not None:
            observations.append(
                Observation(
                    severity=ObservationSeverity.INFO,
                    category="compression",
                    message="Compressed model task loss delta measured.",
                    evidence={"loss_delta": c.get("loss_delta")},
                )
            )

    if qat.section.status == DataStatus.AVAILABLE and qat.section.data.get("enabled"):
        observations.append(
            Observation(
                severity=ObservationSeverity.INFO,
                category="qat",
                message="QAT is enabled on this model.",
                evidence={
                    "bits": qat.section.data.get("bits"),
                    "ste_mode": qat.section.data.get("ste_mode"),
                },
            )
        )

    if training.section.status == DataStatus.AVAILABLE:
        observations.append(
            Observation(
                severity=ObservationSeverity.INFO,
                category="training",
                message="Training position state supplied.",
                evidence=training.section.data,
            )
        )

    if experiment is not None:
        exp_observations = getattr(experiment, "observations", ())
        for item in exp_observations:
            message = getattr(item, "message", str(item))
            category = getattr(item, "category", "experiment")
            caution = getattr(item, "caution", None)
            evidence = {"message": message}
            if caution:
                evidence["caution"] = caution
            observations.append(
                Observation(
                    severity=ObservationSeverity.INFO,
                    category=category,
                    message=message,
                    evidence=evidence,
                )
            )
        metrics = getattr(experiment, "metrics", None)
        if metrics is not None and metrics.compression_ratio is not None:
            observations.append(
                Observation(
                    severity=ObservationSeverity.INFO,
                    category="experiment",
                    message=(
                        f"Experiment measured compression ratio at {metrics.compression_ratio:.2f}x."
                    ),
                    evidence={
                        "compression_ratio": metrics.compression_ratio,
                        "output_rmse": metrics.output_rmse,
                        "loss_delta": metrics.loss_delta,
                    },
                )
            )

    return tuple(observations)


def _build_recommendations(observations: tuple[Observation, ...]) -> tuple[Recommendation, ...]:
    recs: list[Recommendation] = []
    categories = {obs.category for obs in observations}

    if "gradients" in categories:
        recs.append(
            Recommendation(
                message="Inspect affected gradients with Gradient Detective.",
                category="gradients",
                related_observation_categories=("gradients",),
            )
        )
    if "performance" in categories:
        recs.append(
            Recommendation(
                message="Benchmark backward timing before changing implementation.",
                category="performance",
                related_observation_categories=("performance",),
            )
        )
    if "compression" in categories:
        recs.append(
            Recommendation(
                message="Compare multiple bit widths and packing options if deploying compressed models.",
                category="compression",
                related_observation_categories=("compression",),
            )
        )
    if "qat" in categories:
        recs.append(
            Recommendation(
                message="Check whether QAT reduces post-training quantization damage on your task.",
                category="qat",
                related_observation_categories=("qat",),
            )
        )

    return tuple(recs)


def _magnitude_entry(magnitude: Any) -> dict[str, Any] | None:
    if magnitude is None:
        return None
    return {"name": magnitude.name, "magnitude": float(magnitude.magnitude)}


def _serialize_section(section: DiagnosticSection) -> dict[str, Any]:
    return {
        "status": section.status.value,
        "reason": section.reason,
        "data": _json_safe(section.data),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    return str(value)


def _severity_icon(severity: ObservationSeverity) -> str:
    if severity == ObservationSeverity.CRITICAL:
        return "[CRITICAL]"
    if severity == ObservationSeverity.WARNING:
        return "[WARNING]"
    return "[INFO]"

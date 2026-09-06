from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from minitorch.autograd.explorer import GraphExplorer
from minitorch.nn import Linear, Module, Parameter
from minitorch.tensor import Tensor

from .gradient_detective import GradientHealthReport, detect_gradients, format_gradient_report


@dataclass(frozen=True)
class ParameterInfo:
    name: str
    shape: tuple[int, ...]
    numel: int
    trainable: bool
    dtype: str
    memory_bytes: int


@dataclass
class ModelXRayReport:
    """Structured internal inspection combining registry, graph, and gradients."""

    model: Module
    architecture: tuple[tuple[str, str], ...]
    parameter_infos: tuple[ParameterInfo, ...]
    graph_summary: dict[str, Any] | None
    dsa_profile: dict[str, Any] | None
    gradient_report: GradientHealthReport | None
    qat_info: dict[str, Any] | None = None
    output_tensor_id: int | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "architecture_layers": len(self.architecture),
            "parameter_count": len(self.parameter_infos),
            "trainable_parameters": sum(1 for p in self.parameter_infos if p.trainable),
            "graph_vertices": self.graph_summary.get("vertices") if self.graph_summary else None,
            "graph_edges": self.graph_summary.get("edges") if self.graph_summary else None,
            "output_tensor_id": self.output_tensor_id,
            "has_gradient_report": self.gradient_report is not None,
        }

    def parameters_section(self) -> tuple[ParameterInfo, ...]:
        return self.parameter_infos

    def gradients_section(self) -> GradientHealthReport | None:
        return self.gradient_report

    def graph_section(self) -> dict[str, Any] | None:
        return self.graph_summary

    def dsa_section(self) -> dict[str, Any] | None:
        return self.dsa_profile

    def __str__(self) -> str:
        lines = [
            "MODEL X-RAY",
            "===========",
            "",
            "Architecture",
            "------------",
        ]

        for name, layer_type in self.architecture:
            lines.append(f"{name}: {layer_type}")

        lines.extend(["", "Parameters", "----------"])
        for param in self.parameter_infos:
            lines.append(
                f"{param.name}: shape={param.shape}, numel={param.numel}, "
                f"trainable={param.trainable}, bytes={param.memory_bytes}"
            )

        if self.graph_summary is not None:
            lines.extend(
                [
                    "",
                    "Graph",
                    "-----",
                    f"Vertices: {self.graph_summary.get('vertices')}",
                    f"Edges: {self.graph_summary.get('edges')}",
                    f"Depth: {self.graph_summary.get('depth')}",
                    f"Operations: {self.graph_summary.get('operations')}",
                ]
            )

        if self.dsa_profile is not None:
            lines.extend(
                [
                    "",
                    "DSA",
                    "---",
                    f"V = {self.dsa_profile.get('vertices')}",
                    f"E = {self.dsa_profile.get('edges')}",
                    f"DFS visits: {self.dsa_profile.get('dfs_visits')}",
                    f"DFS edge inspections: {self.dsa_profile.get('dfs_edge_inspections')}",
                    f"Topological nodes: {self.dsa_profile.get('topological_processed_nodes')}",
                    f"Complexity: {self.dsa_profile.get('complexity')}",
                ]
            )

        if self.gradient_report is not None:
            lines.extend(["", format_gradient_report(self.gradient_report)])

        if self.qat_info is not None:
            lines.extend(
                [
                    "",
                    "QAT",
                    "---",
                    f"enabled: {self.qat_info.get('enabled')}",
                    f"bits: {self.qat_info.get('bits')}",
                    f"method: {self.qat_info.get('method')}",
                    f"STE: {self.qat_info.get('ste_mode')}",
                    f"fake-quantized layers: {self.qat_info.get('fake_quantized_layers')}",
                ]
            )

        return "\n".join(lines)


def build_model_xray(model: Module, output: Tensor | None = None) -> ModelXRayReport:
    architecture = _discover_architecture(model)
    parameters = _collect_parameter_info(model)

    graph_summary: dict[str, Any] | None = None
    dsa_profile: dict[str, Any] | None = None
    output_id: int | None = None

    if output is not None:
        explorer = GraphExplorer.from_tensor(output)
        graph_summary = explorer.graph.summary()
        graph_summary["depth"] = explorer.depth()
        dsa_profile = explorer.dsa_profile()
        output_id = output.id

    gradient_report = detect_gradients(model)

    from ..qat import qat_summary

    return ModelXRayReport(
        model=model,
        architecture=architecture,
        parameter_infos=parameters,
        graph_summary=graph_summary,
        dsa_profile=dsa_profile,
        gradient_report=gradient_report,
        qat_info=qat_summary(model),
        output_tensor_id=output_id,
    )


def _discover_architecture(model: Module, prefix: str = "") -> tuple[tuple[str, str], ...]:
    layers: list[tuple[str, str]] = []

    for name, child in model._modules.items():
        full_name = f"{prefix}.{name}" if prefix else name
        layers.append((full_name, type(child).__name__))
        layers.extend(_discover_architecture(child, full_name))

    return tuple(layers)


def _collect_parameter_info(model: Module) -> tuple[ParameterInfo, ...]:
    infos: list[ParameterInfo] = []

    for name, parameter in model.named_parameters():
        if not isinstance(parameter, Parameter):
            continue
        numel = int(parameter.size)
        infos.append(
            ParameterInfo(
                name=name,
                shape=parameter.shape,
                numel=numel,
                trainable=parameter.requires_grad,
                dtype=str(parameter.dtype),
                memory_bytes=numel * parameter.data.dtype.itemsize,
            )
        )

    return tuple(infos)


def mri_hook_placeholder() -> str:
    """Documents AI-MRI composition — see ``inspect_model_mri`` / ``model.mri()``."""

    return (
        "AI-MRI is implemented via model.mri() and minitorch.intelligence.inspect_model_mri. "
        "It composes X-Ray, Gradient Detective, Graph Explorer, profiler, compression, and QAT."
    )

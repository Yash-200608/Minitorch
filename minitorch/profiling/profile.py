from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np

from minitorch.autograd.explorer import GraphExplorer
from minitorch.bitpacking import pack_codes, unpack_codes
from minitorch.compression import CompressionConfig, compress_model
from minitorch.dsa.benchmark import benchmark_cache_on_off
from minitorch.dsa.topk import topk
from minitorch.intelligence.graph_optimizer import optimize
from minitorch.nn import Module
from minitorch.quantization import Codebook, compute_error_metrics, quantize_array
from minitorch.rotation import build_rotation_transform, is_power_of_two
from minitorch.tensor import Tensor
from minitorch.training.optimizers import Optimizer

from .timing import Timer, benchmark_timing, TimingBenchmarkResult


class ProfilerMode(str, Enum):
    LIGHTWEIGHT = "lightweight"
    DETAILED = "detailed"


BENCHMARK_WORKLOADS: dict[str, dict[str, Any]] = {
    "small": {"shape": (8, 8), "seed": 0},
    "medium": {"shape": (64, 64), "seed": 1},
}


@dataclass
class PerformanceProfile:
    """Structured performance measurements for MiniTorch workloads."""

    name: str
    mode: str
    stages: dict[str, float] = field(default_factory=dict)
    stage_benchmarks: dict[str, TimingBenchmarkResult] = field(default_factory=dict)
    graph_stats: dict[str, Any] = field(default_factory=dict)
    dsa_stats: dict[str, Any] = field(default_factory=dict)
    operation_timings: dict[str, float] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def dsa_profile(self) -> dict[str, Any]:
        return dict(self.dsa_stats)

    def hottest_stage(self) -> str | None:
        if not self.stages:
            return None
        return max(self.stages, key=self.stages.get)

    def top_operations(self, k: int = 5) -> tuple[tuple[str, float], ...]:
        if not self.operation_timings:
            return ()
        names = list(self.operation_timings.keys())
        values = [self.operation_timings[name] for name in names]
        probe = topk(values, k=min(k, len(values)), largest=True)
        return tuple((names[index], value) for index, value in zip(probe.indices, probe.values))

    def report(self) -> str:
        lines = [
            "MiniTorch Performance Profile",
            "------------------------------",
            "",
            f"Profile: {self.name}",
            f"Mode: {self.mode}",
            "",
        ]

        if self.stages:
            lines.append("STAGES")
            for stage, seconds in sorted(self.stages.items()):
                lines.append(f"  {stage}: {seconds * 1000:.2f} ms")
            lines.append("")

        if self.graph_stats:
            lines.append("GRAPH")
            for key, value in self.graph_stats.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        if self.dsa_stats:
            lines.append("DSA")
            for key, value in self.dsa_stats.items():
                if key != "complexity":
                    lines.append(f"  {key}: {value}")
            lines.append("")

        if self.operation_timings:
            lines.append("OPERATIONS")
            for name, seconds in self.top_operations(k=len(self.operation_timings)):
                lines.append(f"  {name}: {seconds * 1000:.2f} ms")
            lines.append("")

        if self.memory:
            lines.append("MEMORY")
            for key, value in self.memory.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        hottest = self.hottest_stage()
        if hottest is not None:
            lines.extend(
                [
                    "HOTTEST MEASURED STAGE",
                    f"  {hottest}",
                    "",
                    "NOTE",
                    "  This identifies the slowest measured stage.",
                    "  It does not prove why it is slow.",
                ]
            )

        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "mode": self.mode,
            "stages": dict(self.stages),
            "graph_stats": dict(self.graph_stats),
            "dsa_stats": dict(self.dsa_stats),
            "operation_timings": dict(self.operation_timings),
            "memory": dict(self.memory),
            "metadata": dict(self.metadata),
            "hottest_stage": self.hottest_stage(),
        }
        for key, value in self.graph_stats.items():
            payload[key] = value
        for key, value in self.dsa_stats.items():
            if key in ("dfs_visits", "topological_processed_nodes"):
                payload[key] = value
        if "graph_nodes" in self.graph_stats:
            payload["graph_nodes"] = self.graph_stats["graph_nodes"]
        if "graph_edges" in self.graph_stats:
            payload["graph_edges"] = self.graph_stats["graph_edges"]
        if "loss" in self.metadata:
            payload["loss"] = self.metadata["loss"]
        if "processed_nodes" in self.metadata:
            payload["processed_nodes"] = self.metadata["processed_nodes"]
        if "gradient_accumulations" in self.metadata:
            payload["gradient_accumulations"] = self.metadata["gradient_accumulations"]
        if "topological_operations" in self.metadata:
            payload["topological_operations"] = self.metadata["topological_operations"]
        if "dfs_visits" in self.dsa_stats:
            payload["dfs_visits"] = self.dsa_stats["dfs_visits"]
        return payload


def _workload_arrays(workload: str) -> tuple[np.ndarray, np.ndarray]:
    spec = BENCHMARK_WORKLOADS[workload]
    rng = np.random.default_rng(spec["seed"])
    shape = spec["shape"]
    return rng.standard_normal(shape), rng.standard_normal(shape)


def profile_tensor_operation(
    name: str,
    operation: Callable[[Tensor, Tensor], Tensor],
    *,
    workload: str = "small",
    warmup: int = 3,
    repeats: int = 10,
) -> TimingBenchmarkResult:
    left_data, right_data = _workload_arrays(workload)

    def run() -> Tensor:
        left = Tensor(left_data.copy())
        right = Tensor(right_data.copy())
        return operation(left, right)

    return benchmark_timing(name, run, warmup=warmup, repeats=repeats)


def profile_standard_tensor_operations(
    *,
    workload: str = "small",
    warmup: int = 1,
    repeats: int = 3,
) -> dict[str, float]:
    left_data, right_data = _workload_arrays(workload)
    timings: dict[str, float] = {}

    operations: dict[str, Callable[[], Tensor]] = {
        "add": lambda: Tensor(left_data.copy()) + Tensor(right_data.copy()),
        "sub": lambda: Tensor(left_data.copy()) - Tensor(right_data.copy()),
        "mul": lambda: Tensor(left_data.copy()) * Tensor(right_data.copy()),
        "div": lambda: Tensor(left_data.copy()) / (Tensor(right_data.copy()) + 1.0),
        "matmul": lambda: Tensor(left_data.copy()) @ Tensor(right_data.copy()),
        "sum": lambda: Tensor(left_data.copy()).sum(),
        "mean": lambda: Tensor(left_data.copy()).mean(),
        "reshape": lambda: Tensor(left_data.copy()).reshape((left_data.size,)),
        "transpose": lambda: Tensor(left_data.copy()).T,
        "exp": lambda: Tensor(left_data.copy()).exp(),
        "log": lambda: Tensor(np.abs(left_data.copy()) + 0.1).log(),
        "relu": lambda: Tensor(left_data.copy()).relu(),
        "sigmoid": lambda: Tensor(left_data.copy()).sigmoid(),
        "tanh": lambda: Tensor(left_data.copy()).tanh(),
    }

    for op_name, fn in operations.items():
        result = benchmark_timing(op_name, fn, warmup=warmup, repeats=repeats)
        timings[op_name] = result.mean_seconds

    return timings


def profile_training_step(
    model: Module,
    optimizer: Optimizer,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    inputs: Tensor,
    targets: Tensor,
    *,
    mode: ProfilerMode = ProfilerMode.LIGHTWEIGHT,
) -> PerformanceProfile:
    optimizer.zero_grad()

    with Timer() as forward_timer:
        predictions = model(inputs)
    with Timer() as loss_timer:
        loss = loss_fn(predictions, targets)
    with Timer() as backward_timer:
        backward_result = loss.backward()
    with Timer() as optimizer_timer:
        optimizer.step()

    explorer = GraphExplorer.from_tensor(loss)
    dsa_stats = explorer.dsa_profile()

    stages = {
        "forward": forward_timer.elapsed_seconds,
        "loss": loss_timer.elapsed_seconds,
        "backward": backward_timer.elapsed_seconds,
        "optimizer": optimizer_timer.elapsed_seconds,
    }

    operation_timings: dict[str, float] = {}
    if mode == ProfilerMode.DETAILED:
        operation_timings = profile_standard_tensor_operations(
            workload=_infer_workload(inputs.shape),
            warmup=1,
            repeats=1,
        )

    graph_stats = {
        "graph_nodes": len(backward_result.graph.nodes),
        "graph_edges": len(backward_result.graph.edges),
        "nodes": len(backward_result.graph.nodes),
        "edges": len(backward_result.graph.edges),
    }

    memory = {
        "input_nbytes": int(np.prod(inputs.shape) * 8),
        "target_nbytes": int(np.prod(targets.shape) * 8),
        "parameter_nbytes": sum(int(np.prod(param.data.shape) * 8) for param in model.parameters()),
    }

    return PerformanceProfile(
        name="training_step",
        mode=mode.value,
        stages=stages,
        graph_stats=graph_stats,
        dsa_stats=dsa_stats,
        operation_timings=operation_timings,
        memory=memory,
        metadata={
            "loss": float(loss.data),
            "processed_nodes": backward_result.stats.processed_nodes,
            "gradient_accumulations": backward_result.stats.gradient_accumulations,
            "topological_operations": dsa_stats["topological_processed_nodes"],
        },
    )


def _infer_workload(shape: tuple[int, ...]) -> str:
    size = int(np.prod(shape))
    if size <= 64:
        return "small"
    return "medium"


def profile_graph_optimizer_comparison(
    model: Module,
    inputs: Tensor,
    targets: Tensor,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    *,
    warmup: int = 1,
    repeats: int = 3,
) -> PerformanceProfile:
    def original_backward() -> None:
        model.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()

    def optimized_backward() -> None:
        model.zero_grad()
        loss = loss_fn(model(inputs), targets)
        result = optimize(loss)
        result.optimized_output.backward()

    original = benchmark_timing(
        "original_backward",
        original_backward,
        warmup=warmup,
        repeats=repeats,
    )
    optimized = benchmark_timing(
        "optimized_backward",
        optimized_backward,
        warmup=warmup,
        repeats=repeats,
    )

    model.zero_grad()
    loss = loss_fn(model(inputs), targets)
    opt_result = optimize(loss)
    before = opt_result.dsa_profile()

    stages = {
        "original_backward": original.mean_seconds,
        "optimized_backward": optimized.mean_seconds,
    }

    return PerformanceProfile(
        name="graph_optimizer_comparison",
        mode=ProfilerMode.LIGHTWEIGHT.value,
        stages=stages,
        stage_benchmarks={
            "original_backward": original,
            "optimized_backward": optimized,
        },
        graph_stats={
            "original_nodes": before["original_vertices"],
            "original_edges": before["original_edges"],
            "optimized_nodes": before["optimized_vertices"],
            "optimized_edges": before["optimized_edges"],
            "nodes_removed": before["nodes_removed"],
        },
        dsa_stats=before,
        metadata={"accepted": opt_result.accepted},
    )


def profile_quantization_stages(
    data: np.ndarray,
    bits: int,
    *,
    method: str = "uniform",
    packed: bool = True,
    rotation: str | None = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> PerformanceProfile:
    array = np.asarray(data, dtype=np.float64)
    stages: dict[str, float] = {}
    memory: dict[str, Any] = {}

    work = array
    rotation_transform = None
    if rotation is not None:
        dimension = int(array.shape[-1])
        rotation_transform = build_rotation_transform(dimension, kind=rotation)
        with Timer() as timer:
            work = rotation_transform.apply(array)
        stages["rotation"] = timer.elapsed_seconds

    if method == "codebook":
        levels = 2 ** bits
        with Timer() as timer:
            codebook = Codebook.fit(
                work.ravel(),
                levels,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
        stages["codebook_fit"] = timer.elapsed_seconds
        with Timer() as timer:
            codes = codebook.encode(work.ravel())
        stages["encoding"] = timer.elapsed_seconds
        quantized = quantize_array(
            array,
            bits,
            method=method,
            max_iterations=max_iterations,
            tolerance=tolerance,
            packed=False,
            rotation=rotation,
        )
    else:
        with Timer() as timer:
            quantized = quantize_array(
                array,
                bits,
                method=method,
                packed=False,
                rotation=rotation,
            )
        stages["uniform_quantization"] = timer.elapsed_seconds

    flat_codes = quantized.codes.ravel()
    if packed:
        with Timer() as timer:
            packed_bytes = pack_codes(flat_codes, bits)
        stages["packing"] = timer.elapsed_seconds
        with Timer() as timer:
            unpacked_codes = unpack_codes(packed_bytes, bits, flat_codes.size)
        stages["unpacking"] = timer.elapsed_seconds
        memory["packed_nbytes"] = len(packed_bytes)
    else:
        memory["codes_nbytes"] = int(flat_codes.nbytes)

    with Timer() as timer:
        reconstructed = quantized.dequantize().data
    stages["dequantization"] = timer.elapsed_seconds

    if rotation_transform is not None:
        with Timer() as timer:
            rotation_transform.inverse(reconstructed)
        stages["inverse_rotation"] = timer.elapsed_seconds

    error = compute_error_metrics(array, reconstructed)
    report = quantized.report()
    memory.update(
        {
            "tensor_nbytes": int(array.nbytes),
            "representation_bytes": report.memory.total_actual_representation_bytes,
            "reconstruction_rmse": error.rmse,
        }
    )

    return PerformanceProfile(
        name="quantization_stages",
        mode=ProfilerMode.LIGHTWEIGHT.value,
        stages=stages,
        memory=memory,
        metadata={"bits": bits, "method": method, "packed": packed, "rotation": rotation},
    )


def profile_quantization_memory_matrix(
    data: np.ndarray,
    *,
    repeats: int = 1,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for bits in (2, 4, 8):
        for method in ("uniform", "codebook"):
            profile = profile_quantization_stages(
                data,
                bits,
                method=method,
                packed=True,
            )
            quant_seconds = sum(
                profile.stages.get(key, 0.0)
                for key in (
                    "uniform_quantization",
                    "codebook_fit",
                    "encoding",
                )
            )
            packing_seconds = profile.stages.get("packing", 0.0)
            results.append(
                {
                    "bits": bits,
                    "method": method,
                    "packed": True,
                    "rotation": None,
                    "quantization_seconds": quant_seconds,
                    "packing_seconds": packing_seconds,
                    "representation_bytes": profile.memory.get("representation_bytes"),
                    "reconstruction_rmse": profile.memory.get("reconstruction_rmse"),
                }
            )
    return results


def profile_rotation_pipeline(
    data: np.ndarray,
    bits: int = 4,
    *,
    method: str = "uniform",
    packed: bool = True,
) -> PerformanceProfile:
    if not is_power_of_two(int(np.asarray(data).shape[-1])):
        raise ValueError("Rotation profiling requires a power-of-two final axis.")

    profile = profile_quantization_stages(
        data,
        bits,
        method=method,
        packed=packed,
        rotation="combined",
    )
    total = sum(profile.stages.values())
    profile.stages["total"] = total
    profile.name = "rotation_pipeline"
    return profile


def profile_model_compression(
    model: Module,
    config: CompressionConfig | None = None,
    *,
    repeats: int = 1,
) -> PerformanceProfile:
    config = config or CompressionConfig(bits=4, method="uniform", packed=True)
    stages: dict[str, float] = {}

    with Timer() as timer:
        for _ in range(repeats):
            compressed = compress_model(model, config=config)
    stages["compress_model"] = timer.elapsed_seconds / repeats

    with Timer() as timer:
        for _ in range(repeats):
            materialized = compressed.materialize(model)
    stages["materialize"] = timer.elapsed_seconds / repeats

    report = compressed.report()
    memory = {
        "total_original_bytes": report.total_original_bytes,
        "total_compressed_bytes": report.total_compressed_bytes,
        "compression_ratio": report.compression_ratio,
        "aggregate_parameter_rmse": report.aggregate_parameter_rmse,
    }

    return PerformanceProfile(
        name="model_compression",
        mode=ProfilerMode.LIGHTWEIGHT.value,
        stages=stages,
        memory=memory,
        metadata={"config": config},
    )


def profile_compression_tradeoffs(
    model: Module,
    inputs: Tensor,
    targets: Tensor | None = None,
    *,
    configs: list[CompressionConfig] | None = None,
) -> list[dict[str, Any]]:
    if configs is None:
        configs = [
            CompressionConfig(bits=8, method="uniform", packed=True),
            CompressionConfig(bits=4, method="uniform", packed=True),
            CompressionConfig(bits=4, method="codebook", packed=True),
            CompressionConfig(bits=4, method="codebook", packed=True, rotation="combined"),
        ]

    rows: list[dict[str, Any]] = []
    for config in configs:
        profile = profile_model_compression(model, config=config)
        compressed = compress_model(model, config=config)
        report = compressed.report(template=model, inputs=inputs, targets=targets)
        rows.append(
            {
                "configuration": (
                    f"{config.method} {config.bits}bit packed={config.packed} rot={config.rotation}"
                ),
                "memory_bytes": report.total_compressed_bytes,
                "compression_ratio": report.compression_ratio,
                "parameter_rmse": report.aggregate_parameter_rmse,
                "output_rmse": report.output_rmse if report.output_rmse is not None else 0.0,
                "time_seconds": sum(profile.stages.values()),
            }
        )
    return rows


def format_compression_tradeoff_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "Configuration                    Memory   Ratio   Param RMSE  Output RMSE  Time",
        "--------------------------------------------------------------------------------",
    ]
    for row in rows:
        lines.append(
            f"{row['configuration']:<32} {row['memory_bytes']:<8} "
            f"{row['compression_ratio']:.2f}x  {row['parameter_rmse']:.6f}  "
            f"{row['output_rmse']:.6f}  {row['time_seconds']:.4f}s"
        )
    return "\n".join(lines)


def profile_profiling_overhead(
    workload_fn: Callable[[], None],
    *,
    warmup: int = 3,
    repeats: int = 10,
) -> PerformanceProfile:
    none_result = benchmark_timing("profiling_disabled", workload_fn, warmup=warmup, repeats=repeats)

    def light_profile() -> None:
        workload_fn()
        profile_standard_tensor_operations(workload="small", warmup=0, repeats=1)

    light_result = benchmark_timing("lightweight_profile", light_profile, warmup=warmup, repeats=repeats)

    def detailed_profile() -> None:
        workload_fn()
        profile_standard_tensor_operations(workload="medium", warmup=0, repeats=1)

    detailed_result = benchmark_timing("detailed_profile", detailed_profile, warmup=warmup, repeats=repeats)

    return PerformanceProfile(
        name="profiling_overhead",
        mode=ProfilerMode.LIGHTWEIGHT.value,
        stages={
            "none": none_result.mean_seconds,
            "lightweight": light_result.mean_seconds,
            "detailed": detailed_result.mean_seconds,
        },
        stage_benchmarks={
            "none": none_result,
            "lightweight": light_result,
            "detailed": detailed_result,
        },
        metadata={
            "light_overhead_seconds": light_result.mean_seconds - none_result.mean_seconds,
            "detailed_overhead_seconds": detailed_result.mean_seconds - none_result.mean_seconds,
        },
    )


def profile_cache_comparison(
    expression_fn: Callable[[], Tensor],
    *,
    iterations: int = 20,
) -> PerformanceProfile:
    off_result, on_result = benchmark_cache_on_off(expression_fn, iterations=iterations)
    return PerformanceProfile(
        name="cache_comparison",
        mode=ProfilerMode.LIGHTWEIGHT.value,
        stages={
            "cache_disabled": off_result.runtime_seconds,
            "cache_enabled": on_result.runtime_seconds,
        },
        metadata={
            "iterations": iterations,
            "cache_disabled_details": off_result.details,
            "cache_enabled_details": on_result.details,
        },
    )

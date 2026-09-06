from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .dsa.topk import topk
from .losses import MSELoss
from .nn import Linear, Module, Parameter, ReLU, Sequential, Sigmoid, Tanh
from .parallel import ParallelConfig
from .quantization import (
    QuantizedTensor,
    compute_error_metrics,
    quantize_array,
)
from .rotation import is_power_of_two
from .tensor import Tensor

QuantizationMethod = Literal["uniform", "codebook"]
RotationOption = Literal["combined", "hadamard"] | None


@dataclass
class CompressionConfig:
    bits: int = 8
    method: QuantizationMethod = "uniform"
    packed: bool = False
    rotation: RotationOption = None
    max_iterations: int = 100
    tolerance: float = 1e-6


@dataclass(frozen=True)
class ParameterCompressionRecord:
    name: str
    shape: tuple[int, ...]
    original_dtype: str
    original_bytes: int
    compressed_bytes: int
    packed_code_bytes: int
    codebook_bytes: int
    bits: int
    method: str
    packed: bool
    rotation: str | None
    max_absolute_error: float
    mean_absolute_error: float
    rmse: float


@dataclass
class ModelCompressionReport:
    parameter_count: int
    config: CompressionConfig
    total_original_bytes: int
    total_compressed_bytes: int
    compression_ratio: float
    aggregate_parameter_rmse: float
    parameter_records: tuple[ParameterCompressionRecord, ...]
    output_max_absolute_error: float | None = None
    output_mean_absolute_error: float | None = None
    output_rmse: float | None = None
    original_loss: float | None = None
    compressed_loss: float | None = None
    loss_delta: float | None = None

    def parameters(self) -> tuple[ParameterCompressionRecord, ...]:
        return self.parameter_records

    def largest_parameter_errors(self, k: int = 5) -> tuple[ParameterCompressionRecord, ...]:
        if not self.parameter_records:
            return ()
        names = [record.name for record in self.parameter_records]
        rmse_values = [record.rmse for record in self.parameter_records]
        result = topk(rmse_values, k=min(k, len(rmse_values)), largest=True)
        return tuple(self.parameter_records[index] for index in result.indices)

    def format(self) -> str:
        lines = [
            "Model Compression Report",
            "------------------------",
            "",
            f"Parameters: {self.parameter_count}",
            "",
            "Original:",
            f"  {self.total_original_bytes} bytes",
            "",
            "Compressed:",
            f"  {self.total_compressed_bytes} bytes",
            "",
            f"Actual compression ratio: {self.compression_ratio:.2f}x",
            "",
            "Quantization:",
            f"  {self.config.bits}-bit",
            f"  {self.config.method}",
            f"  packed: {self.config.packed}",
            f"  rotation: {self.config.rotation}",
            "",
            f"Aggregate parameter RMSE: {self.aggregate_parameter_rmse}",
        ]
        if self.output_rmse is not None:
            lines.extend(
                [
                    "",
                    "Output error:",
                    f"  max absolute: {self.output_max_absolute_error}",
                    f"  mean absolute: {self.output_mean_absolute_error}",
                    f"  RMSE: {self.output_rmse}",
                ]
            )
        if self.original_loss is not None and self.compressed_loss is not None:
            lines.extend(
                [
                    "",
                    f"Original loss: {self.original_loss}",
                    f"Compressed loss: {self.compressed_loss}",
                    f"Loss delta: {self.loss_delta}",
                ]
            )
        lines.append("")
        lines.append("Status: output change measured from actual forward passes")
        return "\n".join(lines)

    def xray(self) -> dict[str, Any]:
        return {
            "parameter_count": self.parameter_count,
            "total_original_bytes": self.total_original_bytes,
            "total_compressed_bytes": self.total_compressed_bytes,
            "compression_ratio": self.compression_ratio,
            "aggregate_parameter_rmse": self.aggregate_parameter_rmse,
            "output_rmse": self.output_rmse,
            "original_loss": self.original_loss,
            "compressed_loss": self.compressed_loss,
            "loss_delta": self.loss_delta,
            "config": {
                "bits": self.config.bits,
                "method": self.config.method,
                "packed": self.config.packed,
                "rotation": self.config.rotation,
            },
        }


@dataclass
class CompressedModel:
    """In-memory post-training compressed parameter representations."""

    config: CompressionConfig
    parameter_records: dict[str, ParameterCompressionRecord]
    quantized_parameters: dict[str, QuantizedTensor]
    source_parameter_names: tuple[str, ...]

    def decompress_parameter(self, name: str) -> np.ndarray:
        if name not in self.quantized_parameters:
            raise KeyError(f"Parameter {name!r} is not in the compressed model.")
        return self.quantized_parameters[name].dequantize().data.copy()

    def materialize(self, template: Module) -> Module:
        """Build a separate evaluation model with decompressed parameter values."""
        model = clone_module_structure(template)
        for name, parameter in model.named_parameters():
            if name not in self.quantized_parameters:
                raise KeyError(f"Compressed model missing parameter {name!r}.")
            parameter.data = self.decompress_parameter(name)
            parameter.requires_grad = False
            parameter.grad = None
        return model

    def report(
        self,
        template: Module | None = None,
        inputs: Tensor | None = None,
        targets: Tensor | None = None,
    ) -> ModelCompressionReport:
        records = tuple(
            self.parameter_records[name]
            for name in self.source_parameter_names
        )
        total_original = sum(record.original_bytes for record in records)
        total_compressed = sum(record.compressed_bytes for record in records)
        ratio = total_original / total_compressed if total_compressed > 0 else 0.0
        aggregate_rmse = float(
            np.sqrt(np.mean([record.rmse ** 2 for record in records]))
        )

        output_errors = None
        original_loss = None
        compressed_loss = None
        loss_delta = None

        if template is not None and inputs is not None:
            original_outputs = template(inputs)
            reconstructed = self.materialize(template)
            compressed_outputs = reconstructed(inputs)
            output_errors = compute_error_metrics(
                original_outputs.data,
                compressed_outputs.data,
            )
            if targets is not None:
                criterion = MSELoss()
                original_loss = float(criterion(original_outputs, targets).data)
                compressed_loss = float(criterion(compressed_outputs, targets).data)
                loss_delta = compressed_loss - original_loss

        return ModelCompressionReport(
            parameter_count=len(records),
            config=self.config,
            total_original_bytes=total_original,
            total_compressed_bytes=total_compressed,
            compression_ratio=ratio,
            aggregate_parameter_rmse=aggregate_rmse,
            parameter_records=records,
            output_max_absolute_error=(
                output_errors.max_absolute_error if output_errors else None
            ),
            output_mean_absolute_error=(
                output_errors.mean_absolute_error if output_errors else None
            ),
            output_rmse=output_errors.rmse if output_errors else None,
            original_loss=original_loss,
            compressed_loss=compressed_loss,
            loss_delta=loss_delta,
        )

    def xray(self) -> dict[str, Any]:
        return self.report().xray()

    def save(self, path: str) -> None:
        """Persist compressed parameter representations to a MiniTorch artifact."""
        from .persistence import save_compressed_model

        save_compressed_model(self, path)


def compress_model(
    model: Module,
    config: CompressionConfig | None = None,
    *,
    parallel_config: ParallelConfig | None = None,
    parallel: bool = False,
    workers: int | None = None,
) -> CompressedModel:
    """
    Compress each parameter into a separate QuantizedTensor.

    The live model is not modified.
    """
    from .parallel import ParallelConfig, execute_chunked

    config = config or CompressionConfig()
    parallel_settings = _resolve_parallel_config(parallel_config, parallel, workers)
    items = list(model.named_parameters())

    def compress_chunk(start: int, end: int) -> list[tuple[str, ParameterCompressionRecord, QuantizedTensor]]:
        chunk: list[tuple[str, ParameterCompressionRecord, QuantizedTensor]] = []
        for name, parameter in items[start:end]:
            chunk.append(_compress_single_parameter(name, parameter, config, parallel_settings))
        return chunk

    if not parallel_settings.enabled:
        results = compress_chunk(0, len(items))
    else:
        results = []
        for chunk in execute_chunked(len(items), parallel_settings, compress_chunk):
            results.extend(chunk)

    records: dict[str, ParameterCompressionRecord] = {}
    quantized: dict[str, QuantizedTensor] = {}
    names: list[str] = []
    for name, record, quantized_tensor in results:
        names.append(name)
        records[name] = record
        quantized[name] = quantized_tensor

    return CompressedModel(
        config=config,
        parameter_records=records,
        quantized_parameters=quantized,
        source_parameter_names=tuple(names),
    )


def _compress_single_parameter(
    name: str,
    parameter: Parameter,
    config: CompressionConfig,
    parallel_config: ParallelConfig,
) -> tuple[str, ParameterCompressionRecord, QuantizedTensor]:
    original_snapshot = parameter.data.copy()
    quantized_tensor = _quantize_parameter(parameter.data, config, parallel_config)
    reconstructed = quantized_tensor.dequantize().data
    error = compute_error_metrics(original_snapshot, reconstructed)
    memory = quantized_tensor.report().memory

    record = ParameterCompressionRecord(
        name=name,
        shape=tuple(parameter.shape),
        original_dtype=str(parameter.dtype),
        original_bytes=int(original_snapshot.nbytes),
        compressed_bytes=int(memory.total_actual_representation_bytes),
        packed_code_bytes=int(
            memory.actual_packed_bytes if config.packed else memory.actual_code_storage_bytes
        ),
        codebook_bytes=int(memory.codebook_bytes),
        bits=config.bits,
        method=config.method,
        packed=config.packed,
        rotation=config.rotation,
        max_absolute_error=error.max_absolute_error,
        mean_absolute_error=error.mean_absolute_error,
        rmse=error.rmse,
    )

    if not np.array_equal(parameter.data, original_snapshot):
        raise RuntimeError(f"Compression mutated live parameter {name!r}.")

    return name, record, quantized_tensor


def _resolve_parallel_config(
    parallel_config: ParallelConfig | None,
    parallel: bool,
    workers: int | None,
) -> ParallelConfig:
    from .parallel import ParallelConfig as PC

    if parallel_config is not None:
        return parallel_config
    if parallel:
        return PC(enabled=True, workers=workers)
    return PC()


def compare_model_compression(
    model: Module,
    inputs: Tensor,
    targets: Tensor | None = None,
    config: CompressionConfig | None = None,
) -> ModelCompressionReport:
    """Compress, reconstruct, and compare outputs and optional task loss."""
    compressed = compress_model(model, config=config)
    return compressed.report(template=model, inputs=inputs, targets=targets)


def benchmark_model_compression(
    model: Module,
    *,
    configs: list[CompressionConfig] | None = None,
    repeats: int = 3,
) -> list[dict[str, Any]]:
    """Local measurements across compression configurations."""
    if configs is None:
        configs = [
            CompressionConfig(bits=8, method="uniform", packed=False),
            CompressionConfig(bits=4, method="uniform", packed=True),
            CompressionConfig(bits=4, method="codebook", packed=True),
            CompressionConfig(
                bits=4,
                method="codebook",
                packed=True,
                rotation="combined",
            ),
        ]

    results: list[dict[str, Any]] = []
    for config in configs:
        compress_start = time.perf_counter()
        for _ in range(repeats):
            compressed = compress_model(model, config=config)
        compress_seconds = (time.perf_counter() - compress_start) / repeats

        reconstruct_start = time.perf_counter()
        for _ in range(repeats):
            _ = compressed.materialize(model)
        reconstruct_seconds = (time.perf_counter() - reconstruct_start) / repeats

        report = compressed.report()
        results.append(
            {
                "bits": config.bits,
                "method": config.method,
                "packed": config.packed,
                "rotation": config.rotation,
                "compress_seconds": compress_seconds,
                "reconstruct_seconds": reconstruct_seconds,
                "total_original_bytes": report.total_original_bytes,
                "total_compressed_bytes": report.total_compressed_bytes,
                "compression_ratio": report.compression_ratio,
                "aggregate_parameter_rmse": report.aggregate_parameter_rmse,
            }
        )
    return results


def clone_module_structure(module: Module) -> Module:
    """Create a fresh module tree matching supported layer types."""
    if isinstance(module, Sequential):
        children = [clone_module_structure(module._modules[name]) for name in module._layer_order]
        return Sequential(*children)

    if isinstance(module, Linear):
        weight_shape = module.weight.shape
        return Linear(int(weight_shape[1]), int(weight_shape[0]))

    if isinstance(module, (ReLU, Sigmoid, Tanh)):
        return type(module)()

    raise TypeError(
        f"Cannot clone module type {type(module).__name__}. "
        "Supported: Sequential, Linear, ReLU, Sigmoid, Tanh."
    )


def _quantize_parameter(
    data: np.ndarray,
    config: CompressionConfig,
    parallel_config: ParallelConfig | None = None,
) -> QuantizedTensor:
    array = np.asarray(data, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"Cannot compress empty parameter with shape {array.shape}.")
    if config.rotation is not None and not is_power_of_two(int(array.shape[-1])):
        raise ValueError(
            f"Rotation requires power-of-two final axis for shape {array.shape}."
        )
    return quantize_array(
        array,
        config.bits,
        method=config.method,
        max_iterations=config.max_iterations,
        tolerance=config.tolerance,
        packed=config.packed,
        rotation=config.rotation,
        parallel_config=parallel_config,
    )


def format_compression_matrix(results: list[dict[str, Any]]) -> str:
    lines = [
        "Configuration                    Bytes      Ratio    Param RMSE",
        "---------------------------------------------------------------",
    ]
    for entry in results:
        label = (
            f"{entry['method']} {entry['bits']}bit "
            f"packed={entry['packed']} rot={entry['rotation']}"
        )
        lines.append(
            f"{label:<32} {entry['total_compressed_bytes']:<10} "
            f"{entry['compression_ratio']:.2f}x   {entry['aggregate_parameter_rmse']:.6f}"
        )
    return "\n".join(lines)

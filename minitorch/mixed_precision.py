from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .compression import (
    CompressionConfig,
    ModelCompressionReport,
    ParameterCompressionRecord,
    _compress_single_parameter,
    _resolve_parallel_config,
    clone_module_structure,
    compare_model_compression,
)
from .quantization import SUPPORTED_BIT_WIDTHS, QuantizedTensor, compute_error_metrics
from .dsa.topk import topk
from .losses import MSELoss
from .nn import Module, Parameter
from .parallel import ParallelConfig, execute_chunked
from .tensor import Tensor

QuantizationMethod = Literal["uniform", "codebook"]
RotationOption = Literal["combined", "hadamard"] | None


@dataclass
class MixedPrecisionConfig:
    """
    Per-parameter post-training precision assignment.

    Precedence: ``parameter_bits[name]`` overrides ``default_bits``.
    Global ``method``, ``packed``, and ``rotation`` apply unless extended later.
    """

    default_bits: int = 8
    method: QuantizationMethod = "uniform"
    packed: bool = False
    rotation: RotationOption = None
    max_iterations: int = 100
    tolerance: float = 1e-6
    parameter_bits: dict[str, int] = field(default_factory=dict)

    def effective_bits(self, name: str) -> int:
        return int(self.parameter_bits.get(name, self.default_bits))

    def compression_config_for(self, name: str) -> CompressionConfig:
        bits = self.effective_bits(name)
        validate_bits(bits)
        return CompressionConfig(
            bits=bits,
            method=self.method,
            packed=self.packed,
            rotation=self.rotation,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )

    def precision_map(self, model: Module) -> dict[str, int]:
        return {name: self.effective_bits(name) for name, _ in model.named_parameters()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_bits": self.default_bits,
            "method": self.method,
            "packed": self.packed,
            "rotation": self.rotation,
            "parameter_bits": dict(sorted(self.parameter_bits.items())),
        }


@dataclass(frozen=True)
class ParameterSensitivityEntry:
    name: str
    bits: int
    rmse: float
    max_absolute_error: float
    mean_absolute_error: float


@dataclass(frozen=True)
class ParameterSensitivityReport:
    method: str
    packed: bool
    rotation: str | None
    entries: tuple[ParameterSensitivityEntry, ...]

    def by_parameter(self) -> dict[str, tuple[ParameterSensitivityEntry, ...]]:
        grouped: dict[str, list[ParameterSensitivityEntry]] = {}
        for entry in self.entries:
            grouped.setdefault(entry.name, []).append(entry)
        return {name: tuple(items) for name, items in grouped.items()}

    def most_sensitive_parameters(
        self,
        bits: int,
        k: int = 5,
    ) -> tuple[tuple[str, float], ...]:
        """Top-K largest parameter reconstruction RMSE at ``bits`` (not causal importance)."""
        candidates = [e for e in self.entries if e.bits == bits]
        if not candidates:
            return ()
        values = [entry.rmse for entry in candidates]
        result = topk(values, k=min(k, len(values)), largest=True)
        ranked = sorted(
            [(candidates[i].name, candidates[i].rmse) for i in result.indices],
            key=lambda item: item[1],
            reverse=True,
        )
        return tuple(ranked)

    def format(self) -> str:
        lines = [
            "Parameter Quantization Sensitivity",
            "----------------------------------",
            f"method: {self.method}",
            f"packed: {self.packed}",
            f"rotation: {self.rotation}",
            "",
        ]
        for name in sorted(self.by_parameter().keys()):
            lines.append(name)
            for entry in self.by_parameter()[name]:
                lines.append(
                    f"  {entry.bits}-bit RMSE: {entry.rmse:.6f} "
                    f"(max abs: {entry.max_absolute_error:.6f})"
                )
            lines.append("")
        lines.append(
            "Note: sensitivity is parameter reconstruction error under quantization, "
            "not causal influence on model accuracy."
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class OutputSensitivityEntry:
    parameter_name: str
    bits: int
    output_rmse: float
    output_max_absolute_error: float
    loss_delta: float | None


@dataclass(frozen=True)
class OutputSensitivityReport:
    entries: tuple[OutputSensitivityEntry, ...]
    baseline_output_rmse: float = 0.0

    def largest_output_impact(self, k: int = 5) -> tuple[OutputSensitivityEntry, ...]:
        if not self.entries:
            return ()
        values = [entry.output_rmse for entry in self.entries]
        result = topk(values, k=min(k, len(values)), largest=True)
        return tuple(self.entries[index] for index in result.indices)

    def format(self) -> str:
        lines = [
            "Output Sensitivity Experiment",
            "-----------------------------",
            "",
            "Quantizing one parameter group at a time; others remain full precision.",
            "",
        ]
        for entry in sorted(self.entries, key=lambda e: e.output_rmse, reverse=True):
            lines.append(
                f"{entry.parameter_name} ({entry.bits}-bit): "
                f"output RMSE={entry.output_rmse:.6f}, loss_delta={entry.loss_delta}"
            )
        lines.append("")
        lines.append(
            "Interpretation: largest measured output RMSE increase in this experiment, "
            "not a causal importance score."
        )
        return "\n".join(lines)


@dataclass
class MixedPrecisionCompressedModel:
    """Post-training compressed model with per-parameter bit widths."""

    mixed_config: MixedPrecisionConfig
    parameter_records: dict[str, ParameterCompressionRecord]
    quantized_parameters: dict[str, QuantizedTensor]
    source_parameter_names: tuple[str, ...]

    def decompress_parameter(self, name: str) -> np.ndarray:
        if name not in self.quantized_parameters:
            raise KeyError(f"Parameter {name!r} is not in the mixed compressed model.")
        return self.quantized_parameters[name].dequantize().data.copy()

    def materialize(self, template: Module) -> Module:
        model = clone_module_structure(template)
        for name, parameter in model.named_parameters():
            if name not in self.quantized_parameters:
                raise KeyError(f"Mixed compressed model missing parameter {name!r}.")
            parameter.data = self.decompress_parameter(name)
            parameter.requires_grad = False
            parameter.grad = None
        return model

    def report(
        self,
        template: Module | None = None,
        inputs: Tensor | None = None,
        targets: Tensor | None = None,
    ) -> MixedPrecisionReport:
        records = tuple(self.parameter_records[name] for name in self.source_parameter_names)
        total_original = sum(record.original_bytes for record in records)
        total_compressed = sum(record.compressed_bytes for record in records)
        ratio = total_original / total_compressed if total_compressed > 0 else 0.0
        aggregate_rmse = float(np.sqrt(np.mean([record.rmse ** 2 for record in records])))

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

        precision_map = (
            self.mixed_config.precision_map(template)
            if template is not None
            else {name: self.parameter_records[name].bits for name in self.source_parameter_names}
        )

        return MixedPrecisionReport(
            mixed_config=self.mixed_config,
            parameter_count=len(records),
            total_original_bytes=total_original,
            total_compressed_bytes=total_compressed,
            compression_ratio=ratio,
            aggregate_parameter_rmse=aggregate_rmse,
            parameter_records=records,
            precision_map=precision_map,
            output_max_absolute_error=output_errors.max_absolute_error if output_errors else None,
            output_mean_absolute_error=output_errors.mean_absolute_error if output_errors else None,
            output_rmse=output_errors.rmse if output_errors else None,
            original_loss=original_loss,
            compressed_loss=compressed_loss,
            loss_delta=loss_delta,
        )

    def xray(self) -> dict[str, Any]:
        return self.report().xray()


@dataclass
class MixedPrecisionReport:
    mixed_config: MixedPrecisionConfig
    parameter_count: int
    total_original_bytes: int
    total_compressed_bytes: int
    compression_ratio: float
    aggregate_parameter_rmse: float
    parameter_records: tuple[ParameterCompressionRecord, ...]
    precision_map: dict[str, int]
    output_max_absolute_error: float | None = None
    output_mean_absolute_error: float | None = None
    output_rmse: float | None = None
    original_loss: float | None = None
    compressed_loss: float | None = None
    loss_delta: float | None = None

    @property
    def is_mixed(self) -> bool:
        bits = set(self.precision_map.values())
        return len(bits) > 1

    def largest_parameter_errors(self, k: int = 5) -> tuple[ParameterCompressionRecord, ...]:
        if not self.parameter_records:
            return ()
        names = [record.name for record in self.parameter_records]
        rmse_values = [record.rmse for record in self.parameter_records]
        result = topk(rmse_values, k=min(k, len(rmse_values)), largest=True)
        return tuple(self.parameter_records[index] for index in result.indices)

    def as_model_compression_report(self) -> ModelCompressionReport:
        """View for AI-MRI and legacy compression consumers."""
        representative = CompressionConfig(
            bits=self.mixed_config.default_bits,
            method=self.mixed_config.method,
            packed=self.mixed_config.packed,
            rotation=self.mixed_config.rotation,
        )
        return ModelCompressionReport(
            parameter_count=self.parameter_count,
            config=representative,
            total_original_bytes=self.total_original_bytes,
            total_compressed_bytes=self.total_compressed_bytes,
            compression_ratio=self.compression_ratio,
            aggregate_parameter_rmse=self.aggregate_parameter_rmse,
            parameter_records=self.parameter_records,
            output_max_absolute_error=self.output_max_absolute_error,
            output_mean_absolute_error=self.output_mean_absolute_error,
            output_rmse=self.output_rmse,
            original_loss=self.original_loss,
            compressed_loss=self.compressed_loss,
            loss_delta=self.loss_delta,
        )

    def format(self) -> str:
        lines = [
            "Mixed Precision Compression Report",
            "----------------------------------",
            "",
            f"Global default bits: {self.mixed_config.default_bits}",
            f"Mixed assignment: {self.is_mixed}",
            f"method: {self.mixed_config.method}",
            f"packed: {self.mixed_config.packed}",
            f"rotation: {self.mixed_config.rotation}",
            "",
            "Parameter precisions:",
        ]
        for name in sorted(self.precision_map):
            lines.append(f"  {name}: {self.precision_map[name]}-bit")
        lines.extend(
            [
                "",
                f"Original bytes: {self.total_original_bytes}",
                f"Compressed bytes: {self.total_compressed_bytes}",
                f"Compression ratio: {self.compression_ratio:.2f}x",
                f"Aggregate parameter RMSE: {self.aggregate_parameter_rmse:.6f}",
            ]
        )
        if self.output_rmse is not None:
            lines.append(f"Output RMSE: {self.output_rmse:.6f}")
        if self.loss_delta is not None:
            lines.append(f"Loss delta: {self.loss_delta:.6f}")
        return "\n".join(lines)

    def xray(self) -> dict[str, Any]:
        base = self.as_model_compression_report().xray()
        base["mixed_precision"] = True
        base["precision_map"] = dict(self.precision_map)
        base["mixed_config"] = self.mixed_config.to_dict()
        return base


@dataclass(frozen=True)
class MixedPrecisionComparisonRow:
    label: str
    total_bytes: int
    compression_ratio: float
    output_rmse: float | None
    loss_delta: float | None
    aggregate_parameter_rmse: float


@dataclass(frozen=True)
class MixedPrecisionComparisonResult:
    rows: tuple[MixedPrecisionComparisonRow, ...]

    def format_table(self) -> str:
        lines = [
            "Configuration              Bytes    Ratio    Output RMSE    Loss Delta",
            "---------------------------------------------------------------------",
        ]
        for row in self.rows:
            out = f"{row.output_rmse:.6f}" if row.output_rmse is not None else "n/a"
            loss = f"{row.loss_delta:.6f}" if row.loss_delta is not None else "n/a"
            lines.append(
                f"{row.label:<26} {row.total_bytes:<8} {row.compression_ratio:.2f}x   "
                f"{out:<14} {loss}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class GreedyStepDecision:
    parameter_name: str
    from_bits: int
    to_bits: int
    memory_saving_bytes: int
    output_rmse: float
    loss_delta: float | None
    accepted: bool
    reason: str


@dataclass
class GreedyMixedPrecisionResult:
    config: MixedPrecisionConfig
    compressed: MixedPrecisionCompressedModel
    report: MixedPrecisionReport
    steps: tuple[GreedyStepDecision, ...]

    def format(self) -> str:
        lines = ["Greedy Mixed Precision Heuristic", "------------------------------", ""]
        for index, step in enumerate(self.steps, start=1):
            lines.extend(
                [
                    f"Step {index}",
                    f"  Candidate: {step.parameter_name} {step.from_bits} -> {step.to_bits} bits",
                    f"  Memory saving: {step.memory_saving_bytes} bytes",
                    f"  Output RMSE: {step.output_rmse:.6f}",
                    f"  Loss delta: {step.loss_delta}",
                    f"  Decision: {'accepted' if step.accepted else 'rejected'} ({step.reason})",
                    "",
                ]
            )
        lines.append("Note: greedy heuristic, not globally optimal.")
        return "\n".join(lines)


def validate_bits(bits: int) -> None:
    if bits not in SUPPORTED_BIT_WIDTHS:
        supported = ", ".join(str(value) for value in SUPPORTED_BIT_WIDTHS)
        raise ValueError(f"Unsupported bit width {bits}. Supported: {supported}.")


def validate_mixed_precision_config(model: Module, config: MixedPrecisionConfig) -> None:
    validate_bits(config.default_bits)
    known = {name for name, _ in model.named_parameters()}
    for name, bits in config.parameter_bits.items():
        if name not in known:
            raise ValueError(f"Unknown parameter in mixed precision config: {name!r}.")
        validate_bits(bits)
    if config.parameter_bits:
        duplicates = sorted(set(config.parameter_bits.keys()))
        if len(duplicates) != len(config.parameter_bits):
            raise ValueError("Duplicate parameter entries in mixed precision config.")


def compress_model_mixed_precision(
    model: Module,
    config: MixedPrecisionConfig,
    *,
    parallel_config: ParallelConfig | None = None,
    parallel: bool = False,
    workers: int | None = None,
) -> MixedPrecisionCompressedModel:
    """Compress each parameter with its effective mixed-precision configuration."""
    validate_mixed_precision_config(model, config)
    parallel_settings = _resolve_parallel_config(parallel_config, parallel, workers)
    items = list(model.named_parameters())

    def compress_chunk(start: int, end: int) -> list[tuple[str, ParameterCompressionRecord, QuantizedTensor]]:
        chunk: list[tuple[str, ParameterCompressionRecord, QuantizedTensor]] = []
        for name, parameter in items[start:end]:
            param_config = config.compression_config_for(name)
            chunk.append(
                _compress_single_parameter(name, parameter, param_config, parallel_settings)
            )
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

    return MixedPrecisionCompressedModel(
        mixed_config=config,
        parameter_records=records,
        quantized_parameters=quantized,
        source_parameter_names=tuple(names),
    )


def measure_parameter_sensitivity(
    model: Module,
    *,
    bits: tuple[int, ...] = (2, 4, 8),
    method: QuantizationMethod = "uniform",
    packed: bool = False,
    rotation: RotationOption = None,
) -> ParameterSensitivityReport:
    entries: list[ParameterSensitivityEntry] = []
    for name, parameter in model.named_parameters():
        for bit_width in sorted(bits):
            validate_bits(bit_width)
            config = CompressionConfig(
                bits=bit_width,
                method=method,
                packed=packed,
                rotation=rotation,
            )
            _, record, _ = _compress_single_parameter(
                name,
                parameter,
                config,
                ParallelConfig(),
            )
            entries.append(
                ParameterSensitivityEntry(
                    name=name,
                    bits=bit_width,
                    rmse=record.rmse,
                    max_absolute_error=record.max_absolute_error,
                    mean_absolute_error=record.mean_absolute_error,
                )
            )
    return ParameterSensitivityReport(
        method=method,
        packed=packed,
        rotation=rotation,
        entries=tuple(entries),
    )


def output_sensitivity_experiment(
    model: Module,
    inputs: Tensor,
    targets: Tensor | None = None,
    *,
    bits: int = 4,
    method: QuantizationMethod = "uniform",
    packed: bool = False,
    rotation: RotationOption = None,
) -> OutputSensitivityReport:
    validate_bits(bits)
    baseline_outputs = model(inputs)
    baseline_rmse = 0.0
    criterion = MSELoss()
    baseline_loss = float(criterion(baseline_outputs, targets).data) if targets is not None else None

    entries: list[OutputSensitivityEntry] = []
    config = CompressionConfig(bits=bits, method=method, packed=packed, rotation=rotation)

    for name, parameter in model.named_parameters():
        eval_model = clone_module_structure(model)
        for param_name, param in eval_model.named_parameters():
            if param_name == name:
                _, _, quantized = _compress_single_parameter(
                    param_name,
                    parameter,
                    config,
                    ParallelConfig(),
                )
                param.data = quantized.dequantize().data.copy()
            else:
                original = model.parameter(param_name)
                param.data = original.data.copy()

        outputs = eval_model(inputs)
        errors = compute_error_metrics(baseline_outputs.data, outputs.data)
        loss_delta = None
        if targets is not None:
            compressed_loss = float(criterion(outputs, targets).data)
            loss_delta = compressed_loss - baseline_loss

        entries.append(
            OutputSensitivityEntry(
                parameter_name=name,
                bits=bits,
                output_rmse=errors.rmse,
                output_max_absolute_error=errors.max_absolute_error,
                loss_delta=loss_delta,
            )
        )

    return OutputSensitivityReport(entries=tuple(entries), baseline_output_rmse=baseline_rmse)


def compare_mixed_precision(
    model: Module,
    inputs: Tensor,
    targets: Tensor | None = None,
    mixed_config: MixedPrecisionConfig | None = None,
    *,
    global_bits: tuple[int, ...] = (8, 4, 2),
) -> MixedPrecisionComparisonResult:
    rows: list[MixedPrecisionComparisonRow] = []

    if mixed_config is not None:
        compressed = compress_model_mixed_precision(model, mixed_config)
        report = compressed.report(template=model, inputs=inputs, targets=targets)
        rows.append(
            MixedPrecisionComparisonRow(
                label="Mixed precision",
                total_bytes=report.total_compressed_bytes,
                compression_ratio=report.compression_ratio,
                output_rmse=report.output_rmse,
                loss_delta=report.loss_delta,
                aggregate_parameter_rmse=report.aggregate_parameter_rmse,
            )
        )

    fp_original = sum(p.data.nbytes for _, p in model.named_parameters())
    rows.insert(
        0,
        MixedPrecisionComparisonRow(
            label="FP32",
            total_bytes=fp_original,
            compression_ratio=1.0,
            output_rmse=0.0,
            loss_delta=0.0,
            aggregate_parameter_rmse=0.0,
        ),
    )

    for bits in global_bits:
        validate_bits(bits)
        global_report = compare_model_compression(
            model,
            inputs,
            targets,
            config=CompressionConfig(bits=bits, method=mixed_config.method if mixed_config else "uniform"),
        )
        rows.append(
            MixedPrecisionComparisonRow(
                label=f"Uniform {bits}-bit",
                total_bytes=global_report.total_compressed_bytes,
                compression_ratio=global_report.compression_ratio,
                output_rmse=global_report.output_rmse,
                loss_delta=global_report.loss_delta,
                aggregate_parameter_rmse=global_report.aggregate_parameter_rmse,
            )
        )

    return MixedPrecisionComparisonResult(rows=tuple(rows))


def _config_from_bit_map(
    bit_map: dict[str, int],
    *,
    method: QuantizationMethod = "uniform",
    packed: bool = False,
    rotation: RotationOption = None,
) -> MixedPrecisionConfig:
    default_bits = 8
    overrides = {name: bits for name, bits in bit_map.items() if bits != default_bits}
    return MixedPrecisionConfig(
        default_bits=default_bits,
        method=method,
        packed=packed,
        rotation=rotation,
        parameter_bits=overrides,
    )


def greedy_mixed_precision_compress(
    model: Module,
    inputs: Tensor,
    targets: Tensor | None = None,
    *,
    max_output_rmse: float | None = None,
    max_loss_delta: float | None = None,
    method: QuantizationMethod = "uniform",
    packed: bool = False,
    rotation: RotationOption = None,
) -> GreedyMixedPrecisionResult:
    """
    Greedy heuristic: reduce one parameter's precision at a time while satisfying constraints.

    Not globally optimal.
    """
    if max_output_rmse is None and max_loss_delta is None:
        raise ValueError("Provide max_output_rmse and/or max_loss_delta constraint.")

    working_bits = {name: 8 for name, _ in model.named_parameters()}
    steps: list[GreedyStepDecision] = []
    bit_ladder = {8: 4, 4: 2}

    current_compressed = compress_model_mixed_precision(
        model, _config_from_bit_map(working_bits, method=method, packed=packed, rotation=rotation)
    )
    current_report = current_compressed.report(template=model, inputs=inputs, targets=targets)
    current_bytes = current_report.total_compressed_bytes

    while True:
        candidates: list[tuple[str, int, int, int, float, float | None]] = []
        for name in sorted(working_bits):
            current_bits = working_bits[name]
            next_bits = bit_ladder.get(current_bits)
            if next_bits is None:
                continue
            trial_bits = dict(working_bits)
            trial_bits[name] = next_bits
            trial_config = _config_from_bit_map(
                trial_bits,
                method=method,
                packed=packed,
                rotation=rotation,
            )
            trial_compressed = compress_model_mixed_precision(model, trial_config)
            trial_report = trial_compressed.report(template=model, inputs=inputs, targets=targets)
            saving = current_bytes - trial_report.total_compressed_bytes
            candidates.append(
                (
                    name,
                    current_bits,
                    next_bits,
                    saving,
                    trial_report.output_rmse or 0.0,
                    trial_report.loss_delta,
                )
            )

        if not candidates:
            break

        candidates.sort(key=lambda item: (-item[3], item[0]))
        accepted_any = False
        for name, from_bits, to_bits, saving, output_rmse, loss_delta in candidates:
            ok_rmse = max_output_rmse is None or output_rmse <= max_output_rmse
            ok_loss = max_loss_delta is None or (
                loss_delta is not None and loss_delta <= max_loss_delta
            )
            accepted = ok_rmse and ok_loss
            reason = "constraint satisfied" if accepted else "constraint violated"
            steps.append(
                GreedyStepDecision(
                    parameter_name=name,
                    from_bits=from_bits,
                    to_bits=to_bits,
                    memory_saving_bytes=int(saving),
                    output_rmse=output_rmse,
                    loss_delta=loss_delta,
                    accepted=accepted,
                    reason=reason,
                )
            )
            if accepted:
                working_bits[name] = to_bits
                current_bytes -= saving
                accepted_any = True
                break

        if not accepted_any:
            break

    final_config = _config_from_bit_map(
        working_bits,
        method=method,
        packed=packed,
        rotation=rotation,
    )
    final_compressed = compress_model_mixed_precision(model, final_config)
    final_report = final_compressed.report(template=model, inputs=inputs, targets=targets)
    return GreedyMixedPrecisionResult(
        config=final_config,
        compressed=final_compressed,
        report=final_report,
        steps=tuple(steps),
    )


def benchmark_mixed_precision(
    model: Module,
    config: MixedPrecisionConfig,
    *,
    inputs: Tensor | None = None,
    targets: Tensor | None = None,
    repeats: int = 3,
) -> dict[str, Any]:
    start = time.perf_counter()
    compressed = None
    for _ in range(repeats):
        compressed = compress_model_mixed_precision(model, config)
    compress_seconds = (time.perf_counter() - start) / repeats

    reconstruct_seconds = 0.0
    if compressed is not None:
        start = time.perf_counter()
        for _ in range(repeats):
            _ = compressed.materialize(model)
        reconstruct_seconds = (time.perf_counter() - start) / repeats

    report = (
        compressed.report(template=model, inputs=inputs, targets=targets)
        if compressed is not None and inputs is not None
        else compressed.report()
    )
    return {
        "compress_seconds": compress_seconds,
        "reconstruct_seconds": reconstruct_seconds,
        "total_original_bytes": report.total_original_bytes,
        "total_compressed_bytes": report.total_compressed_bytes,
        "compression_ratio": report.compression_ratio,
        "output_rmse": report.output_rmse,
        "loss_delta": report.loss_delta,
        "precision_map": report.precision_map,
    }

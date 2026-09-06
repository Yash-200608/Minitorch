from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .compression import CompressionConfig, compare_model_compression, clone_module_structure
from .kernels.uniform import uniform_codes
from .nn import Linear, Module, ReLU, Sequential, Sigmoid, Tanh
from .quantization import (
    SUPPORTED_BIT_WIDTHS,
    compute_error_metrics,
    validate_bits,
    _dequantize_uniform_codes,
    _validate_quantization_input,
)
from .tensor import Tensor


STEMode = Literal["identity"]
QATTarget = Literal["weights"]


@dataclass(frozen=True)
class QATConfig:
    """Configuration for weight-only uniform fake quantization."""

    bits: int = 4
    method: str = "uniform"
    ste_mode: STEMode = "identity"
    enabled: bool = True
    target: QATTarget = "weights"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "method": self.method,
            "ste_mode": self.ste_mode,
            "enabled": self.enabled,
            "target": self.target,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> QATConfig:
        return QATConfig(
            bits=int(data.get("bits", 4)),
            method=str(data.get("method", "uniform")),
            ste_mode=data.get("ste_mode", "identity"),
            enabled=bool(data.get("enabled", True)),
            target=data.get("target", "weights"),
        )


@dataclass(frozen=True)
class QATExperimentResult:
    bits: int
    ste_mode: str
    normal_fp_loss: float
    normal_quantized_loss: float
    qat_fp_loss: float
    qat_quantized_loss: float
    normal_quantization_damage: float
    qat_quantization_damage: float
    normal_output_rmse: float
    qat_output_rmse: float
    normal_parameter_rmse: float
    qat_parameter_rmse: float
    normal_training_seconds: float
    qat_training_seconds: float
    qat_forward_overhead_ratio: float
    result_label: str

    def format(self) -> str:
        lines = [
            "Quantization-Aware Training Experiment",
            "---------------------------------------",
            "",
            "Configuration:",
            f"  bits: {self.bits}",
            f"  method: uniform",
            f"  STE: {self.ste_mode}",
            "",
            "NORMAL TRAINING",
            f"  FP loss: {self.normal_fp_loss:.6f}",
            f"  Quantized loss: {self.normal_quantized_loss:.6f}",
            "",
            "QAT",
            f"  FP loss: {self.qat_fp_loss:.6f}",
            f"  Quantized loss: {self.qat_quantized_loss:.6f}",
            "",
            "QUANTIZATION DAMAGE",
            f"  post-training: {self.normal_quantization_damage:.6f}",
            f"  QAT: {self.qat_quantization_damage:.6f}",
            "",
            "OUTPUT RMSE (quantized vs FP)",
            f"  post-training: {self.normal_output_rmse:.6f}",
            f"  QAT: {self.qat_output_rmse:.6f}",
            "",
            "PARAMETER RMSE (quantized vs FP)",
            f"  post-training: {self.normal_parameter_rmse:.6f}",
            f"  QAT: {self.qat_parameter_rmse:.6f}",
            "",
            "TRAINING TIME (seconds)",
            f"  normal: {self.normal_training_seconds:.6f}",
            f"  QAT: {self.qat_training_seconds:.6f}",
            "",
            f"QAT forward overhead ratio: {self.qat_forward_overhead_ratio:.3f}x",
            "",
            f"Result: {self.result_label}",
        ]
        return "\n".join(lines)


def validate_ste_mode(ste_mode: str) -> None:
    if ste_mode != "identity":
        raise ValueError(f"Unsupported STE mode {ste_mode!r}. Supported: identity.")


def fake_quantize_uniform_array(data: np.ndarray, bits: int) -> np.ndarray:
    """
    Forward fake quantization: dequantize(uniform_quantize(x)).

    Does not create a QuantizedTensor and does not attach to autograd.
    """
    validate_bits(bits)
    array = _validate_quantization_input(data)
    shape = tuple(array.shape)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    levels = 2 ** bits
    max_code = levels - 1

    if minimum == maximum:
        codes = np.zeros(shape, dtype=np.int32)
        scale = 0.0
    else:
        scale = (maximum - minimum) / max_code
        codes = uniform_codes(
            array,
            minimum=minimum,
            maximum=maximum,
            max_code=max_code,
            backend="vectorized",
        ).reshape(shape)

    return _dequantize_uniform_codes(codes, minimum, maximum, scale, shape)


def walk_modules(model: Module):
    yield model
    for child in model._modules.values():
        yield from walk_modules(child)


def apply_qat_config(model: Module, config: QATConfig) -> None:
    for module in walk_modules(model):
        if isinstance(module, Linear):
            module.qat_config = config


def is_qat_prepared(model: Module) -> bool:
    return get_model_qat_config(model) is not None


def get_model_qat_config(model: Module) -> QATConfig | None:
    for module in walk_modules(model):
        if isinstance(module, Linear) and module.qat_config is not None and module.qat_config.enabled:
            return module.qat_config
    return None


def qat_summary(model: Module) -> dict[str, Any]:
    config = get_model_qat_config(model)
    fake_quant_layers = sum(
        1
        for module in walk_modules(model)
        if isinstance(module, Linear) and module.qat_config is not None and module.qat_config.enabled
    )
    param_count = sum(
        1
        for module in walk_modules(model)
        if isinstance(module, Linear) and module.qat_config is not None and module.qat_config.enabled
    )
    return {
        "enabled": config is not None,
        "bits": config.bits if config else None,
        "method": config.method if config else None,
        "ste_mode": config.ste_mode if config else None,
        "target": config.target if config else None,
        "fake_quantized_layers": fake_quant_layers,
        "fake_quantized_parameters": param_count,
    }


def prepare_qat(model: Module, config: QATConfig | None = None) -> Module:
    """
    Return a copy of ``model`` with weight-only QAT enabled on Linear layers.

    The original model is not modified.
    """
    config = config or QATConfig()
    if config.method != "uniform":
        raise ValueError("Phase O supports uniform fake quantization only.")
    validate_bits(config.bits)
    validate_ste_mode(config.ste_mode)

    clone = clone_module_structure(model)
    for src_name, src_param in model.named_parameters():
        clone.parameter(src_name).data = src_param.data.copy()
    apply_qat_config(clone, config)
    return clone


def benchmark_qat_forward(model: Module, inputs: Tensor, *, repeats: int = 5) -> dict[str, float]:
    model.eval()
    start = time.perf_counter()
    for _ in range(repeats):
        _ = model(inputs)
    normal_seconds = (time.perf_counter() - start) / repeats

    qat_model = prepare_qat(model, get_model_qat_config(model) or QATConfig(bits=4))
    qat_model.train()
    start = time.perf_counter()
    for _ in range(repeats):
        _ = qat_model(inputs)
    qat_seconds = (time.perf_counter() - start) / repeats

    return {
        "normal_forward_seconds": normal_seconds,
        "qat_forward_seconds": qat_seconds,
        "forward_overhead_ratio": qat_seconds / normal_seconds if normal_seconds > 0 else 0.0,
    }


def run_qat_experiment(
    model: Module,
    optimizer_factory,
    loss_fn,
    inputs: Tensor,
    targets: Tensor,
    *,
    epochs: int = 200,
    bits: int = 4,
    ste_mode: STEMode = "identity",
) -> QATExperimentResult:
    """Train normal vs QAT models and compare post-training quantization damage."""
    from .training.trainer import Trainer

    config = CompressionConfig(bits=bits, method="uniform", packed=False)

    normal_model = clone_module_structure(model)
    for name, parameter in model.named_parameters():
        normal_model.parameter(name).data = parameter.data.copy()

    start = time.perf_counter()
    normal_trainer = Trainer(normal_model, optimizer_factory(normal_model), loss_fn)
    normal_trainer.fit(inputs, targets, epochs=epochs)
    normal_train_seconds = time.perf_counter() - start

    qat_model = prepare_qat(model, QATConfig(bits=bits, ste_mode=ste_mode))
    start = time.perf_counter()
    qat_trainer = Trainer(qat_model, optimizer_factory(qat_model), loss_fn)
    qat_trainer.fit(inputs, targets, epochs=epochs)
    qat_train_seconds = time.perf_counter() - start

    normal_fp = float(loss_fn(normal_model(inputs), targets).data)
    qat_fp = float(loss_fn(qat_model(inputs), targets).data)

    normal_report = compare_model_compression(normal_model, inputs, targets, config=config)
    qat_report = compare_model_compression(qat_model, inputs, targets, config=config)

    normal_q_loss = float(normal_report.compressed_loss or 0.0)
    qat_q_loss = float(qat_report.compressed_loss or 0.0)
    normal_damage = normal_q_loss - normal_fp
    qat_damage = qat_q_loss - qat_fp

    if qat_damage < normal_damage - 1e-9:
        label = "QAT improved quantization damage"
    elif abs(qat_damage - normal_damage) <= 1e-9:
        label = "QAT unchanged"
    else:
        label = "QAT worse than post-training quantization"

    overhead = benchmark_qat_forward(normal_model, inputs)

    return QATExperimentResult(
        bits=bits,
        ste_mode=ste_mode,
        normal_fp_loss=normal_fp,
        normal_quantized_loss=normal_q_loss,
        qat_fp_loss=qat_fp,
        qat_quantized_loss=qat_q_loss,
        normal_quantization_damage=normal_damage,
        qat_quantization_damage=qat_damage,
        normal_output_rmse=float(normal_report.output_rmse or 0.0),
        qat_output_rmse=float(qat_report.output_rmse or 0.0),
        normal_parameter_rmse=normal_report.aggregate_parameter_rmse,
        qat_parameter_rmse=qat_report.aggregate_parameter_rmse,
        normal_training_seconds=normal_train_seconds,
        qat_training_seconds=qat_train_seconds,
        qat_forward_overhead_ratio=overhead["forward_overhead_ratio"],
        result_label=label,
    )


def parameter_quantization_rmse(model: Module, bits: int) -> float:
    total_sq = 0.0
    total_count = 0
    for _, parameter in model.named_parameters():
        original = parameter.data
        reconstructed = fake_quantize_uniform_array(original, bits)
        diff = original - reconstructed
        total_sq += float(np.sum(diff * diff))
        total_count += int(original.size)
    if total_count == 0:
        return 0.0
    return float(np.sqrt(total_sq / total_count))

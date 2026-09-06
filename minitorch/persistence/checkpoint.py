from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..nn import Module, Parameter
from ..qat import apply_qat_config, get_model_qat_config, QATConfig
from ..training.optimizers import Adam, Optimizer, SGD
from ..training.trainer import TrainingHistory
from .format import (
    ARTIFACT_CHECKPOINT,
    FRAMEWORK_VERSION,
    ArtifactError,
)
from .io import (
    _deserialize_model,
    _read_artifact,
    _serialize_model,
    _write_artifact,
)
from .format import (
    ARTIFACT_CHECKPOINT,
    dumps_metadata,
)


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass
class TrainingResumeState:
    """
    Restored training position.

    ``completed_epochs`` counts finished epoch iterations.
    ``completed_steps`` counts finished optimizer updates.
    """

    completed_epochs: int
    completed_steps: int
    losses: list[float] = field(default_factory=list)
    history_epochs: list[int] = field(default_factory=list)
    rng_state_present: bool = False
    qat_config: dict[str, Any] | None = None

    def to_history(self) -> TrainingHistory:
        return TrainingHistory(losses=list(self.losses), epochs=list(self.history_epochs))


@dataclass
class TrainingCheckpoint:
    """In-memory training checkpoint loaded from an artifact."""

    path: str | None
    metadata: dict[str, Any]
    payload: bytes
    resume_state: TrainingResumeState
    optimizer_config: dict[str, Any]

    def restore(
        self,
        optimizer: Optimizer | None = None,
        *,
        strict: bool = True,
    ) -> tuple[Module, Optimizer, TrainingResumeState]:
        model = _deserialize_model(self.metadata, self._model_payload())
        _apply_checkpoint_qat_config(model, self.metadata)
        if optimizer is None:
            optimizer = build_optimizer_from_config(model, self.optimizer_config)
        else:
            Optimizer.validate_config_match(self.optimizer_config, optimizer, strict=strict)
            _validate_parameter_names(model, self.metadata.get("parameters", []), strict=strict)

        parameters_by_name = {name: param for name, param in model.named_parameters()}
        _load_optimizer_binary_state(
            optimizer,
            self.metadata.get("optimizer_state", {}),
            self.payload,
            parameters_by_name,
            strict=strict,
        )
        if self.metadata.get("rng_state") is not None:
            _restore_rng_state(self.metadata["rng_state"])
        return model, optimizer, self.resume_state

    def restore_into(
        self,
        model: Module,
        optimizer: Optimizer,
        *,
        strict: bool = True,
    ) -> TrainingResumeState:
        """Load checkpoint state into an existing model and optimizer."""
        _validate_parameter_names(model, self.metadata.get("parameters", []), strict=strict)
        _load_model_payload_into(model, self.metadata.get("parameters", []), self._model_payload())
        Optimizer.validate_config_match(self.optimizer_config, optimizer, strict=strict)
        parameters_by_name = {name: param for name, param in model.named_parameters()}
        _load_optimizer_binary_state(
            optimizer,
            self.metadata.get("optimizer_state", {}),
            self.payload,
            parameters_by_name,
            strict=strict,
        )
        if self.metadata.get("rng_state") is not None:
            _restore_rng_state(self.metadata["rng_state"])
        return self.resume_state

    def report(self) -> str:
        return format_checkpoint_report(self.path or "<memory>", self.metadata, len(self.payload))

    def _model_payload(self) -> bytes:
        model_bytes = int(self.metadata.get("model_payload_bytes", 0))
        return self.payload[:model_bytes]


def save_training_checkpoint(
    model: Module,
    optimizer: Optimizer,
    resume_state: TrainingResumeState,
    path: str | Path,
    *,
    include_rng: bool = True,
) -> None:
    path = Path(path)
    metadata, payload = _serialize_checkpoint(
        model,
        optimizer,
        resume_state,
        include_rng=include_rng,
    )
    _write_artifact(path, ARTIFACT_CHECKPOINT, metadata, payload)


def load_training_checkpoint(path: str | Path) -> TrainingCheckpoint:
    path = Path(path)
    metadata, payload = _read_artifact(path, ARTIFACT_CHECKPOINT)
    training = metadata.get("training", {})
    resume_state = TrainingResumeState(
        completed_epochs=int(training.get("completed_epochs", 0)),
        completed_steps=int(training.get("completed_steps", 0)),
        losses=[float(value) for value in training.get("losses", [])],
        history_epochs=[int(value) for value in training.get("history_epochs", [])],
        rng_state_present=metadata.get("rng_state") is not None,
        qat_config=metadata.get("qat_config"),
    )
    return TrainingCheckpoint(
        path=str(path),
        metadata=metadata,
        payload=payload,
        resume_state=resume_state,
        optimizer_config=dict(metadata.get("optimizer_config", {})),
    )


def build_optimizer_from_config(model: Module, config: dict[str, Any]) -> Optimizer:
    optimizer_type = config.get("type")
    parameters = model.parameters()
    if optimizer_type == "SGD":
        return SGD(
            parameters,
            lr=float(config.get("lr", 0.01)),
            momentum=float(config.get("momentum", 0.0)),
        )
    if optimizer_type == "Adam":
        return Adam(
            parameters,
            lr=float(config.get("lr", 1e-3)),
            beta1=float(config.get("beta1", 0.9)),
            beta2=float(config.get("beta2", 0.999)),
            eps=float(config.get("eps", 1e-8)),
        )
    raise ArtifactError(f"Unsupported checkpoint optimizer type: {optimizer_type!r}")


def format_checkpoint_report(path: str, metadata: dict[str, Any], payload_bytes: int) -> str:
    training = metadata.get("training", {})
    optimizer_config = metadata.get("optimizer_config", {})
    parameter_specs = metadata.get("parameters", [])
    model_param_bytes = sum(int(spec.get("length", 0)) for spec in parameter_specs)
    optimizer_state_bytes = int(metadata.get("optimizer_state_bytes", 0))
    history_losses = training.get("losses", [])
    history_bytes = len(dumps_metadata({"losses": history_losses})) if history_losses else 0

    lines = [
        "MiniTorch Training Checkpoint",
        "------------------------------",
        "",
        f"Path: {path}",
        f"Format version: {metadata.get('format_version', 1)}",
        f"Checkpoint schema: {metadata.get('checkpoint_schema_version', 1)}",
        f"Framework: {metadata.get('framework_version')}",
        "",
        "Artifact: checkpoint",
        f"Model: {metadata.get('module_tree', {}).get('type')}",
        f"Parameters: {len(parameter_specs)}",
        "",
        "Training:",
        f"  completed epochs: {training.get('completed_epochs')}",
        f"  completed steps: {training.get('completed_steps')}",
        f"  loss history entries: {len(history_losses)}",
        "",
        f"Optimizer: {optimizer_config.get('type')}",
        f"  lr: {optimizer_config.get('lr')}",
    ]
    if optimizer_config.get("type") == "SGD":
        lines.append(f"  momentum: {optimizer_config.get('momentum')}")
    if optimizer_config.get("type") == "Adam":
        lines.extend(
            [
                f"  beta1: {optimizer_config.get('beta1')}",
                f"  beta2: {optimizer_config.get('beta2')}",
                f"  eps: {optimizer_config.get('eps')}",
                f"  optimizer step: {metadata.get('optimizer_state', {}).get('step')}",
            ]
        )
    opt_entries = metadata.get("optimizer_state", {}).get("entries", [])
    lines.extend(
        [
            "",
            f"Optimizer state entries: {len(opt_entries)}",
            f"Model parameter bytes: {model_param_bytes}",
            f"Optimizer state bytes: {optimizer_state_bytes}",
            f"History bytes (loss list JSON): {history_bytes}",
            f"RNG state: {'present' if metadata.get('rng_state') is not None else 'absent'}",
            f"Payload bytes: {payload_bytes}",
        ]
    )
    return "\n".join(lines)


def benchmark_checkpoint(
    model: Module,
    optimizer: Optimizer,
    resume_state: TrainingResumeState,
    *,
    repeats: int = 3,
) -> dict[str, Any]:
    import tempfile
    import time
    from pathlib import Path

    from .io import save_model

    results: dict[str, Any] = {"repeats": repeats}
    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "model.mt"
        checkpoint_path = Path(tmp) / "run.mtc"

        save_model(model, model_path)
        results["model_file_bytes"] = model_path.stat().st_size

        start = time.perf_counter()
        for _ in range(repeats):
            save_training_checkpoint(model, optimizer, resume_state, checkpoint_path)
        results["checkpoint_save_seconds"] = (time.perf_counter() - start) / repeats

        start = time.perf_counter()
        for _ in range(repeats):
            _ = load_training_checkpoint(checkpoint_path)
        results["checkpoint_load_seconds"] = (time.perf_counter() - start) / repeats
        results["checkpoint_file_bytes"] = checkpoint_path.stat().st_size
        if results["model_file_bytes"] > 0:
            results["checkpoint_overhead_ratio"] = (
                results["checkpoint_file_bytes"] / results["model_file_bytes"]
            )
    return results


def _serialize_checkpoint(
    model: Module,
    optimizer: Optimizer,
    resume_state: TrainingResumeState,
    *,
    include_rng: bool,
) -> tuple[dict[str, Any], bytes]:
    model_metadata, model_payload = _serialize_model(model)
    optimizer_config = optimizer.config_dict()
    optimizer_payload, optimizer_state_meta = _serialize_optimizer_state(
        model,
        optimizer,
        base_offset=len(model_payload),
    )

    metadata = {
        "framework_version": FRAMEWORK_VERSION,
        "format_version": 1,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "module_tree": model_metadata["module_tree"],
        "parameters": model_metadata["parameters"],
        "gradients_saved": False,
        "model_payload_bytes": len(model_payload),
        "optimizer_config": optimizer_config,
        "optimizer_state": optimizer_state_meta,
        "optimizer_state_bytes": len(optimizer_payload),
        "training": {
            "completed_epochs": resume_state.completed_epochs,
            "completed_steps": resume_state.completed_steps,
            "losses": list(resume_state.losses),
            "history_epochs": list(resume_state.history_epochs),
        },
    }
    if include_rng:
        metadata["rng_state"] = _serialize_rng_state()
    qat_config = get_model_qat_config(model)
    if qat_config is not None:
        metadata["qat_config"] = qat_config.to_dict()
    payload = model_payload + optimizer_payload
    return metadata, payload


def _serialize_optimizer_state(
    model: Module,
    optimizer: Optimizer,
    *,
    base_offset: int,
) -> tuple[bytes, dict[str, Any]]:
    parameters_by_name = {name: param for name, param in model.named_parameters()}
    entries: list[dict[str, Any]] = []
    payload_parts: list[bytes] = []
    offset = base_offset
    meta: dict[str, Any] = {"entries": entries}

    if isinstance(optimizer, Adam):
        meta["step"] = optimizer._step
        for name in sorted(parameters_by_name):
            parameter = parameters_by_name[name]
            m1 = optimizer._moment1[id(parameter)]
            m2 = optimizer._moment2[id(parameter)]
            m1_bytes = m1.tobytes(order="C")
            m2_bytes = m2.tobytes(order="C")
            entries.append(
                {
                    "name": name,
                    "m1_offset": offset,
                    "m1_length": len(m1_bytes),
                    "m2_offset": offset + len(m1_bytes),
                    "m2_length": len(m2_bytes),
                }
            )
            payload_parts.append(m1_bytes)
            payload_parts.append(m2_bytes)
            offset += len(m1_bytes) + len(m2_bytes)
    elif isinstance(optimizer, SGD) and optimizer.momentum:
        for name in sorted(parameters_by_name):
            parameter = parameters_by_name[name]
            velocity = optimizer._velocities[id(parameter)]
            raw = velocity.tobytes(order="C")
            entries.append(
                {
                    "name": name,
                    "velocity_offset": offset,
                    "velocity_length": len(raw),
                }
            )
            payload_parts.append(raw)
            offset += len(raw)

    return b"".join(payload_parts), meta


def _load_optimizer_binary_state(
    optimizer: Optimizer,
    optimizer_state: dict[str, Any],
    payload: bytes,
    parameters_by_name: dict[str, Parameter],
    *,
    strict: bool,
) -> None:
    if isinstance(optimizer, Adam):
        optimizer._step = int(optimizer_state.get("step", 0))
        for entry in optimizer_state.get("entries", []):
            name = entry["name"]
            if name not in parameters_by_name:
                if strict:
                    raise ValueError(f"Unexpected checkpoint optimizer parameter {name!r}.")
                continue
            parameter = parameters_by_name[name]
            m1 = _read_payload_array(
                payload,
                int(entry["m1_offset"]),
                int(entry["m1_length"]),
                parameter.data.shape,
                name,
            )
            m2 = _read_payload_array(
                payload,
                int(entry["m2_offset"]),
                int(entry["m2_length"]),
                parameter.data.shape,
                name,
            )
            optimizer._moment1[id(parameter)] = m1
            optimizer._moment2[id(parameter)] = m2
    elif isinstance(optimizer, SGD) and optimizer.momentum:
        for entry in optimizer_state.get("entries", []):
            name = entry["name"]
            if name not in parameters_by_name:
                if strict:
                    raise ValueError(f"Unexpected checkpoint optimizer parameter {name!r}.")
                continue
            parameter = parameters_by_name[name]
            velocity = _read_payload_array(
                payload,
                int(entry["velocity_offset"]),
                int(entry["velocity_length"]),
                parameter.data.shape,
                name,
            )
            optimizer._velocities[id(parameter)] = velocity


def _read_payload_array(
    payload: bytes,
    offset: int,
    length: int,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    end = offset + length
    if end > len(payload):
        raise ArtifactError(f"Truncated optimizer state payload for parameter {name!r}.")
    expected = int(np.prod(shape)) * 8
    if length != expected:
        raise ArtifactError(f"Invalid optimizer state length for parameter {name!r}.")
    return np.frombuffer(payload[offset:end], dtype=np.float64).reshape(shape).copy()


def _load_model_payload_into(
    model: Module,
    parameter_specs: list[dict[str, Any]],
    payload: bytes,
) -> None:
    for spec in parameter_specs:
        name = spec["name"]
        shape = tuple(int(dim) for dim in spec["shape"])
        dtype = np.dtype(spec["dtype"])
        offset = int(spec["offset"])
        length = int(spec["length"])
        end = offset + length
        if end > len(payload):
            raise ArtifactError(f"Truncated tensor payload for parameter {name!r}.")
        expected = int(np.prod(shape)) * dtype.itemsize
        if length != expected:
            raise ArtifactError(f"Invalid parameter payload length for {name!r}.")
        array = np.frombuffer(payload[offset:end], dtype=dtype).reshape(shape).copy()
        parameter = model.parameter(name)
        if parameter.data.shape != shape:
            raise ValueError(f"Checkpoint parameter shape mismatch for {name!r}.")
        if np.dtype(parameter.data.dtype) != dtype:
            raise ValueError(f"Checkpoint parameter dtype mismatch for {name!r}.")
        parameter.data = array
        parameter.grad = None


def _validate_parameter_names(
    model: Module,
    parameter_specs: list[dict[str, Any]],
    *,
    strict: bool,
) -> None:
    expected = {spec["name"] for spec in parameter_specs}
    actual = {name for name, _ in model.named_parameters()}
    if expected != actual:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        message = []
        if missing:
            message.append(f"missing: {', '.join(missing)}")
        if unexpected:
            message.append(f"unexpected: {', '.join(unexpected)}")
        detail = "; ".join(message)
        if strict:
            raise ValueError(f"Checkpoint parameter mismatch: {detail}")
        raise ArtifactError(f"Checkpoint parameter mismatch: {detail}")


def _apply_checkpoint_qat_config(model: Module, metadata: dict[str, Any]) -> None:
    qat_data = metadata.get("qat_config")
    if qat_data is None:
        return
    apply_qat_config(model, QATConfig.from_dict(qat_data))


def _serialize_rng_state() -> dict[str, Any]:
    state = np.random.get_state()
    keys = state[1]
    return {
        "bit_generator": state[0],
        "keys": keys.tolist() if hasattr(keys, "tolist") else list(keys),
        "pos": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    keys = np.asarray(state["keys"], dtype=np.uint32)
    np.random.set_state(
        (
            state["bit_generator"],
            keys,
            int(state["pos"]),
            int(state["has_gauss"]),
            float(state["cached_gaussian"]),
        )
    )


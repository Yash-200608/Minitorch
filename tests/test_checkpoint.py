import json
import struct
from pathlib import Path

import numpy as np
import pytest

from minitorch import (
    Adam,
    ArtifactError,
    Linear,
    MSELoss,
    Sequential,
    SGD,
    Tensor,
    Trainer,
    format_artifact_report,
    format_checkpoint_report,
    inspect_artifact,
    load_training_checkpoint,
    save_training_checkpoint,
    TrainingResumeState,
)
from minitorch.persistence.format import (
    ARTIFACT_CHECKPOINT,
    MAGIC,
    FORMAT_VERSION,
    dumps_metadata,
    encode_header,
)


def _regression_data():
    x = np.linspace(0.0, 1.0, 8).reshape(8, 1)
    y = 3.0 * x + 2.0
    return Tensor(x), Tensor(y)


def _train_linear(epochs: int, *, optimizer=None, lr=0.1):
    inputs, targets = _regression_data()
    model = Linear(1, 1)
    if optimizer is None:
        optimizer = SGD(model.parameters(), lr=lr)
    trainer = Trainer(model, optimizer, MSELoss().forward)
    trainer.fit(inputs, targets, epochs=epochs)
    return trainer, inputs, targets


def test_checkpoint_save_load_epoch_and_step(tmp_path: Path):
    trainer, _, _ = _train_linear(5)
    path = tmp_path / "run.mtc"
    trainer.save_checkpoint(path)

    checkpoint = load_training_checkpoint(path)
    assert checkpoint.resume_state.completed_epochs == 5
    assert checkpoint.resume_state.completed_steps == 5
    assert len(checkpoint.resume_state.losses) == 5


def test_checkpoint_model_parameter_parity(tmp_path: Path):
    trainer, _, _ = _train_linear(3)
    path = tmp_path / "run.mtc"
    trainer.save_checkpoint(path)

    model, optimizer, _ = load_training_checkpoint(path).restore()
    for name, parameter in trainer.model.named_parameters():
        np.testing.assert_allclose(
            parameter.data,
            model.parameter(name).data,
        )


def test_checkpoint_loaded_arrays_independent(tmp_path: Path):
    trainer, _, _ = _train_linear(2)
    path = tmp_path / "run.mtc"
    trainer.save_checkpoint(path)

    model, _, _ = load_training_checkpoint(path).restore()
    trainer.model.weight.data[0, 0] = 999.0
    assert model.weight.data[0, 0] != 999.0


def test_checkpoint_adam_state_preserved(tmp_path: Path):
    inputs, targets = _regression_data()
    model = Linear(1, 1)
    optimizer = Adam(model.parameters(), lr=0.05)
    trainer = Trainer(model, optimizer, MSELoss().forward)
    trainer.fit(inputs, targets, epochs=4)

    saved_step = optimizer._step
    saved_m1 = optimizer._moment1[id(model.weight)].copy()
    saved_m2 = optimizer._moment2[id(model.weight)].copy()

    path = tmp_path / "adam.mtc"
    trainer.save_checkpoint(path)
    restored_model, restored_optimizer, _ = load_training_checkpoint(path).restore()

    assert restored_optimizer._step == saved_step
    np.testing.assert_allclose(
        restored_optimizer._moment1[id(restored_model.weight)],
        saved_m1,
    )
    np.testing.assert_allclose(
        restored_optimizer._moment2[id(restored_model.weight)],
        saved_m2,
    )


def test_checkpoint_sgd_momentum_preserved(tmp_path: Path):
    inputs, targets = _regression_data()
    model = Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    trainer = Trainer(model, optimizer, MSELoss().forward)
    trainer.fit(inputs, targets, epochs=3)

    saved_velocity = optimizer._velocities[id(model.weight)].copy()
    path = tmp_path / "sgd.mtc"
    trainer.save_checkpoint(path)
    restored_model, restored_optimizer, _ = load_training_checkpoint(path).restore()

    np.testing.assert_allclose(
        restored_optimizer._velocities[id(restored_model.weight)],
        saved_velocity,
    )


def test_checkpoint_history_preserved(tmp_path: Path):
    trainer, _, _ = _train_linear(4)
    path = tmp_path / "run.mtc"
    trainer.save_checkpoint(path)
    _, _, resume_state = load_training_checkpoint(path).restore()
    assert len(resume_state.losses) == 4
    assert resume_state.history_epochs == [0, 1, 2, 3]


def test_checkpoint_optimizer_type_mismatch_rejected(tmp_path: Path):
    trainer, _, _ = _train_linear(2)
    path = tmp_path / "run.mtc"
    trainer.save_checkpoint(path)

    model = Linear(1, 1)
    wrong_optimizer = Adam(model.parameters())
    with pytest.raises(ValueError, match="optimizer type"):
        load_training_checkpoint(path).restore(optimizer=wrong_optimizer)


def test_checkpoint_parameter_mismatch_rejected(tmp_path: Path):
    trainer, _, _ = _train_linear(2)
    path = tmp_path / "run.mtc"
    trainer.save_checkpoint(path)

    other = Sequential(Linear(2, 1))
    optimizer = SGD(other.parameters())
    checkpoint = load_training_checkpoint(path)
    with pytest.raises(ValueError, match="parameter mismatch"):
        checkpoint.restore_into(other, optimizer)


def test_checkpoint_truncated_payload_rejected(tmp_path: Path):
    trainer, _, _ = _train_linear(1)
    path = tmp_path / "run.mtc"
    trainer.save_checkpoint(path)
    data = bytearray(path.read_bytes())
    data = data[:-4]
    path.write_bytes(data)
    with pytest.raises(ArtifactError, match="Truncated"):
        load_training_checkpoint(path)


def test_checkpoint_crc_mismatch_rejected(tmp_path: Path):
    trainer, _, _ = _train_linear(1)
    path = tmp_path / "run.mtc"
    trainer.save_checkpoint(path)
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(data)
    with pytest.raises(ArtifactError, match="checksum"):
        load_training_checkpoint(path)


def test_checkpoint_wrong_magic_rejected(tmp_path: Path):
    trainer, _, _ = _train_linear(1)
    path = tmp_path / "run.mtc"
    trainer.save_checkpoint(path)
    data = bytearray(path.read_bytes())
    data[0:8] = b"BADMAGIC"
    path.write_bytes(data)
    with pytest.raises(ArtifactError, match="magic"):
        load_training_checkpoint(path)


def test_checkpoint_inspection_report(tmp_path: Path):
    trainer, _, _ = _train_linear(3)
    path = tmp_path / "run.mtc"
    trainer.save_checkpoint(path)

    report = format_artifact_report(path)
    assert "checkpoint" in report.lower()
    assert "completed epochs: 3" in report
    assert "completed steps: 3" in report

    info = inspect_artifact(path)
    assert info["artifact_type"] == ARTIFACT_CHECKPOINT


def test_checkpoint_deterministic_bytes(tmp_path: Path):
    trainer_a, _, _ = _train_linear(2)
    trainer_b, _, _ = _train_linear(2)

    path_a = tmp_path / "a.mtc"
    path_b = tmp_path / "b.mtc"
    trainer_a.save_checkpoint(path_a)
    trainer_b.save_checkpoint(path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_malformed_checkpoint_metadata_rejected(tmp_path: Path):
    metadata = {"framework_version": "0.1.0", "format_version": 1}
    payload = b""
    meta_bytes = dumps_metadata(metadata)
    header = encode_header(ARTIFACT_CHECKPOINT, len(meta_bytes), 0, len(payload))
    path = tmp_path / "bad.mtc"
    path.write_bytes(header + meta_bytes + payload)
    checkpoint = load_training_checkpoint(path)
    with pytest.raises(ArtifactError):
        checkpoint.restore()

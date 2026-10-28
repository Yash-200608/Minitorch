import json
import struct
from pathlib import Path

import numpy as np
import pytest

from minitorch import (
    ArtifactError,
    CompressionConfig,
    Linear,
    MSELoss,
    Sequential,
    SGD,
    Tensor,
    Trainer,
    compress_model,
    format_artifact_report,
    inspect_artifact,
    load_compressed_model,
    load_model,
    save_compressed_model,
    save_model,
)
from minitorch.persistence.format import (
    ARTIFACT_COMPRESSED_MODEL,
    ARTIFACT_MODEL,
    MAGIC,
    FORMAT_VERSION,
    dumps_metadata,
    encode_header,
)


def _make_sequential_model() -> Sequential:
    return Sequential(Linear(2, 4), Linear(4, 1))


def test_model_save_load_structure_and_values(tmp_path: Path):
    model = _make_sequential_model()
    for name, parameter in model.named_parameters():
        parameter.data = parameter.data + 1.0
    path = tmp_path / "model.mt"
    save_model(model, path)
    loaded = load_model(path)

    for name, parameter in model.named_parameters():
        original = np.asarray(parameter.data)
        restored = np.asarray(loaded.parameter(name).data)
        assert restored.shape == original.shape
        np.testing.assert_allclose(original, restored)


def test_model_save_load_independence(tmp_path: Path):
    model = Linear(2, 1)
    model.weight.data = np.ones((1, 2))
    path = tmp_path / "model.mt"
    save_model(model, path)
    loaded = load_model(path)

    model.weight.data[0, 0] = 99.0
    assert loaded.weight.data[0, 0] != 99.0


def test_model_module_save_method(tmp_path: Path):
    model = Linear(3, 2)
    path = tmp_path / "via_method.mt"
    model.save(str(path))
    loaded = load_model(path)
    assert loaded.weight.shape == (2, 3)


def test_load_accepts_string_paths(tmp_path: Path):
    model = Linear(2, 1)
    model.weight.data = np.array([[0.5, -1.5]])
    model_path = str(tmp_path / "model.mt")
    save_model(model, model_path)
    np.testing.assert_allclose(load_model(model_path).weight.data, model.weight.data)

    compressed = compress_model(model, config=CompressionConfig(bits=4, method="uniform"))
    compressed_path = str(tmp_path / "compressed.mtc")
    save_compressed_model(compressed, compressed_path)
    loaded = load_compressed_model(compressed_path)
    np.testing.assert_allclose(
        loaded.materialize(model).weight.data,
        compressed.materialize(model).weight.data,
    )


def test_model_deterministic_bytes(tmp_path: Path):
    model = Linear(2, 2)
    path1 = tmp_path / "a.mt"
    path2 = tmp_path / "b.mt"
    save_model(model, path1)
    save_model(model, path2)
    assert path1.read_bytes() == path2.read_bytes()


def test_inspect_artifact_model(tmp_path: Path):
    model = Linear(2, 1)
    path = tmp_path / "model.mt"
    save_model(model, path)
    info = inspect_artifact(path)
    assert info["artifact_type"] == ARTIFACT_MODEL
    assert info["file_bytes"] == path.stat().st_size
    report = format_artifact_report(path)
    assert "MiniTorch Artifact" in report


def test_corruption_wrong_magic(tmp_path: Path):
    path = tmp_path / "bad.mt"
    path.write_bytes(b"NOTTORCH" + b"\0" * 24)
    with pytest.raises(ArtifactError, match="magic"):
        load_model(path)


def test_corruption_unsupported_version(tmp_path: Path):
    header = encode_header(ARTIFACT_MODEL, 10, 0, 0)
    header = header[:8] + struct.pack("<H", 99) + header[10:]
    meta = dumps_metadata({"framework_version": "0.1.0", "module_tree": {"type": "Linear", "in_features": 1, "out_features": 1}, "parameters": []})
    path = tmp_path / "ver.mt"
    path.write_bytes(header + meta)
    with pytest.raises(ArtifactError, match="format version"):
        load_model(path)


def test_corruption_truncated_payload(tmp_path: Path):
    model = Linear(1, 1)
    path = tmp_path / "trunc.mt"
    save_model(model, path)
    data = path.read_bytes()
    path.write_bytes(data[:-4])
    with pytest.raises(ArtifactError):
        load_model(path)


def test_corruption_checksum_mismatch(tmp_path: Path):
    model = Linear(1, 1)
    path = tmp_path / "crc.mt"
    save_model(model, path)
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ArtifactError, match="checksum"):
        load_model(path)


def test_wrong_artifact_type(tmp_path: Path):
    model = Linear(1, 1)
    path = tmp_path / "model.mt"
    save_model(model, path)
    with pytest.raises(ArtifactError, match="artifact type"):
        load_compressed_model(path)


@pytest.mark.parametrize("config", [
    CompressionConfig(bits=8, method="uniform", packed=False),
    CompressionConfig(bits=4, method="uniform", packed=True),
    CompressionConfig(bits=4, method="codebook", packed=True),
])
def test_compressed_save_load_uniform_and_codebook(tmp_path: Path, config: CompressionConfig):
    model = Sequential(Linear(4, 8), Linear(8, 2))
    compressed = compress_model(model, config=config)
    path = tmp_path / "compressed.mtc"
    save_compressed_model(compressed, path)
    loaded = load_compressed_model(path)

    assert loaded.source_parameter_names == compressed.source_parameter_names
    for name in compressed.source_parameter_names:
        orig_q = compressed.quantized_parameters[name]
        load_q = loaded.quantized_parameters[name]
        if orig_q.packed:
            assert np.array_equal(orig_q.packed_bytes, load_q.packed_bytes)
        else:
            assert np.array_equal(orig_q.codes, load_q.codes)
        np.testing.assert_allclose(
            orig_q.dequantize().data,
            load_q.dequantize().data,
            rtol=0,
            atol=1e-12,
        )


def test_compressed_exact_packed_bytes(tmp_path: Path):
    model = Linear(8, 4)
    config = CompressionConfig(bits=4, method="uniform", packed=True)
    compressed = compress_model(model, config=config)
    original_bytes = {
        name: compressed.quantized_parameters[name].packed_bytes.copy()
        for name in compressed.source_parameter_names
    }
    path = tmp_path / "packed.mtc"
    compressed.save(str(path))
    loaded = load_compressed_model(path)
    for name, expected in original_bytes.items():
        assert np.array_equal(expected, loaded.quantized_parameters[name].packed_bytes)


def test_compressed_rotation_parity(tmp_path: Path):
    model = Linear(8, 4)
    config = CompressionConfig(bits=4, method="codebook", packed=True, rotation="combined")
    compressed = compress_model(model, config=config)
    path = tmp_path / "rot.mtc"
    save_compressed_model(compressed, path)
    loaded = load_compressed_model(path)
    inputs = Tensor(np.linspace(0, 1, 32).reshape(4, 8))
    materialized = loaded.materialize(model)
    original_materialized = compressed.materialize(model)
    np.testing.assert_allclose(
        original_materialized(inputs).data,
        materialized(inputs).data,
        rtol=1e-9,
        atol=1e-9,
    )


def test_model_output_parity_after_load(tmp_path: Path):
    model = Sequential(Linear(1, 4), Linear(4, 1))
    trainer = Trainer(model, SGD(model.parameters(), lr=0.1), MSELoss().forward)
    inputs = Tensor(np.linspace(0, 1, 8).reshape(8, 1))
    targets = Tensor(3 * inputs.data + 2)
    trainer.fit(inputs, targets, epochs=20)

    path = tmp_path / "trained.mt"
    save_model(model, path)
    loaded = load_model(path)
    x = Tensor([[0.25], [0.75]])
    np.testing.assert_allclose(model(x).data, loaded(x).data, rtol=1e-9, atol=1e-9)


def test_loaded_model_xray(tmp_path: Path):
    model = Linear(2, 1)
    path = tmp_path / "xray.mt"
    save_model(model, path)
    loaded = load_model(path)
    report = loaded.xray()
    assert len(report.parameter_infos) == 2


def test_benchmark_persistence_runs():
    from minitorch.persistence import benchmark_persistence

    model = Linear(4, 2)
    compressed = compress_model(model)
    results = benchmark_persistence(model, compressed, repeats=1)
    assert results["model_file_bytes"] > 0
    assert results["compressed_file_bytes"] > 0

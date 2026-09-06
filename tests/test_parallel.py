import numpy as np
import pytest

from minitorch import (
    Codebook,
    CompressionConfig,
    Linear,
    ParallelConfig,
    Sequential,
    Tensor,
    benchmark_parallel_comparison,
    chunk_ranges,
    compress_model,
    execute_chunked,
    pack_codes,
    unpack_codes,
)
from minitorch.quantization import quantize_array
from minitorch.parallel import merge_chunk_results


def _collect_indices(ranges: list[tuple[int, int]]) -> list[int]:
    indices: list[int] = []
    for start, end in ranges:
        indices.extend(range(start, end))
    return indices


@pytest.mark.parametrize("count,workers", [
    (0, 4),
    (1, 4),
    (2, 4),
    (3, 4),
    (4, 4),
    (5, 4),
    (7, 3),
    (10, 4),
    (100, 8),
])
def test_chunk_ranges_cover_all_elements_without_overlap(count: int, workers: int):
    ranges = chunk_ranges(count, workers)
    if count == 0:
        assert ranges == []
        return
    collected = _collect_indices(ranges)
    assert collected == list(range(count))
    assert len(collected) == len(set(collected))


def test_chunk_ranges_workers_one_covers_all():
    ranges = chunk_ranges(17, 1)
    assert ranges == [(0, 17)]


def test_parallel_config_workers_one_is_serial_path():
    config = ParallelConfig(enabled=True, workers=1)
    values: list[int] = []

    def worker(start: int, end: int) -> list[int]:
        return list(range(start, end))

    chunks = execute_chunked(8, config, lambda s, e: list(range(s, e)))
    assert chunks == [list(range(8))]


def test_execute_chunked_propagates_exceptions():
    config = ParallelConfig(enabled=True, workers=4)

    def worker(start: int, end: int) -> None:
        if start == 0:
            raise ValueError("worker failed")
        return None

    with pytest.raises(ValueError, match="worker failed"):
        execute_chunked(16, config, worker)


def test_merge_chunk_results_concatenates():
    config = ParallelConfig(enabled=True, workers=4)
    merged = merge_chunk_results(
        8,
        config,
        lambda start, end: list(range(start, end)),
        lambda chunks: sum(chunks, []),
    )
    assert merged == list(range(8))


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_parallel_pack_matches_serial_exactly(bits: int):
    rng = np.random.default_rng(0)
    levels = 2 ** bits
    codes = rng.integers(0, levels, size=997, dtype=np.int32)
    serial = pack_codes(codes, bits)
    parallel = pack_codes(codes, bits, parallel=True, workers=4)
    assert np.array_equal(serial, parallel)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_parallel_pack_unpack_round_trip_exact(bits: int):
    rng = np.random.default_rng(1)
    levels = 2 ** bits
    codes = rng.integers(0, levels, size=500, dtype=np.int32)
    packed = pack_codes(codes, bits, parallel=True, workers=4)
    restored = unpack_codes(packed, bits, codes.size)
    assert np.array_equal(codes, restored)


def test_parallel_codebook_encode_matches_serial():
    rng = np.random.default_rng(2)
    values = rng.standard_normal(4096)
    codebook = Codebook.fit(values, 16, max_iterations=20)
    serial = codebook.encode(values)
    parallel = codebook.encode(values, parallel=True, workers=4)
    assert np.array_equal(serial, parallel)


@pytest.mark.parametrize("method", ["uniform", "codebook"])
@pytest.mark.parametrize("bits", [2, 4, 8])
def test_parallel_quantize_array_matches_serial(method: str, bits: int):
    rng = np.random.default_rng(3)
    data = rng.standard_normal(2048)
    serial = quantize_array(data, bits, method=method, packed=True)
    parallel = quantize_array(
        data,
        bits,
        method=method,
        packed=True,
        parallel=True,
        workers=4,
    )
    assert serial.packed is True and parallel.packed is True
    assert np.array_equal(serial.packed_bytes, parallel.packed_bytes)
    np.testing.assert_allclose(
        serial.dequantize().data,
        parallel.dequantize().data,
        rtol=0,
        atol=1e-12,
    )


def test_parallel_compress_model_matches_serial():
    model = Sequential(Linear(8, 16), Linear(16, 4))
    config = CompressionConfig(bits=4, method="codebook", packed=True)
    serial = compress_model(model, config=config)
    parallel = compress_model(model, config=config, parallel=True, workers=4)
    assert serial.source_parameter_names == parallel.source_parameter_names
    for name in serial.source_parameter_names:
        s_q = serial.quantized_parameters[name]
        p_q = parallel.quantized_parameters[name]
        assert np.array_equal(s_q.packed_bytes, p_q.packed_bytes)
        np.testing.assert_allclose(
            s_q.dequantize().data,
            p_q.dequantize().data,
            rtol=0,
            atol=1e-12,
        )


def test_benchmark_parallel_comparison_structure():
    data = np.linspace(0, 1, 1024)
    result = benchmark_parallel_comparison(
        "pack_codes",
        lambda: pack_codes(data.astype(np.int32), 4),
        lambda: pack_codes(data.astype(np.int32), 4, parallel=True, workers=4),
        workers=4,
        warmup=1,
        repeats=2,
    )
    assert result.serial_seconds >= 0
    assert result.parallel_seconds >= 0
    assert len(result.serial_samples) == 2
    assert "Parallel Profile" in result.report()


def test_tensor_quantize_parallel_flag():
    tensor = Tensor(np.linspace(-1, 1, 64))
    serial = tensor.quantize(bits=4, method="uniform", packed=True)
    parallel = tensor.quantize(bits=4, method="uniform", packed=True, parallel=True, workers=2)
    assert np.array_equal(serial.packed_bytes, parallel.packed_bytes)

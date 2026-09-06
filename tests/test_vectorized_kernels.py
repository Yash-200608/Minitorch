import numpy as np
import pytest

from minitorch import pack_codes, unpack_codes, walsh_hadamard_1d
from minitorch.kernels import (
    nearest_centroid_codes,
    nearest_centroid_codes_reference,
    nearest_centroid_codes_vectorized,
    nearest_centroid_codes_vectorized_chunked,
    pack_codes_reference,
    pack_codes_vectorized,
    uniform_codes,
    uniform_codes_reference,
    uniform_codes_vectorized,
    unpack_codes_reference,
    unpack_codes_vectorized,
    walsh_hadamard_reference,
    walsh_hadamard_vectorized,
    benchmark_vectorization,
)


@pytest.mark.parametrize("count,workers", [
    (0, 4),
    (1, 4),
    (5, 4),
    (17, 3),
])
def test_chunk_ranges_still_used_by_parallel_module(count, workers):
    from minitorch import chunk_ranges

    ranges = chunk_ranges(count, workers)
    if count == 0:
        assert ranges == []
        return
    collected = []
    for start, end in ranges:
        collected.extend(range(start, end))
    assert collected == list(range(count))


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_vectorized_pack_matches_reference(bits: int):
    rng = np.random.default_rng(0)
    levels = 2 ** bits
    codes = rng.integers(0, levels, size=503, dtype=np.int32)
    reference = pack_codes_reference(codes, bits)
    vectorized = pack_codes_vectorized(codes, bits)
    assert np.array_equal(reference, vectorized)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_vectorized_unpack_matches_reference(bits: int):
    rng = np.random.default_rng(1)
    levels = 2 ** bits
    codes = rng.integers(0, levels, size=400, dtype=np.int32)
    packed = pack_codes_reference(codes, bits)
    reference = unpack_codes_reference(packed, bits, codes.size)
    vectorized = unpack_codes_vectorized(packed, bits, codes.size)
    assert np.array_equal(reference, vectorized)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_pack_codes_backend_vectorized(bits: int):
    rng = np.random.default_rng(2)
    levels = 2 ** bits
    codes = rng.integers(0, levels, size=256, dtype=np.int32)
    reference = pack_codes(codes, bits)
    vectorized = pack_codes(codes, bits, backend="vectorized")
    assert np.array_equal(reference, vectorized)


def test_uniform_codes_reference_matches_vectorized():
    rng = np.random.default_rng(3)
    values = rng.standard_normal(1024)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    max_code = 15
    reference = uniform_codes_reference(values, minimum=minimum, maximum=maximum, max_code=max_code)
    vectorized = uniform_codes_vectorized(values, minimum=minimum, maximum=maximum, max_code=max_code)
    assert np.array_equal(reference, vectorized)


def test_codebook_reference_matches_vectorized():
    rng = np.random.default_rng(4)
    values = rng.standard_normal(2048)
    centroids = np.linspace(-2, 2, 16)
    reference = nearest_centroid_codes_reference(values, centroids)
    vectorized = nearest_centroid_codes_vectorized(values, centroids)
    assert np.array_equal(reference, vectorized)


def test_codebook_chunked_matches_vectorized():
    rng = np.random.default_rng(5)
    values = rng.standard_normal(5000)
    centroids = np.linspace(-2, 2, 16)
    full = nearest_centroid_codes_vectorized(values, centroids)
    chunked = nearest_centroid_codes_vectorized_chunked(values, centroids, chunk_size=512)
    assert np.array_equal(full, chunked)


def test_hadamard_vectorized_matches_reference():
    rng = np.random.default_rng(6)
    vector = rng.standard_normal(16)
    reference = walsh_hadamard_reference(vector)
    vectorized = walsh_hadamard_vectorized(vector)
    np.testing.assert_allclose(reference, vectorized, rtol=1e-12, atol=1e-12)


def test_walsh_hadamard_1d_backend_vectorized():
    rng = np.random.default_rng(7)
    vector = rng.standard_normal(8)
    reference = walsh_hadamard_1d(vector, backend="reference")
    vectorized = walsh_hadamard_1d(vector, backend="vectorized")
    np.testing.assert_allclose(reference, vectorized, rtol=1e-12, atol=1e-12)


def test_hadamard_involution():
    vector = np.array([1.0, 2.0, 3.0, 4.0])
    transformed = walsh_hadamard_vectorized(vector)
    restored = walsh_hadamard_vectorized(transformed)
    np.testing.assert_allclose(vector, restored, rtol=1e-10, atol=1e-10)


def test_hadamard_norm_preservation():
    vector = np.linspace(-1, 1, 16)
    transformed = walsh_hadamard_vectorized(vector)
    np.testing.assert_allclose(np.linalg.norm(vector), np.linalg.norm(transformed), rtol=1e-10)


def test_benchmark_vectorization_structure():
    data = np.linspace(0, 15, 256, dtype=np.int32)
    result = benchmark_vectorization(
        "pack_4bit",
        256,
        lambda: pack_codes_reference(data, 4),
        lambda: pack_codes_vectorized(data, 4),
        warmup=1,
        repeats=2,
    )
    assert result.reference_seconds >= 0
    assert result.vectorized_seconds >= 0
    assert "pack_4bit" in result.report()


def test_non_finite_quantization_still_rejected():
    from minitorch.quantization import _validate_quantization_input

    with pytest.raises(ValueError, match="finite"):
        _validate_quantization_input(np.array([np.nan]))


def test_parallel_plus_vectorized_pack_parity():
    rng = np.random.default_rng(8)
    codes = rng.integers(0, 16, size=1000, dtype=np.int32)
    serial = pack_codes(codes, 4)
    parallel = pack_codes(codes, 4, parallel=True, workers=4, backend="vectorized")
    assert np.array_equal(serial, parallel)

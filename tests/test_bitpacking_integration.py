import numpy as np

from minitorch import Tensor, pack_codes, unpack_codes, explain_packing


def test_uniform_logical_matches_packed_codes():
    x = Tensor(np.linspace(0.0, 1.0, 64))
    logical = x.quantize(bits=4, method="uniform")
    packed = x.quantize(bits=4, method="uniform", packed=True)
    np.testing.assert_array_equal(logical.codes, packed.codes)


def test_uniform_logical_matches_packed_reconstruction():
    x = Tensor(np.linspace(0.0, 1.0, 64))
    logical = x.quantize(bits=4, method="uniform").dequantize().data
    packed = x.quantize(bits=4, method="uniform", packed=True).dequantize().data
    np.testing.assert_allclose(logical, packed)


def test_codebook_logical_matches_packed_reconstruction():
    rng = np.random.default_rng(0)
    data = rng.normal(size=128)
    x = Tensor(data)
    logical = x.quantize(bits=4, method="codebook").dequantize().data
    packed = x.quantize(bits=4, method="codebook", packed=True).dequantize().data
    np.testing.assert_allclose(logical, packed)


def test_packed_quantized_tensor_uses_uint8_storage():
    x = Tensor(np.linspace(0.0, 1.0, 1000))
    q = x.quantize(bits=4, packed=True)
    assert q.packed is True
    assert q.packed_bytes is not None
    assert q.packed_bytes.dtype == np.uint8
    assert q._codes is None


def test_packed_memory_report():
    data = np.linspace(0.0, 1.0, 1000)
    x = Tensor(data)
    report = x.quantization_report(bits=4, packed=True)
    assert report.storage_mode == "packed"
    assert report.memory.actual_packed_bytes == report.memory.theoretical_packed_bytes
    assert report.memory.actual_packed_bytes < report.memory.logical_int32_storage_bytes


def test_packed_constant_tensors():
    x = Tensor([5.0, 5.0, 5.0])
    for bits in [2, 4, 8]:
        q = x.quantize(bits=bits, packed=True)
        np.testing.assert_allclose(q.dequantize().data, x.data)


def test_pack_codes_round_trip_integration():
    codes = np.array([10, 3], dtype=np.int32)
    packed = pack_codes(codes, bits=4)
    restored = unpack_codes(packed, bits=4, count=2)
    np.testing.assert_array_equal(restored, codes)


def test_explain_packing_helper():
    text = explain_packing(np.array([1, 3, 0, 2], dtype=np.int32), bits=2)
    assert "Bit Packing" in text

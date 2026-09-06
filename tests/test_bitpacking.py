import numpy as np
import pytest

from minitorch.bitpacking import (
    explain_packing,
    pack_codes,
    packed_byte_count,
    unpack_codes,
    validate_codes,
)
from minitorch.quantization import validate_bits


def test_pack_unpack_4bit_exact_a3():
    codes = np.array([10, 3], dtype=np.int32)
    packed = pack_codes(codes, bits=4)
    assert packed.dtype == np.uint8
    assert packed.tolist() == [0xA3]
    unpacked = unpack_codes(packed, bits=4, count=2)
    np.testing.assert_array_equal(unpacked, codes)


def test_pack_unpack_2bit_exact_pattern():
    codes = np.array([1, 3, 0, 2], dtype=np.int32)
    packed = pack_codes(codes, bits=2)
    assert int(packed[0]) == 0b01110010
    unpacked = unpack_codes(packed, bits=2, count=4)
    np.testing.assert_array_equal(unpacked, codes)


def test_pack_unpack_8bit():
    codes = np.array([0, 127, 255], dtype=np.int32)
    packed = pack_codes(codes, bits=8)
    np.testing.assert_array_equal(packed, codes.astype(np.uint8))
    unpacked = unpack_codes(packed, bits=8, count=3)
    np.testing.assert_array_equal(unpacked, codes)


@pytest.mark.parametrize("count", [1, 2, 3, 5, 7, 9])
def test_2bit_non_byte_aligned_counts(count):
    codes = np.arange(count, dtype=np.int32) % 4
    packed = pack_codes(codes, bits=2)
    assert packed.size == packed_byte_count(count, 2)
    unpacked = unpack_codes(packed, bits=2, count=count)
    np.testing.assert_array_equal(unpacked, codes)


@pytest.mark.parametrize("count", [1, 3, 5, 7])
def test_4bit_non_byte_aligned_counts(count):
    codes = np.arange(count, dtype=np.int32) % 16
    packed = pack_codes(codes, bits=4)
    assert packed.size == packed_byte_count(count, 4)
    unpacked = unpack_codes(packed, bits=4, count=count)
    np.testing.assert_array_equal(unpacked, codes)


def test_invalid_4bit_code_raises():
    with pytest.raises(ValueError, match="0 <= code"):
        pack_codes(np.array([16], dtype=np.int32), bits=4)
    with pytest.raises(ValueError, match="0 <= code"):
        pack_codes(np.array([-1], dtype=np.int32), bits=4)


def test_invalid_2bit_code_raises():
    with pytest.raises(ValueError, match="0 <= code"):
        pack_codes(np.array([4], dtype=np.int32), bits=2)


def test_invalid_8bit_code_raises():
    with pytest.raises(ValueError, match="0 <= code"):
        pack_codes(np.array([256], dtype=np.int32), bits=8)


def test_truncated_packed_data_raises():
    packed = pack_codes(np.array([1, 2, 3], dtype=np.int32), bits=4)
    with pytest.raises(ValueError, match="required"):
        unpack_codes(packed[:1], bits=4, count=3)


def test_explain_packing_contains_binary():
    text = explain_packing(np.array([10, 3], dtype=np.int32), bits=4)
    assert "1010" in text
    assert "0xA3" in text.upper() or "0xa3" in text.lower()


def test_large_tensor_packed_smaller_than_int32():
    codes = np.arange(1000, dtype=np.int32) % 16
    packed = pack_codes(codes, bits=4)
    assert packed.nbytes < codes.nbytes
    assert packed.nbytes == packed_byte_count(1000, 4)

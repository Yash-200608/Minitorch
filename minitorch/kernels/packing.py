from __future__ import annotations

import math

import numpy as np

from .backend import NumericalBackend

SUPPORTED_BIT_WIDTHS: tuple[int, ...] = (2, 4, 8)
BIT_MASKS: dict[int, int] = {2: 0b11, 4: 0b1111, 8: 0b11111111}


def _validate_bits(bits: int) -> None:
    if bits not in SUPPORTED_BIT_WIDTHS:
        supported = ", ".join(str(value) for value in SUPPORTED_BIT_WIDTHS)
        raise ValueError(f"Unsupported bit width {bits}. Supported values: {supported}.")


def _code_mask(bits: int) -> int:
    _validate_bits(bits)
    return BIT_MASKS[bits]


def _packed_byte_count(count: int, bits: int) -> int:
    _validate_bits(bits)
    if count < 0:
        raise ValueError("count must be non-negative.")
    if count == 0:
        return 0
    return int(math.ceil(count * bits / 8))


def pack_codes_reference(flat: np.ndarray, bits: int) -> np.ndarray:
    """Scalar MSB-first packing reference (matches original MiniTorch layout)."""
    _validate_bits(bits)
    count = int(flat.size)
    if count == 0:
        return np.array([], dtype=np.uint8)

    packed = np.zeros(_packed_byte_count(count, bits), dtype=np.uint8)

    if bits == 8:
        packed[:] = flat.astype(np.uint8)
        return packed

    for index, code in enumerate(flat):
        code_value = int(code)
        if bits == 4:
            byte_index = index // 2
            if index % 2 == 0:
                packed[byte_index] |= np.uint8(code_value << 4)
            else:
                packed[byte_index] |= np.uint8(code_value)
        elif bits == 2:
            byte_index = index // 4
            shift = 6 - (index % 4) * 2
            packed[byte_index] |= np.uint8(code_value << shift)

    return packed


def pack_codes_vectorized(flat: np.ndarray, bits: int) -> np.ndarray:
    """NumPy bitwise vectorized packing with identical byte layout."""
    _validate_bits(bits)
    codes = np.asarray(flat, dtype=np.int64).ravel()
    count = int(codes.size)
    if count == 0:
        return np.array([], dtype=np.uint8)

    if bits == 8:
        return codes.astype(np.uint8)

    if bits == 4:
        high = codes[0::2].astype(np.uint8) << 4
        low = codes[1::2].astype(np.uint8)
        packed = np.zeros(_packed_byte_count(count, bits), dtype=np.uint8)
        if low.size:
            packed[:low.size] = high[:low.size] | low
        if count % 2 == 1:
            packed[low.size] = high[low.size]
        return packed

    if bits == 2:
        mask = _code_mask(bits)
        packed = np.zeros(_packed_byte_count(count, bits), dtype=np.uint8)
        full_groups = count // 4
        if full_groups:
            grouped = codes[:full_groups * 4].reshape(full_groups, 4).astype(np.uint8)
            weights = np.array([64, 16, 4, 1], dtype=np.uint8)
            packed[:full_groups] = np.sum(grouped * weights, axis=1, dtype=np.uint8)

        tail_start = full_groups * 4
        if tail_start < count:
            tail = codes[tail_start:]
            tail_packed = pack_codes_reference(tail, bits)
            packed[full_groups:full_groups + tail_packed.size] = tail_packed
        return packed

    raise ValueError(f"Unsupported bit width {bits}.")


def unpack_codes_reference(
    byte_array: np.ndarray,
    bits: int,
    count: int,
) -> np.ndarray:
    _validate_bits(bits)
    mask = _code_mask(bits)
    codes = np.empty(count, dtype=np.int32)

    if bits == 8:
        codes[:] = byte_array[:count]
        return codes

    if bits == 4:
        for index in range(count):
            byte_index = index // 2
            byte_value = int(byte_array[byte_index])
            if index % 2 == 0:
                codes[index] = (byte_value >> 4) & mask
            else:
                codes[index] = byte_value & mask
        return codes

    for index in range(count):
        byte_index = index // 4
        byte_value = int(byte_array[byte_index])
        shift = 6 - (index % 4) * 2
        codes[index] = (byte_value >> shift) & mask
    return codes


def unpack_codes_vectorized(
    byte_array: np.ndarray,
    bits: int,
    count: int,
) -> np.ndarray:
    _validate_bits(bits)
    mask = _code_mask(bits)
    bytes_needed = _packed_byte_count(count, bits)
    packed = np.asarray(byte_array[:bytes_needed], dtype=np.uint8)
    codes = np.empty(count, dtype=np.int32)

    if bits == 8:
        codes[:] = packed[:count]
        return codes

    if bits == 4:
        high = (packed >> 4) & mask
        low = packed & mask
        codes[0::2] = high[: (count + 1) // 2][: ((count + 1) // 2)]
        if count // 2:
            codes[1::2] = low[: count // 2]
        return codes

    if bits == 2:
        full_groups = count // 4
        if full_groups:
            grouped_bytes = packed[:full_groups]
            shifts = np.array([6, 4, 2, 0], dtype=np.int32)
            for group_index, byte_value in enumerate(grouped_bytes):
                for slot in range(4):
                    codes[group_index * 4 + slot] = (int(byte_value) >> shifts[slot]) & mask

        tail_start = full_groups * 4
        if tail_start < count:
            tail_codes = unpack_codes_reference(
                packed[full_groups:],
                bits,
                count - tail_start,
            )
            codes[tail_start:] = tail_codes
        return codes

    raise ValueError(f"Unsupported bit width {bits}.")


def pack_codes(
    flat: np.ndarray,
    bits: int,
    *,
    backend: NumericalBackend = "reference",
) -> np.ndarray:
    if backend == "vectorized":
        return pack_codes_vectorized(flat, bits)
    return pack_codes_reference(flat, bits)


def unpack_codes(
    byte_array: np.ndarray,
    bits: int,
    count: int,
    *,
    backend: NumericalBackend = "reference",
) -> np.ndarray:
    if backend == "vectorized":
        return unpack_codes_vectorized(byte_array, bits, count)
    return unpack_codes_reference(byte_array, bits, count)

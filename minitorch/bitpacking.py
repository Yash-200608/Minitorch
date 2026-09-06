from __future__ import annotations

import math
from typing import Any

import numpy as np

from concurrent.futures import ThreadPoolExecutor

from .parallel import ParallelConfig
from .kernels.backend import DEFAULT_PACKING_BACKEND, NumericalBackend, resolve_backend
from .kernels.packing import pack_codes as kernel_pack_codes
from .kernels.packing import unpack_codes as kernel_unpack_codes

SUPPORTED_BIT_WIDTHS: tuple[int, ...] = (2, 4, 8)

BIT_MASKS: dict[int, int] = {
    2: 0b11,
    4: 0b1111,
    8: 0b11111111,
}

CODES_PER_BYTE: dict[int, int] = {
    2: 4,
    4: 2,
    8: 1,
}


def validate_bits(bits: int) -> None:
    if bits not in SUPPORTED_BIT_WIDTHS:
        supported = ", ".join(str(value) for value in SUPPORTED_BIT_WIDTHS)
        raise ValueError(
            f"Unsupported bit width {bits}. Supported values: {supported}."
        )


def code_mask(bits: int) -> int:
    validate_bits(bits)
    return BIT_MASKS[bits]


def codes_per_byte(bits: int) -> int:
    validate_bits(bits)
    return CODES_PER_BYTE[bits]


def packed_byte_count(count: int, bits: int) -> int:
    validate_bits(bits)
    if count < 0:
        raise ValueError("count must be non-negative.")
    if count == 0:
        return 0
    return int(math.ceil(count * bits / 8))


def validate_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    validate_bits(bits)
    array = np.asarray(codes)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("codes must be an integer array.")
    flat = array.ravel()
    levels = 2 ** bits
    if np.any(flat < 0) or np.any(flat >= levels):
        raise ValueError(
            f"All codes must satisfy 0 <= code < {levels} for {bits}-bit packing."
        )
    return flat.astype(np.int64, copy=False)


def _resolve_parallel_config(
    parallel_config: ParallelConfig | None,
    parallel: bool,
    workers: int | None,
) -> ParallelConfig:
    if parallel_config is not None:
        return parallel_config
    if parallel:
        return ParallelConfig(enabled=True, workers=workers)
    return ParallelConfig()


def _pack_codes_flat(flat: np.ndarray, bits: int) -> np.ndarray:
    """Pack a validated 1-D code array using the reference kernel."""
    return kernel_pack_codes(flat, bits, backend="reference")


def pack_codes(
    codes: np.ndarray,
    bits: int,
    *,
    parallel_config: ParallelConfig | None = None,
    parallel: bool = False,
    workers: int | None = None,
    backend: NumericalBackend | None = None,
) -> np.ndarray:
    """
    Pack integer codes into a compact uint8 byte array.

    Codes are packed from lowest index to highest, MSB-first within each byte.
    Unused bits in the final byte are zero-padded.

    Parallel mode partitions contiguous code ranges across workers. Chunk outputs
    are concatenated, preserving the serial byte stream exactly.
    """
    flat = validate_codes(codes, bits)
    config = _resolve_parallel_config(parallel_config, parallel, workers)
    pack_backend = resolve_backend(backend, DEFAULT_PACKING_BACKEND)
    count = int(flat.size)
    if count == 0:
        return np.array([], dtype=np.uint8)

    if not config.enabled:
        return kernel_pack_codes(flat, bits, backend=pack_backend)

    from .parallel import chunk_ranges_aligned

    ranges = chunk_ranges_aligned(count, config.resolve_workers(count), codes_per_byte(bits))
    if len(ranges) <= 1:
        return kernel_pack_codes(flat, bits, backend=pack_backend)

    chunks: list[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        futures = [
            executor.submit(kernel_pack_codes, flat[start:end], bits, backend=pack_backend)
            for start, end in ranges
        ]
        for future in futures:
            chunks.append(future.result())
    return np.concatenate(chunks)


def unpack_codes(
    packed: np.ndarray,
    bits: int,
    count: int,
    *,
    backend: NumericalBackend | None = None,
) -> np.ndarray:
    """Unpack logical codes from packed uint8 bytes."""
    validate_bits(bits)
    if count < 0:
        raise ValueError("count must be non-negative.")
    if count == 0:
        return np.array([], dtype=np.int32)

    byte_array = np.asarray(packed, dtype=np.uint8)
    required_bytes = packed_byte_count(count, bits)
    if byte_array.size < required_bytes:
        raise ValueError(
            f"Packed data has {byte_array.size} bytes but {required_bytes} "
            f"are required for {count} codes at {bits} bits."
        )

    unpack_backend = resolve_backend(backend, DEFAULT_PACKING_BACKEND)
    return kernel_unpack_codes(byte_array, bits, count, backend=unpack_backend)


def explain_packing(codes: np.ndarray, bits: int) -> str:
    """Educational helper showing codes, binary patterns, and packed bytes."""
    flat = validate_codes(codes, bits)
    packed = pack_codes(flat, bits)
    mask = code_mask(bits)

    lines = [
        "Bit Packing Explanation",
        "-----------------------",
        f"Bits per code: {bits}",
        f"Mask: {bin(mask)}",
        f"Code count: {flat.size}",
        "",
        "Codes:",
        " ".join(str(int(value)) for value in flat),
        "",
        "Binary (per code):",
        " ".join(format(int(value), f"0{bits}b") for value in flat),
        "",
        "Packed bytes:",
    ]
    for index, byte_value in enumerate(packed):
        lines.append(f"  byte[{index}] = 0x{int(byte_value):02X} ({bin(int(byte_value))})")
    return "\n".join(lines)


def benchmark_bitpacking(
    codes: np.ndarray,
    bits: int,
    *,
    repeats: int = 3,
) -> dict[str, Any]:
    """Local measurements for pack/unpack timing and byte sizes."""
    import time

    flat = validate_codes(codes, bits)
    count = int(flat.size)
    logical_bytes = int(flat.astype(np.int32).nbytes)

    pack_start = time.perf_counter()
    packed = None
    for _ in range(repeats):
        packed = pack_codes(flat, bits)
    pack_seconds = (time.perf_counter() - pack_start) / repeats

    unpack_start = time.perf_counter()
    for _ in range(repeats):
        unpacked = unpack_codes(packed, bits, count)
    unpack_seconds = (time.perf_counter() - unpack_start) / repeats

    packed_bytes = int(packed.nbytes)
    return {
        "bits": bits,
        "count": count,
        "logical_int32_bytes": logical_bytes,
        "packed_bytes": packed_bytes,
        "theoretical_packed_bytes": packed_byte_count(count, bits),
        "pack_seconds": pack_seconds,
        "unpack_seconds": unpack_seconds,
        "round_trip_exact": np.array_equal(unpacked, flat.astype(np.int32)),
    }

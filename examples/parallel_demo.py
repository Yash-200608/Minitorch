"""MiniTorch parallel execution demonstration."""

import numpy as np

from minitorch import (
    benchmark_parallel_comparison,
    pack_codes,
)
from minitorch.quantization import quantize_array


def main() -> None:
    rng = np.random.default_rng(0)
    # Large workload so parallel comparison is meaningful on some machines.
    codes = rng.integers(0, 16, size=200_000, dtype=np.int32)
    data = rng.standard_normal(200_000)

    pack_result = benchmark_parallel_comparison(
        "bit_packing_4bit",
        lambda: pack_codes(codes, 4),
        lambda: pack_codes(codes, 4, parallel=True, workers=4),
        workers=4,
        warmup=2,
        repeats=5,
    )
    print(pack_result.report())
    serial_packed = pack_codes(codes, 4)
    parallel_packed = pack_codes(codes, 4, parallel=True, workers=4)
    print(f"Pack byte parity: {np.array_equal(serial_packed, parallel_packed)}")
    print()

    encode_result = benchmark_parallel_comparison(
        "codebook_quantize_4bit",
        lambda: quantize_array(data, 4, method="codebook", packed=True),
        lambda: quantize_array(
            data,
            4,
            method="codebook",
            packed=True,
            parallel=True,
            workers=4,
        ),
        workers=4,
        warmup=1,
        repeats=3,
    )
    print(encode_result.report())
    serial_q = quantize_array(data, 4, method="codebook", packed=True)
    parallel_q = quantize_array(
        data,
        4,
        method="codebook",
        packed=True,
        parallel=True,
        workers=4,
    )
    print(f"Packed bytes equal: {np.array_equal(serial_q.packed_bytes, parallel_q.packed_bytes)}")
    print(
        "Reconstruction close: "
        f"{np.allclose(serial_q.dequantize().data, parallel_q.dequantize().data)}"
    )


if __name__ == "__main__":
    main()

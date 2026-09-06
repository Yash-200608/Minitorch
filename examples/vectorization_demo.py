"""MiniTorch vectorization benchmark demonstration."""

import numpy as np

from minitorch import pack_codes
from minitorch.kernels import (
    nearest_centroid_codes_reference,
    nearest_centroid_codes_vectorized,
    pack_codes_reference,
    pack_codes_vectorized,
    uniform_codes_reference,
    uniform_codes_vectorized,
    benchmark_vectorization,
)


def main() -> None:
    rng = np.random.default_rng(0)

    print("Uniform Quantization")
    print("--------------------")
    values = rng.standard_normal(100_000)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    max_code = 15
    uniform_result = benchmark_vectorization(
        "uniform_4bit",
        values.size,
        lambda: uniform_codes_reference(values, minimum=minimum, maximum=maximum, max_code=max_code),
        lambda: uniform_codes_vectorized(values, minimum=minimum, maximum=maximum, max_code=max_code),
        memory_notes="vectorized uses NumPy ufuncs over contiguous float64",
        warmup=2,
        repeats=5,
    )
    print(uniform_result.report())
    codes_match = np.array_equal(
        uniform_codes_reference(values, minimum=minimum, maximum=maximum, max_code=max_code),
        uniform_codes_vectorized(values, minimum=minimum, maximum=maximum, max_code=max_code),
    )
    print(f"Codes identical: {'PASS' if codes_match else 'FAIL'}")
    print()

    print("Codebook Encoding")
    print("-----------------")
    centroids = np.linspace(-2, 2, 16)
    encode_values = rng.standard_normal(100_000)
    encode_result = benchmark_vectorization(
        "codebook_encode",
        encode_values.size,
        lambda: nearest_centroid_codes_reference(encode_values, centroids),
        lambda: nearest_centroid_codes_vectorized(encode_values, centroids),
        memory_notes="vectorized builds N×K distance matrix temporarily",
        warmup=2,
        repeats=5,
    )
    print(encode_result.report())
    encode_match = np.array_equal(
        nearest_centroid_codes_reference(encode_values, centroids),
        nearest_centroid_codes_vectorized(encode_values, centroids),
    )
    print(f"Codes identical: {'PASS' if encode_match else 'FAIL'}")
    print()

    print("4-bit Bit Packing")
    print("-----------------")
    codes = rng.integers(0, 16, size=200_000, dtype=np.int32)
    pack_result = benchmark_vectorization(
        "pack_4bit",
        codes.size,
        lambda: pack_codes_reference(codes, 4),
        lambda: pack_codes_vectorized(codes, 4),
        memory_notes="reference uses Python loop; vectorized uses bitwise NumPy ops",
        warmup=2,
        repeats=5,
    )
    print(pack_result.report())
    pack_match = np.array_equal(pack_codes_reference(codes, 4), pack_codes_vectorized(codes, 4))
    api_match = np.array_equal(pack_codes(codes, 4), pack_codes(codes, 4, backend="vectorized"))
    print(f"Kernel bytes identical: {'PASS' if pack_match else 'FAIL'}")
    print(f"API backend bytes identical: {'PASS' if api_match else 'FAIL'}")


if __name__ == "__main__":
    main()

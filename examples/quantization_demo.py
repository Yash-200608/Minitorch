"""Demonstrate uniform scalar quantization and memory/error tradeoffs."""

import numpy as np

from minitorch import Tensor


def main() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(1000, 128))
    x = Tensor(data)

    print("MiniTorch Quantization Demo")
    print("===========================")
    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")
    print(f"Input nbytes: {x.data.nbytes}")
    print()

    for bits in [2, 4, 8]:
        q = x.quantize(bits=bits)
        reconstructed = q.dequantize()
        report = q.report()

        print(report.format())
        print(
            f"Actual code dtype: {q.codes.dtype}, "
            f"codes nbytes: {q.codes.nbytes}, "
            f"logical {bits}-bit codes != {bits}-bit physical storage yet"
        )
        print(
            f"Round-trip max abs diff: "
            f"{np.max(np.abs(x.data - reconstructed.data)):.6f}"
        )
        print()


if __name__ == "__main__":
    main()

"""Compare quantization error with and without rotation."""

import numpy as np

from minitorch import Tensor, compare_quantization_rotation


def main() -> None:
    rng = np.random.default_rng(0)
    uniform_data = rng.uniform(-1.0, 1.0, size=(4, 16))
    clustered = np.concatenate(
        [rng.normal(-2.0, 0.2, size=(4, 16)), rng.normal(2.0, 0.2, size=(4, 16))],
        axis=0,
    )

    print("MiniTorch Rotation + Quantization Demo")
    print("======================================")

    for name, data in [("uniform-like", uniform_data), ("clustered", clustered)]:
        print(f"\n--- {name} data ---")
        x = Tensor(data)
        uniform_exp = compare_quantization_rotation(
            x.data,
            bits=4,
            method="uniform",
            rotation="combined",
        )
        codebook_exp = compare_quantization_rotation(
            x.data,
            bits=4,
            method="codebook",
            rotation="combined",
        )

        transform = x.quantize(bits=4, rotation="combined").rotation_transform
        norm_diff = transform.l2_norm_difference(x.data)

        print(f"L2 norm difference after rotation: {norm_diff:.6e}")
        print(uniform_exp.format())
        print()
        print(codebook_exp.format())

        packed = x.quantize(bits=4, method="codebook", rotation="combined", packed=True)
        report = packed.report()
        print(
            f"Packed bytes: {report.memory.actual_packed_bytes}, "
            f"codebook bytes: {report.memory.codebook_bytes}, "
            f"total: {report.memory.total_actual_representation_bytes}"
        )


if __name__ == "__main__":
    main()

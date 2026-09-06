"""Compare uniform vs learned codebook quantization."""

import numpy as np

from minitorch import Tensor


def main() -> None:
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=-2.0, scale=0.25, size=500)
    cluster_b = rng.normal(loc=2.0, scale=0.25, size=500)
    data = np.concatenate([cluster_a, cluster_b])
    x = Tensor(data)

    print("MiniTorch Codebook Demo")
    print("=======================")
    print(f"Input shape: {x.shape}")
    print(f"Input nbytes: {x.data.nbytes}")
    print()

    for bits in [4]:
        uniform = x.quantize(bits=bits, method="uniform")
        codebook = x.quantize(bits=bits, method="codebook")

        uniform_report = uniform.report()
        codebook_report = codebook.report()

        print(f"--- {bits}-bit uniform ---")
        print(f"RMSE: {uniform_report.error.rmse:.6f}")
        print(f"Theoretical packed bytes: {uniform_report.memory.theoretical_packed_bytes}")
        print()

        print(f"--- {bits}-bit codebook ---")
        print(f"RMSE: {codebook_report.error.rmse:.6f}")
        print(f"Iterations: {codebook_report.codebook_fit.iterations}")
        print(f"Converged: {codebook_report.codebook_fit.converged}")
        print(f"Codebook bytes: {codebook_report.memory.codebook_bytes}")
        print(f"Total theoretical representation: "
              f"{codebook_report.memory.total_theoretical_representation_bytes} bytes")
        print()
        print(codebook_report.format())
        print()


if __name__ == "__main__":
    main()

"""Train a model, compress parameters, and compare memory and behavior."""

import numpy as np

from minitorch import (
    CompressionConfig,
    Linear,
    MSELoss,
    SGD,
    Tensor,
    Trainer,
    benchmark_model_compression,
    compare_model_compression,
    format_compression_matrix,
)


def main() -> None:
    inputs = Tensor(np.linspace(-1.0, 1.0, 16).reshape(16, 1))
    targets = Tensor(3.0 * inputs.data + 2.0)

    model = Linear(1, 1)
    trainer = Trainer(model, SGD(model.parameters(), lr=0.1), MSELoss().forward)
    trainer.fit(inputs, targets, epochs=200)

    print("MiniTorch Model Compression Demo")
    print("==============================")
    print(f"Trained model parameter bytes: {sum(p.data.nbytes for p in model.parameters())}")

    configs = [
        ("FP32 baseline", None),
        (
            "4-bit uniform packed",
            CompressionConfig(bits=4, method="uniform", packed=True),
        ),
        (
            "4-bit codebook packed",
            CompressionConfig(bits=4, method="codebook", packed=True),
        ),
        (
            "4-bit codebook + rotation packed",
            CompressionConfig(
                bits=4,
                method="codebook",
                packed=True,
                rotation="combined",
            ),
        ),
    ]

    print("\nConfiguration comparison")
    print("------------------------")
    for label, config in configs:
        if config is None:
            original_loss = float(MSELoss()(model(inputs), targets).data)
            print(f"{label}: bytes={sum(p.data.nbytes for p in model.parameters())}, "
                  f"loss={original_loss:.6f}")
            continue

        report = compare_model_compression(model, inputs, targets, config=config)
        print(f"{label}:")
        print(f"  bytes={report.total_compressed_bytes}, ratio={report.compression_ratio:.2f}x")
        print(f"  param RMSE={report.aggregate_parameter_rmse:.6f}")
        print(f"  output RMSE={report.output_rmse:.6f}")
        print(f"  loss delta={report.loss_delta:.6f}")

    benchmark_configs = [
        CompressionConfig(bits=4, method="uniform", packed=True),
        CompressionConfig(bits=4, method="codebook", packed=True),
        CompressionConfig(bits=4, method="codebook", packed=True, rotation="combined"),
    ]
    matrix_model = Linear(8, 16)
    results = benchmark_model_compression(matrix_model, configs=benchmark_configs)
    print("\nBenchmark matrix (Linear 8x16)")
    print(format_compression_matrix(results))


if __name__ == "__main__":
    main()

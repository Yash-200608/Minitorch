"""MiniTorch model and compressed-model persistence demonstration."""

import numpy as np

from minitorch import (
    CompressionConfig,
    Linear,
    MSELoss,
    Sequential,
    SGD,
    Tensor,
    Trainer,
    compress_model,
    format_artifact_report,
    inspect_artifact,
    load_compressed_model,
    load_model,
)


def main() -> None:
    model = Sequential(Linear(2, 8), Linear(8, 1))
    trainer = Trainer(model, SGD(model.parameters(), lr=0.05), MSELoss().forward)
    inputs = Tensor(np.linspace(0, 1, 16).reshape(8, 2))
    targets = Tensor(3 * inputs.data.sum(axis=1, keepdims=True) + 2)
    trainer.fit(inputs, targets, epochs=30)

    model_path = "demo_model.mt"
    model.save(model_path)
    loaded = load_model(model_path)
    sample = Tensor([[0.2, 0.4]])
    np.testing.assert_allclose(model(sample).data, loaded(sample).data, rtol=1e-9, atol=1e-9)
    print(format_artifact_report(model_path))
    print()

    config = CompressionConfig(bits=4, method="codebook", packed=True, rotation="combined")
    compressed = compress_model(model, config=config)
    compressed_path = "demo_model.mtc"
    compressed.save(compressed_path)
    loaded_compressed = load_compressed_model(compressed_path)
    materialized = loaded_compressed.materialize(model)
    np.testing.assert_allclose(
        compressed.materialize(model)(sample).data,
        materialized(sample).data,
        rtol=1e-9,
        atol=1e-9,
    )
    print(format_artifact_report(compressed_path))
    print()

    model_bytes = inspect_artifact(model_path)["file_bytes"]
    compressed_bytes = inspect_artifact(compressed_path)["file_bytes"]
    print(f"Model artifact: {model_bytes} bytes")
    print(f"Compressed artifact: {compressed_bytes} bytes")
    if model_bytes > 0:
        print(f"File-size ratio (model/compressed): {model_bytes / compressed_bytes:.2f}x")


if __name__ == "__main__":
    main()

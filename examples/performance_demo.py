"""MiniTorch performance profiling demonstration."""

import numpy as np

from minitorch import (
    Linear,
    MSELoss,
    ProfilerMode,
    Sequential,
    SGD,
    Tensor,
    Trainer,
    format_compression_tradeoff_report,
    profile_compression_tradeoffs,
    profile_graph_optimizer_comparison,
    profile_quantization_memory_matrix,
    profile_rotation_pipeline,
)


def main() -> None:
    rng = np.random.default_rng(0)
    inputs = Tensor(rng.standard_normal((8, 4)))
    targets = Tensor(rng.standard_normal((8, 2)))
    model = Sequential(Linear(4, 8), Linear(8, 2))

    trainer = Trainer(model, SGD(model.parameters(), lr=0.05), MSELoss().forward)
    step_profile = trainer.profile_step(inputs, targets, mode=ProfilerMode.LIGHTWEIGHT)
    print(step_profile.report())
    print()

    detailed = trainer.profile_step(inputs, targets, mode=ProfilerMode.DETAILED)
    print("Top measured operations:")
    for name, seconds in detailed.top_operations(k=5):
        print(f"  {name}: {seconds * 1000:.2f} ms")
    print()

    graph_profile = profile_graph_optimizer_comparison(
        model,
        inputs,
        targets,
        MSELoss().forward,
        repeats=2,
    )
    print("Graph optimizer comparison:")
    print(f"  original nodes: {graph_profile.graph_stats['original_nodes']}")
    print(f"  optimized nodes: {graph_profile.graph_stats['optimized_nodes']}")
    print(f"  original backward: {graph_profile.stages['original_backward'] * 1000:.2f} ms")
    print(f"  optimized backward: {graph_profile.stages['optimized_backward'] * 1000:.2f} ms")
    print()

    quant_data = rng.standard_normal(32)
    matrix = profile_quantization_memory_matrix(quant_data)
    print("Quantization memory/time matrix (subset):")
    for entry in matrix[:3]:
        print(
            f"  {entry['method']} {entry['bits']}bit "
            f"quant={entry['quantization_seconds'] * 1000:.2f}ms "
            f"bytes={entry['representation_bytes']} "
            f"rmse={entry['reconstruction_rmse']:.6f}"
        )
    print()

    rotation_data = rng.standard_normal(16)
    rotation_profile = profile_rotation_pipeline(rotation_data, bits=4)
    print("Rotation pipeline stages:")
    for stage, seconds in rotation_profile.stages.items():
        print(f"  {stage}: {seconds * 1000:.2f} ms")
    print()

    tradeoffs = profile_compression_tradeoffs(model, inputs, targets)
    print(format_compression_tradeoff_report(tradeoffs))


if __name__ == "__main__":
    main()

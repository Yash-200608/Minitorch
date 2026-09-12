"""MiniTorch Phase S — compact end-to-end final evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from minitorch import (
    CompressionConfig,
    MSELoss,
    MixedPrecisionConfig,
    compress_model,
    compress_model_mixed_precision,
)
from minitorch.experiments import (
    Experiment,
    ExperimentConfig,
    ExperimentTechnique,
    TrainingConfig,
    build_baseline_dataset,
    build_baseline_model,
    save_experiment_result,
    seed_everything,
)
from minitorch.training import Adam, Trainer

from benchmarks.canonical_suite import (
    _initialize_model_weights,
    format_benchmark_report,
    run_canonical_suite,
)
from benchmarks.test_matrix import build_test_summary, format_test_summary


def main() -> None:
    seed = 42
    results_dir = Path("benchmarks/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== MiniTorch Final Evaluation ===\n")
    print(format_test_summary(build_test_summary(Path.cwd())))
    print()

    # Compact canonical suite (fast mode for demo)
    report = run_canonical_suite(seed=seed, epochs=30, fast=True)
    print(format_benchmark_report(report))
    report_path = results_dir / "final_evaluation.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (results_dir / "final_evaluation.txt").write_text(
        format_benchmark_report(report), encoding="utf-8"
    )

    # Experiment Lab + AI-MRI on a fresh short run
    seed_everything(seed)
    inputs, targets = build_baseline_dataset(seed)
    model = build_baseline_model()
    _initialize_model_weights(model, seed)
    trainer = Trainer(model, Adam(model.parameters(), lr=0.05), MSELoss().forward)
    trainer.fit(inputs, targets, epochs=20)

    compressed = compress_model(model, CompressionConfig(bits=4, method="codebook"))
    compression_report = compressed.report(template=model, inputs=inputs, targets=targets)
    mixed = compress_model_mixed_precision(
        model, MixedPrecisionConfig(default_bits=8, parameter_bits={"2.weight": 4})
    )
    mixed_report = mixed.report(template=model, inputs=inputs, targets=targets)

    exp = Experiment(
        ExperimentConfig(
            name="final_evaluation_fp32",
            technique=ExperimentTechnique.FP32_BASELINE,
            training=TrainingConfig(epochs=0),
            seed=seed,
        )
    ).run()
    save_experiment_result(exp, results_dir / f"experiment_{exp.experiment_id}.json")

    output = model(inputs)
    mri = model.mri(output=output, compression=mixed_report, experiment=exp)
    print()
    print(str(mri))
    print()
    print("Compression (codebook 4-bit) measured:")
    print(f"  bytes={compression_report.total_compressed_bytes} ratio={compression_report.compression_ratio}")
    print(f"  param RMSE={compression_report.aggregate_parameter_rmse}")
    print(f"  output RMSE={compression_report.output_rmse}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()

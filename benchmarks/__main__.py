"""Canonical MiniTorch benchmark entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.canonical_suite import format_benchmark_report, run_and_save
from benchmarks.test_matrix import build_test_summary, format_test_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MiniTorch Phase S canonical benchmarks.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Shorter QAT epochs and fewer vectorization sizes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results"),
        help="Directory for JSON/text outputs.",
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Only emit the test matrix summary.",
    )
    args = parser.parse_args()

    summary = build_test_summary(Path.cwd())
    print(format_test_summary(summary))
    args.output.mkdir(parents=True, exist_ok=True)
    import json

    (args.output / "test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if args.skip_benchmarks:
        return

    report = run_and_save(args.output, fast=args.fast)
    print()
    print(format_benchmark_report(report))
    print()
    print(f"Wrote: {args.output / 'canonical_benchmark.json'}")


if __name__ == "__main__":
    main()

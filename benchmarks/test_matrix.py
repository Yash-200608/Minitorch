"""Generate MiniTorch test summary by subsystem classification."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Map test module stems to subsystem labels (first match wins via explicit file map).
_FILE_TO_SUBSYSTEM: dict[str, str] = {
    "test_tensor": "Core",
    "test_autograd": "Autograd",
    "test_gradcheck": "Autograd",
    "test_graph": "DSA / Graph",
    "test_dsa": "DSA / Graph",
    "test_topk": "DSA / Graph",
    "test_benchmark": "DSA / Graph",
    "test_registry": "DSA / Graph",
    "test_cache": "Core / Cache",
    "test_proof": "Glass Box",
    "test_explorer": "Glass Box",
    "test_gradient_detective": "Glass Box",
    "test_model_xray": "Glass Box",
    "test_graph_optimizer": "Glass Box",
    "test_ai_mri": "Glass Box / AI-MRI",
    "test_training": "Training",
    "test_optimizers": "Training",
    "test_nn_layers": "Training",
    "test_activations": "Training",
    "test_losses": "Training",
    "test_dataloader": "Training",
    "test_quantization": "Quantization",
    "test_quantization_codebook": "Codebook",
    "test_bitpacking": "Bit Packing",
    "test_bitpacking_integration": "Bit Packing",
    "test_rotation": "Rotation",
    "test_rotation_quantization": "Rotation",
    "test_model_compression": "Compression",
    "test_mixed_precision": "Mixed Precision",
    "test_mixed_precision_experiments": "Mixed Precision",
    "test_qat": "QAT",
    "test_qat_training": "QAT",
    "test_qat_checkpoint": "QAT",
    "test_profiling": "Performance / Profiling",
    "test_parallel": "Parallelism",
    "test_vectorized_kernels": "Vectorization",
    "test_persistence": "Persistence",
    "test_checkpoint": "Checkpointing",
    "test_checkpoint_resume": "Checkpointing",
    "test_experiments": "Experiment Lab",
    "test_experiment_serialization": "Experiment Lab",
    "test_experiment_comparisons": "Experiment Lab",
    "test_experiment_mri": "Experiment Lab / AI-MRI",
}


def _collect_test_nodes(repo_root: Path) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    nodes: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if "::" in line and not line.startswith("="):
            nodes.append(line.split()[0])
    return nodes


def build_test_summary(repo_root: Path | None = None) -> dict:
    root = repo_root or Path(__file__).resolve().parents[1]
    nodes = _collect_test_nodes(root)
    by_subsystem: dict[str, list[str]] = defaultdict(list)
    by_file: dict[str, int] = defaultdict(int)

    for node in nodes:
        file_part = node.split("::")[0]
        stem = Path(file_part).stem
        subsystem = _FILE_TO_SUBSYSTEM.get(stem, "Other")
        by_subsystem[subsystem].append(node)
        by_file[stem] += 1

    return {
        "total_collected": len(nodes),
        "by_subsystem": {k: len(v) for k, v in sorted(by_subsystem.items())},
        "subsystem_tests": {k: sorted(v) for k, v in sorted(by_subsystem.items())},
        "by_file": dict(sorted(by_file.items())),
    }


def format_test_summary(summary: dict) -> str:
    lines = [
        "MiniTorch Test Summary",
        "----------------------",
        "",
        f"Total collected: {summary['total_collected']}",
        "",
    ]
    for subsystem, count in sorted(summary["by_subsystem"].items()):
        lines.append(f"{subsystem}: {count}")
    lines.append("")
    lines.append("By file:")
    for stem, count in summary["by_file"].items():
        lines.append(f"  {stem}: {count}")
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = build_test_summary(root)
    text = format_test_summary(summary)
    print(text)
    out = root / "benchmarks" / "results" / "test_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()

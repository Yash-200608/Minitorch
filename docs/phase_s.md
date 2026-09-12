# Phase S — Final Validation & Benchmark Suite

Phase S is the validation and evidence layer for MiniTorch. It does **not** add a new major subsystem.

## Deliverables

| Artifact | Purpose |
| --- | --- |
| `benchmarks/` | Canonical reproducible benchmark suite |
| `benchmarks/test_matrix.py` | Subsystem test classification summary |
| `examples/final_evaluation.py` | Compact end-to-end evaluation + AI-MRI |
| `docs/final_report.md` | Technical report with measured findings |
| `tests/test_phase_s_benchmarks.py` | Structural tests (no timing thresholds) |

## Commands

```bash
python -m pytest tests/ -q
python -m benchmarks --fast
python examples/final_evaluation.py
```

## Canonical workload

- Model: `Sequential(Linear(1,8), ReLU, Linear(8,1))`
- Dataset: deterministic regression `y = 3x + 2`
- Seed: 42
- Weight init: non-degenerate random (Linear defaults to zeros)

## What is measured

Training baseline, quantization matrix (uniform/codebook 2/4/8), packing parity, rotation, mixed precision, QAT, graph optimizer (Linear-only; ReLU not replayable), parallelism, vectorized packing, persistence file sizes.

## Important distinctions

- Training technique (normal vs QAT) ≠ storage technique (uniform/codebook/packing/mixed)
- In-memory representation bytes ≠ serialized `.mtc` file size
- Complexity ≠ runtime ≠ hardware ISA claims
- Findings are scoped: “in this benchmark / workload / configuration”

## Core project status

**Complete through Phase S.** Further work belongs in optional research tracks (`docs/future.md`, `docs/final_report.md`).

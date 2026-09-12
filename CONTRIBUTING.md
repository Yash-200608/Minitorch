# Contributing

MiniTorch is an educational Glass Box framework. Changes should stay
inspectable, measurable, and honest.

## Setup

```bash
python -m pip install -e ".[dev]"
pytest tests/ -q
```

Python 3.11+ and NumPy 1.26+ are required.

## Before you change code

1. Read [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`docs/architecture.md`](docs/architecture.md).
2. Do not rewrite Phases A–S to “clean things up.”
3. Do not add a new optimizer, quantizer, graph, or profiler unless the change
   is a necessary fix.
4. Prefer composing existing APIs (compression, QAT, Experiment Lab, MRI).

## Tests

- Run `pytest tests/ -q` before opening a PR.
- Do **not** add `assert runtime < X` for machine-dependent timings.
- Unit tests should stay tiny and deterministic. Heavy benches belong in
  [`benchmarks/`](benchmarks/) or [`examples/`](examples/).
- Do not invent metrics or fill missing values with zeros.

## Experiment Lab fairness

When comparing techniques:

- same model, data, seed, and epoch budget where practical
- keep training technique (normal vs QAT) separate from storage technique
- scope conclusions to the measured workload

## Style

- Match the surrounding module. Avoid drive-by refactors.
- Public APIs live in [`minitorch/__init__.py`](minitorch/__init__.py).
  Do not export internal helpers without a reason.
- ASCII in console reports is preferred (Windows consoles).

## Pull requests

Use the PR template. Include:

- what changed and why
- how you tested it
- docs updates if the change is user-facing

## Security

See [`SECURITY.md`](SECURITY.md). Do not commit secrets, `.env` files, or
machine-specific `benchmarks/results/`.

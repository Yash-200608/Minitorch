# Phase P — AI-MRI

AI-MRI is MiniTorch's **unified Glass Box diagnostic composition layer**.

## What it is

An aggregator that combines structured outputs from:

- Model X-Ray (parameters, architecture)
- Graph Explorer (DAG structure, DSA counters)
- Gradient Detective (gradient health, Top-K)
- Performance Profiler (supplied profiles only)
- Compression reports (supplied experiments only)
- QAT state / experiment results

## What it is not

- Not a learned health predictor
- Not an arbitrary 0–100 health score
- Not a duplicate implementation of subsystems
- Not an automatic trainer/compressor/benchmarker

## API

```python
mri = model.mri(output=loss, performance=profile, compression=report)
print(mri)
mri.to_dict()
mri.to_json()

mri.prove(parameter)      # delegates to Proof Engine
mri.graph_explorer()        # Graph Explorer when output supplied
mri.gradients_report()      # Gradient Detective report
mri.performance_profile()   # supplied profile only
```

## Missing data semantics

Each section uses:

- `available` — real data present
- `unavailable` — not supplied (e.g. no graph without `output=`)
- `not_applicable` — e.g. QAT on a non-QAT model

No fabricated zeros.

## Observations and recommendations

Evidence-based observations with severity `INFO`, `WARNING`, or `CRITICAL`.

Recommendations are conservative and do not mutate the model.

Language avoids unsupported causal claims.

## Performance cost

`mri()` is lightweight by default. It does not silently profile, compress, or train.

Pass precomputed `performance`, `compression`, and `qat_experiment` objects.

## Snapshot semantics

MRI assumes supplied model/report objects are not mutated concurrently during inspection.

## Future

Structured `to_dict()` / `to_json()` support dashboards and experiment comparison without redesign.

GUI/dashboard is out of scope for Phase P.

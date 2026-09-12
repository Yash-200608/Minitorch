# Future work — optional research tracks

Core MiniTorch is **complete through Phase S** (validation + Experiment Lab + final report).

The items below are **optional**. They are not a mandatory Phase T/U/V sequence.

## Track A — Advanced QAT

- Codebook QAT
- Activation quantization
- Mixed-precision QAT

## Track B — Native performance

- Specialized kernels beyond NumPy vectorization
- Actual ISA-specific SIMD (only if measured and claimed carefully)

## Track C — Storage

- Memory mapping
- Streaming artifacts
- Large checkpoint handling

## Track D — Glass Box UX

- Interactive graph visualization
- AI-MRI dashboard

## Track E — Research algorithms

- New quantization algorithms
- Broader graph optimizer replay coverage (e.g. ReLU)
- New compression strategies

## Integration rules if pursued

1. Correctness and inspectability before raw speed
2. Opt-in experimental flags — never silent behavior changes
3. Real benchmarks via Experiment Lab / `benchmarks/`
4. Each feature must name the DSA it demonstrates
5. Scope claims to measured workloads

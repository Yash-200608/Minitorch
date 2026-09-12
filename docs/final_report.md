# MiniTorch Final Technical Report

**Phase S — Final Validation + Benchmark Suite + Research Evidence**

Framework version: `0.1.0`  
Report generated from measured runs on the Phase S canonical suite.  
Environment for the numbers below: Python 3.12.10, NumPy 1.26.4, Windows-11, MiniTorch 0.1.0.

> Re-run `python -m benchmarks --fast` (or without `--fast`) to regenerate machine-local results under `benchmarks/results/`.

---

## Abstract

MiniTorch is a small, educational **Glass Box** autodiff and neural-network framework. It implements reverse-mode automatic differentiation on an explicit computational DAG, exposes graph/DSA introspection, and integrates post-training quantization, packing, rotation experiments, mixed precision, weight-only QAT, profiling, persistence, and a reproducible Experiment Lab. This report documents what was built, how subsystems connect, what was measured on a deterministic canonical workload, and which claims are supported by evidence.

---

## Motivation

Production frameworks hide graph structure, memory accounting, and optimization tradeoffs. MiniTorch exists to make those internals **visible, measurable, and comparable**:

- Build it (correct core systems)
- Measure it (Experiment Lab + benchmarks)
- Explain it (Proof Engine, X-Ray, AI-MRI)
- Compare it (fair seed/model/data control)

---

## Architecture

```text
                         MiniTorch
                      "Glass Box AI"
                             |
       +---------------------+---------------------+
       |                     |                     |
       v                     v                     v
     Tensor               Autograd                NN
       |                     |                     |
       |                     v                     v
       |              Computational DAG         Training
       |                     |                     |
       |              +------+------+              |
       |              v             v              |
       |             DFS           Topo            |
       +---------------------+---------------------+
                             v
                       Glass Box Layer
                             |
        +--------------------+--------------------+
        v                    v                    v
      Proof              Detective              X-Ray
        |                    |                    |
        +--------------------+--------------------+
                             v
                           AI-MRI
                             |
       +---------------------+---------------------+
       v                     v                     v
 Quantization            Compression           Performance
       |                     |                     |
 Codebook               Mixed Precision       Parallelism
 Packing                Model Storage          Vectorization
 Rotation
       |                     |                     |
       +---------------------+---------------------+
                             v
                         Persistence
                             |
                       Experiment Lab
```

Package layout is documented in [`architecture.md`](architecture.md). Core MiniTorch is complete through **Phase S**.

---

## DSA

| Operation | Structure / Algorithm | Time | Space | Actual usage |
| --- | --- | --- | --- | --- |
| Graph traversal | DFS (explicit stack) | O(V+E) | O(V) | Explorer, backward prep, MRI graph section |
| Topological order | Kahn | O(V+E) | O(V) | Autograd scheduling, explorer topo |
| Top-K selection | Binary heap | O(N log K) | O(K) | Gradient Detective, compression largest errors |
| Parameter lookup | HashMap registry | Avg O(1) | O(P) | Module parameters, named access |
| Computation cache | HashMap memoization | Avg O(1) | O(entries) | Optional forward memoization |
| Chunk partitioning | range splitting | O(chunks) | O(1) | Parallel compression workers |

These complexities match the implementations in `minitorch/dsa/` and callers; they are asymptotic statements, not runtime guarantees.

---

## Autograd

Tensors build a parent DAG during forward ops. `backward()` runs reverse-mode accumulation using topological order. Gradient checking (`gradcheck`) validates local rules numerically. The Proof Engine replays the same local rules for human-readable explanations.

---

## Glass Box

| Tool | Role |
| --- | --- |
| Proof Engine | Explain a parameter gradient path |
| Graph Explorer | DFS / topo / DSA counters |
| Gradient Detective | Magnitude health + Top-K |
| Model X-Ray | Architecture + parameters + optional graph |
| Graph Optimizer | Opt-in algebraic simplification (limited op set) |
| AI-MRI | Compose existing diagnostics; does not recompute compression |

---

## Training

`Module` / `Parameter` / `Linear` / `Sequential` / activations / `MSELoss` / SGD / Adam / `Dataset` / `DataLoader` / `Trainer` / checkpoints. Linear defaults to **zero initialization**; Phase S benchmarks intentionally apply non-degenerate random init so compression metrics are meaningful.

---

## Numerical Representation

Uniform and codebook quantization (2/4/8-bit), bit packing (logical vs physical), rotation transforms with inverse, and memory reports separating codes, codebooks, and packed storage.

---

## Compression & Mixed Precision

Post-training `compress_model` and per-parameter `compress_model_mixed_precision`. Original models are not mutated. Mixed-precision QAT is **not implemented**.

---

## QAT

Weight-only uniform fake quantization with identity STE. Experimental and opt-in. Distinct from post-training compression.

---

## Performance

Timing, training-step profiling, parallel chunk execution, and NumPy-backed vectorized kernels. Vectorization here means array kernels (not claimed ISA-specific SIMD such as AVX2).

---

## Persistence & Experiment Lab

Versioned `.mtc` artifacts (model / compressed / checkpoint) with CRC32. Experiment Lab provides deterministic IDs, structured results, JSON serialization, comparisons, and MRI attachment.

---

## Results (canonical benchmark)

**Workload:** `Sequential(Linear(1,8), ReLU, Linear(8,1))`, deterministic regression `y = 3x + 2`, seed `42`, Adam lr `0.05`, 35 epochs, non-zero weight init. Fast-mode QAT used 25 epochs.

### Baseline training

| Metric | Measured |
| --- | --- |
| Initial loss | ~12.92 |
| Final loss | ~0.25 |
| Training time | ~0.017 s |
| Model bytes (in-memory FP64 params) | 200 |
| Hottest training stage | backward |

### Quantization matrix (same trained model)

| Configuration | Bytes | Ratio | Out RMSE | Loss Δ |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 200 | 1.00 | 0 | 0 |
| Uniform 8-bit | 100 | 2.00 | 0.00519 | -0.00439 |
| Uniform 4-bit | 100 | 2.00 | 0.0256 | 0.0190 |
| Uniform 2-bit | 100 | 2.00 | 0.0782 | 0.0722 |
| Codebook 8-bit | 8292 | 0.024 | 0 | 0 |
| Codebook 4-bit | 612 | 0.327 | 0.00393 | 0.00324 |
| Codebook 2-bit | 228 | 0.877 | 0.0895 | 0.0877 |
| Mixed precision | 100 | 2.00 | 0.0172 | -0.0148 |

Notes for this workload:

- Unpacked uniform codes still use byte-oriented storage accounting; **packing** further reduced uniform 4-bit from 100 → 13 bytes with **zero** output difference.
- Codebook 4-bit reduced output RMSE vs uniform 4-bit (0.00393 vs 0.0256) but **increased** total bytes due to codebook overhead (512 codebook bytes) — ratio < 1 for this tiny model.
- Codebook 8-bit expands storage dramatically (8292 bytes) for negligible error benefit on 25 parameters.

### Packing

Logical vs packed reconstruction matched (`max_output_difference = 0`) for uniform and codebook 4-bit. Packing changed storage, not numerical semantics, in this experiment.

### Rotation

On `0.weight`, rotation labels were **unchanged** for uniform/codebook at 4/8-bit in this run (RMSE identical within measurement). Earlier Phase H demos showed rotation can worsen some tensors; that remains possible on other data.

### Mixed precision

Global 4-bit out RMSE ~0.0256; mixed (default 8, `2.weight` at 4) out RMSE ~0.0172 at the same reported 100 uncompressed-code bytes in this accounting mode. On this tiny model, mixed precision improved error vs global 4-bit without changing the measured unpacked byte total.

### QAT

| Path | FP loss | Quantized loss | Damage | Out RMSE |
| --- | ---: | ---: | ---: | ---: |
| Normal → PTQ | 0.0104 | 0.0124 | 0.00196 | 0.0309 |
| QAT → quantize | 0.0116 | 0.0093 | -0.00231 | 0.0211 |

Observed label: **QAT improved quantization damage** for this seed/workload. Forward overhead ratio ~3.8× in the QAT forward microbench.

### Graph optimizer

On `Linear(1,1)` (ReLU graphs are not replayable by the current optimizer): nodes 10→10, edges 10→10. Backward time **worse** (~0.00025 s → ~0.00125 s). Structural reduction was not observed; overhead of optimize+replay dominated.

### Parallelism (4-bit packed compress)

| Workers | Speedup (parallel/serial) |
| ---: | ---: |
| 1 | ~0.96 |
| 2 | ~0.17 |
| 4 | ~0.14 |

On this small model, parallelism **hurt**. No break-even toward speedup was observed in the tested worker range.

### Vectorization (`pack_codes`)

| Size | Speedup (vectorized / reference) |
| ---: | ---: |
| 64 | ~4.7× |
| 256 | ~15.4× |

Vectorized packing helped even at small sizes in this NumPy environment.

### Persistence file sizes (distinct from in-memory bytes)

| Artifact | File size (bytes) |
| --- | ---: |
| Model `.mtc` | 845 |
| Compressed codebook `.mtc` | 2452 |
| Checkpoint `.mtc` | 8796 |
| Experiment JSON | 2631 |
| In-memory FP params | 200 |
| In-memory compressed (codebook 4) | 612 |

### Nonlinear spot check

`Sequential(Linear(2,8), Sigmoid, Linear(8,1))`: codebook 4-bit output RMSE (0.00217) was lower than uniform 4-bit (0.0216) in this measurement — consistent with the regression workload’s error trend, still subject to codebook overhead on small models.

---

## Research questions (answered for this benchmark)

1. **Does codebook quantization reduce error?**  
   Yes for 4-bit on the canonical model (output RMSE 0.00393 vs 0.0256 uniform).

2. **Does codebook overhead matter for small models?**  
   Yes — codebook 4/8-bit ratios were < 1 (storage expanded).

3. **Does bit packing provide real storage savings?**  
   Yes — uniform 4-bit 100 → 13 bytes with identical outputs.

4. **Does rotation help the tested workload?**  
   Unchanged on measured `0.weight` cases in this run.

5. **Does mixed precision improve the memory/error tradeoff?**  
   Lower output RMSE than global 4-bit at similar unpacked bytes in this run; not a general optimality claim.

6. **Does QAT reduce quantization damage?**  
   Observed improvement for this seed/epochs/bits configuration.

7. **Does graph optimization improve runtime?**  
   No on the Linear-only measured graph (worse backward time; no node removal).

8. **When does parallelism help?**  
   Not on this tiny compression workload; overhead dominated.

9. **When does vectorization help?**  
   Packing kernels showed clear speedups at n=64 and n=256 here.

10. **Dominant tradeoffs?**  
    Error vs codebook storage on tiny models; packing helps physical size; QAT can help damage at training-time cost; parallel workers are not free.

---

## Limitations

- Small educational framework (NumPy backend)
- Limited layers / losses / activations
- Linear zero-init by default (benchmarks re-init)
- Weight-only QAT; no codebook QAT; no activation quantization; no mixed-precision QAT
- Graph optimizer cannot replay ReLU (and some other ops)
- No GPU / distributed training / cloud experiment tracking
- No production-scale validation
- Vectorization ≠ proven AVX2/SIMD ISA claims
- Timings are machine-local; do not treat them as absolute thresholds

---

## Status of capabilities

| Area | Status |
| --- | --- |
| Core autodiff + training | Implemented |
| Glass Box + AI-MRI | Implemented |
| Quantization / packing / rotation | Implemented |
| Compression / mixed precision | Implemented (post-training) |
| QAT | Experimental, opt-in |
| Experiment Lab | Implemented |
| Phase S benchmarks | Implemented (`benchmarks/`) |
| Mixed-precision QAT | Not implemented |
| GPU / distributed | Not implemented |

---

## Future work (optional research tracks)

These are **not** mandatory phases:

- **Track A — Advanced QAT:** codebook QAT, activation QAT, mixed-precision QAT  
- **Track B — Native performance:** specialized kernels, true ISA-specific SIMD  
- **Track C — Storage:** memory mapping, streaming artifacts  
- **Track D — Glass Box UX:** interactive visualization / dashboard  
- **Track E — Research:** new quantizers, graph rules, compression strategies  

---

## How to reproduce

```bash
python -m pytest tests/ -q
python -m benchmarks --fast
python examples/final_evaluation.py
```

Unit tests do not assert timing thresholds. Benchmark JSON lives in `benchmarks/results/` (gitignored by default).

---

## Conclusion

MiniTorch’s value is not a long feature list. It is a coherent stack where DSA, autodiff, training, representation, compression, performance, persistence, and evaluation reinforce each other — and where findings can be **scoped to measurements**. Phase S closes the core project loop: validate, measure, document limitations, and leave optional research tracks explicit rather than inventing infinite mandatory phases.

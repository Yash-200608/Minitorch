# Architecture

MiniTorch: Tensor core, DAG, DSA, autograd, Phase A utilities, and Phase B
Glass Box inspection layers.

## Package layout

```text
minitorch/
  tensor.py                 ops, inspect/lineage, explore/proof/explain
  graph.py                  ComputationGraph
  autograd/
    engine.py               reverse-mode autograd
    proof.py                gradient proof replay
    explorer.py             graph explorer + DSA profile
  gradcheck.py
  cache.py
  registry.py               ParameterRegistry
  nn.py                     Module, layers, gradient_report, xray
  losses.py                 MSELoss
  data.py                   Dataset, TensorDataset, DataLoader
  training/
    optimizers.py           SGD, Adam
    trainer.py              Trainer, TrainingHistory
  quantization.py           QuantizedTensor, Codebook, uniform + codebook modes
  bitpacking.py             pack_codes, unpack_codes
  rotation.py               transforms + rotation quantization experiments
  compression.py            post-training model compression, uint8 physical storage
  intelligence/
    gradient_detective.py   gradient health + heap Top-K ranking
    model_xray.py           combined inspection report
    graph_optimizer.py      verified DAG simplification
    ai_mri.py               unified MRI composition layer
  mixed_precision.py        per-parameter compression experiments
  experiments/              Experiment Lab (config, runner, serialization, canonical suites)
    config.py               serializable ExperimentConfig
    runner.py               Experiment.run() evaluation pipeline
    canonical.py            quantization matrix, QAT, rotation, performance suites
    serialization.py        JSON save/load with schema validation
  qat.py                    weight-only QAT
  dsa/                      DFS, topo sort, topk, benchmarks
```

## Glass Box data flow

```text
forward → Tensor.parents → ComputationGraph
       → backward (engine) → Tensor.grad
       → proof (replay same local rules)
       → explorer (DFS / topo on graph)
       → gradient_report (parameter grads + topk)
       → xray (registry + graph + gradients)
       → mri (unified composition + optional experiment result)
       → Experiment Lab (train → compress → measure → compare → persist)
       → Trainer.fit (zero_grad → forward → loss → backward → step)
```

## Experiment Lab data flow

```text
ExperimentConfig → seed_everything → train → evaluate FP
  → compress / mixed precision / QAT comparison (existing APIs)
  → ExperimentResult (metrics, timings, memory, observations)
  → JSON artifact / compare_experiments / model.mri(experiment=...)
```

## DSA reuse

| Feature | Algorithm |
| --- | --- |
| Explorer `dsa_profile()` | existing DFS + Kahn counters |
| Gradient Top-K | `dsa.topk` binary heap |
| Parameter lookup | HashMap registry |
| Cache | HashMap memoization |
| Optimizer state | HashMap keyed by parameter id |
| Training profiling | graph + DFS + topo counters via `Trainer.profile_step` |
| Experiment registry | `CANONICAL_EXPERIMENTS` dict lookup |
| Sensitivity Top-K in compression reports | `dsa.topk` |

## Training data flow

```text
Dataset → DataLoader → Trainer.train_step
  → optimizer.zero_grad (grad = None)
  → model.forward
  → loss_fn
  → loss.backward
  → optimizer.step (mutates Parameter.data, outside autograd)
```

## Not implemented

See [`future.md`](future.md) for optional research tracks (mixed-precision QAT, cloud tracking, SIMD).

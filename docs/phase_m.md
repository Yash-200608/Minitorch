# Phase M — Persistence & Versioned Artifacts

Phase M adds safe, versioned serialization for models and compressed models.

## Philosophy

> The model should not just be saved — its structure and numerical representation should remain understandable.

No pickle as the primary format. Artifacts use explicit metadata + binary payloads.

## Artifact layout (v1)

```text
Header (32 bytes)
  magic:        8 bytes  "MINITORC"
  format_version: uint16
  artifact_type:  uint8  (1=model, 2=compressed_model)
  metadata_length, payload_crc32, payload_length

Metadata: JSON (UTF-8, sort_keys=True)
Payload:  concatenated binary blobs with per-field offsets in metadata
```

## Versioning

| Field | Purpose |
|-------|---------|
| `format_version` | Artifact layout version (currently `1`) |
| `framework_version` | MiniTorch package version (`0.1.0`) |

Unknown format versions are rejected with `ArtifactError`.

## What is saved

### Model artifact (`model`)
- Module tree (Sequential, Linear, ReLU, Sigmoid, Tanh)
- Parameter names, shapes, dtypes, `requires_grad`
- Raw parameter bytes (C-contiguous)
- **Gradients are not saved** by default

### Compressed artifact (`compressed_model`)
- `CompressionConfig`
- Per-parameter quantization metadata
- **Exact packed bytes** (no unpack/repack round trip)
- Codebook centroids as float64 bytes
- Rotation metadata for deterministic rebuild

## API

```python
save_model(model, "model.mt")
loaded = load_model("model.mt")

compressed.save("model.mtc")
loaded_compressed = load_compressed_model("model.mtc")

info = inspect_artifact("model.mt")
print(format_artifact_report("model.mtc"))

model.save("path.mt")
```

## Safety

- CRC32 payload checksum
- Magic and version validation
- No arbitrary Python object execution
- Loaded arrays are **copies** (no shared memory with original)

## Limitations (v1)

- No optimizer state
- No training history
- No computation cache persistence
- No cross-version migration framework yet

## Example

```bash
py examples/persistence_demo.py
```

## Tests

`tests/test_persistence.py` — parity, corruption, packed-byte equality, rotation.

# eegdino-rs

[![Crates.io](https://img.shields.io/crates/v/eegdino.svg)](https://crates.io/crates/eegdino)
[![Docs.rs](https://docs.rs/eegdino/badge.svg)](https://docs.rs/eegdino)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Rust inference crate for the [EEG-DINO](https://github.com/miraclefish/EEG-DINO) foundation model, built on [Burn](https://burn.dev).

EEG-DINO learns robust EEG representations via hierarchical self-distillation on 9 000+ hours of EEG data. This crate provides a numerically verified port of the encoder and classification head with NRMSE < 1e-6 against the original PyTorch implementation.

## Features

- **Three model sizes** --- Small (4.6 M), Medium (33 M), Large (201 M params)
- **Multiple backends** --- CPU (ndarray + optional Accelerate BLAS), GPU (wgpu Metal/Vulkan), GPU f16
- **On-device spectral embedding** --- DFT basis cached as device tensors, no CPU round-trips
- **Builder API** --- ergonomic construction with auto-detection, configurable normalization
- **Batch encoding** --- `encode_batch` packs N signals into a single forward pass

## Quick start

```rust
use eegdino_rs::prelude::*;
use burn::backend::NdArray;

type B = NdArray;

// Build encoder (auto-detects model size from weights)
let encoder = EegDinoEncoder::<B>::builder()
    .weights("weights/eeg_dino_small.safetensors")
    .device(Default::default())
    .build()?;

// Encode a 10-second EEG recording (19 channels @ 200 Hz)
let signal = vec![0.0f32; 19 * 2000];
let result = encoder.encode_raw(&signal, 1, 19, 2000)?;
println!("{:?}", result.shape);
// [1, 191, 200]  (1 global token + 19 channels x 10 patches)
```

### Batch encoding

```rust
let signals: Vec<Vec<f32>> = load_recordings();

// Single batched forward pass (fastest):
let result = encoder.encode_batch(&signals, 19, 2000)?;
// result.shape == [N, 191, 200]

// Or one-by-one:
let results = encoder.encode_many(&signals, 19, 2000);
```

### Shorthand

```rust
let (encoder, load_ms) = EegDinoEncoder::<B>::load(
    "weights/eeg_dino_small.safetensors".as_ref(),
    None,   // auto-detect size from weights
    device,
)?;
```

## Model variants

| Variant | Params | d_model | Heads | Layers | FFN dim | Weights |
|---------|--------|---------|-------|--------|---------|---------|
| Small   | 4.6 M  | 200     | 8     | 12     | 512     | 17 MB   |
| Medium  | 33 M   | 512     | 16    | 16     | 1 024   | 129 MB  |
| Large   | 201 M  | 1 024   | 16    | 24     | 2 048   | 770 MB  |

## Backends

| Feature | Backend | Notes |
|---------|---------|-------|
| `ndarray` (default) | CPU | Rayon multi-threaded + SIMD |
| `blas-accelerate` | CPU + Accelerate | Recommended on Apple Silicon |
| `wgpu` | GPU f32 | Metal (macOS) / Vulkan (Linux) |
| `wgpu-f16` | GPU f16 | Half-precision, 2x less memory |

```bash
cargo build --release                                          # CPU
cargo build --release --features blas-accelerate               # CPU + Accelerate
cargo build --release --no-default-features --features wgpu    # GPU f32
cargo build --release --no-default-features --features wgpu-f16 # GPU f16
```

## Benchmarks

Apple M4 Mac Mini, 19 channels, 2 000 samples (10 s @ 200 Hz).
See [ABLATION.md](ABLATION.md) for the full study.

| Backend | Small B=1 | Small B=8 | Large B=1 | Large B=8 | Peak |
|---------|-----------|-----------|-----------|-----------|------|
| CPU ndarray | 40 ms | 284 ms | 527 ms | 3.66 s | 28 samp/s |
| CPU + Accelerate | 33 ms | 234 ms | 262 ms | 1.72 s | 34 samp/s |
| GPU f32 | 111 ms | 194 ms | 401 ms | 1.45 s | 43 samp/s |
| **GPU f16** | **85 ms** | **152 ms** | **287 ms** | **962 ms** | **53 samp/s** |

## Numerical parity

Verified against PyTorch on identical inputs. Errors are at the f32 accumulation floor:

| Variant | Max abs error | NRMSE |
|---------|---------------|-------|
| Small | 8.5e-7 | 5.5e-7 |
| Medium | 2.1e-6 | 8.8e-7 |
| Large | 4.8e-6 | 5.9e-7 |

```bash
python scripts/parity_test.py
cargo run --release --example parity_check
```

## Weight conversion

Convert pretrained PyTorch `.pt` checkpoints to safetensors:

```bash
pip install torch safetensors

# All three sizes
python scripts/convert_weights.py --all \
    --input-dir path/to/EEG-DINO/pre-trained-models \
    --output-dir weights

# Single checkpoint
python scripts/convert_weights.py \
    --input model_EEG_DINO_S.pt \
    --output weights/eeg_dino_small.safetensors
```

The script strips teacher/projector/loss weights, renames Sequential keys, and transposes linear weights to Burn's `[in, out]` layout.

## Architecture

```
Raw EEG [B, 19, P, 200]
    |
    +-- Temporal: 3-layer Conv2d + GroupNorm + GELU
    +-- Spectral: on-device DFT matmul -> Linear(101, D)
    +-- Channel:  cached one-hot -> Linear(19, D)
    |
    v  (sum + depthwise time encoding)
Patch embeddings [B, 19*P, D]
    |
    v  Transformer encoder (N layers, pre-norm, fused QKV bias)
    +-- Global tokens injected after layer 1
    |
    v
Embeddings [B, 1 + 19*P, D]
```

`EmbeddingCache` stores DFT basis and channel one-hot as on-device tensors,
created once at load time.

## Examples

```bash
# Inference
cargo run --release --example infer -- --weights weights/eeg_dino_small.safetensors

# Benchmark
cargo run --release --example bench -- --batch 1,4,8

# Parity check
cargo run --release --example parity_check

# Full ablation study
./scripts/ablation.sh
```

## Project layout

```
src/
  lib.rs              Public API + prelude
  prelude.rs          use eegdino_rs::prelude::*
  config.rs           ModelConfig (S / M / L)
  weights.rs          safetensors -> burn weight loader
  inference.rs        EegDinoEncoder (builder, batch, many), EegDinoClassifier
  model/
    embedding.rs      PatchEmbedding + EmbeddingCache
    attention.rs      Multi-head attention with fused QKV bias
    mlp.rs            Feed-forward with GELU
    transformer.rs    TransformerEncoderLayer (pre-norm)
    encoder.rs        EEGEncoder (cached + uncached forward)
    classifier.rs     ClassificationModel
```

## License

[MIT](LICENSE)

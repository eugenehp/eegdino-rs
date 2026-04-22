# Ablation Study

Apple M4 Mac Mini. Input: 19 channels, 2 000 samples (10 s @ 200 Hz).
30 timed iterations after 10 warmup (CPU) / 20 warmup (GPU). Median latency.

## Backend comparison

| Backend | Model | B=1 | B=4 | B=8 | Peak samp/s |
|---------|-------|-----|-----|-----|-------------|
| **CPU ndarray** | Small | 40 ms | 143 ms | 284 ms | 28.2 |
| | Medium | 144 ms | 512 ms | 1.03 s | 7.8 |
| | Large | 527 ms | 1.83 s | 3.66 s | 2.2 |
| **CPU + Accelerate** | Small | 33 ms | 117 ms | 234 ms | 34.2 |
| | Medium | 95 ms | 352 ms | 700 ms | 11.5 |
| | Large | 262 ms | 882 ms | 1.72 s | 4.6 |
| **GPU wgpu f32** | Small | 111 ms | 140 ms | 194 ms | 42.5 |
| | Medium | 190 ms | 308 ms | 490 ms | 16.4 |
| | Large | 401 ms | 866 ms | 1.45 s | 5.6 |
| **GPU wgpu f16** | Small | 85 ms | 120 ms | 152 ms | **53.0** |
| | Medium | 143 ms | 235 ms | 369 ms | **22.4** |
| | Large | 287 ms | 586 ms | 962 ms | **8.3** |

## Speedup vs CPU ndarray baseline

| Backend | Small B=1 | Small B=8 | Large B=1 | Large B=8 |
|---------|-----------|-----------|-----------|-----------|
| + Accelerate | 1.2x | 1.2x | **2.0x** | **2.1x** |
| GPU f32 | 0.4x | 1.5x | 1.3x | **2.5x** |
| GPU f16 | 0.5x | 1.9x | **1.8x** | **3.8x** |

## Observations

- **Accelerate** gives a consistent 1.2-2.1x speedup over baseline ndarray by
  replacing the generic matmul with Apple's vectorised BLAS. The gain grows with
  model size because matmul becomes a larger fraction of total compute.
- **GPU f32 (Metal)** has higher B=1 latency due to kernel launch overhead, but
  scales better with batch size. At B=8 it processes the Large model 2.5x faster
  than CPU baseline.
- **GPU f16** is the fastest configuration. Half-precision halves memory bandwidth
  and enables wider SIMD. The Large model at B=8 runs 3.8x faster than CPU
  baseline and 1.5x faster than GPU f32.
- CPU throughput saturates around B=4 (memory-bandwidth-bound). GPU continues to
  scale through B=8 and beyond.

## Optimizations applied

| Optimization | Effect |
|--------------|--------|
| Pre-computed DFT basis | `DftBasis` caches cos/sin matrices (40 K floats) at model init, avoiding trig recomputation every forward call |
| Fused QKV bias | `fuse_qkv_bias()` bakes `[q_bias, 0, v_bias]` into `Linear.bias` after weight loading, so forward uses burn's fused linear+bias path |
| Pre-computed channel one-hot | `ChannelOneHot` caches the identity matrix at init |
| Conv-Norm fusion | `ConvNormBlock::forward()` chains conv, norm, gelu in one call |
| Rayon-parallel FFT | CPU spectral path processes all 190 patches concurrently |
| On-device DFT matmul | GPU spectral path computes rfft magnitudes entirely on Metal via `x @ cos_basis^T`, `x @ sin_basis^T` — no CPU round-trip |
| f16 backend | Half-precision GPU reduces memory bandwidth 2x |

## Numerical parity

All optimizations preserve numerical parity with the Python reference (NRMSE < 1e-6):

| Model | Max abs error | NRMSE |
|-------|---------------|-------|
| Small | 8.5 e-7 | 5.5 e-7 |
| Medium | 2.1 e-6 | 8.8 e-7 |
| Large | 4.8 e-6 | 5.9 e-7 |

## Reproducing

```bash
# Full ablation (builds each backend, ~10 min)
./scripts/ablation.sh

# Single model
./scripts/ablation.sh small

# Individual backends
cargo run --release --example bench                                            # CPU ndarray
cargo run --release --example bench --features blas-accelerate                 # CPU + Accelerate
cargo build --release --example bench --no-default-features --features wgpu
./target/release/examples/bench --device gpu                                   # GPU f32
cargo build --release --example bench --no-default-features --features wgpu-f16
./target/release/examples/bench --device gpu-f16                               # GPU f16
```

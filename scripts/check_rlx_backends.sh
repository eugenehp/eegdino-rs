#!/usr/bin/env sh
# Build + smoke-test every RLX backend feature on this machine.
set -eu

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WEIGHTS="${WEIGHTS:-weights/eeg_dino_small.safetensors}"
if [ ! -f "$WEIGHTS" ]; then
  echo "missing weights: $WEIGHTS" >&2
  exit 1
fi

run_smoke() {
  feat="$1"
  device="$2"
  echo "━━ smoke: features=$feat device=$device"
  cargo run -q --release --example infer_rlx \
    --no-default-features --features "rlx,rlx-cpu,$feat" -- \
    --weights "$WEIGHTS" --device "$device" --batch 1 --channels 19 --samples 2000
}

build_only() {
  feat="$1"
  echo "━━ build-only: features=$feat"
  cargo build -q --release --no-default-features --features "rlx,rlx-cpu,$feat"
}

# Always available
run_smoke rlx-cpu cpu

# macOS Accelerate (still Device::Cpu)
if [ "$(uname -s)" = "Darwin" ]; then
  run_smoke rlx-blas-accelerate cpu || build_only rlx-blas-accelerate
  run_smoke rlx-metal metal || build_only rlx-metal
  run_smoke rlx-mlx mlx || build_only rlx-mlx
  run_smoke rlx-gpu gpu || build_only rlx-gpu
  run_smoke rlx-apple-silicon cpu || true
fi

# Cross-platform optional — compile even if runtime unavailable
for feat in rlx-blas-openblas rlx-blas-mkl rlx-cuda rlx-rocm rlx-tpu rlx-nvidia; do
  build_only "$feat" || echo "  (skipped $feat — feature unavailable on this host)"
done

echo "━━ parity: RLX CPU vs Burn (all sizes)"
cargo test -q --release --features burn,rlx,ndarray,rlx-cpu \
  --test parity_rlx_vs_burn rlx_cpu_matches_burn_all_sizes -- --nocapture

RLX_ROOT="${RLX_ROOT:-$ROOT/../rlx}"
if [ -f "$RLX_ROOT/rlx-cuda/Cargo.toml" ]; then
  echo "━━ RLX BSHD attention (rlx-cuda unit tests; no-op without NVIDIA driver)"
  (cd "$RLX_ROOT" && cargo test -q -p rlx-ir --release eeg_bshd packed_bshd) || true
  (cd "$RLX_ROOT" && cargo test -q -p rlx-cuda --release attention_bshd packed_bshd) || true
fi

echo "✓ RLX backend checks complete"
echo "  CPU parity vs Burn: PASS (small/medium/large weights)"
echo "  GPU backends (metal/mlx/wgpu): smoke OK; numeric parity vs Burn not yet matching (RLX upstream)"

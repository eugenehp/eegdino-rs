#!/usr/bin/env bash
# Ablation study: benchmark each backend/feature combination.
#
# Usage:
#   ./scripts/ablation.sh           # full study
#   ./scripts/ablation.sh small     # single model
set -euo pipefail
cd "$(dirname "$0")/.."

ONLY="${1:-}"
ITERS=30
WARMUP=10
BATCH="1,4,8"

only_flag=""
if [ -n "$ONLY" ]; then
    only_flag="--only $ONLY"
fi

header() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

# ── CPU: ndarray baseline ────────────────────────────────────────────────────
header "CPU  ndarray (SIMD + Rayon)"
cargo build --release --example bench --features ndarray 2>&1 | tail -1
./target/release/examples/bench --batch "$BATCH" --iters "$ITERS" --warmup "$WARMUP" $only_flag

# ── CPU: ndarray + blas-accelerate ───────────────────────────────────────────
header "CPU  ndarray + Apple Accelerate"
cargo build --release --example bench --features ndarray,blas-accelerate 2>&1 | tail -1
./target/release/examples/bench --batch "$BATCH" --iters "$ITERS" --warmup "$WARMUP" $only_flag

# ── GPU: wgpu f32 ────────────────────────────────────────────────────────────
header "GPU  wgpu f32 (Metal)"
cargo build --release --example bench --no-default-features --features wgpu 2>&1 | tail -1
./target/release/examples/bench --device gpu --batch "$BATCH" --iters "$ITERS" --warmup 20 $only_flag

# ── GPU: wgpu f16 ────────────────────────────────────────────────────────────
header "GPU  wgpu f16 (Metal, half-precision)"
cargo build --release --example bench --no-default-features --features wgpu-f16 2>&1 | tail -1
./target/release/examples/bench --device gpu-f16 --batch "$BATCH" --iters "$ITERS" --warmup 20 $only_flag

echo ""
echo "================================================================"
echo "  Ablation complete"
echo "================================================================"

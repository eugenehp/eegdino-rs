#!/usr/bin/env sh
# Benchmark EEG-DINO RLX backends + Burn vs RLX parity.
set -eu

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WEIGHTS="${WEIGHTS:-weights/eeg_dino_small.safetensors}"
RUNS="${RUNS:-5}"
WARMUP="${WARMUP:-3}"
BATCH="${BATCH:-1,2,4,8}"

if [ ! -f "$WEIGHTS" ]; then
  echo "missing weights: $WEIGHTS" >&2
  exit 1
fi

FEATS="burn,rlx,ndarray,rlx-cpu,rlx-blas-accelerate,rlx-metal,rlx-mlx,rlx-gpu,rlx-cuda,rlx-rocm,rlx-tpu"

echo "━━━ Build backend_compare + bench_rlx"
cargo build -q --release --no-default-features --features "$FEATS" \
  --example backend_compare --example bench_rlx

echo ""
echo "━━━ Burn vs RLX parity + latency (all compiled backends)"
cargo run -q --release --no-default-features --features "$FEATS" \
  --example backend_compare -- --weights "$WEIGHTS" --runs "$RUNS" --warmup "$WARMUP"

echo ""
echo "━━━ RLX throughput by backend (small model)"
for dev in cpu mps mlx wgpu cuda; do
  echo "--- device=$dev ---"
  if cargo run -q --release --no-default-features --features "$FEATS" \
      --example bench_rlx -- --device "$dev" --weights "$WEIGHTS" \
      --batch "$BATCH" --iters "$RUNS" --warmup "$WARMUP" --only small 2>&1; then
    :
  else
    echo "  (skipped $dev)"
  fi
  echo ""
done

echo ""
echo "━━━ CPU parity gate (small/medium/large weights)"
cargo test -q --release --features burn,rlx,ndarray,rlx-cpu \
  --test parity_rlx_vs_burn rlx_cpu_matches_burn_all_sizes -- --nocapture

echo ""
echo "✓ benchmark complete"

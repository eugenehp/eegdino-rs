//! Parity: RLX `EegDinoEncoder` vs Burn reference on identical inputs.
//!
//! ```text
//! cargo test --release --features burn,rlx,ndarray,rlx-cpu \
//!     --test parity_rlx_vs_burn -- --nocapture
//! ```

//! Optional: RLX vs Burn reference (enable with `--features burn,ndarray`).
#![cfg(all(feature = "burn", feature = "rlx", feature = "ndarray"))]

use std::f32::consts::PI;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use eegdino_rs::{init_threads, EegDinoEncoder as BurnEncoder, ModelSize};

type B = NdArray;

fn deterministic_signal(batch: usize, channels: usize, samples: usize) -> Vec<f32> {
    let n = batch * channels * samples;
    (0..n)
        .map(|i| ((i as f32 * 0.013) % (2.0 * PI)).sin() * 50.0)
        .collect()
}

fn compare(name: &str, burn: &[f32], rlx: &[f32], max_tol: f32, nrmse_tol: f32) -> bool {
    assert_eq!(burn.len(), rlx.len(), "{name}: length mismatch");
    let mut max_abs = 0.0f32;
    let mut sum_sq = 0.0f64;
    let mut sum_ref_sq = 0.0f64;
    for (&b, &r) in burn.iter().zip(rlx.iter()) {
        let e = (b - r).abs();
        max_abs = max_abs.max(e);
        sum_sq += (e as f64) * (e as f64);
        sum_ref_sq += (b as f64) * (b as f64);
    }
    let nrmse = if sum_ref_sq > 0.0 {
        (sum_sq / sum_ref_sq).sqrt()
    } else {
        0.0
    };
    let pass = max_abs <= max_tol && nrmse <= f64::from(nrmse_tol);
    eprintln!(
        "[{name:>6}] {}  max_abs={max_abs:.3e}  nrmse={nrmse:.3e}  n={}",
        if pass { "PASS" } else { "FAIL" },
        burn.len()
    );
    pass
}

fn run_pair(name: &str, size: ModelSize, weights: &str, device: rlx::Device) -> bool {
    let weights_path = std::path::Path::new(weights);
    if !weights_path.exists() {
        eprintln!("[{name}] SKIP — {weights} not found");
        return true;
    }

    let signal = deterministic_signal(1, 19, 2000);
    let device_cpu = NdArrayDevice::Cpu;

    let burn = BurnEncoder::<B>::builder()
        .weights(weights)
        .size(size)
        .device(device_cpu)
        .build()
        .expect("burn build");
    let burn_out = burn.encode_raw(&signal, 1, 19, 2000).expect("burn encode");

    let (mut rlx, _) =
        eegdino_rs::EegDinoEncoder::load(weights_path, None, device).expect("rlx load");
    let rlx_out = rlx.encode_raw(&signal, 1, 19, 2000).expect("rlx encode");

    assert_eq!(burn_out.shape, rlx_out.shape, "{name}: shape mismatch");
    let (max_tol, nrmse_tol) = cpu_parity_tol(name);
    compare(
        name,
        &burn_out.embeddings,
        &rlx_out.embeddings,
        max_tol,
        nrmse_tol,
    )
}

fn cpu_parity_tol(model: &str) -> (f32, f32) {
    match model.to_lowercase().as_str() {
        "small" => (3e-6, 2e-6),
        "medium" => (6e-6, 4e-6),
        "large" => (1.2e-5, 2e-6),
        _ => (1.2e-5, 4e-6),
    }
}

#[test]
fn rlx_cpu_matches_burn_all_sizes() {
    init_threads(Some(4));

    let cases = [
        (
            "small",
            ModelSize::Small,
            "weights/eeg_dino_small.safetensors",
        ),
        (
            "medium",
            ModelSize::Medium,
            "weights/eeg_dino_medium.safetensors",
        ),
        (
            "large",
            ModelSize::Large,
            "weights/eeg_dino_large.safetensors",
        ),
    ];

    let mut all_pass = true;
    for (name, size, path) in cases {
        if !run_pair(name, size, path, rlx::Device::Cpu) {
            all_pass = false;
        }
    }
    assert!(all_pass, "RLX CPU vs Burn parity failed");
}

#[test]
#[cfg(feature = "rlx-blas-accelerate")]
fn rlx_accelerate_matches_burn_small() {
    init_threads(Some(4));
    assert!(run_pair(
        "accel",
        ModelSize::Small,
        "weights/eeg_dino_small.safetensors",
        rlx::Device::Cpu,
    ));
}

#[test]
#[cfg(feature = "rlx-metal")]
fn rlx_metal_matches_burn_small() {
    init_threads(Some(4));
    assert!(run_pair(
        "metal",
        ModelSize::Small,
        "weights/eeg_dino_small.safetensors",
        rlx::Device::Metal,
    ));
}

#[test]
#[cfg(feature = "rlx-mlx")]
fn rlx_mlx_matches_burn_small() {
    init_threads(Some(4));
    assert!(run_pair(
        "mlx",
        ModelSize::Small,
        "weights/eeg_dino_small.safetensors",
        rlx::Device::Mlx,
    ));
}

#[test]
#[cfg(feature = "rlx-gpu")]
#[ignore = "run with --ignored when Burn + weights available"]
fn rlx_gpu_matches_burn_small() {
    init_threads(Some(4));
    assert!(run_pair(
        "gpu",
        ModelSize::Small,
        "weights/eeg_dino_small.safetensors",
        rlx::Device::Gpu,
    ));
}

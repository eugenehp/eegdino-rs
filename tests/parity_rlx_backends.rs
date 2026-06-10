//! RLX backend parity vs RLX CPU reference (small encoder).

use std::f32::consts::PI;

use eegdino_rs::{init_threads, is_device_available, EegDinoEncoder};

fn deterministic_signal() -> Vec<f32> {
    (0..19 * 2000)
        .map(|i| ((i as f32 * 0.013) % (2.0 * PI)).sin() * 50.0)
        .collect()
}

fn compare(cpu: &[f32], other: &[f32], max_tol: f32, nrmse_tol: f32, cos_tol: f64) -> bool {
    assert_eq!(cpu.len(), other.len());
    let mut max_abs = 0.0f32;
    let mut sum_sq = 0.0f64;
    let mut sum_ref_sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&a, &b) in cpu.iter().zip(other.iter()) {
        let e = (a - b).abs();
        max_abs = max_abs.max(e);
        sum_sq += (e as f64) * (e as f64);
        sum_ref_sq += (a as f64) * (a as f64);
        let ad = a as f64;
        let bd = b as f64;
        dot += ad * bd;
        norm_a += ad * ad;
        norm_b += bd * bd;
    }
    let nrmse = if sum_ref_sq > 0.0 {
        (sum_sq / sum_ref_sq).sqrt()
    } else {
        0.0
    };
    let denom = norm_a.sqrt() * norm_b.sqrt();
    let cos_dist = if denom > 0.0 { 1.0 - dot / denom } else { 0.0 };
    let pass = max_abs <= max_tol && nrmse <= f64::from(nrmse_tol) && cos_dist <= cos_tol;
    eprintln!(
        "max_abs={max_abs:.3e} nrmse={nrmse:.3e} cos_dist={cos_dist:.3e} {}",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn run_device(
    name: &str,
    device: rlx::Device,
    weights: &str,
    max_tol: f32,
    nrmse_tol: f32,
    cos_tol: f64,
) -> bool {
    let weights_path = std::path::Path::new(weights);
    if !weights_path.exists() {
        eprintln!("[{name}] SKIP — {weights} not found");
        return true;
    }
    if !is_device_available(device) {
        eprintln!("[{name}] SKIP — not available on this host");
        return true;
    }

    let signal = deterministic_signal();
    let (mut cpu_enc, _) =
        EegDinoEncoder::load(weights_path, None, rlx::Device::Cpu).expect("cpu load");
    let cpu = cpu_enc
        .encode_raw(&signal, 1, 19, 2000)
        .expect("cpu encode");

    let (mut enc, _) = EegDinoEncoder::load(weights_path, None, device).expect("device load");
    let out = enc.encode_raw(&signal, 1, 19, 2000).expect("device encode");

    eprint!("[{name}] ");
    compare(
        &cpu.embeddings,
        &out.embeddings,
        max_tol,
        nrmse_tol,
        cos_tol,
    )
}

#[test]
fn rlx_cpu_baseline() {
    init_threads(Some(4));
    assert!(run_device(
        "cpu",
        rlx::Device::Cpu,
        "weights/eeg_dino_small.safetensors",
        3e-6,
        2e-6,
        1e-5,
    ));
}

#[test]
#[cfg(feature = "rlx-metal")]
fn rlx_metal_matches_cpu() {
    init_threads(Some(4));
    assert!(run_device(
        "metal",
        rlx::Device::Metal,
        "weights/eeg_dino_small.safetensors",
        3e-6,
        2e-6,
        1e-5,
    ));
}

#[test]
#[cfg(feature = "rlx-mlx")]
fn rlx_mlx_matches_cpu() {
    init_threads(Some(4));
    assert!(run_device(
        "mlx",
        rlx::Device::Mlx,
        "weights/eeg_dino_small.safetensors",
        3e-6,
        2e-6,
        1e-5,
    ));
}

#[test]
#[cfg(feature = "rlx-gpu")]
fn rlx_gpu_matches_cpu() {
    init_threads(Some(4));
    assert!(run_device(
        "gpu",
        rlx::Device::Gpu,
        "weights/eeg_dino_small.safetensors",
        3e-6,
        2e-6,
        1e-5,
    ));
}

#[test]
#[cfg(feature = "rlx-cuda")]
fn rlx_cuda_matches_cpu() {
    init_threads(Some(4));
    // Strict f32 matmul + tiled kernel path (see rlx-cuda RLX_CUDA_PARITY).
    unsafe { std::env::set_var("RLX_CUDA_PARITY", "1") };
    assert!(run_device(
        "cuda",
        rlx::Device::Cuda,
        "weights/eeg_dino_small.safetensors",
        3e-6,
        2e-6,
        1e-5,
    ));
}

#[test]
#[cfg(feature = "rlx-rocm")]
fn rlx_rocm_matches_cpu() {
    init_threads(Some(4));
    assert!(run_device(
        "rocm",
        rlx::Device::Rocm,
        "weights/eeg_dino_small.safetensors",
        3e-6,
        2e-6,
        1e-5,
    ));
}

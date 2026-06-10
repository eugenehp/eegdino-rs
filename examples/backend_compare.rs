//! Compare RLX backends: latency + parity vs RLX CPU (and optional Burn).
//!
//! ```text
//! cargo run --release --features all-backends --example backend_compare -- \
//!     --runs 5
//! ```

mod common;

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use eegdino_rs::prelude::*;

use common::{
    bench_encode, compare_embeddings, cpu_parity_tol, deterministic_signal, fmt_metrics,
    passes_parity, NUM_CHANNELS, NUM_SAMPLES,
};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "weights/eeg_dino_small.safetensors")]
    weights: PathBuf,

    #[arg(long, default_value_t = 1)]
    batch: usize,

    #[arg(long, default_value_t = 3)]
    warmup: usize,

    #[arg(long, default_value_t = 5)]
    runs: usize,

    #[arg(long)]
    devices: Option<String>,

    /// Also compare each backend against Burn (requires `--features burn,ndarray`).
    #[arg(long)]
    burn: bool,
}

struct Row {
    label: String,
    load_ms: f64,
    median_ms: f64,
    mean_ms: f64,
    cpu_metrics: Option<common::Metrics>,
    burn_metrics: Option<common::Metrics>,
    ok_cpu: bool,
    ok_burn: bool,
}

fn rlx_cpu_reference(weights: &Path, signal: &[f32], batch: usize) -> anyhow::Result<Vec<f32>> {
    let (mut enc, _) = EegDinoEncoder::load(weights, None, rlx::Device::Cpu)?;
    let out = enc.encode_raw(signal, batch, NUM_CHANNELS, NUM_SAMPLES)?;
    Ok(out.embeddings)
}

#[cfg(feature = "burn")]
fn burn_reference(weights: &Path, signal: &[f32], batch: usize) -> anyhow::Result<Vec<f32>> {
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;
    use eegdino_rs::BurnEegDinoEncoder;
    let enc = BurnEegDinoEncoder::<NdArray>::builder()
        .weights(weights)
        .device(NdArrayDevice::Cpu)
        .build()?;
    let out = enc.encode_raw(signal, batch, NUM_CHANNELS, NUM_SAMPLES)?;
    Ok(out.embeddings)
}

fn default_devices() -> Vec<(&'static str, rlx::Device)> {
    let mut v = Vec::new();
    #[cfg(feature = "rlx-cpu")]
    {
        let name = if cfg!(feature = "rlx-blas-accelerate") {
            "CPU+Accelerate"
        } else {
            "CPU"
        };
        v.push((name, rlx::Device::Cpu));
    }
    #[cfg(feature = "rlx-metal")]
    v.push(("Metal/MPS", rlx::Device::Metal));
    #[cfg(feature = "rlx-mlx")]
    v.push(("MLX", rlx::Device::Mlx));
    #[cfg(feature = "rlx-gpu")]
    v.push(("wgpu", rlx::Device::Gpu));
    v.retain(|(_, d)| is_device_available(*d));
    v
}

fn try_row(
    label: &str,
    device: rlx::Device,
    weights: &Path,
    signal: &[f32],
    batch: usize,
    cpu_ref: &[f32],
    burn_ref: Option<&[f32]>,
    warmup: usize,
    runs: usize,
) -> Row {
    let mut row = Row {
        label: label.into(),
        load_ms: 0.0,
        median_ms: 0.0,
        mean_ms: 0.0,
        cpu_metrics: None,
        burn_metrics: None,
        ok_cpu: false,
        ok_burn: true,
    };

    if !is_device_available(device) {
        return row;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let t0 = Instant::now();
        let (mut enc, _) = EegDinoEncoder::load(weights, None, device)?;
        row.load_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let (median, mean, _) = bench_encode(warmup, runs, || {
            enc.encode_raw(signal, batch, NUM_CHANNELS, NUM_SAMPLES)?;
            Ok(())
        });
        row.median_ms = median;
        row.mean_ms = mean;

        let out = enc.encode_raw(signal, batch, NUM_CHANNELS, NUM_SAMPLES)?;
        let (max_tol, nrmse_tol) = cpu_parity_tol("small");
        let m = compare_embeddings(cpu_ref, &out.embeddings);
        row.ok_cpu = passes_parity(&m, max_tol, nrmse_tol);
        if device != rlx::Device::Cpu {
            row.cpu_metrics = Some(m);
        }
        if let Some(bref) = burn_ref {
            let bm = compare_embeddings(bref, &out.embeddings);
            row.ok_burn = passes_parity(&bm, max_tol, nrmse_tol);
            row.burn_metrics = Some(bm);
        }
        anyhow::Ok(())
    }));

    if result.is_err() {
        row.ok_cpu = false;
        row.ok_burn = false;
    }
    row
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    init_threads(None);

    if !args.weights.exists() {
        anyhow::bail!("weights not found: {}", args.weights.display());
    }

    let signal = deterministic_signal(args.batch);
    eprint!("RLX CPU reference … ");
    let cpu_ref = rlx_cpu_reference(&args.weights, &signal, args.batch)?;
    eprintln!("ok ({} values)", cpu_ref.len());

    let burn_ref = if args.burn {
        #[cfg(feature = "burn")]
        {
            eprint!("Burn reference … ");
            let b = burn_reference(&args.weights, &signal, args.batch)?;
            eprintln!("ok");
            Some(b)
        }
        #[cfg(not(feature = "burn"))]
        {
            anyhow::bail!("--burn requires `--features burn,ndarray`");
        }
    } else {
        None
    };

    let backends: Vec<(&str, rlx::Device)> = if let Some(ref ds) = args.devices {
        ds.split(',')
            .map(|s| s.trim())
            .map(|s| {
                let d = parse_device(s)?;
                Ok((device_label(d), d))
            })
            .collect::<anyhow::Result<_>>()?
    } else {
        default_devices()
    };

    println!();
    println!("=== EEG-DINO RLX backend compare ===");
    println!("  weights : {}", args.weights.display());
    println!("  batch   : {}", args.batch);
    println!(
        "  parity  : vs RLX-CPU{}",
        if args.burn { " + Burn" } else { "" }
    );
    println!();

    println!(
        "  {:<18}  {:>8}  {:>10}  {:>10}  {}",
        "Backend", "Load", "Median", "Mean", "Parity"
    );
    println!("  {}", "-".repeat(72));

    for (label, device) in backends {
        eprint!("  {label:<18} … ");
        let row = try_row(
            label,
            device,
            &args.weights,
            &signal,
            args.batch,
            &cpu_ref,
            burn_ref.as_deref(),
            args.warmup,
            args.runs,
        );
        if let Some(ref m) = row.cpu_metrics {
            let cpu_tag = if row.ok_cpu { "PASS" } else { "FAIL" };
            let burn_tag =
                row.burn_metrics
                    .as_ref()
                    .map(|bm| if row.ok_burn { "PASS" } else { "FAIL" });
            eprintln!(
                "CPU:{cpu_tag}  {}  {}",
                fmt_metrics(m),
                burn_tag
                    .map(|t| format!(
                        "Burn:{t} max={:.2e}",
                        row.burn_metrics.as_ref().unwrap().max_abs
                    ))
                    .unwrap_or_default()
            );
        } else if device == rlx::Device::Cpu {
            eprintln!("(reference)");
        } else {
            eprintln!("SKIP");
        }
    }

    Ok(())
}

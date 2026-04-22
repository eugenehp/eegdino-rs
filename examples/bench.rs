/// Benchmark EEG-DINO encoder latency and throughput across model sizes and
/// batch sizes.
///
/// Usage:
///   cargo run --release --example bench
///   cargo run --release --example bench -- --batch 1,2,4,8 --iters 30
///   cargo run --release --example bench -- --only small
///   cargo run --release --example bench --features blas-accelerate
///   cargo run --release --example bench --no-default-features --features wgpu -- --device gpu
///   cargo run --release --example bench --no-default-features --features wgpu-f16 -- --device gpu-f16
use std::path::Path;
use std::time::Instant;

use clap::Parser;
use eegdino_rs::{EegDinoEncoder, ModelSize, init_threads};

#[derive(Parser)]
#[command(name = "eegdino-bench")]
struct Args {
    /// Device: cpu, gpu, gpu-f16.
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Comma-separated batch sizes.
    #[arg(long, default_value = "1,2,4,8")]
    batch: String,

    /// Timed iterations per configuration.
    #[arg(long, default_value_t = 20)]
    iters: usize,

    /// Warmup iterations.
    #[arg(long, default_value_t = 3)]
    warmup: usize,

    /// Only bench a specific size: small, medium, large.
    #[arg(long)]
    only: Option<String>,
}

const NUM_CHANNELS: usize = 19;
const NUM_SAMPLES: usize = 2000;

fn bench_batch<B: burn::prelude::Backend>(
    encoder: &EegDinoEncoder<B>,
    signal: &[f32],
    batch_size: usize,
    warmup: usize,
    iters: usize,
) -> anyhow::Result<(f64, f64, f64, f64, f64)> {
    let batched: Vec<f32> = signal.iter().copied().cycle()
        .take(signal.len() * batch_size).collect();

    // Warmup
    for _ in 0..warmup {
        let _ = encoder.encode_raw(&batched, batch_size, NUM_CHANNELS, NUM_SAMPLES)?;
    }

    // Timed
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let _ = encoder.encode_raw(&batched, batch_size, NUM_CHANNELS, NUM_SAMPLES)?;
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = times.len();
    let mean = times.iter().sum::<f64>() / n as f64;
    let median = times[n / 2];
    let min = times[0];
    let max = *times.last().unwrap();
    let throughput = (batch_size as f64) * 1000.0 / mean;

    Ok((median, mean, min, max, throughput))
}

fn run_bench<B: burn::prelude::Backend>(
    models: &[(&str, ModelSize, &str)],
    signal: &[f32],
    device: B::Device,
    batch_sizes: &[usize],
    warmup: usize,
    iters: usize,
) -> anyhow::Result<()> {
    for &(name, size, weights_path) in models {
        if !Path::new(weights_path).exists() {
            println!("[{name}] SKIP - {weights_path} not found\n");
            continue;
        }

        let t0 = Instant::now();
        let encoder = EegDinoEncoder::<B>::builder()
            .weights(weights_path)
            .size(size)
            .device(device.clone())
            .build()?;
        let load_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let cfg = &encoder.config;
        println!("--- {name} (d={}, h={}, L={}, ffn={}) ---",
            cfg.feature_size, cfg.num_heads, cfg.num_layers, cfg.dim_feedforward);
        println!("  Load: {load_ms:.1} ms");
        println!();
        println!("  {:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>12}",
            "Batch", "Median", "Mean", "Min", "Max", "Throughput");

        for &bs in batch_sizes {
            let (median, mean, min, max, throughput) =
                bench_batch::<B>(&encoder, signal, bs, warmup, iters)?;
            println!("  {:>5}  {:>9.2}ms  {:>9.2}ms  {:>9.2}ms  {:>9.2}ms  {:>8.1} samp/s",
                bs, median, mean, min, max, throughput);
        }
        println!();
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    init_threads(None);

    let batch_sizes: Vec<usize> = args.batch
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<_, _>>()?;

    let all_models: Vec<(&str, ModelSize, &str)> = vec![
        ("Small", ModelSize::Small, "weights/eeg_dino_small.safetensors"),
        ("Medium", ModelSize::Medium, "weights/eeg_dino_medium.safetensors"),
        ("Large", ModelSize::Large, "weights/eeg_dino_large.safetensors"),
    ];

    let models: Vec<_> = match &args.only {
        Some(f) => {
            let f = f.to_lowercase();
            all_models.into_iter().filter(|(n, _, _)| n.to_lowercase() == f).collect()
        }
        None => all_models,
    };

    let signal: Vec<f32> = (0..NUM_CHANNELS * NUM_SAMPLES)
        .map(|i| ((i as f32) * 0.001).sin() * 50.0)
        .collect();

    println!("EEG-DINO Inference Benchmark");
    println!("  Device:  {}", args.device);
    println!("  Input:   {NUM_CHANNELS} ch x {NUM_SAMPLES} samples (10 s @ 200 Hz)");
    println!("  Batches: {:?}", batch_sizes);
    println!("  Warmup:  {}, Timed: {}", args.warmup, args.iters);
    println!();

    match args.device.as_str() {
        "cpu" => {
            #[cfg(feature = "ndarray")]
            { run_bench::<burn::backend::NdArray>(&models, &signal,
                burn::backend::ndarray::NdArrayDevice::Cpu,
                &batch_sizes, args.warmup, args.iters)? }
            #[cfg(not(feature = "ndarray"))]
            anyhow::bail!("ndarray feature not enabled");
        }
        "gpu" => {
            #[cfg(feature = "wgpu")]
            { run_bench::<burn::backend::Wgpu>(&models, &signal,
                burn::backend::wgpu::WgpuDevice::default(),
                &batch_sizes, args.warmup, args.iters)? }
            #[cfg(not(feature = "wgpu"))]
            anyhow::bail!("wgpu feature not enabled");
        }
        "gpu-f16" => {
            #[cfg(feature = "wgpu-f16")]
            { run_bench::<burn::backend::Wgpu<half::f16, i32>>(&models, &signal,
                burn::backend::wgpu::WgpuDevice::default(),
                &batch_sizes, args.warmup, args.iters)? }
            #[cfg(not(feature = "wgpu-f16"))]
            anyhow::bail!("wgpu-f16 feature not enabled");
        }
        other => anyhow::bail!("unknown device: {other} (expected cpu, gpu, gpu-f16)"),
    }

    Ok(())
}

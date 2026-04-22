/// EEG-DINO encoder inference example.
///
/// Usage:
///   cargo run --release --example infer -- --weights weights/eeg_dino_small.safetensors
///   cargo run --release --example infer -- --weights weights/eeg_dino_large.safetensors --size large
///   cargo run --release --example infer --no-default-features --features wgpu -- \
///       --weights weights/eeg_dino_small.safetensors --device gpu
use std::path::PathBuf;

use clap::Parser;
use eegdino_rs::{EegDinoEncoder, ModelSize, init_threads};

#[derive(Parser)]
#[command(name = "eegdino-infer", about = "EEG-DINO encoder inference")]
struct Args {
    /// Path to the safetensors weight file.
    #[arg(long)]
    weights: PathBuf,

    /// Model size: small, medium, large.  Auto-detected if omitted.
    #[arg(long)]
    size: Option<String>,

    /// Device: cpu, gpu, gpu-f16.
    #[arg(long, default_value = "cpu")]
    device: String,
}

fn parse_size(s: &str) -> anyhow::Result<ModelSize> {
    match s.to_lowercase().as_str() {
        "small" | "s" => Ok(ModelSize::Small),
        "medium" | "m" => Ok(ModelSize::Medium),
        "large" | "l" => Ok(ModelSize::Large),
        _ => anyhow::bail!("unknown model size: {s} (expected small, medium, large)"),
    }
}

fn run<B: burn::prelude::Backend>(args: &Args, device: B::Device) -> anyhow::Result<()> {
    let mut builder = EegDinoEncoder::<B>::builder()
        .weights(&args.weights)
        .device(device);

    if let Some(ref s) = args.size {
        builder = builder.size(parse_size(s)?);
    }

    let encoder = builder.build()?;

    let cfg = &encoder.config;
    println!(
        "Loaded {} model (d={}, heads={}, layers={}, ffn={})",
        match cfg.model_size {
            ModelSize::Small => "Small",
            ModelSize::Medium => "Medium",
            ModelSize::Large => "Large",
        },
        cfg.feature_size, cfg.num_heads, cfg.num_layers, cfg.dim_feedforward,
    );

    // Encode a dummy 10-second EEG recording (19 channels @ 200 Hz)
    let num_channels = 19;
    let num_samples = 2000;
    let signal = vec![0.0f32; num_channels * num_samples];

    let result = encoder.encode_raw(&signal, 1, num_channels, num_samples)?;

    println!("Output shape: {:?}", result.shape);
    println!("Encode time:  {:.1} ms", result.ms_encode);

    let n = result.embeddings.len().min(8);
    print!("First {n} values: [");
    for (i, v) in result.embeddings[..n].iter().enumerate() {
        if i > 0 { print!(", "); }
        print!("{v:.4}");
    }
    println!("]");

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    init_threads(None);

    match args.device.as_str() {
        "cpu" => {
            #[cfg(feature = "ndarray")]
            { run::<burn::backend::NdArray>(&args, burn::backend::ndarray::NdArrayDevice::Cpu) }
            #[cfg(not(feature = "ndarray"))]
            anyhow::bail!("ndarray feature not enabled")
        }
        "gpu" => {
            #[cfg(feature = "wgpu")]
            { run::<burn::backend::Wgpu>(&args, burn::backend::wgpu::WgpuDevice::default()) }
            #[cfg(not(feature = "wgpu"))]
            anyhow::bail!("wgpu feature not enabled")
        }
        "gpu-f16" => {
            #[cfg(feature = "wgpu-f16")]
            { run::<burn::backend::Wgpu<half::f16, i32>>(&args, burn::backend::wgpu::WgpuDevice::default()) }
            #[cfg(not(feature = "wgpu-f16"))]
            anyhow::bail!("wgpu-f16 feature not enabled")
        }
        other => anyhow::bail!("unknown device: {other} (expected cpu, gpu, gpu-f16)"),
    }
}

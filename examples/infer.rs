//! EEG-DINO encoder inference (RLX).
//!
//! ```text
//! cargo run --release --example infer -- \
//!     --weights weights/eeg_dino_small.safetensors --device metal
//! ```
use std::path::PathBuf;

use clap::Parser;
use eegdino_rs::prelude::*;

#[derive(Parser)]
#[command(name = "eegdino-infer", about = "EEG-DINO encoder inference (RLX)")]
struct Args {
    #[arg(long, default_value = "weights/eeg_dino_small.safetensors")]
    weights: PathBuf,

    /// Model size: small, medium, large (auto-detected from weights if omitted).
    #[arg(long)]
    size: Option<String>,

    /// Backend: cpu | metal | mps | mlx | gpu | wgpu | cuda | rocm | tpu
    #[arg(long, default_value = "cpu")]
    device: String,

    #[arg(long, default_value_t = 1)]
    batch: usize,

    #[arg(long, default_value_t = 19)]
    channels: usize,

    #[arg(long, default_value_t = 2000)]
    samples: usize,
}

fn parse_size(s: &str) -> anyhow::Result<ModelSize> {
    match s.to_lowercase().as_str() {
        "small" | "s" => Ok(ModelSize::Small),
        "medium" | "m" => Ok(ModelSize::Medium),
        "large" | "l" => Ok(ModelSize::Large),
        _ => anyhow::bail!("unknown model size: {s} (expected small, medium, large)"),
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    init_threads(None);

    let device = parse_device(&args.device)?;
    if !is_device_available(device) {
        anyhow::bail!(
            "device {:?} ({}) is not available — build with `--features {}`",
            args.device,
            device_label(device),
            feature_for(device),
        );
    }

    let mut builder = EegDinoEncoder::builder()
        .weights(&args.weights)
        .device(device);
    if let Some(ref s) = args.size {
        builder = builder.size(parse_size(s)?);
    }
    let mut encoder = builder.build()?;

    println!(
        "Loaded on {} (d={}, heads={}, layers={})",
        device_label(device),
        encoder.cfg.feature_size,
        encoder.cfg.num_heads,
        encoder.cfg.num_layers,
    );

    let len = args.batch * args.channels * args.samples;
    let signal = vec![0.0f32; len];
    let result = encoder.encode_raw(&signal, args.batch, args.channels, args.samples)?;

    println!("Output shape: {:?}", result.shape);
    println!("Encode time:  {:.1} ms", result.ms_encode);

    let n = result.embeddings.len().min(8);
    print!("First {n} values: [");
    for (i, v) in result.embeddings[..n].iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{v:.4}");
    }
    println!("]");

    Ok(())
}

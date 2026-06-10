//! RLX encoder benchmark across backends.
//!
//! ```text
//! cargo run --release --features all-backends --example bench -- \
//!     --device metal --batch 1,2,4,8
//! ```

mod common;

use std::env;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use clap::Parser;
use eegdino_rs::prelude::*;

use common::{bench_encode, deterministic_signal, NUM_CHANNELS, NUM_SAMPLES};

/// Auto `--isolate` when any batch in a multi-size sweep exceeds this (VRAM safety).
const AUTO_ISOLATE_ABOVE: usize = 128;

#[derive(Parser)]
#[command(name = "eegdino-bench")]
struct Args {
    #[arg(long, default_value = "cpu")]
    device: String,

    #[arg(long, default_value = "weights/eeg_dino_small.safetensors")]
    weights: String,

    #[arg(long, default_value = "1,2,4,8")]
    batch: String,

    #[arg(long, default_value_t = 20)]
    iters: usize,

    #[arg(long, default_value_t = 3)]
    warmup: usize,

    #[arg(long)]
    only: Option<String>,

    /// Run each batch size in a fresh process (frees GPU memory between shapes).
    #[arg(long)]
    isolate: bool,

    /// Emit one JSON object per batch line (for CI / rig trends).
    #[arg(long)]
    json: bool,
}

fn run_isolated(args: &Args, batch_sizes: &[usize]) -> anyhow::Result<()> {
    let exe = env::current_exe()?;
    for &bs in batch_sizes {
        let mut cmd = Command::new(&exe);
        cmd.arg("--device")
            .arg(&args.device)
            .arg("--weights")
            .arg(&args.weights)
            .arg("--batch")
            .arg(bs.to_string())
            .arg("--iters")
            .arg(args.iters.to_string())
            .arg("--warmup")
            .arg(args.warmup.to_string());
        if args.json {
            cmd.arg("--json");
        }
        if let Some(ref only) = args.only {
            cmd.arg("--only").arg(only);
        }
        let status = cmd.status()?;
        if !status.success() {
            anyhow::bail!("isolated bench failed for batch={bs}");
        }
    }
    Ok(())
}

fn should_auto_isolate(batch_sizes: &[usize]) -> bool {
    batch_sizes.len() > 1 && batch_sizes.iter().any(|&b| b > AUTO_ISOLATE_ABOVE)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    init_threads(None);

    let mut batch_sizes: Vec<usize> = args
        .batch
        .split(',')
        .map(|s| s.trim().parse())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    batch_sizes.sort_by(|a, b| b.cmp(a));

    let isolate = args.isolate || should_auto_isolate(&batch_sizes);
    if isolate && batch_sizes.len() > 1 {
        if !args.isolate && !args.json {
            eprintln!(
                "note: auto --isolate (batch > {AUTO_ISOLATE_ABOVE} in multi-batch sweep)"
            );
        }
        return run_isolated(&args, &batch_sizes);
    }

    let device = parse_device(&args.device)?;
    if !is_device_available(device) {
        anyhow::bail!(
            "device {} ({}) is not available — compile with `--features {}`",
            args.device,
            device_label(device),
            feature_for(device),
        );
    }

    let models = [
        (
            "Small",
            ModelSize::Small,
            "weights/eeg_dino_small.safetensors",
        ),
        (
            "Medium",
            ModelSize::Medium,
            "weights/eeg_dino_medium.safetensors",
        ),
        (
            "Large",
            ModelSize::Large,
            "weights/eeg_dino_large.safetensors",
        ),
    ];

    if !args.json {
        println!("EEG-DINO RLX Benchmark");
        println!("  Device:  {} ({})", args.device, device_label(device));
        println!("  Input:   {NUM_CHANNELS} ch × {NUM_SAMPLES} samples");
        println!("  Batches: {:?}", batch_sizes);
        println!("  Warmup:  {}, Timed: {}", args.warmup, args.iters);
        println!();
    }

    for (name, _size, path) in models {
        if let Some(ref only) = args.only {
            if name.to_lowercase() != only.to_lowercase() {
                continue;
            }
        }
        if !Path::new(path).exists() {
            if args.json {
                println!(
                    r#"{{"model":"{name}","skipped":true,"reason":"weights missing"}}"#
                );
            } else {
                println!("[{name}] SKIP — {path} not found\n");
            }
            continue;
        }

        let t0 = Instant::now();
        let mut enc = EegDinoEncoder::builder()
            .weights(path)
            .device(device)
            .max_cached_shapes(1)
            .build()?;
        let load_ms = t0.elapsed().as_secs_f64() * 1000.0;

        if !args.json {
            println!("--- {name} (d={}) ---", enc.cfg.feature_size);
            println!("  Load: {load_ms:.1} ms");
            println!();
            println!(
                "  {:>5}  {:>10}  {:>10}  {:>10}  {:>12}",
                "Batch", "Median", "Mean", "Min", "Throughput"
            );
        }

        let mut out_buf = Vec::new();

        for &bs in &batch_sizes {
            enc.clear_cache();
            let signal = deterministic_signal(bs);
            enc.encode_raw_into(&signal, bs, NUM_CHANNELS, NUM_SAMPLES, &mut out_buf)?;
            let (median, mean, min) = bench_encode(args.warmup, args.iters, || {
                enc.encode_raw_into(&signal, bs, NUM_CHANNELS, NUM_SAMPLES, &mut out_buf)?;
                Ok(())
            });
            let throughput = (bs as f64) * 1000.0 / mean;
            if args.json {
                let line = format!(
                    r#"{{"model":"{name}","device":"{}","batch":{bs},"median_ms":{median:.4},"mean_ms":{mean:.4},"min_ms":{min:.4},"throughput_samp_s":{throughput:.2},"channels":{NUM_CHANNELS},"samples":{NUM_SAMPLES}}}"#,
                    args.device
                );
                println!("{line}");
            } else {
                println!(
                    "  {:>5}  {:>9.2}ms  {:>9.2}ms  {:>9.2}ms  {:>8.1} samp/s",
                    bs, median, mean, min, throughput
                );
            }
            enc.clear_cache();
        }
        if !args.json {
            println!();
        }
    }

    if args.json {
        std::io::stdout().flush()?;
    }

    Ok(())
}

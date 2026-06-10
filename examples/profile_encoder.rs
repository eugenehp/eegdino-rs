//! Per-stage encoder timing (prefix subgraph compile → run) for bottleneck analysis.
//!
//! ```text
//! cargo run --release --features rlx,rlx-cpu,rlx-cuda --example profile_encoder -- \
//!     --device cuda --batch 16 --stages early
//! ```

mod common;

use std::path::Path;

use clap::{Parser, ValueEnum};
use eegdino_rs::config::ModelConfig;
use eegdino_rs::prelude::*;
use eegdino_rs::rlx::graph::{build_encoder_graph_with_taps, EncoderSpec};
use eegdino_rs::rlx::weights::{apply_params, load_safetensors, prepare_params};

use common::{bench_encode, deterministic_signal, NUM_CHANNELS, NUM_SAMPLES};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Stages {
    /// conv → patch_emb → spectral → pre_transformer + EegDinoEncoder full/transformer split.
    Early,
    /// All prefix checkpoints (11 separate compiles).
    All,
}

#[derive(Parser)]
#[command(name = "eegdino-profile")]
struct Args {
    #[arg(long, default_value = "cuda")]
    device: String,

    #[arg(long, default_value = "weights/eeg_dino_small.safetensors")]
    weights: String,

    #[arg(long, default_value_t = 16)]
    batch: usize,

    #[arg(long, default_value_t = 5)]
    warmup: usize,

    #[arg(long, default_value_t = 30)]
    iters: usize,

    #[arg(long, value_enum, default_value_t = Stages::All)]
    stages: Stages,
}

fn prepare_input(signal: &[f32], norm: f32) -> Vec<f32> {
    signal.iter().map(|v| v / norm).collect()
}

fn seq_at_layer_attention(cfg: &ModelConfig, spec: &EncoderSpec, layer: usize) -> usize {
    let h_tokens = spec.c * spec.p;
    if layer >= cfg.global_token_layer {
        h_tokens + cfg.num_global_tokens
    } else {
        h_tokens
    }
}

fn expected_elements(cfg: &ModelConfig, spec: &EncoderSpec, label: &str) -> usize {
    let b = spec.b;
    let c = spec.c;
    let p = spec.p;
    let d = cfg.feature_size;
    let h_tokens = c * p;
    let k = cfg.spectral_bins();
    let hd = cfg.num_heads * (d / cfg.num_heads);
    let ff = cfg.dim_feedforward;
    let seq = h_tokens + cfg.num_global_tokens;
    match label {
        "conv3_gn_gelu" => b * cfg.conv_channels[2] * h_tokens * 8,
        "patch_emb" => b * c * p * d,
        "spectral_mag" => b * c * p * k,
        "pre_transformer" => b * h_tokens * d,
        "layer_0_attention" => b * seq_at_layer_attention(cfg, spec, 0) * hd,
        "layer_5_attention" => b * seq_at_layer_attention(cfg, spec, 5) * hd,
        "layer_11_attention" => b * seq_at_layer_attention(cfg, spec, 11) * hd,
        "concat_global" | "full_encoder" => b * seq * d,
        "qkv_l0" => b * h_tokens * 3 * hd,
        "mlp_fc1_l0" => b * h_tokens * ff,
        _ => 0,
    }
}

fn validate_output(out: &[f32], label: &str, expected: usize) -> anyhow::Result<()> {
    if out.len() != expected {
        anyhow::bail!(
            "stage {label}: output len {} != expected {expected}",
            out.len()
        );
    }
    let energy: f64 = out.iter().map(|v| f64::from(*v).abs()).sum();
    if !energy.is_finite() || energy < 1e-12 {
        anyhow::bail!("stage {label}: output energy too small ({energy:e})");
    }
    Ok(())
}

fn time_stage(
    label: &str,
    cfg: &ModelConfig,
    spec: &EncoderSpec,
    params: &eegdino_rs::rlx::weights::ParamMap,
    device: rlx::Device,
    output: rlx::NodeId,
    x: &[f32],
    warmup: usize,
    iters: usize,
) -> anyhow::Result<f64> {
    let expected = expected_elements(cfg, spec, label);
    let mut g = build_encoder_graph_with_taps(cfg, spec).0;
    g.set_outputs(vec![output]);
    let sess = rlx::Session::new(device);
    let mut compiled = sess.compile(g);
    apply_params(&mut compiled, cfg, spec, params)?;

    let run_once = || -> anyhow::Result<()> {
        let outs = compiled.run(&[("x", x)]);
        let out = outs
            .first()
            .ok_or_else(|| anyhow::anyhow!("stage {label}: no outputs"))?;
        validate_output(out, label, expected)?;
        std::hint::black_box(out[0]);
        Ok(())
    };

    let (median, _, _) = bench_encode(warmup, iters, run_once);
    Ok(median)
}

fn configure_cuda_profiling(device: rlx::Device) {
    if device != rlx::Device::Cuda {
        return;
    }
    if std::env::var_os("RLX_CUDA_EXEC_MODE").is_none() {
        unsafe { std::env::set_var("RLX_CUDA_EXEC_MODE", "stream") };
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    init_threads(None);

    let device = parse_device(&args.device)?;
    configure_cuda_profiling(device);
    if !is_device_available(device) {
        anyhow::bail!("device {} not available", args.device);
    }
    let weights = Path::new(&args.weights);
    if !weights.exists() {
        anyhow::bail!("weights not found: {}", weights.display());
    }

    let cfg = ModelConfig::from_size(eegdino_rs::config::ModelSize::Small);
    let spec = EncoderSpec {
        b: args.batch,
        c: NUM_CHANNELS,
        p: NUM_SAMPLES / cfg.patch_size,
    };
    let params = prepare_params(&cfg, load_safetensors(weights.to_str().unwrap())?)?;
    let signal = deterministic_signal(args.batch);
    let x = prepare_input(&signal, 100.0);

    let (_g, taps) = build_encoder_graph_with_taps(&cfg, &spec);
    let checkpoints: Vec<(&str, rlx::NodeId)> = match args.stages {
        Stages::Early => taps.checkpoints_early().to_vec(),
        Stages::All => taps.checkpoints().to_vec(),
    };

    println!(
        "EEG-DINO stage profile — device={} ({}) batch={} warmup={} iters={} stages={:?}",
        args.device,
        device_label(device),
        args.batch,
        args.warmup,
        args.iters,
        args.stages,
    );
    if device == rlx::Device::Cuda {
        let mode = std::env::var("RLX_CUDA_EXEC_MODE").unwrap_or_else(|_| "stream".into());
        println!("  RLX_CUDA_EXEC_MODE={mode}");
    }
    println!("  (median ms per isolated prefix compile)\n");
    println!("  {:28} {:>10}", "Stage", "Median ms");
    println!("  {}", "-".repeat(42));

    let mut rows = Vec::new();
    for (label, id) in &checkpoints {
        let ms = time_stage(
            label,
            &cfg,
            &spec,
            &params,
            device,
            *id,
            &x,
            args.warmup,
            args.iters,
        )?;
        eprintln!("  {label:28} {ms:9.2} ms");
        rows.push((label.to_string(), ms));
    }

    let mut enc = EegDinoEncoder::builder()
        .weights(weights)
        .device(device)
        .max_cached_shapes(1)
        .build()?;
    let (enc_median, _, _) = bench_encode(args.warmup, args.iters, || {
        let r = enc.encode_raw(&signal, args.batch, NUM_CHANNELS, NUM_SAMPLES)?;
        validate_output(&r.embeddings, "encoder.encode_raw", expected_elements(&cfg, &spec, "full_encoder"))?;
        Ok(())
    });
    println!("\n  EegDinoEncoder full encode: {enc_median:.2} ms");

    if matches!(args.stages, Stages::Early) {
        let pre = rows
            .iter()
            .find(|(l, _)| l == "pre_transformer")
            .map(|(_, m)| *m)
            .unwrap_or(0.0);
        let transformer = (enc_median - pre).max(0.0);
        let pct = 100.0 * transformer / enc_median.max(1e-9);
        println!(
            "  Transformer (full − pre_transformer prefix): {transformer:.2} ms ({pct:.1}% of full)"
        );
        let patch = rows
            .iter()
            .find(|(l, _)| l == "patch_emb")
            .map(|(_, m)| *m)
            .unwrap_or(0.0);
        println!("\n  Groups (prefix compile / full encode):");
        println!(
            "    patch embedding (conv→patch_emb)  {:5.1}%",
            100.0 * patch / enc_median
        );
        println!(
            "    + spectral/pre_transformer        {:5.1}%",
            100.0 * (pre - patch).max(0.0) / enc_median
        );
        println!("    transformer (estimated)           {:5.1}%", pct);
        return Ok(());
    }

    if let Some((_, subgraph_ms)) = rows.iter().find(|(l, _)| l == "full_encoder") {
        let ratio = subgraph_ms / enc_median.max(1e-9);
        println!(
            "\n  Sanity: full_encoder subgraph {subgraph_ms:.2} ms vs EegDinoEncoder {enc_median:.2} ms (ratio {ratio:.2})"
        );
    }

    if let Some((_, full_ms)) = rows.iter().find(|(l, _)| l == "full_encoder") {
        if *full_ms <= 0.0 {
            anyhow::bail!("full_encoder median time is zero — check RLX_CUDA_EXEC_MODE=stream");
        }
        let mut prev = 0.0f64;
        println!("\n  Incremental cost (when prefix times increase monotonically):");
        for (label, ms) in &rows {
            if label == "full_encoder" {
                continue;
            }
            let delta = if *ms + 1e-9 >= prev {
                let d = *ms - prev;
                prev = *ms;
                Some(d)
            } else {
                prev = *ms;
                None
            };
            match delta {
                Some(d) => {
                    let pct = 100.0 * d / full_ms;
                    println!("    {label:26} +{d:7.2} ms  ({pct:4.1}% of full)");
                }
                None => {
                    println!("    {label:26}   (n/a — isolated compile faster than prior stage)");
                }
            }
        }
    }

    let patch = rows
        .iter()
        .find(|(l, _)| l == "patch_emb")
        .map(|(_, m)| *m)
        .unwrap_or(0.0);
    let pre = rows
        .iter()
        .find(|(l, _)| l == "pre_transformer")
        .map(|(_, m)| *m)
        .unwrap_or(0.0);
    let full = rows
        .iter()
        .find(|(l, _)| l == "full_encoder")
        .map(|(_, m)| *m)
        .unwrap_or(enc_median);
    println!("\n  Groups (% of full encode):");
    println!(
        "    patch embedding (conv→patch_emb)  {:5.1}%",
        100.0 * patch / full
    );
    println!(
        "    + spectral/pre_transformer        {:5.1}%",
        100.0 * (pre - patch).max(0.0) / full
    );
    println!(
        "    transformer (pre→full)            {:5.1}%",
        100.0 * (enc_median - pre).max(0.0) / full
    );

    Ok(())
}

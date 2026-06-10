//! Compare RLX encoder output: CPU reference vs a chosen device.
//!
//! ```text
//! cargo run --release --features all-backends --example debug_parity -- --device metal
//! ```
use std::f32::consts::PI;
use std::path::PathBuf;

use clap::Parser;
use eegdino_rs::prelude::*;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "weights/eeg_dino_small.safetensors")]
    weights: PathBuf,

    #[arg(long, default_value = "cpu")]
    device: String,

    /// Also run RLX-Metal and report max_abs vs CPU (requires `all-backends`).
    #[arg(long)]
    compare_metal: bool,
}

fn signal() -> Vec<f32> {
    (0..19 * 2000)
        .map(|i| ((i as f32 * 0.013) % (2.0 * PI)).sin() * 50.0)
        .collect()
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    init_threads(Some(4));
    let sig = signal();

    let (mut cpu_enc, _) = EegDinoEncoder::load(&args.weights, None, rlx::Device::Cpu)?;
    let cpu = cpu_enc.encode_raw(&sig, 1, 19, 2000)?;

    let device = parse_device(&args.device)?;
    if device == rlx::Device::Cpu {
        println!("device=cpu (reference only)");
        println!("shape={:?} first8={:?}", cpu.shape, &cpu.embeddings[..8]);
        return Ok(());
    }
    if !is_device_available(device) {
        anyhow::bail!(
            "device {} not available — enable `--features {}`",
            args.device,
            feature_for(device),
        );
    }

    let (mut dev_enc, _) = EegDinoEncoder::load(&args.weights, None, device)?;
    let out = dev_enc.encode_raw(&sig, 1, 19, 2000)?;

    println!(
        "device={} ({}) shape={:?}",
        args.device,
        device_label(device),
        out.shape
    );
    println!("cpu  first8: {:?}", &cpu.embeddings[..8]);
    println!("dev  first8: {:?}", &out.embeddings[..8]);
    let max_abs = cpu
        .embeddings
        .iter()
        .zip(out.embeddings.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let nrmse = {
        let mut sum_sq = 0.0f64;
        let mut sum_ref_sq = 0.0f64;
        for (&a, &b) in cpu.embeddings.iter().zip(out.embeddings.iter()) {
            let e = (a - b) as f64;
            sum_sq += e * e;
            sum_ref_sq += (a as f64) * (a as f64);
        }
        if sum_ref_sq > 0.0 {
            (sum_sq / sum_ref_sq).sqrt()
        } else {
            0.0
        }
    };
    let cosine_distance = {
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for (&a, &b) in cpu.embeddings.iter().zip(out.embeddings.iter()) {
            let ad = a as f64;
            let bd = b as f64;
            dot += ad * bd;
            norm_a += ad * ad;
            norm_b += bd * bd;
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom > 0.0 {
            1.0 - dot / denom
        } else {
            0.0
        }
    };
    println!("max_abs={max_abs:.6} nrmse={nrmse:.6} cos_dist={cosine_distance:.6}");
    let rel_l2 = if nrmse > 0.0 { nrmse } else { 0.0 };
    println!(
        "note: cos_dist≈0 only means direction match; max_abs/nrmse={rel_l2:.6} catch element-wise drift"
    );
    let mut worst_i = 0usize;
    let mut worst_e = 0.0f32;
    for (i, (a, b)) in cpu.embeddings.iter().zip(out.embeddings.iter()).enumerate() {
        let e = (a - b).abs();
        if e > worst_e {
            worst_e = e;
            worst_i = i;
        }
    }
    let a = cpu.embeddings[worst_i];
    let b = out.embeddings[worst_i];
    println!("worst_idx={worst_i} cpu={a:.6} gpu={b:.6} err={worst_e:.6}");

    if args.compare_metal && device != rlx::Device::Metal && is_device_available(rlx::Device::Metal)
    {
        let (mut metal_enc, _) = EegDinoEncoder::load(&args.weights, None, rlx::Device::Metal)?;
        let metal = metal_enc.encode_raw(&sig, 1, 19, 2000)?;
        let (mi, me) = out
            .embeddings
            .iter()
            .zip(metal.embeddings.iter())
            .enumerate()
            .map(|(i, (g, m))| (i, (g - m).abs()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        let metal_cos = {
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            for (&g, &m) in out.embeddings.iter().zip(metal.embeddings.iter()) {
                let gd = g as f64;
                let md = m as f64;
                dot += gd * md;
                na += gd * gd;
                nb += md * md;
            }
            let d = na.sqrt() * nb.sqrt();
            if d > 0.0 {
                1.0 - dot / d
            } else {
                0.0
            }
        };
        eprintln!(
            "gpu vs metal: max_abs={:.6} cos_dist={:.6} at idx={mi} gpu={:.6} metal={:.6}",
            me, metal_cos, out.embeddings[mi], metal.embeddings[mi]
        );
    }
    Ok(())
}

//! Numerical parity check: RLX encoder vs Python reference tensors.
//!
//! Requires `python scripts/parity_test.py` to have been run first.
//!
//! ```text
//! cargo run --release --example parity_check
//! ```
use std::path::Path;

use eegdino_rs::{init_threads, EegDinoEncoder, ModelSize};

fn load_f32(st: &safetensors::SafeTensors, key: &str) -> (Vec<f32>, Vec<usize>) {
    let view = st.tensor(key).unwrap();
    let shape = view.shape().to_vec();
    let data: Vec<f32> = view
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (data, shape)
}

fn main() -> anyhow::Result<()> {
    init_threads(None);

    let models = [
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

    for (name, size, weights_path) in &models {
        let parity_path = format!("tests/parity_data/parity_{name}.safetensors");
        if !Path::new(&parity_path).exists() {
            println!("[{name}] SKIP - {parity_path} not found");
            continue;
        }

        let ref_bytes = std::fs::read(&parity_path)?;
        let ref_st = safetensors::SafeTensors::deserialize(&ref_bytes)?;
        let (input_data, input_shape) = load_f32(&ref_st, "input");
        let (ref_output, output_shape) = load_f32(&ref_st, "output");

        let (mut encoder, _) =
            EegDinoEncoder::load(Path::new(weights_path), Some(size.into()), rlx::Device::Cpu)?;

        let num_samples = input_shape[2] * input_shape[3];
        let result =
            encoder.encode_raw(&input_data, input_shape[0], input_shape[1], num_samples)?;

        assert_eq!(result.shape, output_shape, "shape mismatch");

        let mut max_abs: f32 = 0.0;
        let mut sum_sq: f64 = 0.0;
        let mut sum_ref_sq: f64 = 0.0;
        for (&r, &p) in result.embeddings.iter().zip(ref_output.iter()) {
            let e = (r - p).abs();
            max_abs = max_abs.max(e);
            sum_sq += (e as f64) * (e as f64);
            sum_ref_sq += (p as f64) * (p as f64);
        }
        let nrmse = if sum_ref_sq > 0.0 {
            (sum_sq / sum_ref_sq).sqrt()
        } else {
            0.0
        };

        let pass = max_abs < 1e-3 && nrmse < 1e-5;
        let status = if pass {
            "PASS"
        } else {
            all_pass = false;
            "FAIL"
        };

        println!(
            "[{name:>6}] {status}  max_abs={max_abs:.2e}  nrmse={nrmse:.2e}  shape={:?}",
            result.shape
        );
    }

    if !all_pass {
        std::process::exit(1);
    }
    Ok(())
}

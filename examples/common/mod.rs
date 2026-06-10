//! Shared helpers for RLX benchmark / compare examples.

use std::f32::consts::PI;
use std::time::Instant;

pub const NUM_CHANNELS: usize = 19;
pub const NUM_SAMPLES: usize = 2000;

pub fn deterministic_signal(batch: usize) -> Vec<f32> {
    let n = batch * NUM_CHANNELS * NUM_SAMPLES;
    (0..n)
        .map(|i| ((i as f32 * 0.013) % (2.0 * PI)).sin() * 50.0)
        .collect()
}

#[derive(Clone, Debug)]
pub struct Metrics {
    pub max_abs: f32,
    pub nrmse: f64,
    pub mean_abs: f64,
    /// `1 - cos_sim` where cos_sim = dot(a,b) / (||a|| ||b||).
    pub cosine_distance: f64,
}

pub fn compare_embeddings(burn: &[f32], rlx: &[f32]) -> Metrics {
    assert_eq!(burn.len(), rlx.len());
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut sum_ref_sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&b, &r) in burn.iter().zip(rlx.iter()) {
        let e = (b - r).abs();
        max_abs = max_abs.max(e);
        sum_abs += e as f64;
        sum_sq += (e as f64) * (e as f64);
        sum_ref_sq += (b as f64) * (b as f64);
        let bd = b as f64;
        let rd = r as f64;
        dot += bd * rd;
        norm_a += bd * bd;
        norm_b += rd * rd;
    }
    let n = burn.len() as f64;
    let denom = norm_a.sqrt() * norm_b.sqrt();
    let cosine_distance = if denom > 0.0 {
        1.0 - (dot / denom)
    } else {
        0.0
    };
    Metrics {
        max_abs,
        mean_abs: sum_abs / n,
        nrmse: if sum_ref_sq > 0.0 {
            (sum_sq / sum_ref_sq).sqrt()
        } else {
            0.0
        },
        cosine_distance,
    }
}

pub fn bench_encode<F>(warmup: usize, iters: usize, mut f: F) -> (f64, f64, f64)
where
    F: FnMut() -> anyhow::Result<()>,
{
    for _ in 0..warmup {
        f().expect("warmup encode");
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f().expect("timed encode");
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let median = times[times.len() / 2];
    let min = times[0];
    (median, mean, min)
}

pub fn fmt_metrics(m: &Metrics) -> String {
    format!(
        "max={:.2e}  mean={:.2e}  nrmse={:.2e}  cos_dist={:.2e}",
        m.max_abs, m.mean_abs, m.nrmse, m.cosine_distance
    )
}

/// CPU parity gate vs Burn (f32 accumulation through depth).
pub fn cpu_parity_tol(model: &str) -> (f32, f32) {
    match model.to_lowercase().as_str() {
        "small" => (3e-6, 2e-6),
        "medium" => (6e-6, 4e-6),
        "large" => (1.2e-5, 2e-6),
        _ => (1.2e-5, 4e-6),
    }
}

pub fn passes_parity(m: &Metrics, max_tol: f32, nrmse_tol: f32) -> bool {
    m.max_abs <= max_tol && m.nrmse <= f64::from(nrmse_tol)
}

//! Bisect RLX CUDA vs CPU divergence on the EEG-DINO encoder graph.

use std::f32::consts::PI;
use std::path::Path;

use eegdino_rs::config::ModelConfig;
use eegdino_rs::init_threads;
use eegdino_rs::is_device_available;
use eegdino_rs::rlx::graph::{build_encoder_graph, EncoderSpec};
use eegdino_rs::rlx::weights::{apply_params, load_safetensors, prepare_params};

fn signal() -> Vec<f32> {
    (0..19 * 2000)
        .map(|i| ((i as f32 * 0.013) % (2.0 * PI)).sin() * 50.0)
        .collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn prepare_input(signal: &[f32], norm: f32) -> Vec<f32> {
    signal.iter().map(|v| v / norm).collect()
}

fn run_stage(
    weights: &Path,
    output_node: rlx::NodeId,
    label: &str,
    device: rlx::Device,
) -> anyhow::Result<f32> {
    if !is_device_available(device) {
        eprintln!("SKIP cuda bisect — {device:?} not available");
        return Ok(0.0);
    }

    let cfg = ModelConfig::from_size(eegdino_rs::config::ModelSize::Small);
    let spec = EncoderSpec { b: 1, c: 19, p: 10 };
    let mut g = build_encoder_graph(&cfg, &spec);
    g.set_outputs(vec![output_node]);

    let params = prepare_params(&cfg, load_safetensors(weights.to_str().unwrap())?)?;
    let x = prepare_input(&signal(), 100.0);

    let cpu_sess = rlx::Session::new(rlx::Device::Cpu);
    let mut cpu_c = cpu_sess.compile(g.clone());
    apply_params(&mut cpu_c, &cfg, &spec, &params)?;
    let cpu_out = cpu_c.run(&[("x", &x)]).into_iter().next().unwrap();

    let dev_sess = rlx::Session::new(device);
    let mut dev_c = dev_sess.compile(g);
    apply_params(&mut dev_c, &cfg, &spec, &params)?;
    let dev_out = dev_c.run(&[("x", &x)]).into_iter().next().unwrap();

    let err = max_abs(&cpu_out, &dev_out);
    eprintln!(
        "[bisect] {label}: max_abs={err:.6e} len={} first8_cpu={:?} first8_dev={:?}",
        cpu_out.len(),
        &cpu_out[..8.min(cpu_out.len())],
        &dev_out[..8.min(dev_out.len())],
    );
    Ok(err)
}

fn patch_embed_checkpoints(g: &rlx::Graph) -> Vec<(String, rlx::NodeId)> {
    let mut out = Vec::new();
    let mut conv_stack = 0usize;
    for node in g.nodes() {
        let dims: Vec<usize> = node
            .shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        if dims == [1, 25, 190, 8] {
            conv_stack += 1;
            out.push((format!("conv_stack_{conv_stack}_{:?}", node.op), node.id));
        }
        if dims == [1, 19, 10, 200] && matches!(&node.op, rlx::Op::Reshape { .. }) {
            out.push(("patch_emb".into(), node.id));
        }
    }
    out
}

fn transformer_checkpoints(g: &rlx::Graph) -> Vec<(String, rlx::NodeId)> {
    let mut out = Vec::new();
    let mut attn_i = 0usize;
    let mut ln_i = 0usize;
    let mut ln190_i = 0usize;
    let mut mm_191 = 0usize;
    for node in g.nodes() {
        let dims: Vec<usize> = node
            .shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        if matches!(&node.op, rlx::Op::Attention { .. }) {
            out.push((format!("layer_{attn_i}_attention"), node.id));
            attn_i += 1;
        }
        if dims == [1, 190, 200] && matches!(&node.op, rlx::Op::Reshape { .. }) {
            out.push(("pre_transformer".into(), node.id));
        }
        if dims == [1, 190, 200] && matches!(&node.op, rlx::Op::LayerNorm { .. }) {
            out.push((format!("layernorm_s190_{ln190_i}"), node.id));
            ln190_i += 1;
        }
        if dims == [1, 191, 200] {
            match &node.op {
                rlx::Op::Concat { .. } => {
                    out.push(("concat_global".into(), node.id));
                    if node.inputs.len() >= 2 {
                        out.push(("pre_concat_x".into(), node.inputs[1]));
                    }
                }
                rlx::Op::LayerNorm { .. } => {
                    out.push((format!("layernorm_{ln_i}"), node.id));
                    ln_i += 1;
                }
                rlx::Op::MatMul => {
                    out.push((format!("matmul_s191_{mm_191}"), node.id));
                    mm_191 += 1;
                }
                _ => {}
            }
        }
        if dims == [1, 191, 600] {
            out.push(("qkv_mm_out".into(), node.id));
        }
    }
    out
}

fn run_checkpoints(weights: &Path, checkpoints: &[(String, rlx::NodeId)]) {
    for (label, id) in checkpoints {
        let err = run_stage(weights, *id, label, rlx::Device::Cuda).expect("stage run");
        eprintln!("  => {label}: {err:.6e}");
    }
}

#[test]
#[cfg(feature = "rlx-cuda")]
#[ignore = "run manually on CUDA rig: needs weights/eeg_dino_small.safetensors"]
fn bisect_patch_embed_cuda() {
    init_threads(Some(4));
    let weights = Path::new("weights/eeg_dino_small.safetensors");
    if !weights.exists() {
        eprintln!("SKIP — weights not found");
        return;
    }
    let cfg = ModelConfig::from_size(eegdino_rs::config::ModelSize::Small);
    let spec = EncoderSpec { b: 1, c: 19, p: 10 };
    let g = build_encoder_graph(&cfg, &spec);
    run_checkpoints(weights, &patch_embed_checkpoints(&g));
}

#[test]
#[cfg(feature = "rlx-cuda")]
#[ignore = "run manually on CUDA rig"]
fn bisect_transformer_cuda() {
    init_threads(Some(4));
    let weights = Path::new("weights/eeg_dino_small.safetensors");
    if !weights.exists() {
        return;
    }
    let cfg = ModelConfig::from_size(eegdino_rs::config::ModelSize::Small);
    let spec = EncoderSpec { b: 1, c: 19, p: 10 };
    let g = build_encoder_graph(&cfg, &spec);
    run_checkpoints(weights, &transformer_checkpoints(&g));
}

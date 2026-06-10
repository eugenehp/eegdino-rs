//! Profile tap nodes: shapes and full_encoder prefix matches production encode.

use eegdino_rs::config::{ModelConfig, ModelSize};
use eegdino_rs::prelude::*;
use eegdino_rs::rlx::graph::{build_encoder_graph_with_taps, EncoderSpec};
use eegdino_rs::rlx::weights::{apply_params, load_safetensors, prepare_params};

const CH: usize = 19;
const SAMPLES: usize = 2000;

fn weights_path() -> Option<String> {
    let p = "weights/eeg_dino_small.safetensors";
    if std::path::Path::new(p).exists() {
        Some(p.to_string())
    } else {
        None
    }
}

#[test]
fn profile_tap_shapes_match_graph() {
    let cfg = ModelConfig::from_size(ModelSize::Small);
    let spec = EncoderSpec {
        b: 2,
        c: CH,
        p: SAMPLES / cfg.patch_size,
    };
    let (g, taps) = build_encoder_graph_with_taps(&cfg, &spec);

    let h_tokens = CH * spec.p;
    let d = cfg.feature_size;
    let k = cfg.spectral_bins();
    let hd = cfg.num_heads * (d / cfg.num_heads);
    let seq = h_tokens + cfg.num_global_tokens;

    let expect = |id, elems: usize| {
        let n = g.shape(id).num_elements().unwrap();
        assert_eq!(n, elems, "node {:?}", id);
    };

    expect(taps.conv3_gn_gelu, spec.b * cfg.conv_channels[2] * h_tokens * 8);
    expect(taps.patch_emb, spec.b * CH * spec.p * d);
    expect(taps.spectral_mag, spec.b * CH * spec.p * k);
    expect(taps.pre_transformer, spec.b * h_tokens * d);
    expect(taps.layer_0_attention, spec.b * h_tokens * hd);
    expect(taps.layer_5_attention, spec.b * seq * hd);
    expect(taps.full_encoder, spec.b * seq * d);
}

#[test]
fn full_encoder_prefix_matches_encode_raw() {
    let Some(path) = weights_path() else {
        eprintln!("skip: weights not found");
        return;
    };

    let cfg = ModelConfig::from_size(ModelSize::Small);
    let b = 1usize;
    let spec = EncoderSpec {
        b,
        c: CH,
        p: SAMPLES / cfg.patch_size,
    };

    let signal: Vec<f32> = (0..b * CH * SAMPLES)
        .map(|i| ((i as f32 * 0.013) % std::f32::consts::TAU).sin() * 50.0)
        .collect();
    let norm = 100.0f32;
    let x: Vec<f32> = signal.iter().map(|v| v / norm).collect();

    let params = prepare_params(&cfg, load_safetensors(&path).expect("load")).expect("prepare");
    let (mut g, taps) = build_encoder_graph_with_taps(&cfg, &spec);
    g.set_outputs(vec![taps.full_encoder]);
    let sess = rlx::Session::new(rlx::Device::Cpu);
    let mut compiled = sess.compile(g);
    apply_params(&mut compiled, &cfg, &spec, &params).expect("params");
    let prefix = compiled.run(&[("x", &x)]).into_iter().next().expect("out");

    let mut enc = EegDinoEncoder::builder()
        .weights(&path)
        .device(rlx::Device::Cpu)
        .build()
        .expect("encoder");
    let got = enc
        .encode_raw(&signal, b, CH, SAMPLES)
        .expect("encode")
        .embeddings;

    assert_eq!(prefix.len(), got.len());
    let max_abs = prefix
        .iter()
        .zip(got.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e-5,
        "prefix full_encoder vs encode_raw max_abs={max_abs}"
    );
}

//! wgpu mid-axis broadcast add parity (EEG channel embedding pattern).

use rlx::ir::op::BinaryOp;

#[test]
#[cfg(feature = "rlx-gpu")]
fn wgpu_channel_broadcast_add_matches_cpu() {
    if !eegdino_rs::is_device_available(rlx::Device::Gpu) {
        return;
    }
    let (b, c, p, d) = (1, 19, 10, 200);
    let mut g = rlx::Graph::new("chan_broadcast");
    let emb = g.input("emb", rlx::Shape::new(&[b, c, p, d], rlx::DType::F32));
    let ch = g.input("ch", rlx::Shape::new(&[b, c, 1, d], rlx::DType::F32));
    let out = g.add_node(
        rlx::Op::Binary(BinaryOp::Add),
        vec![emb, ch],
        rlx::Shape::new(&[b, c, p, d], rlx::DType::F32),
    );
    g.set_outputs(vec![out]);

    let n_emb = b * c * p * d;
    let n_ch = b * c * d;
    let emb_v: Vec<f32> = (0..n_emb).map(|i| ((i as f32 * 0.013).sin())).collect();
    let ch_v: Vec<f32> = (0..n_ch).map(|i| ((i as f32 * 0.07).cos() * 0.1)).collect();

    let mut cpu = rlx::Session::new(rlx::Device::Cpu);
    let mut cc = cpu.compile(g.clone());
    let want = cc
        .run(&[("emb", &emb_v), ("ch", &ch_v)])
        .into_iter()
        .next()
        .unwrap();

    let mut gpu = rlx::Session::new(rlx::Device::Gpu);
    let mut gc = gpu.compile(g);
    let got = gc
        .run(&[("emb", &emb_v), ("ch", &ch_v)])
        .into_iter()
        .next()
        .unwrap();

    let err = want
        .iter()
        .zip(got.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        err < 1e-5,
        "channel broadcast add max_abs={err:.3e} (cpu[100]={} gpu[100]={})",
        want[100],
        got[100]
    );
}

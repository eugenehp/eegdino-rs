//! CUDA GELU on EEG conv tensor shape vs CPU.

#[test]
#[cfg(feature = "rlx-cuda")]
fn cuda_gelu_eeg_tensor_matches_cpu() {
    if !eegdino_rs::is_device_available(rlx::Device::Cuda) {
        return;
    }
    use rlx::prelude::*;
    let n = 1 * 25 * 190 * 8;
    let x: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.017).sin() * 2.0)).collect();
    let mut g = Graph::new("gelu");
    let xi = g.input("x", Shape::new(&[1, 25, 190, 8], DType::F32));
    let y = g.activation(
        Activation::Gelu,
        xi,
        Shape::new(&[1, 25, 190, 8], DType::F32),
    );
    g.set_outputs(vec![y]);
    let want = Session::new(Device::Cpu)
        .compile(g.clone())
        .run(&[("x", &x)])
        .into_iter()
        .next()
        .unwrap();
    let got = Session::new(Device::Cuda)
        .compile(g)
        .run(&[("x", &x)])
        .into_iter()
        .next()
        .unwrap();
    let err = want
        .iter()
        .zip(got.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(err < 1e-5, "GELU max_abs={err:.3e}");
}

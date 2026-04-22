pub mod embedding;
pub mod attention;
pub mod mlp;
pub mod transformer;
pub mod encoder;
pub mod classifier;

use burn::prelude::*;
use burn::module::{Param, ParamId};
use burn::nn::Linear;

/// Create a [`Linear`] layer with zero-filled weights.
///
/// Weights will be immediately overwritten from safetensors.
/// `Tensor::zeros` is essentially free compared to random init.
pub fn linear_zeros<B: Backend>(
    d_input: usize,
    d_output: usize,
    bias: bool,
    device: &B::Device,
) -> Linear<B> {
    let weight = Param::initialized(
        ParamId::new(),
        Tensor::zeros([d_input, d_output], device),
    );
    let bias = if bias {
        Some(Param::initialized(
            ParamId::new(),
            Tensor::zeros([d_output], device),
        ))
    } else {
        None
    };
    Linear { weight, bias }
}

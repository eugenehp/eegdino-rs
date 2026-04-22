//! # eegdino-rs
//!
//! Rust inference crate for the
//! [EEG-DINO](https://github.com/miraclefish/EEG-DINO) foundation model,
//! built on the [Burn](https://burn.dev) ML framework.
//!
//! EEG-DINO learns robust EEG representations via hierarchical self-distillation
//! on 9 000+ hours of EEG data.  This crate provides a faithful port of the
//! encoder architecture with verified numerical parity (NRMSE < 1e-6) against
//! the original PyTorch implementation.
//!
//! ## Model sizes
//!
//! | Variant | Params | d_model | Heads | Layers | FFN dim |
//! |---------|--------|---------|-------|--------|---------|
//! | Small   | 4.6 M  | 200     | 8     | 12     | 512     |
//! | Medium  | 33 M   | 512     | 16    | 16     | 1 024   |
//! | Large   | 201 M  | 1 024   | 16    | 24     | 2 048   |
//!
//! ## Quick start (builder)
//!
//! ```rust,ignore
//! use eegdino_rs::prelude::*;
//! use burn::backend::NdArray;
//!
//! type B = NdArray;
//!
//! let encoder = EegDinoEncoder::<B>::builder()
//!     .weights("weights/eeg_dino_small.safetensors")
//!     .size(ModelSize::Small)
//!     .device(Default::default())
//!     .build()?;
//!
//! let signal = vec![0.0f32; 19 * 2000];
//! let result = encoder.encode_raw(&signal, 1, 19, 2000)?;
//! // result.shape == [1, 191, 200]
//! ```
//!
//! ## Batch encoding
//!
//! ```rust,ignore
//! let signals: Vec<Vec<f32>> = load_recordings();
//! // Single batched forward pass (fastest):
//! let result = encoder.encode_batch(&signals, 19, 2000)?;
//! // Or one-by-one:
//! let results = encoder.encode_many(&signals, 19, 2000);
//! ```
//!
//! ## Backends
//!
//! | Feature | Backend | Notes |
//! |---------|---------|-------|
//! | `ndarray` (default) | CPU | Multi-threaded via Rayon + SIMD |
//! | `blas-accelerate` | CPU + Accelerate | Recommended on Apple Silicon |
//! | `wgpu` | GPU | Metal (macOS) / Vulkan (Linux) |
//! | `wgpu-f16` | GPU f16 | Half-precision, 2x less memory |

pub mod config;
pub mod error;
pub mod model;
pub(crate) mod weights;
pub mod inference;
pub mod prelude;

pub use config::{ModelConfig, ModelSize};
pub use error::{EegDinoError, Result};
pub use inference::{
    EegDinoEncoder, EegDinoEncoderBuilder, EncodingResult,
    EegDinoClassifier, ClassificationResult,
    detect_model_size,
};
pub use model::encoder::EEGEncoder;
pub use model::classifier::ClassificationModel;
pub use model::embedding::{EmbeddingCache, PatchEmbedding};

/// Configure the Rayon thread pool.  Call once before model use.
pub fn init_threads(n: Option<usize>) {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if let Some(n) = n {
        builder = builder.num_threads(n);
    }
    builder.build_global().ok();
}

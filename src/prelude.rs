//! Convenience re-exports for common usage patterns.
//!
//! ```rust,ignore
//! use eegdino_rs::prelude::*;
//! ```

pub use crate::config::{ModelConfig, ModelSize};
pub use crate::error::{EegDinoError, Result};
pub use crate::inference::{
    ClassificationResult, EegDinoClassifier, EegDinoEncoder, EegDinoEncoderBuilder,
    EncodingResult,
};
pub use crate::model::classifier::ClassificationModel;
pub use crate::model::embedding::{EmbeddingCache, PatchEmbedding};
pub use crate::model::encoder::EEGEncoder;
pub use crate::init_threads;

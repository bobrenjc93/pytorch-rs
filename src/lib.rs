//! A native Rust tensor and deep-learning library pursuing `PyTorch` semantics
//! and performance.
//!
//! The crate provides a small, well-tested typed CPU tensor core. Burner grows
//! the supported surface monotonically while correctness, benchmark integrity,
//! and existing performance remain merge gates.

mod python;
mod tensor;

pub use tensor::{DType, Device, Scalar, Tensor, TensorData, TensorDataRef, TensorError};

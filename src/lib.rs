//! A native Rust tensor and deep-learning library pursuing `PyTorch` semantics
//! and performance.
//!
//! The initial crate intentionally starts with a small, well-tested CPU tensor
//! core. Burner grows the supported surface monotonically while correctness,
//! benchmark integrity, and existing performance remain merge gates.

mod python;
mod tensor;

pub use tensor::{Tensor, TensorError};

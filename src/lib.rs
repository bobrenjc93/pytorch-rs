//! A native Rust tensor and deep-learning library pursuing `PyTorch` semantics
//! and performance.
//!
//! The initial crate intentionally starts with a small, well-tested CPU tensor
//! core. Burner grows the supported surface monotonically while correctness,
//! benchmark integrity, and existing performance remain merge gates.

mod dtype;
#[cfg(feature = "python-bindings")]
mod python;
#[cfg(feature = "python-bindings")]
mod python_layout;
mod tensor;

#[cfg(feature = "python-bindings")]
pub(crate) use tensor::{enter_no_grad, exit_no_grad};

pub use dtype::DType;
pub use tensor::{
    Device, LogicalValues, MemoryFormat, NoGradGuard, Tensor, TensorError, is_grad_enabled, no_grad,
};

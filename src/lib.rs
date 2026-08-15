//! A native Rust tensor and deep-learning library pursuing `PyTorch` semantics
//! and performance.
//!
//! The initial crate intentionally starts with a small, well-tested CPU tensor
//! core. Burner grows the supported surface monotonically while correctness,
//! benchmark integrity, and existing performance remain merge gates.

mod device;
mod dtype;
mod grad_mode;
mod memory_format;
#[cfg(feature = "python-bindings")]
mod python;
#[cfg(feature = "python-bindings")]
mod python_layout;
mod tensor;

#[cfg(feature = "python-bindings")]
pub(crate) use grad_mode::{enter_no_grad, exit_no_grad};

pub use device::Device;
pub use dtype::DType;
pub use grad_mode::{NoGradGuard, is_grad_enabled, no_grad};
pub use memory_format::MemoryFormat;
pub use tensor::{LogicalValues, Tensor, TensorError};

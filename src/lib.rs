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
// Python API docstrings intentionally contain Python examples and are tested
// through the Python suite. They are private Rust implementation modules, so
// omit them while rustdoc is collecting Rust doctests.
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_device;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_dtype;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_grad_mode;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_layout;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_memory_format;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_no_argument_builtins;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_variable_functions;
mod storage;
mod tensor;
mod tensor_error;

#[cfg(feature = "python-bindings")]
pub(crate) use grad_mode::{enter_no_grad, exit_no_grad};

pub use device::Device;
pub use dtype::DType;
pub use grad_mode::{NoGradGuard, is_grad_enabled, no_grad};
pub use memory_format::MemoryFormat;
pub use tensor::{LogicalValues, Tensor};
pub use tensor_error::TensorError;

//! A native Rust tensor and deep-learning library pursuing `PyTorch` semantics
//! and performance.
//!
//! The initial crate intentionally starts with a small, well-tested CPU tensor
//! core. Burner grows the supported surface monotonically while correctness,
//! benchmark integrity, and existing performance remain merge gates.

mod autograd_node;
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
mod python_argument_schema;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_cpython_compat;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_device;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_dtype;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_finfo;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_grad_mode;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_layout;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_memory_format;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_nn_functional;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_no_argument_builtins;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_scalar_conversions;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_size;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_tensor_alternate_layout;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_tensor_data;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_tensor_devices;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_tensor_dtype;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_tensor_errors;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_tensor_lazy_bits;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_tensor_leaf_grad;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_tensor_name;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_tensor_queries;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_tensor_shape;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_tensor_storage;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_torch_function_mode;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_torch_function_probe;
#[cfg(all(feature = "python-bindings", not(doc)))]
mod python_variable_functions;
mod storage;
mod tensor;
mod tensor_error;

#[cfg(feature = "python-bindings")]
pub(crate) use grad_mode::{enter_enable_grad, enter_no_grad, exit_grad_mode};

pub use device::Device;
pub use dtype::{DType, FloatingPointInfo};
pub use grad_mode::{EnableGradGuard, NoGradGuard, enable_grad, is_grad_enabled, no_grad};
pub use memory_format::MemoryFormat;
pub use tensor::{LogicalValues, Tensor};
pub use tensor_error::TensorError;

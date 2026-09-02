use std::ffi::{CStr, c_char};
use std::os::raw::c_long;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{
    PyIndexError, PyMemoryError, PyNotImplementedError, PyOverflowError, PyRuntimeError,
    PyTypeError, PyUserWarning, PyValueError,
};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyAny, PyBool, PyBytes, PyCFunction, PyComplex, PyDict, PyEllipsis, PyFloat, PyInt, PyList,
    PyMapping, PyMemoryView, PyModule, PySequence, PySlice, PyString, PyTuple, PyType,
};

use crate::{
    DType, Device, MemoryFormat, Tensor as CoreTensor, TensorError,
    grad_mode::is_grad_enabled,
    python_cpython_compat as cpython_compat,
    python_device::{PyDevice, device_argument_type_error, parse_device_value},
    python_dtype::{PyDType, add_default_dtype_validator, dtype_object},
    python_finfo::finfo_type_object,
    python_grad_mode::add_grad_mode_contexts,
    python_layout::{LayoutObjects as PyLayoutObjects, create_layout_objects},
    python_memory_format::{PyMemoryFormat, memory_format_object},
    python_nn_functional::add_nn_functional_bridges,
    python_no_argument_builtins::add_no_argument_builtins,
    python_scalar_conversions::register_scalar_conversions,
    python_size::size_type_object,
    python_tensor_errors::{item_error, permute_error, tensor_error, transpose_error},
    python_tensor_queries::add_tensor_queries,
    python_torch_function_mode as torch_function_mode_stack,
    python_torch_function_probe::{add_torch_function_probe, is_disabled_torch_function_handler},
    python_variable_functions::{add_variable_functions, variable_function},
};

static LAYOUT_OBJECTS: PyOnceLock<PyLayoutObjects> = PyOnceLock::new();
static T_NON_MATRIX_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);
static T_SCALAR_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);
static H_SCALAR_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);
static MT_SCALAR_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);
static MH_SCALAR_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);
static ADJOINT_SCALAR_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);
static TORCH_FUNCTION_PLAIN_METHOD_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);
static ASARRAY_REQUIRES_GRAD_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);
static WARN_ALWAYS_ENABLED: AtomicBool = AtomicBool::new(false);
static CUDNN_ENABLED: AtomicBool = AtomicBool::new(true);
static CUDNN_BENCHMARK: AtomicBool = AtomicBool::new(false);
static CUDNN_BENCHMARK_LIMIT: AtomicI32 = AtomicI32::new(10);
static CUDNN_DETERMINISTIC: AtomicBool = AtomicBool::new(false);
static CUDNN_ALLOW_TF32: AtomicBool = AtomicBool::new(true);
static CUBLAS_ALLOW_TF32: AtomicBool = AtomicBool::new(false);
static CUBLAS_ALLOW_FP16_REDUCED_PRECISION_REDUCTION: AtomicBool = AtomicBool::new(true);
static CUBLAS_ALLOW_FP16_REDUCED_PRECISION_REDUCTION_SPLIT_K: AtomicBool = AtomicBool::new(true);
static CUBLAS_ALLOW_BF16_REDUCED_PRECISION_REDUCTION: AtomicBool = AtomicBool::new(true);
static CUBLAS_ALLOW_BF16_REDUCED_PRECISION_REDUCTION_SPLIT_K: AtomicBool = AtomicBool::new(true);
static FLASH_SDP_ENABLED: AtomicBool = AtomicBool::new(true);
static MEM_EFFICIENT_SDP_ENABLED: AtomicBool = AtomicBool::new(true);
static MATH_SDP_ENABLED: AtomicBool = AtomicBool::new(true);
const AS_TENSOR_MAX_SEQUENCE_DIMENSIONS: usize = 128;
static CUDNN_SDP_ENABLED: AtomicBool = AtomicBool::new(true);
static MATH_SDP_ALLOW_FP16_BF16_REDUCTION: AtomicBool = AtomicBool::new(false);
static NNPACK_ENABLED: AtomicBool = AtomicBool::new(true);
static GUARD_COLLECTIVES_ENABLED: AtomicBool = AtomicBool::new(false);
const BROADCAST_TENSORS_EXACT_TENSORS_ERROR: &str =
    "broadcast_tensors() only supports exact native Tensor inputs";
const BROADCAST_TENSORS_EXPANSION_ERROR: &str =
    "torch_rs.broadcast_tensors does not support shape expansion";

// These are compile-time facts about the native Cargo build. Keep them native
// so importing the Python package never probes the host or imports another
// tensor runtime to infer capabilities.
const NATIVE_BUILD_CAPABILITIES: [(&str, bool); 9] = [
    ("_GLIBCXX_USE_CXX11_ABI", false),
    ("_has_cudnn", false),
    ("_has_cuda", false),
    ("_has_cusparselt", false),
    ("_has_kleidiai", false),
    ("has_openmp", false),
    ("has_mkl", false),
    ("has_lapack", false),
    ("has_spectral", false),
];
const PRIVATE_NATIVE_EXPORTS: [&str; 8] = [
    "_GLIBCXX_USE_CXX11_ABI",
    "_get_cpu_capability",
    "_has_cudnn",
    "_has_cuda",
    "_has_cusparselt",
    "_has_kleidiai",
    "_is_ck_sdpa_available",
    "_is_flash_attention_available",
];
const NATIVE_CPU_CAPABILITY: &str = "DEFAULT";
const NATIVE_CK_SDPA_AVAILABLE: bool = false;
const NATIVE_FLASH_ATTENTION_AVAILABLE: bool = false;

#[pyfunction(name = "_get_cpu_capability", signature = (), text_signature = None)]
fn get_cpu_capability_native() -> &'static str {
    NATIVE_CPU_CAPABILITY
}

#[pyfunction(name = "_is_ck_sdpa_available", signature = (), text_signature = None)]
fn is_ck_sdpa_available_native() -> bool {
    NATIVE_CK_SDPA_AVAILABLE
}

#[pyfunction(
    name = "_is_flash_attention_available",
    signature = (),
    text_signature = None
)]
fn is_flash_attention_available_native() -> bool {
    NATIVE_FLASH_ATTENTION_AVAILABLE
}

const IS_TENSOR_SOURCE: &CStr = cr#"
import copy as _copy
import sys as _sys
from typing import Any as _Any

try:
    from typing_extensions import TypeIs as _TypeIs
except ModuleNotFoundError as _type_is_error:
    if _type_is_error.name != "typing_extensions":
        raise
    _TypeIs = None


torch = _sys.modules.get("torch_rs")


def is_tensor(obj, /):
    r"""Returns True if `obj` is a PyTorch tensor.

    Args:
        obj (object): Object to test
    Example::

        >>> x = torch.tensor([1, 2, 3])
        >>> torch.is_tensor(x)
        True

    """
    return isinstance(obj, torch.Tensor)


if _TypeIs is not None:
    # Do not share a mutable ForwardRef cache with another torch implementation.
    is_tensor.__annotations__ = {
        "obj": _Any,
        "return": _copy.deepcopy(_TypeIs["torch.Tensor"]),
    }
is_tensor.__module__ = "torch_rs"
"#;

#[cfg(target_os = "macos")]
const T_NON_MATRIX_WARNING: &CStr = c"The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4317.)";
#[cfg(target_os = "macos")]
const T_SCALAR_WARNING: &CStr = c"Tensor.T is deprecated on 0-D tensors. This function is the identity in these cases. (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4322.)";
#[cfg(target_os = "macos")]
const H_SCALAR_WARNING: &CStr = c"Tensor.H is deprecated on 0-D tensors. Consider using x.conj(). (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4336.)";
#[cfg(target_os = "macos")]
const MT_SCALAR_WARNING: &CStr = c"Tensor.mT is deprecated on 0-D tensors. This function is the identity in these cases. (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4374.)";
#[cfg(target_os = "macos")]
const MH_SCALAR_WARNING: &CStr = c"Tensor.mH is deprecated on 0-D tensors. Consider using x.conj(). (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4383.)";
#[cfg(target_os = "macos")]
const ADJOINT_SCALAR_WARNING: &CStr = c"adjoint() is deprecated on 0-D tensors. Consider using x.conj(). (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4391.)";
#[cfg(target_os = "macos")]
const TORCH_FUNCTION_PLAIN_METHOD_WARNING: &CStr = c"Defining your `__torch_function__` as a plain method is deprecated and will be an error in future, please define it as a classmethod. (Triggered internally at /Users/runner/work/pytorch/pytorch/torch/csrc/utils/python_arg_parser.cpp:359.)";

#[cfg(target_os = "linux")]
const T_NON_MATRIX_WARNING: &CStr = c"The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at /__w/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4314.)";
#[cfg(target_os = "linux")]
const T_SCALAR_WARNING: &CStr = c"Tensor.T is deprecated on 0-D tensors. This function is the identity in these cases. (Triggered internally at /__w/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4321.)";
#[cfg(target_os = "linux")]
const H_SCALAR_WARNING: &CStr = c"Tensor.H is deprecated on 0-D tensors. Consider using x.conj(). (Triggered internally at /__w/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4335.)";
#[cfg(target_os = "linux")]
const MT_SCALAR_WARNING: &CStr = c"Tensor.mT is deprecated on 0-D tensors. This function is the identity in these cases. (Triggered internally at /__w/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4373.)";
#[cfg(target_os = "linux")]
const MH_SCALAR_WARNING: &CStr = c"Tensor.mH is deprecated on 0-D tensors. Consider using x.conj(). (Triggered internally at /__w/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4382.)";
#[cfg(target_os = "linux")]
const ADJOINT_SCALAR_WARNING: &CStr = c"adjoint() is deprecated on 0-D tensors. Consider using x.conj(). (Triggered internally at /__w/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4390.)";
#[cfg(target_os = "linux")]
const TORCH_FUNCTION_PLAIN_METHOD_WARNING: &CStr = c"Defining your `__torch_function__` as a plain method is deprecated and will be an error in future, please define it as a classmethod. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/utils/python_arg_parser.cpp:359.)";

#[cfg(target_os = "windows")]
const T_NON_MATRIX_WARNING: &CStr = c"The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at C:\\actions-runner\\_work\\pytorch\\pytorch\\aten\\src\\ATen\\native\\TensorShape.cpp:4317.)";
#[cfg(target_os = "windows")]
const T_SCALAR_WARNING: &CStr = c"Tensor.T is deprecated on 0-D tensors. This function is the identity in these cases. (Triggered internally at C:\\actions-runner\\_work\\pytorch\\pytorch\\aten\\src\\ATen\\native\\TensorShape.cpp:4322.)";
#[cfg(target_os = "windows")]
const H_SCALAR_WARNING: &CStr = c"Tensor.H is deprecated on 0-D tensors. Consider using x.conj(). (Triggered internally at C:\\actions-runner\\_work\\pytorch\\pytorch\\aten\\src\\ATen\\native\\TensorShape.cpp:4336.)";
#[cfg(target_os = "windows")]
const MT_SCALAR_WARNING: &CStr = c"Tensor.mT is deprecated on 0-D tensors. This function is the identity in these cases. (Triggered internally at C:\\actions-runner\\_work\\pytorch\\pytorch\\aten\\src\\ATen\\native\\TensorShape.cpp:4374.)";
#[cfg(target_os = "windows")]
const MH_SCALAR_WARNING: &CStr = c"Tensor.mH is deprecated on 0-D tensors. Consider using x.conj(). (Triggered internally at C:\\actions-runner\\_work\\pytorch\\pytorch\\aten\\src\\ATen\\native\\TensorShape.cpp:4383.)";
#[cfg(target_os = "windows")]
const ADJOINT_SCALAR_WARNING: &CStr = c"adjoint() is deprecated on 0-D tensors. Consider using x.conj(). (Triggered internally at C:\\actions-runner\\_work\\pytorch\\pytorch\\aten\\src\\ATen\\native\\TensorShape.cpp:4391.)";
#[cfg(target_os = "windows")]
const TORCH_FUNCTION_PLAIN_METHOD_WARNING: &CStr = c"Defining your `__torch_function__` as a plain method is deprecated and will be an error in future, please define it as a classmethod. (Triggered internally at C:\\actions-runner\\_work\\pytorch\\pytorch\\torch\\csrc\\utils\\python_arg_parser.cpp:359.)";

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const T_NON_MATRIX_WARNING: &CStr = c"The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor.";
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const T_SCALAR_WARNING: &CStr =
    c"Tensor.T is deprecated on 0-D tensors. This function is the identity in these cases.";
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const H_SCALAR_WARNING: &CStr = c"Tensor.H is deprecated on 0-D tensors. Consider using x.conj().";
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const MT_SCALAR_WARNING: &CStr =
    c"Tensor.mT is deprecated on 0-D tensors. This function is the identity in these cases.";
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const MH_SCALAR_WARNING: &CStr =
    c"Tensor.mH is deprecated on 0-D tensors. Consider using x.conj().";
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const ADJOINT_SCALAR_WARNING: &CStr =
    c"adjoint() is deprecated on 0-D tensors. Consider using x.conj().";
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const TORCH_FUNCTION_PLAIN_METHOD_WARNING: &CStr = c"Defining your `__torch_function__` as a plain method is deprecated and will be an error in future, please define it as a classmethod.";

fn device_ordinal(device: Device) -> PyResult<i64> {
    device
        .index()
        .map_or(Ok(-1), |index| i64::try_from(index).map_err(Into::into))
}

fn imag_non_complex_error() -> PyErr {
    PyRuntimeError::new_err("imag is not implemented for tensors with non-complex dtypes.")
}

// Internal descriptor owner matching PyTorch's native tensor base class.
#[pyclass(
    name = "TensorBase",
    module = "torch._C",
    subclass,
    skip_from_py_object
)]
pub(crate) struct PyTensorBase;

#[pymethods]
impl PyTensorBase {
    #[getter]
    fn layout(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) =
            dispatch_tensorbase_mode(slf.py(), tensor, TensorBaseModeTarget::GetSet("layout"))?
        {
            return Ok(result);
        }

        Ok(strided_object(slf.py())?.clone_ref(slf.py()))
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nIs ``True`` if this Tensor is non-leaf and its :attr:`grad` is enabled to be\npopulated during :func:`backward`, ``False`` otherwise.\n"]
    #[getter]
    fn retains_grad(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_mode(
            slf.py(),
            tensor,
            TensorBaseModeTarget::GetSet("retains_grad"),
        )? {
            return Ok(result);
        }

        tensor
            .try_borrow()?
            .inner
            .retains_grad()
            .into_py_any(slf.py())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nretain_grad() -> None\n\nEnables this Tensor to have their :attr:`grad` populated during\n:func:`backward`. This is a no-op for leaf tensors.\n"]
    // Keep the method as METH_NOARGS with no embedded signature. CPython 3.13+
    // derives `($self, /)` from that descriptor shape, while older runtimes
    // leave `__text_signature__` unset; PyTorch follows the same split.
    #[pyo3(text_signature = None)]
    fn retain_grad(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_mode(
            slf.py(),
            tensor,
            TensorBaseModeTarget::Method("retain_grad"),
        )? {
            return Ok(result);
        }

        let tensor = tensor.try_borrow()?;
        if !tensor.inner.requires_grad() {
            return Err(PyRuntimeError::new_err(
                "can't retain_grad on Tensor that has requires_grad=False",
            ));
        }
        if !tensor.inner.is_leaf() {
            return Err(PyRuntimeError::new_err(
                "retain_grad(): retaining gradients for non-leaf tensors is not supported",
            ));
        }

        Ok(slf.py().None())
    }

    #[getter]
    fn output_nr(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) =
            dispatch_tensorbase_mode(slf.py(), tensor, TensorBaseModeTarget::GetSet("output_nr"))?
        {
            return Ok(result);
        }

        tensor.try_borrow()?.inner.output_nr().into_py_any(slf.py())
    }

    fn __getitem__(slf: &Bound<'_, Self>, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        // Keep the common integer-index path allocation-free when no mode is active.
        if !torch_function_mode_stack::is_empty() {
            let args = PyTuple::new(slf.py(), [index.clone()])?;
            if let Some(result) = dispatch_tensorbase_method_mode(
                slf.py(),
                tensor,
                "__getitem__",
                "torch.Tensor.__getitem__",
                &args,
                None,
            )? {
                return Ok(result);
            }
        }

        let tensor = tensor.try_borrow()?;
        let inner = if index.is_none() {
            tensor.inner.unsqueeze_front()
        } else if index.is_instance_of::<PyEllipsis>() {
            tensor.inner.metadata_alias()
        } else if let Ok(indices) = index.cast::<PyTuple>() {
            // PyTorch materializes tuple subclasses through their Python iterator
            // before interpreting any index elements.
            let normalized_indices;
            let indices = if index.is_exact_instance_of::<PyTuple>() {
                indices
            } else {
                normalized_indices = slf
                    .py()
                    .get_type::<PyTuple>()
                    .call1((index,))?
                    .cast_into::<PyTuple>()?;
                &normalized_indices
            };
            if let Some(indexed_dimensions) = metadata_alias_tuple_dimensions(indices)? {
                if indexed_dimensions > tensor.inner.shape().len() {
                    return Err(too_many_indices(tensor.inner.shape().len()));
                }
                tensor.inner.metadata_alias()
            } else if indices.len() == 2
                && indices.get_item(0)?.is_instance_of::<PyEllipsis>()
                && indices.get_item(1)?.is_none()
            {
                tensor.inner.unsqueeze_back()
            } else if indices.len() > tensor.inner.shape().len() {
                return Err(too_many_indices(tensor.inner.shape().len()));
            } else if let Some(indices) = parse_leading_integer_full_slice(&tensor.inner, indices)?
            {
                tensor.inner.index(indices)
            } else {
                let indices = parse_integer_indices(&tensor.inner, indices.len(), indices.iter())?;
                tensor.inner.index(indices)
            }
        } else if is_exact_full_slice(index)? {
            tensor.inner.index_full_slice()
        } else if is_fast_integer_index(index)? {
            let index = parse_integer_index(index)?;
            tensor.inner.index_integer(index)
        } else {
            if tensor.inner.shape().is_empty() {
                return Err(too_many_indices(0));
            }
            let index = parse_integer_index(index)?;
            tensor.inner.index([index])
        }
        .map_err(|error| tensor_error(&error))?;
        Ok(Py::new(slf.py(), PyTensor::new(inner))?.into_any())
    }

    #[doc = "\nunsqueeze(dim) -> Tensor\n\nReturns a new tensor with a dimension of size one inserted at the\nspecified position.\n\nThis implementation supports exact native CPU float32 tensors for every valid\ninsertion dimension. dtype/device extensions, tensor subclasses, broader\n``None`` indexing expansion, and ``__torch_function__`` modes remain\nunsupported.\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn unsqueeze(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let ([dimension], keyword_error) =
            bind_tensor_arguments("unsqueeze", args, kwargs, ["dim"])?;
        validate_dimension_swap_dimension(
            "unsqueeze",
            "dim",
            dimension.position,
            &dimension.value,
        )?;
        if let Some(keyword_error) = keyword_error {
            return Err(keyword_error);
        }
        if !torch_function_mode_stack::is_empty() {
            return Err(unsqueeze_torch_function_mode_error());
        }

        let tensor = slf.as_any();
        if !tensor.is_exact_instance_of::<PyTensor>() {
            return Err(unsqueeze_unsupported_native_input());
        }
        let tensor = tensor.cast::<PyTensor>()?;
        let dimension = extract_dimension_swap_dimension(&dimension.value)?;
        apply_unsqueeze(slf.py(), tensor, dimension)
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nunbind(dim=0) -> seq\n\nSee :func:`torch.unbind`\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn unbind(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let dimension = bind_unbind_dimension(args, kwargs)?;
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_method_mode(
            slf.py(),
            tensor,
            "unbind",
            "torch.Tensor.unbind",
            args,
            kwargs,
        )? {
            return Ok(result);
        }

        let dimension = dimension.map_or(Ok(0), |dimension| {
            extract_dimension_swap_dimension(&dimension.value)
        })?;
        unbind_dimension(slf.py(), tensor, dimension)
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nselect(dim, index) -> Tensor\n\nSee :func:`torch.select`\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn select(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let ([dimension, index], keyword_error) = bind_select_arguments(args, kwargs)?;
        if let Some(keyword_error) = keyword_error {
            return Err(keyword_error);
        }

        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_method_mode(
            slf.py(),
            tensor,
            "select",
            "torch.Tensor.select",
            args,
            kwargs,
        )? {
            return Ok(result);
        }

        // Generated bindings convert the SymInt-like index before the plain
        // integer dimension. Keep that observable order after mode dispatch.
        let index = extract_select_index(&index.value)?;
        let dimension = extract_dimension_swap_dimension(&dimension.value)?;
        select_dimension(slf.py(), tensor, dimension, index)
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nIs ``True`` if the Tensor is quantized, ``False`` otherwise.\n"]
    #[getter]
    fn is_quantized(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        Ok(tensor.inner.is_quantized())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nto_dense(dtype=None, *, masked_grad=True) -> Tensor\n\nCreates a strided copy of :attr:`self` if :attr:`self` is not a strided tensor, otherwise returns :attr:`self`.\n\nKeyword args:\n    {dtype}\n    masked_grad (bool, optional): If set to ``True`` (default) and\n      :attr:`self` has a sparse layout then the backward of\n      :meth:`to_dense` returns ``grad.sparse_mask(self)``.\n\nExample::\n\n    >>> s = torch.sparse_coo_tensor(\n    ...        torch.tensor([[1, 1],\n    ...                      [0, 2]]),\n    ...        torch.tensor([9, 10]),\n    ...        size=(3, 3))\n    >>> s.to_dense()\n    tensor([[ 0,  0,  0],\n            [ 9,  0, 10],\n            [ 0,  0,  0]])\n"]
    // Keep the variadic descriptor shape used by PyTorch even though only the
    // empty call is supported. A METH_NOARGS descriptor gains a synthesized
    // `($self, /)` signature on CPython 3.13+, unlike PyTorch's native method.
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn to_dense(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        if !args.is_empty() {
            return Err(PyTypeError::new_err(format!(
                "to_dense() takes 0 positional arguments but {} {} given",
                args.len(),
                if args.len() == 1 { "was" } else { "were" }
            )));
        }
        if let Some(kwargs) = kwargs
            && let Some((key, _)) = kwargs.iter().next()
        {
            let key = key.extract::<String>()?;
            return Err(PyTypeError::new_err(format!(
                "to_dense() got an unexpected keyword argument '{key}'"
            )));
        }

        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_method_mode(
            slf.py(),
            tensor,
            "to_dense",
            "torch.Tensor.to_dense",
            args,
            kwargs,
        )? {
            return Ok(result);
        }

        // Strided CPU storage is the only supported layout. The no-argument
        // dense conversion is therefore the exact receiver, without a storage
        // borrow, metadata rewrite, copy, or autograd operation. Sparse storage
        // and the dtype and masked_grad overloads remain outside this surface.
        Ok(tensor.clone().unbind().into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nfloat(memory_format=torch.preserve_format) -> Tensor\n\n``self.float()`` is equivalent to ``self.to(torch.float32)``. See :func:`to`.\n\nArgs:\n    memory_format (:class:`torch.memory_format`, optional): the desired memory format of\n        returned Tensor. Default: ``torch.preserve_format``.\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn float(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyTensor>> {
        if !args.is_empty() {
            return Err(PyTypeError::new_err(format!(
                "float() takes 0 positional arguments but {} {} given",
                args.len(),
                if args.len() == 1 { "was" } else { "were" }
            )));
        }

        let mut memory_format = MemoryFormat::Preserve;
        if let Some(kwargs) = kwargs {
            // PyTorch converts the recognized argument before reporting any
            // extra keywords, independent of keyword insertion order.
            if let Some(value) = kwargs.get_item("memory_format")? {
                memory_format = parse_float_memory_format(&value)?;
            }
            for (key, _) in kwargs {
                let key = key.extract::<String>()?;
                if key != "memory_format" {
                    return Err(PyTypeError::new_err(format!(
                        "float() got an unexpected keyword argument '{key}'"
                    )));
                }
            }
        }

        let tensor = slf.as_any().cast::<PyTensor>()?;
        // Float32 is the only supported dtype. Preserve is therefore an exact
        // no-op for every tensor, including arbitrary non-contiguous views.
        if memory_format == MemoryFormat::Preserve {
            return Ok(tensor.clone().unbind());
        }

        let inner = {
            let tensor_ref = tensor.try_borrow()?;
            if tensor_ref.inner.suggested_memory_format() == memory_format {
                return Ok(tensor.clone().unbind());
            }
            tensor_ref
                .inner
                .try_copy_with_memory_format(memory_format)
                .map_err(|error| tensor_error(&error))?
        };
        Py::new(slf.py(), PyTensor::new(inner))
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\ncpu(memory_format=torch.preserve_format) -> Tensor\n\nReturns a copy of this object in CPU memory.\n\nIf this object is already in CPU memory,\nthen no copy is performed and the original object is returned.\n\nArgs:\n    memory_format (:class:`torch.memory_format`, optional): the desired memory format of\n        returned Tensor. Default: ``torch.preserve_format``.\n\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn cpu(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyTensor>> {
        if !args.is_empty() {
            return Err(PyTypeError::new_err(format!(
                "cpu() takes 0 positional arguments but {} {} given",
                args.len(),
                if args.len() == 1 { "was" } else { "were" }
            )));
        }

        let mut memory_format = MemoryFormat::Preserve;
        if let Some(kwargs) = kwargs {
            // PyTorch converts the recognized argument before reporting any
            // extra keywords, independent of keyword insertion order.
            if let Some(value) = kwargs.get_item("memory_format")? {
                memory_format = parse_cpu_memory_format(&value)?;
            }
            for (key, _) in kwargs {
                let key = key.extract::<String>()?;
                if key != "memory_format" {
                    return Err(PyTypeError::new_err(format!(
                        "cpu() got an unexpected keyword argument '{key}'"
                    )));
                }
            }
        }

        let tensor = slf.as_any().cast::<PyTensor>()?;
        // CPU is the only supported device. Preserve never normalizes an
        // existing CPU tensor, including arbitrary non-contiguous views.
        if memory_format == MemoryFormat::Preserve {
            return Ok(tensor.clone().unbind());
        }

        let inner = {
            let tensor_ref = tensor.try_borrow()?;
            if tensor_ref.inner.suggested_memory_format() == memory_format {
                return Ok(tensor.clone().unbind());
            }
            tensor_ref
                .inner
                .try_copy_with_memory_format(memory_format)
                .map_err(|error| tensor_error(&error))?
        };
        Py::new(slf.py(), PyTensor::new(inner))
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nget_device() -> Device ordinal (Integer)\n\nFor CUDA tensors, this function returns the device ordinal of the GPU on which the tensor resides.\nFor CPU tensors, this function returns `-1`.\n\nExample::\n\n    >>> x = torch.randn(3, 4, 5, device='cuda:0')\n    >>> x.get_device()\n    0\n    >>> x.cpu().get_device()\n    -1\n"]
    #[pyo3(text_signature = None)]
    fn get_device(slf: &Bound<'_, Self>) -> PyResult<i64> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        device_ordinal(tensor.inner.device())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nReturns a view of a matrix (2-D tensor) conjugated and transposed.\n\n``x.H`` is equivalent to ``x.transpose(0, 1).conj()`` for complex matrices and\n``x.transpose(0, 1)`` for real matrices.\n\n.. seealso::\n\n        :attr:`~.Tensor.mH`: An attribute that also works on batches of matrices.\n"]
    #[getter(H)]
    fn conjugate_transpose(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) =
            dispatch_tensorbase_mode(slf.py(), tensor, TensorBaseModeTarget::GetSet("H"))?
        {
            return Ok(result);
        }

        let rank = tensor.try_borrow()?.inner.shape().len();
        if rank > 2 {
            return Err(PyRuntimeError::new_err(format!(
                "tensor.H is only supported on matrices (2-D tensors). Got {rank}-D tensor. For batches of matrices, consider using tensor.mH"
            )));
        }

        matrix_adjoint(
            slf.py(),
            tensor,
            &H_SCALAR_WARNING_EMITTED,
            H_SCALAR_WARNING,
            "tensor.H is only supported on matrices (2-D tensors). Got 1-D tensor.",
        )
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nadjoint() -> Tensor\n\nAlias for :func:`adjoint`\n"]
    #[pyo3(text_signature = None)]
    fn adjoint(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) =
            dispatch_tensorbase_mode(slf.py(), tensor, TensorBaseModeTarget::Method("adjoint"))?
        {
            return Ok(result);
        }

        matrix_adjoint(
            slf.py(),
            tensor,
            &ADJOINT_SCALAR_WARNING_EMITTED,
            ADJOINT_SCALAR_WARNING,
            "tensor.adjoint() is only supported on matrices or batches of matrices. Got 1-D tensor.",
        )
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nAccessing this property is equivalent to calling :func:`adjoint`.\n"]
    #[getter(mH)]
    fn conjugate_matrix_transpose(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) =
            dispatch_tensorbase_mode(slf.py(), tensor, TensorBaseModeTarget::GetSet("mH"))?
        {
            return Ok(result);
        }

        matrix_adjoint(
            slf.py(),
            tensor,
            &MH_SCALAR_WARNING_EMITTED,
            MH_SCALAR_WARNING,
            "tensor.mH is only supported on matrices or batches of matrices. Got 1-D tensor.",
        )
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nReturns a new tensor containing real values of the :attr:`self` tensor for a complex-valued input tensor.\nThe returned tensor and :attr:`self` share the same underlying storage.\n\nReturns :attr:`self` if :attr:`self` is a real-valued tensor.\n\nExample::\n\n    >>> x=torch.randn(4, dtype=torch.cfloat)\n    >>> x\n    tensor([(0.3100+0.3553j), (-0.5445-0.7896j), (-1.6492-0.0633j), (-0.0638-0.8119j)])\n    >>> x.real\n    tensor([ 0.3100, -0.5445, -1.6492, -0.0638])\n\n"]
    #[getter]
    fn real(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) =
            dispatch_tensorbase_mode(slf.py(), tensor, TensorBaseModeTarget::GetSet("real"))?
        {
            return Ok(result);
        }

        // Float32 is the only supported dtype, so every Tensor is already real.
        // Preserve the wrapper itself without inspecting storage or autograd state.
        Ok(tensor.clone().unbind().into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nReturns a new tensor containing imaginary values of the :attr:`self` tensor.\nThe returned tensor and :attr:`self` share the same underlying storage.\n\n.. warning::\n    :func:`imag` is only supported for tensors with complex dtypes.\n\nExample::\n\n    >>> x=torch.randn(4, dtype=torch.cfloat)\n    >>> x\n    tensor([(0.3100+0.3553j), (-0.5445-0.7896j), (-1.6492-0.0633j), (-0.0638-0.8119j)])\n    >>> x.imag\n    tensor([ 0.3553, -0.7896, -0.0633, -0.8119])\n\n"]
    #[getter]
    fn imag(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) =
            dispatch_tensorbase_mode(slf.py(), tensor, TensorBaseModeTarget::GetSet("imag"))?
        {
            return Ok(result);
        }

        apply_tensor_imag(tensor)
    }

    #[setter(imag)]
    fn set_imag(_slf: &Bound<'_, Self>, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        Err(imag_non_complex_error())
    }

    #[deleter(imag)]
    fn delete_imag(_slf: &Bound<'_, Self>) -> PyResult<()> {
        Err(imag_non_complex_error())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nReturns true if this tensor resides in pinned memory.\nBy default, the device pinned memory on will be the current :ref:`accelerator<accelerators>`.\n"]
    // PyTorch retains a variadic native descriptor for its deprecated device
    // argument. Match that descriptor metadata while exposing only the stable
    // no-argument query supported by this pageable CPU storage model.
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn is_pinned(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        if args.len() > 1 {
            return Err(PyTypeError::new_err(format!(
                "is_pinned() takes from 0 to 1 positional arguments but {} were given",
                args.len()
            )));
        }
        if let Some(kwargs) = kwargs {
            for (key, _) in kwargs {
                let key = key.extract::<String>()?;
                if key != "device" {
                    return Err(PyTypeError::new_err(format!(
                        "is_pinned() got an unexpected keyword argument '{key}'"
                    )));
                }
            }
        }
        if !args.is_empty() {
            return Err(PyTypeError::new_err(
                "is_pinned() takes 0 positional arguments but 1 was given",
            ));
        }
        if kwargs.is_some_and(|kwargs| !kwargs.is_empty()) {
            return Err(PyTypeError::new_err(
                "is_pinned() got an unexpected keyword argument 'device'",
            ));
        }

        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_method_mode(
            slf.py(),
            tensor,
            "is_pinned",
            "torch.Tensor.is_pinned",
            args,
            kwargs,
        )? {
            return Ok(result);
        }

        tensor.try_borrow()?.inner.is_pinned().into_py_any(slf.py())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nis_inference() -> bool\n\nSee :func:`torch.is_inference`\n"]
    // Keep the method as METH_NOARGS with no embedded signature. CPython 3.13+
    // derives `($self, /)` from that descriptor shape, while older runtimes
    // leave `__text_signature__` unset; PyTorch follows the same split.
    #[pyo3(text_signature = None)]
    fn is_inference(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_mode(
            slf.py(),
            tensor,
            TensorBaseModeTarget::Method("is_inference"),
        )? {
            return Ok(result);
        }

        let result = tensor.try_borrow()?.inner.is_inference();
        result.into_py_any(slf.py())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nabs() -> Tensor\n\nSee :func:`torch.abs`\n"]
    #[pyo3(text_signature = None)]
    fn abs(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "abs")? {
            return Ok(result);
        }

        let output = {
            let tensor = tensor.try_borrow()?;
            tensor.inner.abs().map_err(|error| tensor_error(&error))?
        };
        Ok(Py::new(slf.py(), PyTensor::new(output))?.into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nabsolute() -> Tensor\n\nAlias for :func:`abs`\n"]
    #[pyo3(text_signature = None)]
    fn absolute(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "absolute")? {
            return Ok(result);
        }

        let output = {
            let tensor = tensor.try_borrow()?;
            tensor.inner.abs().map_err(|error| tensor_error(&error))?
        };
        Ok(Py::new(slf.py(), PyTensor::new(output))?.into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nexp() -> Tensor\n\nSee :func:`torch.exp`\n"]
    #[pyo3(text_signature = None)]
    fn exp(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "exp")? {
            return Ok(result);
        }

        let output = {
            let tensor = tensor.try_borrow()?;
            tensor.inner.exp().map_err(|error| tensor_error(&error))?
        };
        Ok(Py::new(slf.py(), PyTensor::new(output))?.into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nfloor() -> Tensor\n\nSee :func:`torch.floor`\n"]
    #[pyo3(text_signature = None)]
    fn floor(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "floor")? {
            return Ok(result);
        }

        let output = {
            let tensor = tensor.try_borrow()?;
            tensor.inner.floor().map_err(|error| tensor_error(&error))?
        };
        Ok(Py::new(slf.py(), PyTensor::new(output))?.into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nceil() -> Tensor\n\nSee :func:`torch.ceil`\n"]
    #[pyo3(text_signature = None)]
    fn ceil(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "ceil")? {
            return Ok(result);
        }

        let output = {
            let tensor = tensor.try_borrow()?;
            tensor.inner.ceil().map_err(|error| tensor_error(&error))?
        };
        Ok(Py::new(slf.py(), PyTensor::new(output))?.into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\ntrunc() -> Tensor\n\nSee :func:`torch.trunc`\n"]
    #[pyo3(text_signature = None)]
    fn trunc(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "trunc")? {
            return Ok(result);
        }

        let output = {
            let tensor = tensor.try_borrow()?;
            tensor.inner.trunc().map_err(|error| tensor_error(&error))?
        };
        Ok(Py::new(slf.py(), PyTensor::new(output))?.into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nfix() -> Tensor\n\nSee :func:`torch.fix`.\n"]
    #[pyo3(text_signature = None)]
    fn fix(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "fix")? {
            return Ok(result);
        }

        let output = {
            let tensor = tensor.try_borrow()?;
            tensor.inner.trunc().map_err(|error| tensor_error(&error))?
        };
        Ok(Py::new(slf.py(), PyTensor::new(output))?.into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nsigmoid() -> Tensor\n\nSee :func:`torch.sigmoid`\n"]
    #[pyo3(text_signature = None)]
    fn sigmoid(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "sigmoid")? {
            return Ok(result);
        }

        let output = {
            let tensor = tensor.try_borrow()?;
            tensor
                .inner
                .sigmoid()
                .map_err(|error| tensor_error(&error))?
        };
        Ok(Py::new(slf.py(), PyTensor::new(output))?.into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\ntanh() -> Tensor\n\nSee :func:`torch.tanh`\n"]
    #[pyo3(text_signature = None)]
    fn tanh(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "tanh")? {
            return Ok(result);
        }

        let output = {
            let tensor = tensor.try_borrow()?;
            tensor.inner.tanh().map_err(|error| tensor_error(&error))?
        };
        Ok(Py::new(slf.py(), PyTensor::new(output))?.into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nsin() -> Tensor\n\nSee :func:`torch.sin`\n"]
    #[pyo3(text_signature = None)]
    fn sin(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "sin")? {
            return Ok(result);
        }

        let output = {
            let tensor = tensor.try_borrow()?;
            tensor.inner.sin().map_err(|error| tensor_error(&error))?
        };
        Ok(Py::new(slf.py(), PyTensor::new(output))?.into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\ncos() -> Tensor\n\nSee :func:`torch.cos`\n"]
    #[pyo3(text_signature = None)]
    fn cos(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "cos")? {
            return Ok(result);
        }

        let output = {
            let tensor = tensor.try_borrow()?;
            tensor.inner.cos().map_err(|error| tensor_error(&error))?
        };
        Ok(Py::new(slf.py(), PyTensor::new(output))?.into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nsqrt() -> Tensor\n\nSee :func:`torch.sqrt`\n"]
    #[pyo3(text_signature = None)]
    fn sqrt(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "sqrt")? {
            return Ok(result);
        }

        let output = {
            let tensor = tensor.try_borrow()?;
            tensor.inner.sqrt().map_err(|error| tensor_error(&error))?
        };
        Ok(Py::new(slf.py(), PyTensor::new(output))?.into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nsquare() -> Tensor\n\nSee :func:`torch.square`\n"]
    #[pyo3(text_signature = None)]
    fn square(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "square")? {
            return Ok(result);
        }

        let output = {
            let tensor = tensor.try_borrow()?;
            tensor
                .inner
                .square()
                .map_err(|error| tensor_error(&error))?
        };
        Ok(Py::new(slf.py(), PyTensor::new(output))?.into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nreciprocal() -> Tensor\n\nSee :func:`torch.reciprocal`\n"]
    #[pyo3(text_signature = None)]
    fn reciprocal(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "reciprocal")?
        {
            return Ok(result);
        }

        let output = {
            let tensor = tensor.try_borrow()?;
            tensor
                .inner
                .reciprocal()
                .map_err(|error| tensor_error(&error))?
        };
        Ok(Py::new(slf.py(), PyTensor::new(output))?.into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nrsqrt() -> Tensor\n\nSee :func:`torch.rsqrt`\n"]
    #[pyo3(text_signature = None)]
    fn rsqrt(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "rsqrt")? {
            return Ok(result);
        }

        let output = {
            let tensor = tensor.try_borrow()?;
            tensor.inner.rsqrt().map_err(|error| tensor_error(&error))?
        };
        Ok(Py::new(slf.py(), PyTensor::new(output))?.into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nlog() -> Tensor\n\nSee :func:`torch.log`\n"]
    #[pyo3(text_signature = None)]
    fn log(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "log")? {
            return Ok(result);
        }

        let output = {
            let tensor = tensor.try_borrow()?;
            tensor.inner.log().map_err(|error| tensor_error(&error))?
        };
        Ok(Py::new(slf.py(), PyTensor::new(output))?.into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\npositive() -> Tensor\n\nSee :func:`torch.positive`\n"]
    #[pyo3(text_signature = None)]
    fn positive(slf: &Bound<'_, Self>) -> PyResult<Py<PyTensor>> {
        // Unary positive is an identity for the supported float32 tensors. Return
        // the existing wrapper so storage, layout, and autograd state are untouched.
        Ok(slf.as_any().cast::<PyTensor>()?.clone().unbind())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nneg() -> Tensor\n\nSee :func:`torch.neg`\n"]
    #[pyo3(text_signature = None)]
    fn neg(slf: &Bound<'_, Self>) -> PyResult<PyTensor> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        tensor.negated()
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nnegative() -> Tensor\n\nSee :func:`torch.negative`\n"]
    #[pyo3(text_signature = None)]
    fn negative(slf: &Bound<'_, Self>) -> PyResult<PyTensor> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        tensor.negated()
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nmultiply(value) -> Tensor\n\nSee :func:`torch.multiply`.\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn multiply(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyTensor> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        tensor.multiplication_method(MultiplicationOperation::Multiply, args, kwargs)
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nadd(other, *, alpha=1) -> Tensor\n\nAdd a scalar or tensor to :attr:`self` tensor. If both :attr:`alpha`\nand :attr:`other` are specified, each element of :attr:`other` is scaled by\n:attr:`alpha` before being used.\n\nWhen :attr:`other` is a tensor, the shape of :attr:`other` must be\n:ref:`broadcastable <broadcasting-semantics>` with the shape of the underlying\ntensor\n\nSee :func:`torch.add`\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn add(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        tensor_add_sub_method(AddSubMethodOperation::Add, slf, args, kwargs)
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\ndiv(value, *, rounding_mode=None) -> Tensor\n\nSee :func:`torch.div`\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn div(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        tensor_division_method(DivisionOperation::Div, slf, args, kwargs)
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\ndivide(value, *, rounding_mode=None) -> Tensor\n\nSee :func:`torch.divide`\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn divide(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        tensor_division_method(DivisionOperation::Divide, slf, args, kwargs)
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nsub(other, *, alpha=1) -> Tensor\n\nSee :func:`torch.sub`.\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn sub(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        tensor_add_sub_method(AddSubMethodOperation::Sub, slf, args, kwargs)
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nsubtract(other, *, alpha=1) -> Tensor\n\nSee :func:`torch.subtract`.\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn subtract(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        tensor_add_sub_method(AddSubMethodOperation::Subtract, slf, args, kwargs)
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nmatmul(tensor2) -> Tensor\n\nSee :func:`torch.matmul`\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn matmul(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let (argument, keyword_error) = bind_matmul_argument(args, kwargs)?;
        let other = parse_tensor_or_torch_function_argument("matmul", "other", &argument)?;
        if let Some(keyword_error) = keyword_error {
            return Err(keyword_error);
        }
        let tensor = slf.as_any().cast::<PyTensor>()?;
        dispatch_matmul(slf.py(), tensor, &other, args, kwargs)
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\npermute(*dims) -> Tensor\n\nReturns a view of the tensor with its dimensions permuted.\n\nArgs:\n    dims (torch.Size, int..., tuple of int or list of int): the desired ordering of dimensions.\n\nExample:\n    >>> x = torch.randn(2, 3, 5)\n    >>> x.size()\n    torch.Size([2, 3, 5])\n    >>> x.permute(2, 0, 1).size()\n    torch.Size([5, 2, 3])\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn permute(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyTensor> {
        let dimensions = bind_permute_dimensions(args, kwargs)?;
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        permute_tensor(&tensor.inner, dimensions).map(PyTensor::new)
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nmovedim(source, destination) -> Tensor\n\nSee :func:`torch.movedim`\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn movedim(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        dimension_move_tensor_method(DimensionMoveOperation::Movedim, slf, args, kwargs)
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nmoveaxis(source, destination) -> Tensor\n\nSee :func:`torch.moveaxis`\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn moveaxis(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        dimension_move_tensor_method(DimensionMoveOperation::Moveaxis, slf, args, kwargs)
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nreshape_as(other) -> Tensor\n\nReturns this tensor as the same shape as :attr:`other`.\n``self.reshape_as(other)`` is equivalent to ``self.reshape(other.sizes())``.\nThis method returns a view if ``other.sizes()`` is compatible with the current\nshape. See :meth:`torch.Tensor.view` on when it is possible to return a view.\n\nPlease see :meth:`reshape` for more information about ``reshape``.\n\nArgs:\n    other (:class:`torch.Tensor`): The result tensor has the same shape\n        as :attr:`other`.\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn reshape_as(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyTensor> {
        let (arguments, keyword_error) =
            bind_tensor_arguments("reshape_as", args, kwargs, ["other"])?;
        let other = parse_tensor_argument("reshape_as", "other", &arguments[0])?;
        if let Some(keyword_error) = keyword_error {
            return Err(keyword_error);
        }

        let shape = tensor_shape_as_i64(other)?;

        slf.as_any()
            .cast::<PyTensor>()?
            .try_borrow()?
            .inner
            .reshape(shape)
            .map(PyTensor::new)
            .map_err(|error| tensor_error(&error))
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = r"
view(*shape) -> Tensor

Returns a new tensor with the same data as the :attr:`self` tensor but of a
different :attr:`shape`.

The returned tensor shares the same data and must have the same number
of elements, but may have a different size. For a tensor to be viewed, the new
view size must be compatible with its original size and stride, i.e., each new
view dimension must either be a subspace of an original dimension, or only span
across original dimensions :math:`d, d+1, \dots, d+k` that satisfy the following
contiguity-like condition that :math:`\forall i = d, \dots, d+k-1`,

.. math::

  \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]

Otherwise, it will not be possible to view :attr:`self` tensor as :attr:`shape`
without copying it (e.g., via :meth:`contiguous`). When it is unclear whether a
:meth:`view` can be performed, it is advisable to use :meth:`reshape`, which
returns a view if the shapes are compatible, and copies (equivalent to calling
:meth:`contiguous`) otherwise.

Args:
    shape (torch.Size or int...): the desired size

Example::

    >>> x = torch.randn(4, 4)
    >>> x.size()
    torch.Size([4, 4])
    >>> y = x.view(16)
    >>> y.size()
    torch.Size([16])
    >>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
    >>> z.size()
    torch.Size([2, 8])

    >>> a = torch.randn(1, 2, 3, 4)
    >>> a.size()
    torch.Size([1, 2, 3, 4])
    >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
    >>> b.size()
    torch.Size([1, 3, 2, 4])
    >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
    >>> c.size()
    torch.Size([1, 3, 2, 4])
    >>> torch.equal(b, c)
    False


.. method:: view(dtype) -> Tensor
   :noindex:

Returns a new tensor with the same data as the :attr:`self` tensor but of a
different :attr:`dtype`.

If the element size of :attr:`dtype` is different than that of ``self.dtype``,
then the size of the last dimension of the output will be scaled
proportionally.  For instance, if :attr:`dtype` element size is twice that of
``self.dtype``, then each pair of elements in the last dimension of
:attr:`self` will be combined, and the size of the last dimension of the output
will be half that of :attr:`self`. If :attr:`dtype` element size is half that
of ``self.dtype``, then each element in the last dimension of :attr:`self` will
be split in two, and the size of the last dimension of the output will be
double that of :attr:`self`. For this to be possible, the following conditions
must be true:

    * ``self.dim()`` must be greater than 0.
    * ``self.stride(-1)`` must be 1.

Additionally, if the element size of :attr:`dtype` is greater than that of
``self.dtype``, the following conditions must be true as well:

    * ``self.size(-1)`` must be divisible by the ratio between the element
      sizes of the dtypes.
    * ``self.storage_offset()`` must be divisible by the ratio between the
      element sizes of the dtypes.
    * The strides of all dimensions, except the last dimension, must be
      divisible by the ratio between the element sizes of the dtypes.

If any of the above conditions are not met, an error is thrown.

.. warning::

    This overload is not supported by TorchScript, and using it in a Torchscript
    program will cause undefined behavior.


Args:
    dtype (:class:`torch.dtype`): the desired dtype

Example::

    >>> x = torch.randn(4, 4)
    >>> x
    tensor([[ 0.9482, -0.0310,  1.4999, -0.5316],
            [-0.1520,  0.7472,  0.5617, -0.8649],
            [-2.4724, -0.0334, -0.2976, -0.8499],
            [-0.2109,  1.9913, -0.9607, -0.6123]])
    >>> x.dtype
    torch.float32

    >>> y = x.view(torch.int32)
    >>> y
    tensor([[ 1064483442, -1124191867,  1069546515, -1089989247],
            [-1105482831,  1061112040,  1057999968, -1084397505],
            [-1071760287, -1123489973, -1097310419, -1084649136],
            [-1101533110,  1073668768, -1082790149, -1088634448]],
        dtype=torch.int32)
    >>> y[0, 0] = 1000000000
    >>> x
    tensor([[ 0.0047, -0.0310,  1.4999, -0.5316],
            [-0.1520,  0.7472,  0.5617, -0.8649],
            [-2.4724, -0.0334, -0.2976, -0.8499],
            [-0.2109,  1.9913, -0.9607, -0.6123]])

    >>> x.view(torch.cfloat)
    tensor([[ 0.0047-0.0310j,  1.4999-0.5316j],
            [-0.1520+0.7472j,  0.5617-0.8649j],
            [-2.4724-0.0334j, -0.2976-0.8499j],
            [-0.2109+1.9913j, -0.9607-0.6123j]])
    >>> x.view(torch.cfloat).size()
    torch.Size([4, 2])

    >>> x.view(torch.uint8)
    tensor([[  0, 202, 154,  59, 182, 243, 253, 188, 185, 252, 191,  63, 240,  22,
               8, 191],
            [227, 165,  27, 190, 128,  72,  63,  63, 146, 203,  15,  63,  22, 106,
              93, 191],
            [205,  59,  30, 192, 112, 206,   8, 189,   7,  95, 152, 190,  12, 147,
              89, 191],
            [ 43, 246,  87, 190, 235, 226, 254,  63, 111, 240, 117, 191, 177, 191,
              28, 191]], dtype=torch.uint8)
    >>> x.view(torch.uint8).size()
    torch.Size([4, 16])
"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn view(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let argument = bind_view_argument(args, kwargs)?;
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_method_mode(
            slf.py(),
            tensor,
            "view",
            "torch.Tensor.view",
            args,
            kwargs,
        )? {
            return Ok(result);
        }

        let inner = match argument {
            ViewArgument::Shape(shape) => {
                let shape = parse_view_shape_argument(shape)?;
                tensor
                    .try_borrow()?
                    .inner
                    .view(shape)
                    .map_err(|error| tensor_error(&error))?
            }
            ViewArgument::DType(dtype) => {
                let dtype = dtype.try_borrow()?.inner();
                let tensor = tensor.try_borrow()?;
                if dtype != tensor.inner.dtype() {
                    return Err(unsupported_view_dtype_error());
                }
                tensor
                    .inner
                    .detach()
                    .map_err(|error| tensor_error(&error))?
            }
        };
        Ok(Py::new(slf.py(), PyTensor::new(inner))?.into_any())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nview_as(other) -> Tensor\n\nView this tensor as the same size as :attr:`other`.\n``self.view_as(other)`` is equivalent to ``self.view(other.size())``.\n\nPlease see :meth:`~Tensor.view` for more information about ``view``.\n\nArgs:\n    other (:class:`torch.Tensor`): The result tensor has the same size\n        as :attr:`other`.\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn view_as(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let (arguments, keyword_error) = bind_tensor_arguments("view_as", args, kwargs, ["other"])?;
        let other = parse_tensor_or_torch_function_argument("view_as", "other", &arguments[0])?;
        if let Some(keyword_error) = keyword_error {
            return Err(keyword_error);
        }

        let tensor = slf.as_any().cast::<PyTensor>()?;
        dispatch_view_as(slf.py(), tensor, &other, args, kwargs)
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\ntype_as(tensor) -> Tensor\n\nReturns this tensor cast to the type of the given tensor.\n\nThis is a no-op if the tensor is already of the correct type. This is\nequivalent to ``self.type(tensor.type())``\n\nArgs:\n    tensor (Tensor): the tensor which has the desired type\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn type_as(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyTensor>> {
        let (arguments, keyword_error) = bind_tensor_arguments("type_as", args, kwargs, ["other"])?;
        parse_tensor_argument("type_as", "other", &arguments[0])?;
        if let Some(keyword_error) = keyword_error {
            return Err(keyword_error);
        }

        // Float32 on CPU is the only supported tensor type, so matching
        // PyTorch's no-op path also preserves the exact Python wrapper and its
        // storage and autograd state.
        Ok(slf.as_any().cast::<PyTensor>()?.clone().unbind())
    }
}

/// Python-facing tensor backed by the native Rust tensor core.
#[pyclass(
    name = "Tensor",
    module = "torch_rs",
    extends = PyTensorBase,
    skip_from_py_object
)]
pub(crate) struct PyTensor {
    inner: CoreTensor,
    grad_cache: PyOnceLock<Py<PyTensor>>,
}

impl From<PyTensor> for PyClassInitializer<PyTensor> {
    fn from(tensor: PyTensor) -> Self {
        PyClassInitializer::from(PyTensorBase).add_subclass(tensor)
    }
}

// PyO3 deliberately leaves conversion unspecified for native subclasses because
// their base initializer is application-defined. Every Tensor owns the same
// stateless TensorBase portion, so construction can provide it consistently.
impl<'py> IntoPyObject<'py> for PyTensor {
    type Target = Self;
    type Output = Bound<'py, Self>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Bound::new(py, self)
    }
}

impl PyTensor {
    pub(crate) fn new(inner: CoreTensor) -> Self {
        Self {
            inner,
            grad_cache: PyOnceLock::new(),
        }
    }

    pub(crate) const fn inner(&self) -> &CoreTensor {
        &self.inner
    }

    pub(crate) const fn grad_cache(&self) -> &PyOnceLock<Py<PyTensor>> {
        &self.grad_cache
    }
}

pub(crate) fn get_device_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = bind_legacy_single_tensor_argument("get_device", args, kwargs)?;
    let tensor = input
        .value
        .cast::<PyTensor>()
        .expect("the get_device input type was checked while binding")
        .try_borrow()?;
    device_ordinal(tensor.inner.device())?.into_py_any(py)
}

pub(crate) fn as_tensor_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let arguments = bind_as_tensor_arguments(args, kwargs)?;
    let Some(data) = arguments.data else {
        return Err(PyTypeError::new_err(
            "as_tensor() missing 1 required positional arguments: \"data\"",
        ));
    };

    let dtype = parse_as_tensor_dtype(arguments.dtype.as_ref())?;
    validate_as_tensor_device_type("as_tensor", arguments.device.as_ref())?;
    if let Some(keyword_error) = arguments.keyword_error {
        return Err(keyword_error);
    }
    if let Some(result) = dispatch_as_tensor_mode(py, args, kwargs)? {
        return Ok(result);
    }
    let device = parse_as_tensor_device("as_tensor", arguments.device.as_ref())?;

    if dtype != DType::Float32 || !device.is_cpu() {
        return Err(PyNotImplementedError::new_err(
            "as_tensor(): only identity conversion for CPU float32 tensors is supported",
        ));
    }
    if !data.value.is_exact_instance_of::<PyTensor>() {
        if let Some(value) = extract_exact_python_float_scalar(&data.value)? {
            return Ok(
                Py::new(py, rank_zero_scalar_tensor(value, dtype, device, false)?)?.into_any(),
            );
        }
        if let Some((flattened, shape)) = as_tensor_float_sequence(&data.value)? {
            return Ok(Py::new(
                py,
                CoreTensor::from_vec_with_metadata(flattened, shape, dtype, device)
                    .map(PyTensor::new)
                    .map_err(|error| tensor_error(&error))?,
            )?
            .into_any());
        }
        return Err(PyNotImplementedError::new_err(
            "as_tensor(): only exact native CPU float32 Tensor inputs, Python float scalars, or exact list/tuple sequences of Python floats are supported; NumPy arrays/scalars, integer and boolean inference, and other conversions are not implemented",
        ));
    }
    Ok(data.value.unbind())
}

pub(crate) fn asarray_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let arguments = bind_asarray_arguments(args, kwargs)?;
    let Some(obj) = arguments.obj else {
        return Err(PyTypeError::new_err(
            "asarray() missing 1 required positional arguments: \"obj\"",
        ));
    };

    validate_identity_dtype_type("asarray", arguments.dtype.as_ref())?;
    validate_as_tensor_device_type("asarray", arguments.device.as_ref())?;
    validate_asarray_bool("copy", arguments.copy.as_ref())?;
    validate_asarray_bool("requires_grad", arguments.requires_grad.as_ref())?;
    if let Some(keyword_error) = arguments.keyword_error {
        return Err(keyword_error);
    }
    if let Some(result) = dispatch_asarray_mode(py, args, kwargs)? {
        return Ok(result);
    }

    let dtype = parse_identity_dtype("asarray", arguments.dtype.as_ref())?;
    validate_asarray_device_string_syntax(arguments.device.as_ref())?;
    if is_exact_list_or_tuple(&obj.value) {
        validate_asarray_sequence_copy(arguments.copy.as_ref())?;
    }
    let device = parse_as_tensor_device("asarray", arguments.device.as_ref())?;
    let is_exact_native_tensor = obj.value.is_exact_instance_of::<PyTensor>();
    if !is_exact_native_tensor {
        validate_asarray_copy(arguments.copy.as_ref())?;
    }
    validate_asarray_requires_grad(arguments.requires_grad.as_ref())?;

    if dtype != DType::Float32 || !device.is_cpu() {
        return Err(PyNotImplementedError::new_err(
            "asarray(): only identity conversion for CPU float32 tensors is supported",
        ));
    }
    if !is_exact_native_tensor {
        if let Some(value) = extract_exact_python_float_scalar(&obj.value)? {
            validate_asarray_scalar_copy(arguments.copy.as_ref())?;
            return Ok(
                Py::new(py, rank_zero_scalar_tensor(value, dtype, device, false)?)?.into_any(),
            );
        }
        if let Some((flattened, shape)) = as_tensor_float_sequence(&obj.value)? {
            return Ok(Py::new(
                py,
                CoreTensor::from_vec_with_metadata(flattened, shape, dtype, device)
                    .map(PyTensor::new)
                    .map_err(|error| tensor_error(&error))?,
            )?
            .into_any());
        }
        return Err(PyNotImplementedError::new_err(
            "asarray(): only exact native CPU float32 Tensor inputs, Python float scalars, or exact list/tuple sequences of Python floats are supported; NumPy arrays/scalars, integer and boolean inference, and other conversions are not implemented",
        ));
    }
    let source_requires_grad = {
        let tensor = obj.value.cast::<PyTensor>()?.try_borrow()?;
        if tensor.inner.dtype() != DType::Float32 || !tensor.inner.device().is_cpu() {
            return Err(PyNotImplementedError::new_err(
                "asarray(): only identity conversion for CPU float32 tensors is supported",
            ));
        }
        tensor.inner.requires_grad()
    };
    if source_requires_grad {
        warn_once(
            py,
            &ASARRAY_REQUIRES_GRAD_WARNING_EMITTED,
            c"torch.asarray: unspecified requires_grad now defaults to obj.requires_grad instead of False. Pass requires_grad=False explicitly to get the old behavior and silence this warning. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:1737.)",
        )?;
    }
    if asarray_copy_requested(arguments.copy.as_ref())? {
        let copied = obj
            .value
            .cast::<PyTensor>()?
            .try_borrow()?
            .inner
            .try_clone()
            .map(|tensor| tensor.with_requires_grad(source_requires_grad))
            .map(PyTensor::new)
            .map_err(|error| tensor_error(&error))?;
        return Ok(Py::new(py, copied)?.into_any());
    }
    Ok(obj.value.unbind())
}

pub(crate) fn scalar_tensor_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    Ok(Bound::new(py, scalar_tensor_impl(args, kwargs)?)?
        .into_any()
        .unbind())
}

pub(crate) fn arange_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    if !torch_function_mode_stack::is_empty() {
        let function = variable_function(py, "arange")?;
        let types = PyTuple::empty(py);
        let active_mode = torch_function_mode_stack::pop();
        if let Some(mode) = active_mode.get() {
            validate_torch_function_mode_handler(mode.bind(py))?;
            let handler = mode.bind(py).getattr("__torch_function__")?;
            let result =
                call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
            if !is_not_implemented(py, &result) {
                return Ok(result);
            }
            return Err(torch_function_dispatch_error(
                py,
                "torch.arange",
                Some(mode),
                None,
            )?);
        }
    }

    Ok(Bound::new(py, arange_impl(args, kwargs)?)?
        .into_any()
        .unbind())
}

pub(crate) fn ones_like_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (input, requires_grad) = parse_like_factory_arguments(
        "ones_like",
        bind_like_factory_arguments("ones_like", args, kwargs)?,
    )?;
    let shape = like_factory_input_shape("ones_like", &input)?;
    let inner = CoreTensor::ones_with_metadata(shape, DType::Float32, Device::Cpu)
        .map_err(|error| tensor_error(&error))?;
    Ok(Py::new(py, PyTensor::new(inner.with_requires_grad(requires_grad)))?.into_any())
}

pub(crate) fn empty_like_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (input, requires_grad) = parse_like_factory_arguments(
        "empty_like",
        bind_like_factory_arguments("empty_like", args, kwargs)?,
    )?;
    let shape = like_factory_input_shape("empty_like", &input)?;
    let inner = CoreTensor::empty_with_metadata(shape, DType::Float32, Device::Cpu)
        .map_err(|error| tensor_error(&error))?;
    Ok(Py::new(py, PyTensor::new(inner.with_requires_grad(requires_grad)))?.into_any())
}

pub(crate) fn full_like_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (input, fill_value, requires_grad) =
        parse_full_like_arguments(bind_full_like_arguments(args, kwargs)?)?;
    let shape = like_factory_input_shape("full_like", &input)?;
    let fill_value = fill_value.into_f32()?;
    let inner = CoreTensor::full_with_metadata(shape, fill_value, DType::Float32, Device::Cpu)
        .map_err(|error| tensor_error(&error))?;
    Ok(Py::new(py, PyTensor::new(inner.with_requires_grad(requires_grad)))?.into_any())
}

pub(crate) fn zeros_like_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (input, requires_grad) = parse_like_factory_arguments(
        "zeros_like",
        bind_like_factory_arguments("zeros_like", args, kwargs)?,
    )?;
    let shape = like_factory_input_shape("zeros_like", &input)?;
    let inner = CoreTensor::zeros_with_metadata(shape, DType::Float32, Device::Cpu)
        .map_err(|error| tensor_error(&error))?;
    Ok(Py::new(py, PyTensor::new(inner.with_requires_grad(requires_grad)))?.into_any())
}

fn dispatch_empty_variadic_tensor_input(
    py: Python<'_>,
    name: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<Py<PyAny>>> {
    if args.len() != 1 || kwargs.is_some_and(|kwargs| !kwargs.is_empty()) {
        return Ok(None);
    }

    let input = args.get_item(0)?;
    if !input.is_exact_instance_of::<PyTuple>() || !input.cast::<PyTuple>()?.is_empty() {
        return Ok(None);
    }
    if torch_function_mode_stack::is_empty() {
        return Ok(Some(input.unbind()));
    }

    let function = variable_function(py, name)?;
    let types = PyTuple::empty(py);
    let active_mode = torch_function_mode_stack::pop();
    let Some(mode) = active_mode.get() else {
        return Ok(Some(input.unbind()));
    };
    validate_torch_function_mode_handler(mode.bind(py))?;
    // Generated variable functions omit kwargs on the initial call, while
    // explicit forwarding with `**{}` supplies an observable empty dictionary.
    let result = if kwargs.is_none() {
        cpython_compat::call_torch_function_mode_handler(
            py,
            mode.bind(py),
            &function,
            &types,
            args,
        )?
    } else {
        let handler = mode.bind(py).getattr("__torch_function__")?;
        call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?
    };
    if !is_not_implemented(py, &result) {
        return Ok(Some(result));
    }

    Err(torch_function_dispatch_error(
        py,
        &format!("torch.{name}"),
        Some(mode),
        None,
    )?)
}

pub(crate) fn atleast_1d_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    if let Some(result) = dispatch_empty_variadic_tensor_input(py, "atleast_1d", args, kwargs)? {
        return Ok(result);
    }
    if args.len() != 1 || kwargs.is_some_and(|kwargs| !kwargs.is_empty()) {
        return Err(PyTypeError::new_err(
            "atleast_1d() only supports a single Tensor input",
        ));
    }

    let input = args.get_item(0)?;
    if input.is_instance_of::<PyTuple>() || input.is_instance_of::<PyList>() {
        return Err(PyTypeError::new_err(
            "atleast_1d() only supports a single Tensor input",
        ));
    }
    let Ok(tensor) = input.cast::<PyTensor>() else {
        let actual = python_type_name(&input)?;
        return Err(PyTypeError::new_err(format!(
            "atleast_1d() received an invalid combination of arguments - got ({actual}), but expected one of:\n * (Tensor input)\n      didn't match because some of the arguments have invalid types: (!{actual}!)\n * (tuple of Tensors tensors)\n      didn't match because some of the arguments have invalid types: (!{actual}!)\n"
        )));
    };

    let inner = {
        let tensor = tensor.try_borrow()?;
        if !tensor.inner.shape().is_empty() {
            return Ok(input.unbind());
        }
        tensor
            .inner
            .reshape([1])
            .map_err(|error| tensor_error(&error))?
    };
    Ok(Py::new(py, PyTensor::new(inner))?.into_any())
}

pub(crate) fn atleast_2d_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    if let Some(result) = dispatch_empty_variadic_tensor_input(py, "atleast_2d", args, kwargs)? {
        return Ok(result);
    }
    if args.len() != 1 || kwargs.is_some_and(|kwargs| !kwargs.is_empty()) {
        return Err(PyTypeError::new_err(
            "atleast_2d() only supports a single Tensor input",
        ));
    }

    let input = args.get_item(0)?;
    if input.is_instance_of::<PyTuple>() || input.is_instance_of::<PyList>() {
        return Err(PyTypeError::new_err(
            "atleast_2d() only supports a single Tensor input",
        ));
    }
    let Ok(tensor) = input.cast::<PyTensor>() else {
        let actual = python_type_name(&input)?;
        return Err(PyTypeError::new_err(format!(
            "atleast_2d() received an invalid combination of arguments - got ({actual}), but expected one of:\n * (Tensor input)\n      didn't match because some of the arguments have invalid types: (!{actual}!)\n * (tuple of Tensors tensors)\n      didn't match because some of the arguments have invalid types: (!{actual}!)\n"
        )));
    };

    let inner = {
        let tensor = tensor.try_borrow()?;
        match tensor.inner.shape().len() {
            0 => tensor.inner.reshape([1, 1]),
            1 => tensor.inner.unsqueeze_front(),
            _ => return Ok(input.unbind()),
        }
        .map_err(|error| tensor_error(&error))?
    };
    Ok(Py::new(py, PyTensor::new(inner))?.into_any())
}

pub(crate) fn atleast_3d_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    if let Some(result) = dispatch_empty_variadic_tensor_input(py, "atleast_3d", args, kwargs)? {
        return Ok(result);
    }
    if args.len() != 1 || kwargs.is_some_and(|kwargs| !kwargs.is_empty()) {
        return Err(PyTypeError::new_err(
            "atleast_3d() only supports a single Tensor input",
        ));
    }

    let input = args.get_item(0)?;
    if input.is_instance_of::<PyTuple>() || input.is_instance_of::<PyList>() {
        return Err(PyTypeError::new_err(
            "atleast_3d() only supports a single Tensor input",
        ));
    }
    let Ok(tensor) = input.cast::<PyTensor>() else {
        let actual = python_type_name(&input)?;
        return Err(PyTypeError::new_err(format!(
            "atleast_3d() received an invalid combination of arguments - got ({actual}), but expected one of:\n * (Tensor input)\n      didn't match because some of the arguments have invalid types: (!{actual}!)\n * (tuple of Tensors tensors)\n      didn't match because some of the arguments have invalid types: (!{actual}!)\n"
        )));
    };

    let inner = {
        let tensor = tensor.try_borrow()?;
        match tensor.inner.shape().len() {
            0 => tensor.inner.reshape([1, 1, 1]),
            1 => tensor
                .inner
                .unsqueeze_front()
                .and_then(|tensor| tensor.unsqueeze_back()),
            2 => tensor.inner.unsqueeze_back(),
            _ => return Ok(input.unbind()),
        }
        .map_err(|error| tensor_error(&error))?
    };
    Ok(Py::new(py, PyTensor::new(inner))?.into_any())
}

pub(crate) fn broadcast_tensors_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    if let Some(result) =
        dispatch_empty_variadic_tensor_input(py, "broadcast_tensors", args, kwargs)?
    {
        return Ok(result);
    }
    if args.len() != 1 || kwargs.is_some_and(|kwargs| !kwargs.is_empty()) {
        return Err(PyTypeError::new_err(BROADCAST_TENSORS_EXACT_TENSORS_ERROR));
    }

    let inputs = args.get_item(0)?;
    if !inputs.is_exact_instance_of::<PyTuple>() {
        return Err(PyTypeError::new_err(BROADCAST_TENSORS_EXACT_TENSORS_ERROR));
    }
    let inputs = inputs.cast::<PyTuple>()?;
    if inputs.is_empty() {
        return Ok(inputs.clone().into_any().unbind());
    }
    for input in inputs {
        if !input.is_exact_instance_of::<PyTensor>() {
            return Err(PyTypeError::new_err(BROADCAST_TENSORS_EXACT_TENSORS_ERROR));
        }
    }

    let first = inputs.get_item(0)?.cast_into::<PyTensor>()?.try_borrow()?;
    for input in inputs.iter().skip(1) {
        let input = input.cast_into::<PyTensor>()?.try_borrow()?;
        if !first.inner().is_same_size(input.inner()) {
            return Err(PyNotImplementedError::new_err(
                BROADCAST_TENSORS_EXPANSION_ERROR,
            ));
        }
    }

    Ok(inputs.clone().into_any().unbind())
}

pub(crate) fn adjoint_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = bind_legacy_single_tensor_or_override_argument("adjoint", args, kwargs)?;
    dispatch_adjoint(py, &input, args, kwargs)
}

pub(crate) fn positive_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = bind_legacy_single_tensor_or_override_argument("positive", args, kwargs)?;
    dispatch_single_tensor_override(
        SingleTensorOverrideOperation::POSITIVE,
        py,
        &input,
        args,
        kwargs,
    )
}

pub(crate) fn detach_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = bind_legacy_single_tensor_or_override_argument("detach", args, kwargs)?;
    dispatch_single_tensor_override(
        SingleTensorOverrideOperation::DETACH,
        py,
        &input,
        args,
        kwargs,
    )
}

pub(crate) fn ravel_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = bind_legacy_single_tensor_or_override_argument("ravel", args, kwargs)?;
    dispatch_single_tensor_override(
        SingleTensorOverrideOperation::RAVEL,
        py,
        &input,
        args,
        kwargs,
    )
}

pub(crate) fn reshape_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let ([input, shape], keyword_error) = bind_top_level_reshape_arguments(args, kwargs)?;
    let input = parse_tensor_or_torch_function_argument("reshape", "input", &input)?;
    let shape = bind_top_level_reshape_shape(&shape)?;
    if let Some(keyword_error) = keyword_error {
        return Err(keyword_error);
    }
    dispatch_top_level_reshape(py, &input, &shape, args, kwargs)
}

pub(crate) fn abs_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    unary_out_variable_function(UnaryOutOperation::ABS, py, args, kwargs)
}

pub(crate) fn absolute_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    unary_out_variable_function(UnaryOutOperation::ABSOLUTE, py, args, kwargs)
}

pub(crate) fn reciprocal_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    unary_out_variable_function(UnaryOutOperation::RECIPROCAL, py, args, kwargs)
}

pub(crate) fn rsqrt_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    unary_out_variable_function(UnaryOutOperation::RSQRT, py, args, kwargs)
}

pub(crate) fn log_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    unary_out_variable_function(UnaryOutOperation::LOG, py, args, kwargs)
}

pub(crate) fn exp_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    unary_out_variable_function(UnaryOutOperation::EXP, py, args, kwargs)
}

pub(crate) fn floor_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    unary_out_variable_function(UnaryOutOperation::FLOOR, py, args, kwargs)
}

pub(crate) fn ceil_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    unary_out_variable_function(UnaryOutOperation::CEIL, py, args, kwargs)
}

pub(crate) fn trunc_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    unary_out_variable_function(UnaryOutOperation::TRUNC, py, args, kwargs)
}

pub(crate) fn fix_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    unary_out_variable_function(UnaryOutOperation::FIX, py, args, kwargs)
}

pub(crate) fn neg_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    unary_out_variable_function(UnaryOutOperation::NEG, py, args, kwargs)
}

pub(crate) fn negative_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    unary_out_variable_function(UnaryOutOperation::NEGATIVE, py, args, kwargs)
}

pub(crate) fn sin_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    unary_out_variable_function(UnaryOutOperation::SIN, py, args, kwargs)
}

pub(crate) fn cos_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    unary_out_variable_function(UnaryOutOperation::COS, py, args, kwargs)
}

pub(crate) fn sqrt_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    unary_out_variable_function(UnaryOutOperation::SQRT, py, args, kwargs)
}

pub(crate) fn sigmoid_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    unary_out_variable_function(UnaryOutOperation::SIGMOID, py, args, kwargs)
}

pub(crate) fn square_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    unary_out_variable_function(UnaryOutOperation::SQUARE, py, args, kwargs)
}

pub(crate) fn sum_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let call = bind_top_level_sum_arguments(args, kwargs)?;
    dispatch_top_level_sum(py, &call, args, kwargs)
}

pub(crate) fn mean_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let call = bind_top_level_mean_arguments(args, kwargs)?;
    dispatch_top_level_mean(py, &call, args, kwargs)
}

pub(crate) fn tanh_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    unary_out_variable_function(UnaryOutOperation::TANH, py, args, kwargs)
}

fn unary_out_variable_function(
    operation: UnaryOutOperation,
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let call = bind_unary_out_arguments(operation, args, kwargs)?;
    dispatch_top_level_unary_out(operation, py, &call, args, kwargs)
}

pub(crate) fn is_conj_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = bind_legacy_single_tensor_or_override_argument("is_conj", args, kwargs)?;
    dispatch_is_conj(py, &input, args, kwargs)
}

pub(crate) fn is_inference_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = bind_legacy_single_tensor_or_override_argument("is_inference", args, kwargs)?;
    dispatch_is_inference(py, &input, args, kwargs)
}

pub(crate) fn conj_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = bind_legacy_single_tensor_or_override_argument("conj", args, kwargs)?;
    dispatch_single_tensor_override(
        SingleTensorOverrideOperation::CONJ,
        py,
        &input,
        args,
        kwargs,
    )
}

pub(crate) fn real_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = bind_legacy_single_tensor_or_override_argument("real", args, kwargs)?;
    dispatch_single_tensor_override(
        SingleTensorOverrideOperation::REAL,
        py,
        &input,
        args,
        kwargs,
    )
}

pub(crate) fn imag_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = bind_legacy_single_tensor_or_override_argument("imag", args, kwargs)?;
    dispatch_single_tensor_override(
        SingleTensorOverrideOperation::IMAG,
        py,
        &input,
        args,
        kwargs,
    )
}

pub(crate) fn resolve_conj_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = bind_legacy_single_tensor_or_override_argument("resolve_conj", args, kwargs)?;
    dispatch_single_tensor_override(
        SingleTensorOverrideOperation::RESOLVE_CONJ,
        py,
        &input,
        args,
        kwargs,
    )
}

pub(crate) fn resolve_neg_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = bind_legacy_single_tensor_or_override_argument("resolve_neg", args, kwargs)?;
    dispatch_single_tensor_override(
        SingleTensorOverrideOperation::RESOLVE_NEG,
        py,
        &input,
        args,
        kwargs,
    )
}

pub(crate) fn unbind_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (input, dimension) = bind_top_level_unbind_arguments(args, kwargs)?;
    dispatch_top_level_unbind(py, &input, dimension.as_ref(), args, kwargs)
}

pub(crate) fn unsqueeze_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let ([input, dimension], keyword_error) = bind_top_level_unsqueeze_arguments(args, kwargs)?;
    let input = parse_exact_native_unsqueeze_tensor_argument("input", &input)?;
    validate_dimension_swap_dimension("unsqueeze", "dim", dimension.position, &dimension.value)?;
    if let Some(keyword_error) = keyword_error {
        return Err(keyword_error);
    }
    if !torch_function_mode_stack::is_empty() {
        return Err(unsqueeze_torch_function_mode_error());
    }

    let dimension = extract_dimension_swap_dimension(&dimension.value)?;
    apply_unsqueeze(py, input, dimension)
}

pub(crate) fn select_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (input, [dimension, index], keyword_error) = bind_top_level_select_arguments(args, kwargs)?;
    if let Some(keyword_error) = keyword_error {
        return Err(keyword_error);
    }
    dispatch_top_level_select(py, &input, &dimension, &index, args, kwargs)
}

pub(crate) fn permute_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let ([input, dimensions], keyword_error) = bind_top_level_permute_arguments(args, kwargs)?;
    let input = parse_tensor_argument("permute", "input", &input)?;
    let Some(dimension_arguments) = permute_sequence_arguments(&dimensions.value) else {
        return Err(permute_argument_type_error(
            &dimensions.value,
            dimensions.position,
        )?);
    };
    validate_permute_sequence_first(&dimension_arguments, &dimensions.value, dimensions.position)?;
    if let Some(keyword_error) = keyword_error {
        return Err(keyword_error);
    }
    let dimensions = parse_permute_dimension_arguments(dimension_arguments)?;
    let tensor = input.try_borrow()?;
    Ok(Bound::new(
        py,
        permute_tensor(&tensor.inner, dimensions).map(PyTensor::new)?,
    )?
    .into_any()
    .unbind())
}

pub(crate) fn movedim_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    dimension_move_variable_function(DimensionMoveOperation::Movedim, py, args, kwargs)
}

pub(crate) fn moveaxis_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    dimension_move_variable_function(DimensionMoveOperation::Moveaxis, py, args, kwargs)
}

fn tensor_shape_as_i64(tensor: &Bound<'_, PyTensor>) -> PyResult<Vec<i64>> {
    let tensor = tensor.try_borrow()?;
    let mut shape = try_size_vector(tensor.inner.shape().len())?;
    for &dimension in tensor.inner.shape() {
        let dimension = i64::try_from(dimension).map_err(|_| {
            PyOverflowError::new_err("tensor dimension exceeds the signed 64-bit shape limit")
        })?;
        try_push_size(&mut shape, dimension)?;
    }
    Ok(shape)
}

fn dimension_move_tensor_method(
    operation: DimensionMoveOperation,
    slf: &Bound<'_, PyTensorBase>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let [source, destination] = bind_movedim_arguments(operation, args, kwargs)?;
    let tensor = slf.as_any().cast::<PyTensor>()?;
    if let Some(result) = dispatch_tensorbase_method_mode(
        slf.py(),
        tensor,
        operation.name(),
        operation.tensor_qualified_name(),
        args,
        kwargs,
    )? {
        return Ok(result);
    }

    let [source, destination] = parse_dimension_swap_dimensions(
        operation.name(),
        ["source", "destination"],
        &source,
        &destination,
    )?;
    let inner = movedim_tensor(&tensor.try_borrow()?.inner, source, destination)?;
    Ok(Py::new(slf.py(), PyTensor::new(inner))?.into_any())
}

fn dimension_move_variable_function(
    operation: DimensionMoveOperation,
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (input, [source, destination]) = bind_top_level_movedim_arguments(operation, args, kwargs)?;
    dispatch_top_level_movedim(operation, py, &input, &source, &destination, args, kwargs)
}

pub(crate) fn matmul_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let ([input, other], keyword_error) = bind_legacy_binary_arguments(
        "matmul",
        args,
        kwargs,
        LegacyBinaryInputKind::TensorOrTorchFunction,
    )?;
    let input = parse_tensor_or_torch_function_argument("matmul", "input", &input)?;
    let other = parse_tensor_or_torch_function_argument("matmul", "other", &other)?;
    if let Some(keyword_error) = keyword_error {
        return Err(keyword_error);
    }
    dispatch_top_level_matmul(py, &input, &other, args, kwargs)
}

pub(crate) fn mm_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (arguments, out, keyword_error) = bind_top_level_mm_arguments(args, kwargs)?;
    let input = parse_exact_native_mm_tensor_or_torch_function_argument("input", &arguments[0])?;
    let mat2 = parse_exact_native_mm_tensor_or_torch_function_argument("mat2", &arguments[1])?;
    let out = parse_top_level_mm_out(out)?;
    if let Some(keyword_error) = keyword_error {
        return Err(keyword_error);
    }
    let call = BoundTopLevelMmCall { input, mat2, out };
    dispatch_top_level_mm(py, &call, args, kwargs)
}

pub(crate) fn mul_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    multiplication_variable_function(MultiplicationOperation::Mul, py, args, kwargs)
}

pub(crate) fn multiply_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    multiplication_variable_function(MultiplicationOperation::Multiply, py, args, kwargs)
}

pub(crate) fn add_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (arguments, alpha, out, keyword_error) = bind_top_level_add_arguments(args, kwargs)?;
    let input = parse_top_level_add_operand("input", &arguments[0])?;
    let other = parse_top_level_add_operand("other", &arguments[1])?;
    let alpha = parse_top_level_add_alpha(alpha.as_ref())?;
    let out = parse_top_level_add_out(out)?;
    if let Some(keyword_error) = keyword_error {
        return Err(keyword_error);
    }
    let call = BoundTopLevelAdditionCall {
        input,
        other,
        alpha,
        out,
    };
    dispatch_top_level_addition(py, &call, args, kwargs)
}

pub(crate) fn div_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    division_variable_function(DivisionOperation::Div, py, args, kwargs)
}

pub(crate) fn divide_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    division_variable_function(DivisionOperation::Divide, py, args, kwargs)
}

pub(crate) fn sub_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    subtraction_variable_function(SubtractionOperation::Sub, py, args, kwargs)
}

pub(crate) fn subtract_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    subtraction_variable_function(SubtractionOperation::Subtract, py, args, kwargs)
}

pub(crate) fn can_cast_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    dtype_binary_variable_function(DTypeBinaryOperation::CanCast, py, args, kwargs)
}

pub(crate) fn promote_types_variable_function(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    dtype_binary_variable_function(DTypeBinaryOperation::PromoteTypes, py, args, kwargs)
}

fn dtype_binary_variable_function(
    operation: DTypeBinaryOperation,
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let ([first, second], consumed_keywords) =
        bind_dtype_binary_arguments(operation, args, kwargs)?;
    validate_dtype_binary_keywords(operation, args.len(), kwargs, &consumed_keywords)?;
    dispatch_dtype_binary(operation, py, &first, &second, args, kwargs)
}

fn multiplication_variable_function(
    operation: MultiplicationOperation,
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let ([input, other], keyword_error) = bind_legacy_binary_arguments(
        operation.name(),
        args,
        kwargs,
        LegacyBinaryInputKind::Multiplication(operation),
    )?;
    let input = parse_top_level_multiplication_operand(operation, "input", &input, args, kwargs)?;
    let other = parse_top_level_multiplication_operand(operation, "other", &other, args, kwargs)?;
    if let Some(keyword_error) = keyword_error {
        return Err(keyword_error);
    }
    dispatch_top_level_multiplication(operation, py, &input, &other, args, kwargs)
}

fn division_variable_function(
    operation: DivisionOperation,
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (arguments, rounding_mode, out, keyword_error) =
        bind_top_level_division_arguments(operation, args, kwargs)?;
    let input = parse_top_level_division_input(operation, &arguments[0], args, kwargs)?;
    let other = parse_top_level_division_other(operation, &arguments[1], args, kwargs)?;
    let rounding_mode =
        parse_top_level_division_rounding_mode(operation, rounding_mode.as_ref(), args, kwargs)?;
    let out = parse_top_level_division_out(operation, out, args, kwargs)?;
    if let Some(keyword_error) = keyword_error {
        return Err(keyword_error);
    }

    let call = BoundTopLevelDivisionCall {
        input,
        other,
        rounding_mode,
        out,
    };
    dispatch_top_level_division(operation, py, &call, args, kwargs)
}

fn subtraction_variable_function(
    operation: SubtractionOperation,
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (arguments, alpha, out, keyword_error) =
        bind_top_level_subtraction_arguments(operation, args, kwargs)?;
    let alpha_is_positional = alpha
        .as_ref()
        .is_some_and(|alpha| alpha.position == Some(3));
    let input = parse_top_level_subtraction_input(operation, &arguments[0], args, kwargs)?;
    let other = parse_top_level_subtraction_other(operation, &arguments[1], args, kwargs)?;
    let alpha = parse_top_level_subtraction_alpha(operation, alpha.as_ref(), args, kwargs)?;
    let out = parse_top_level_subtraction_out(operation, out, args, kwargs)?;
    if let Some(keyword_error) = keyword_error {
        return Err(keyword_error);
    }
    let call = BoundTopLevelSubtractionCall {
        input,
        other,
        alpha,
        out,
        alpha_is_positional,
    };
    dispatch_top_level_subtraction(operation, py, &call, args, kwargs)
}

enum ParsedFillValue {
    Float(f64),
    SignedInteger(i64),
    UnsignedInteger(u64),
    TensorScalar(f32),
}

enum ParsedArithmeticScalar {
    PythonBool(bool),
    Number(ParsedFillValue),
    WideNumpyUnsigned,
}

#[derive(Clone, Copy)]
struct StrictBool(bool);

impl<'a, 'py> FromPyObject<'a, 'py> for StrictBool {
    type Error = PyErr;

    fn extract(object: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        parse_requires_grad("tensor", &object.to_owned()).map(Self)
    }
}

pub(crate) struct ParsedCallArgument<'py> {
    pub(crate) value: Bound<'py, PyAny>,
    position: Option<usize>,
}

struct ConsumedDTypeKeyword<'py> {
    key: Bound<'py, PyAny>,
    position: usize,
}

#[derive(Clone)]
struct ProbedTorchFunctionOverride<'py> {
    receiver: Bound<'py, PyAny>,
    dispatch_type: Bound<'py, PyAny>,
    precedence_type: Bound<'py, PyType>,
}

enum BoundTensorOrTorchFunction<'py> {
    Tensor(Bound<'py, PyTensor>),
    Override(ProbedTorchFunctionOverride<'py>),
}

struct BoundTopLevelNativeReshapeShape<'py> {
    dimensions: Vec<Bound<'py, PyAny>>,
}

enum BoundTopLevelReshapeShape<'py> {
    Native(BoundTopLevelNativeReshapeShape<'py>),
    Override(Vec<ProbedTorchFunctionOverride<'py>>),
}

type SingleTensorNativeCallback = fn(Python<'_>, &Bound<'_, PyTensor>) -> PyResult<Py<PyAny>>;

#[derive(Clone, Copy)]
struct SingleTensorOverrideOperation {
    name: &'static str,
    qualified_name: &'static str,
    apply_native: SingleTensorNativeCallback,
}

impl SingleTensorOverrideOperation {
    const POSITIVE: Self = Self {
        name: "positive",
        qualified_name: "torch.positive",
        apply_native: apply_top_level_positive,
    };

    const RAVEL: Self = Self {
        name: "ravel",
        qualified_name: "torch.ravel",
        apply_native: apply_top_level_ravel,
    };

    const DETACH: Self = Self {
        name: "detach",
        qualified_name: "torch.detach",
        apply_native: apply_top_level_detach,
    };

    const RESOLVE_CONJ: Self = Self {
        name: "resolve_conj",
        qualified_name: "torch.resolve_conj",
        apply_native: apply_top_level_lazy_bit_identity,
    };

    const RESOLVE_NEG: Self = Self {
        name: "resolve_neg",
        qualified_name: "torch.resolve_neg",
        apply_native: apply_top_level_lazy_bit_identity,
    };

    const CONJ: Self = Self {
        name: "conj",
        qualified_name: "torch.conj",
        apply_native: apply_top_level_lazy_bit_identity,
    };

    const REAL: Self = Self {
        name: "real",
        qualified_name: "torch.real",
        apply_native: apply_top_level_real,
    };

    const IMAG: Self = Self {
        name: "imag",
        qualified_name: "torch.imag",
        apply_native: apply_top_level_imag,
    };
}

type UnaryOutApplication = fn(&CoreTensor) -> Result<CoreTensor, TensorError>;

#[derive(Clone, Copy)]
struct UnaryOutOperation {
    name: &'static str,
    qualified_name: &'static str,
    dispatch_allocation_error: &'static str,
    out_unsupported_error: &'static str,
    apply: UnaryOutApplication,
}

impl UnaryOutOperation {
    const ABS: Self = Self {
        name: "abs",
        qualified_name: "torch.abs",
        dispatch_allocation_error: "unable to allocate abs dispatch operands",
        out_unsupported_error: "abs(): the 'out' argument is not supported",
        apply: CoreTensor::abs,
    };

    const ABSOLUTE: Self = Self {
        name: "absolute",
        qualified_name: "torch.absolute",
        dispatch_allocation_error: "unable to allocate absolute dispatch operands",
        out_unsupported_error: "absolute(): the 'out' argument is not supported",
        apply: CoreTensor::abs,
    };

    const NEG: Self = Self {
        name: "neg",
        qualified_name: "torch.neg",
        dispatch_allocation_error: "unable to allocate neg dispatch operands",
        out_unsupported_error: "neg(): the 'out' argument is not supported",
        apply: CoreTensor::negate,
    };

    const NEGATIVE: Self = Self {
        name: "negative",
        qualified_name: "torch.negative",
        dispatch_allocation_error: "unable to allocate negative dispatch operands",
        out_unsupported_error: "negative(): the 'out' argument is not supported",
        apply: CoreTensor::negate,
    };

    const EXP: Self = Self {
        name: "exp",
        qualified_name: "torch.exp",
        dispatch_allocation_error: "unable to allocate exp dispatch operands",
        out_unsupported_error: "exp(): the 'out' argument is not supported",
        apply: CoreTensor::exp,
    };

    const FLOOR: Self = Self {
        name: "floor",
        qualified_name: "torch.floor",
        dispatch_allocation_error: "unable to allocate floor dispatch operands",
        out_unsupported_error: "floor(): the 'out' argument is not supported",
        apply: CoreTensor::floor,
    };

    const CEIL: Self = Self {
        name: "ceil",
        qualified_name: "torch.ceil",
        dispatch_allocation_error: "unable to allocate ceil dispatch operands",
        out_unsupported_error: "ceil(): the 'out' argument is not supported",
        apply: CoreTensor::ceil,
    };

    const TRUNC: Self = Self {
        name: "trunc",
        qualified_name: "torch.trunc",
        dispatch_allocation_error: "unable to allocate trunc dispatch operands",
        out_unsupported_error: "trunc(): the 'out' argument is not supported",
        apply: CoreTensor::trunc,
    };

    const FIX: Self = Self {
        name: "fix",
        qualified_name: "torch.fix",
        dispatch_allocation_error: "unable to allocate fix dispatch operands",
        out_unsupported_error: "fix(): the 'out' argument is not supported",
        apply: CoreTensor::trunc,
    };

    const RECIPROCAL: Self = Self {
        name: "reciprocal",
        qualified_name: "torch.reciprocal",
        dispatch_allocation_error: "unable to allocate reciprocal dispatch operands",
        out_unsupported_error: "reciprocal(): the 'out' argument is not supported",
        apply: CoreTensor::reciprocal,
    };

    const RSQRT: Self = Self {
        name: "rsqrt",
        qualified_name: "torch.rsqrt",
        dispatch_allocation_error: "unable to allocate rsqrt dispatch operands",
        out_unsupported_error: "rsqrt(): the 'out' argument is not supported",
        apply: CoreTensor::rsqrt,
    };

    const LOG: Self = Self {
        name: "log",
        qualified_name: "torch.log",
        dispatch_allocation_error: "unable to allocate log dispatch operands",
        out_unsupported_error: "log(): the 'out' argument is not supported",
        apply: CoreTensor::log,
    };

    const SIN: Self = Self {
        name: "sin",
        qualified_name: "torch.sin",
        dispatch_allocation_error: "unable to allocate sin dispatch operands",
        out_unsupported_error: "sin(): the 'out' argument is not supported",
        apply: CoreTensor::sin,
    };

    const COS: Self = Self {
        name: "cos",
        qualified_name: "torch.cos",
        dispatch_allocation_error: "unable to allocate cos dispatch operands",
        out_unsupported_error: "cos(): the 'out' argument is not supported",
        apply: CoreTensor::cos,
    };

    const SQRT: Self = Self {
        name: "sqrt",
        qualified_name: "torch.sqrt",
        dispatch_allocation_error: "unable to allocate sqrt dispatch operands",
        out_unsupported_error: "sqrt(): the 'out' argument is not supported",
        apply: CoreTensor::sqrt,
    };

    const SIGMOID: Self = Self {
        name: "sigmoid",
        qualified_name: "torch.sigmoid",
        dispatch_allocation_error: "unable to allocate sigmoid dispatch operands",
        out_unsupported_error: "sigmoid(): the 'out' argument is not supported",
        apply: CoreTensor::sigmoid,
    };

    const SQUARE: Self = Self {
        name: "square",
        qualified_name: "torch.square",
        dispatch_allocation_error: "unable to allocate square dispatch operands",
        out_unsupported_error: "square(): the 'out' argument is not supported",
        apply: CoreTensor::square,
    };

    const TANH: Self = Self {
        name: "tanh",
        qualified_name: "torch.tanh",
        dispatch_allocation_error: "unable to allocate tanh dispatch operands",
        out_unsupported_error: "tanh(): the 'out' argument is not supported",
        apply: CoreTensor::tanh,
    };
}

struct BoundUnaryOutCall<'py> {
    input: BoundTensorOrTorchFunction<'py>,
    out: Option<BoundTensorOrTorchFunction<'py>>,
}

enum BoundTopLevelSumDType<'py> {
    Native,
    Override(ProbedTorchFunctionOverride<'py>),
}

struct BoundTopLevelSumCall<'py> {
    input: BoundTensorOrTorchFunction<'py>,
    dtype: BoundTopLevelSumDType<'py>,
    out: Option<BoundTensorOrTorchFunction<'py>>,
    full_reduction: bool,
    keepdim_full_reduction: bool,
}

enum BoundTopLevelMeanDType<'py> {
    Native,
    Override(ProbedTorchFunctionOverride<'py>),
}

struct BoundTopLevelMeanCall<'py> {
    input: BoundTensorOrTorchFunction<'py>,
    dtype: BoundTopLevelMeanDType<'py>,
    out: Option<BoundTensorOrTorchFunction<'py>>,
    full_reduction: bool,
    keepdim_full_reduction: bool,
}

struct BoundTopLevelMmCall<'py> {
    input: BoundTensorOrTorchFunction<'py>,
    mat2: BoundTensorOrTorchFunction<'py>,
    out: Option<BoundTensorOrTorchFunction<'py>>,
}

struct BoundMethodReductionCall {
    keepdim_full_reduction: bool,
}

struct BoundTopLevelSubtractionCall<'py> {
    input: BoundSubOperand<'py>,
    other: BoundSubOperand<'py>,
    alpha: BoundSubAlpha<'py>,
    out: Option<BoundTensorOrTorchFunction<'py>>,
    alpha_is_positional: bool,
}

struct BoundTopLevelAdditionCall<'py> {
    input: BoundAddOperand<'py>,
    other: BoundAddOperand<'py>,
    alpha: BoundSubAlpha<'py>,
    out: Option<BoundTensorOrTorchFunction<'py>>,
}

struct BoundTopLevelDivisionCall<'py> {
    input: BoundDivOperand<'py>,
    other: BoundDivOperand<'py>,
    rounding_mode: BoundDivisionRoundingMode<'py>,
    out: Option<BoundTensorOrTorchFunction<'py>>,
}

struct BoundTensorMethodDivisionCall<'py> {
    input: BoundDivOperand<'py>,
    other: BoundDivOperand<'py>,
    rounding_mode: BoundDivisionRoundingMode<'py>,
}

type BoundTopLevelAdditionArguments<'py> = (
    [ParsedCallArgument<'py>; 2],
    Option<ParsedCallArgument<'py>>,
    Option<ParsedCallArgument<'py>>,
    Option<PyErr>,
);

type BoundTopLevelSubtractionArguments<'py> = (
    [ParsedCallArgument<'py>; 2],
    Option<ParsedCallArgument<'py>>,
    Option<ParsedCallArgument<'py>>,
    Option<PyErr>,
);

type BoundTopLevelDivisionArguments<'py> = (
    [ParsedCallArgument<'py>; 2],
    Option<ParsedCallArgument<'py>>,
    Option<ParsedCallArgument<'py>>,
    Option<PyErr>,
);

type BoundTopLevelMmArguments<'py> = (
    [ParsedCallArgument<'py>; 2],
    Option<ParsedCallArgument<'py>>,
    Option<PyErr>,
);

type BoundTensorMethodDivisionArguments<'py> = (
    ParsedCallArgument<'py>,
    Option<ParsedCallArgument<'py>>,
    Option<PyErr>,
);

struct BoundTensorMethodAddSubCall<'py> {
    input: BoundSubOperand<'py>,
    other: BoundSubOperand<'py>,
    alpha: BoundSubAlpha<'py>,
    alpha_precedes_other: bool,
}

type BoundTensorMethodAddSubArguments<'py> = (
    ParsedCallArgument<'py>,
    Option<ParsedCallArgument<'py>>,
    Option<PyErr>,
);

type BoundTensorMethodSubtractionArguments<'py> = BoundTensorMethodAddSubArguments<'py>;

enum BoundSubOperand<'py> {
    Tensor(Bound<'py, PyTensor>),
    Scalar(Bound<'py, PyAny>),
    Override(ProbedTorchFunctionOverride<'py>),
}

enum BoundSubAlpha<'py> {
    Default,
    PythonBool,
    NonDefault,
    Override(ProbedTorchFunctionOverride<'py>),
}

enum BoundAddOperand<'py> {
    Tensor(Bound<'py, PyTensor>),
    Scalar(Bound<'py, PyAny>),
    Override(ProbedTorchFunctionOverride<'py>),
}

enum BoundDivOperand<'py> {
    Tensor(Bound<'py, PyTensor>),
    Scalar(Bound<'py, PyAny>),
    Override(ProbedTorchFunctionOverride<'py>),
}

enum BoundDivisionRoundingMode<'py> {
    TrueDivision,
    Rounding,
    Override(ProbedTorchFunctionOverride<'py>),
}

enum BoundMulOperand<'py> {
    Tensor(Bound<'py, PyTensor>),
    Scalar(Bound<'py, PyAny>),
    Override(ProbedTorchFunctionOverride<'py>),
}

enum BoundDTypeOperand<'py> {
    DType(DType),
    Override(ProbedTorchFunctionOverride<'py>),
}

#[derive(Clone, Copy)]
enum DTypeBinaryOperation {
    CanCast,
    PromoteTypes,
}

impl DTypeBinaryOperation {
    const fn name(self) -> &'static str {
        match self {
            Self::CanCast => "can_cast",
            Self::PromoteTypes => "promote_types",
        }
    }

    const fn argument_names(self) -> [&'static str; 2] {
        match self {
            Self::CanCast => ["from_", "to"],
            Self::PromoteTypes => ["type1", "type2"],
        }
    }

    const fn qualified_name(self) -> &'static str {
        match self {
            Self::CanCast => "torch.can_cast",
            Self::PromoteTypes => "torch.promote_types",
        }
    }

    const fn dispatch_allocation_error(self) -> &'static str {
        match self {
            Self::CanCast => "unable to allocate can_cast dispatch operands",
            Self::PromoteTypes => "unable to allocate promote_types dispatch operands",
        }
    }
}

fn matrix_adjoint(
    py: Python<'_>,
    tensor: &Bound<'_, PyTensor>,
    scalar_warning_emitted: &AtomicBool,
    scalar_warning: &CStr,
    vector_error: &'static str,
) -> PyResult<Py<PyAny>> {
    match tensor.try_borrow()?.inner.shape().len() {
        0 => {
            warn_once(py, scalar_warning_emitted, scalar_warning)?;
            Ok(tensor.clone().unbind().into_any())
        }
        1 => Err(PyRuntimeError::new_err(vector_error)),
        _ => {
            // Float32 is real-valued, so conjugation is an identity. H, mH,
            // and adjoint() therefore share this exact checked final-two-axis
            // transpose view and its autograd history.
            let inner = tensor
                .try_borrow()?
                .inner
                .matrix_transpose()
                .map_err(|error| transpose_error(&error))?;
            Ok(Py::new(py, PyTensor::new(inner))?.into_any())
        }
    }
}

#[derive(Clone, Copy)]
enum TensorBaseModeTarget {
    Method(&'static str),
    GetSet(&'static str),
}

#[allow(
    unsafe_code,
    reason = "PyTorch binding parity requires CPython's exception-suppressing legacy attribute probe"
)]
fn probe_torch_function_override<'py>(
    value: &Bound<'py, PyAny>,
) -> Option<ProbedTorchFunctionOverride<'py>> {
    // PyTorch's argument parser uses the legacy, exception-suppressing
    // attribute probe here. If the initial probe fails, its tensor-type
    // fallback retries once before rejecting the input. The actual callable is
    // deliberately resolved only after the active mode has declined, so keep
    // only the dispatch type metadata from this lookup.
    let handler = probe_torch_function_handler(value, true)?;
    if is_disabled_torch_function_handler(&handler) {
        return None;
    }
    Some(probed_torch_function_override(value))
}

#[allow(
    unsafe_code,
    reason = "dtype argument parity requires CPython's one-shot exception-suppressing attribute probe"
)]
fn probe_dtype_torch_function_override<'py>(
    value: &Bound<'py, PyAny>,
) -> Option<ProbedTorchFunctionOverride<'py>> {
    // Unlike tensor arguments, PyTorch's dtype parser does not retry a failed
    // __torch_function__ lookup through a tensor-type fallback.
    let handler = probe_torch_function_handler(value, false)?;
    if is_disabled_torch_function_handler(&handler) {
        return None;
    }
    Some(probed_torch_function_override(value))
}

#[allow(
    unsafe_code,
    reason = "PyTorch suppresses errors while probing the __torch_function__ descriptor"
)]
fn probe_torch_function_handler<'py>(
    value: &Bound<'py, PyAny>,
    retry_failed_lookup: bool,
) -> Option<Bound<'py, PyAny>> {
    for attempt in 0..=usize::from(retry_failed_lookup) {
        // SAFETY: `value` is live for this call and the attribute name is a
        // static, NUL-terminated string. A non-null result is a new reference.
        let handler =
            unsafe { ffi::PyObject_GetAttrString(value.as_ptr(), c"__torch_function__".as_ptr()) };
        if !handler.is_null() {
            // SAFETY: PyObject_GetAttrString returned a new owned reference.
            return Some(unsafe { Bound::<PyAny>::from_owned_ptr(value.py(), handler) });
        }
        // SAFETY: the GIL is held and PyTorch's legacy probe suppresses
        // descriptor lookup failures.
        unsafe { ffi::PyErr_Clear() };
        if attempt == 0 && !retry_failed_lookup {
            break;
        }
    }
    None
}

fn probed_torch_function_override<'py>(
    value: &Bound<'py, PyAny>,
) -> ProbedTorchFunctionOverride<'py> {
    let precedence_type = value.get_type();
    let dispatch_type = if value.cast::<PyType>().is_ok() {
        value.clone()
    } else {
        precedence_type.clone().into_any()
    };
    ProbedTorchFunctionOverride {
        receiver: value.clone(),
        dispatch_type,
        precedence_type,
    }
}

#[allow(
    unsafe_code,
    reason = "PyTorch suppresses errors while checking a handler's __self__ identity"
)]
fn has_receiver_as_self(handler: &Bound<'_, PyAny>, receiver: &Bound<'_, PyAny>) -> bool {
    // SAFETY: `handler` is live for the call and the attribute name is a
    // static, NUL-terminated string. A non-null result is a new reference.
    let handler_self =
        unsafe { ffi::PyObject_GetAttrString(handler.as_ptr(), c"__self__".as_ptr()) };
    if handler_self.is_null() {
        // PyTorch treats every lookup failure as a non-matching `__self__`.
        // SAFETY: clearing the current Python exception is valid while the GIL
        // is held, including if a broken descriptor returned null without one.
        unsafe { ffi::PyErr_Clear() };
        return false;
    }
    // SAFETY: PyObject_GetAttrString returned a new owned reference above.
    let handler_self = unsafe { Bound::<PyAny>::from_owned_ptr(handler.py(), handler_self) };
    handler_self.is(receiver)
}

fn resolve_torch_function_override<'py>(
    py: Python<'py>,
    probed: &ProbedTorchFunctionOverride<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let handler = probed.receiver.getattr("__torch_function__")?;
    if has_receiver_as_self(&handler, &probed.receiver) {
        warn_once(
            py,
            &TORCH_FUNCTION_PLAIN_METHOD_WARNING_EMITTED,
            TORCH_FUNCTION_PLAIN_METHOD_WARNING,
        )?;
    }
    Ok(handler)
}

fn validate_torch_function_mode_handler(mode: &Bound<'_, PyAny>) -> PyResult<()> {
    let handler = mode.getattr("__torch_function__")?;
    if !has_receiver_as_self(&handler, mode) {
        return Err(PyRuntimeError::new_err(
            "Defining your mode's `__torch_function__` as a classmethod is not supported, please make it a plain method",
        ));
    }
    Ok(())
}

fn call_torch_function_handler(
    py: Python<'_>,
    handler: &Bound<'_, PyAny>,
    function: &Py<PyAny>,
    types: &Bound<'_, PyTuple>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let result = if let Some(kwargs) = kwargs {
        handler.call1((
            function.clone_ref(py),
            types.clone(),
            args.clone(),
            kwargs.clone(),
        ))?
    } else {
        handler.call1((function.clone_ref(py), types.clone(), args.clone()))?
    };
    Ok(result.unbind())
}

fn is_not_implemented(py: Python<'_>, result: &Py<PyAny>) -> bool {
    result.as_ptr() == py.NotImplemented().as_ptr()
}

fn dispatch_tensorbase_mode(
    py: Python<'_>,
    tensor: &Bound<'_, PyTensor>,
    target: TensorBaseModeTarget,
) -> PyResult<Option<Py<PyAny>>> {
    if torch_function_mode_stack::is_empty() {
        return Ok(None);
    }

    let tensor_base = py.get_type::<PyTensorBase>();
    let function = match target {
        TensorBaseModeTarget::Method(name) => tensor_base.getattr(name)?.unbind(),
        TensorBaseModeTarget::GetSet(name) => {
            tensor_base.getattr(name)?.getattr("__get__")?.unbind()
        }
    };
    let types = PyTuple::new(py, [tensor.get_type().into_any()])?;
    let args = PyTuple::new(py, [tensor.clone().into_any()])?;

    let mut active_mode = torch_function_mode_stack::pop();
    let Some(mode) = active_mode.get() else {
        return Ok(None);
    };
    validate_torch_function_mode_handler(mode.bind(py))?;
    let result = cpython_compat::call_torch_function_mode_handler(
        py,
        mode.bind(py),
        &function,
        &types,
        &args,
    )?;
    if !is_not_implemented(py, &result) {
        return Ok(Some(result));
    }
    let legacy_no_argument_method = cpython_compat::uses_legacy_tensorbase_redispatch(py)
        && matches!(
            target,
            TensorBaseModeTarget::Method(
                "abs"
                    | "absolute"
                    | "ceil"
                    | "const_data_ptr"
                    | "cos"
                    | "exp"
                    | "fix"
                    | "floor"
                    | "log"
                    | "reciprocal"
                    | "rsqrt"
                    | "sigmoid"
                    | "sin"
                    | "sqrt"
                    | "square"
                    | "tanh"
                    | "trunc"
            )
        );
    if legacy_no_argument_method {
        cpython_compat::probe_tensorbase_legacy_redispatch(py)?;
    }

    // TensorBase's fallback retries the descriptor after restoring the active
    // mode. That intentionally re-enters a declining top mode, matching
    // PyTorch's recursion behavior for a mode returning NotImplemented. A mode
    // that wants to reach the next mode instead calls `func(*args)` itself
    // while the current mode is disabled.
    active_mode.restore();
    // Keep the retry in a Python frame so configured `sys.setrecursionlimit`
    // values and mode side effects match TensorBase's recursive fallback.
    let caller = cpython_compat::torch_function_descriptor_caller(py)?;
    let _redispatch_depth =
        legacy_no_argument_method.then(cpython_compat::enter_tensorbase_legacy_redispatch);
    Ok(Some(caller.bind(py).call1((function, args))?.unbind()))
}

pub(crate) fn dispatch_tensorbase_getset_mode(
    py: Python<'_>,
    tensor: &Bound<'_, PyTensor>,
    property: &'static str,
) -> PyResult<Option<Py<PyAny>>> {
    dispatch_tensorbase_mode(py, tensor, TensorBaseModeTarget::GetSet(property))
}

pub(crate) fn dispatch_tensorbase_no_argument_mode(
    py: Python<'_>,
    tensor: &Bound<'_, PyTensor>,
    method: &'static str,
) -> PyResult<Option<Py<PyAny>>> {
    dispatch_tensorbase_mode(py, tensor, TensorBaseModeTarget::Method(method))
}

fn unbind_dimension(
    py: Python<'_>,
    tensor: &Bound<'_, PyTensor>,
    dimension: i64,
) -> PyResult<Py<PyAny>> {
    let axis = normalize_unbind_dimension(dimension, tensor.try_borrow()?.inner.shape().len())?;
    let outputs = tensor
        .try_borrow()?
        .inner
        .unbind_dimension(axis)
        .map_err(|error| tensor_error(&error))?;
    Ok(PyTuple::new(py, outputs.into_iter().map(PyTensor::new))?
        .into_any()
        .unbind())
}

fn apply_unsqueeze(
    py: Python<'_>,
    tensor: &Bound<'_, PyTensor>,
    dimension: i64,
) -> PyResult<Py<PyAny>> {
    let inner = {
        let tensor = tensor.try_borrow()?;
        let rank = tensor.inner.shape().len();
        let axis = normalize_unsqueeze_dimension(dimension, rank)?;
        tensor
            .inner
            .unsqueeze_axis(axis)
            .map_err(|error| tensor_error(&error))?
    };
    Ok(Py::new(py, PyTensor::new(inner))?.into_any())
}

fn normalize_unsqueeze_dimension(dimension: i64, rank: usize) -> PyResult<usize> {
    let insertion_slots = rank
        .checked_add(1)
        .ok_or_else(|| PyOverflowError::new_err("tensor rank exceeds the platform limit"))?;
    let signed_slots = i64::try_from(insertion_slots)
        .map_err(|_| PyOverflowError::new_err("tensor rank exceeds the platform limit"))?;
    if dimension < -signed_slots || dimension >= signed_slots {
        return Err(PyIndexError::new_err(format!(
            "Dimension out of range (expected to be in range of [{}, {}], but got {dimension})",
            -signed_slots,
            signed_slots - 1
        )));
    }

    usize::try_from(if dimension < 0 {
        dimension + signed_slots
    } else {
        dimension
    })
    .map_err(|_| PyOverflowError::new_err("tensor dimension exceeds the platform limit"))
}

fn unsqueeze_unsupported_native_input() -> PyErr {
    PyNotImplementedError::new_err(
        "unsqueeze(): only exact native CPU float32 Tensor inputs are supported",
    )
}

fn unsqueeze_torch_function_mode_error() -> PyErr {
    PyNotImplementedError::new_err("unsqueeze(): __torch_function__ modes are not supported")
}

fn select_dimension(
    py: Python<'_>,
    tensor: &Bound<'_, PyTensor>,
    dimension: i64,
    index: i64,
) -> PyResult<Py<PyAny>> {
    let tensor = tensor.try_borrow()?;
    let shape = tensor.inner.shape();
    if shape.is_empty() {
        return Err(PyIndexError::new_err(
            "select() cannot be applied to a 0-dim tensor.",
        ));
    }
    let axis = normalize_dimension(dimension, shape.len())?;

    let inner = tensor.inner.select_dimension(axis, index).map_err(|error| {
        if let TensorError::IndexOutOfBounds {
            index, dimension, ..
        } = &error
        {
            PyIndexError::new_err(format!(
                "select(): index {index} out of range for tensor of size {shape:?} at dimension {dimension}"
            ))
        } else {
            tensor_error(&error)
        }
    })?;
    Ok(Py::new(py, PyTensor::new(inner))?.into_any())
}

fn dispatch_top_level_unbind(
    py: Python<'_>,
    input: &BoundTensorOrTorchFunction<'_>,
    dimension: Option<&ParsedCallArgument<'_>>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    if torch_function_mode_stack::is_empty()
        && let BoundTensorOrTorchFunction::Tensor(tensor) = input
    {
        let dimension = dimension.map_or(Ok(0), |dimension| {
            extract_dimension_swap_dimension(&dimension.value)
        })?;
        return unbind_dimension(py, tensor, dimension);
    }

    let function = variable_function(py, "unbind")?;
    let types = match input {
        BoundTensorOrTorchFunction::Tensor(_) => PyTuple::empty(py),
        BoundTensorOrTorchFunction::Override(probed) => {
            PyTuple::new(py, [probed.dispatch_type.clone()])?
        }
    };

    // Integer conversion and dimension range checks remain deferred until
    // every active torch-function handler has had an opportunity to replace
    // the valid generated call.
    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    match input {
        BoundTensorOrTorchFunction::Override(probed) => {
            let handler = resolve_torch_function_override(py, probed)?;
            let result =
                call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
            if !is_not_implemented(py, &result) {
                return Ok(result);
            }
            Err(torch_function_dispatch_error(
                py,
                "torch.unbind",
                active_mode.get(),
                Some(probed.dispatch_type.as_unbound()),
            )?)
        }
        BoundTensorOrTorchFunction::Tensor(tensor) => {
            if active_mode.get().is_some() {
                return Err(torch_function_dispatch_error(
                    py,
                    "torch.unbind",
                    active_mode.get(),
                    None,
                )?);
            }
            let dimension = dimension.map_or(Ok(0), |dimension| {
                extract_dimension_swap_dimension(&dimension.value)
            })?;
            unbind_dimension(py, tensor, dimension)
        }
    }
}

fn dispatch_top_level_select(
    py: Python<'_>,
    input: &BoundTensorOrTorchFunction<'_>,
    dimension: &ParsedCallArgument<'_>,
    index: &ParsedCallArgument<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    if torch_function_mode_stack::is_empty()
        && let BoundTensorOrTorchFunction::Tensor(tensor) = input
    {
        let index = extract_select_index(&index.value)?;
        let dimension = extract_dimension_swap_dimension(&dimension.value)?;
        return select_dimension(py, tensor, dimension, index);
    }

    let function = variable_function(py, "select")?;
    let types = match input {
        BoundTensorOrTorchFunction::Tensor(_) => PyTuple::empty(py),
        BoundTensorOrTorchFunction::Override(probed) => {
            PyTuple::new(py, [probed.dispatch_type.clone()])?
        }
    };

    // Concrete integer conversion and tensor bounds checks remain deferred
    // until every torch-function handler has had an opportunity to replace
    // the otherwise valid generated call.
    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    match input {
        BoundTensorOrTorchFunction::Override(probed) => {
            let handler = resolve_torch_function_override(py, probed)?;
            let result =
                call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
            if !is_not_implemented(py, &result) {
                return Ok(result);
            }
            Err(torch_function_dispatch_error(
                py,
                "torch.select",
                active_mode.get(),
                Some(probed.dispatch_type.as_unbound()),
            )?)
        }
        BoundTensorOrTorchFunction::Tensor(tensor) => {
            if active_mode.get().is_some() {
                return Err(torch_function_dispatch_error(
                    py,
                    "torch.select",
                    active_mode.get(),
                    None,
                )?);
            }
            let index = extract_select_index(&index.value)?;
            let dimension = extract_dimension_swap_dimension(&dimension.value)?;
            select_dimension(py, tensor, dimension, index)
        }
    }
}

pub(crate) fn dispatch_tensorbase_method_mode(
    py: Python<'_>,
    tensor: &Bound<'_, PyTensor>,
    method: &'static str,
    qualified_method: &'static str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<Py<PyAny>>> {
    if torch_function_mode_stack::is_empty() {
        return Ok(None);
    }

    let function = py.get_type::<PyTensorBase>().getattr(method)?.unbind();
    // Parsed method arguments are metadata or options rather than overloaded
    // tensor operands, so PyTorch supplies no dispatch types even though the
    // receiver remains in args.
    let types = PyTuple::empty(py);
    let argument_count = args.len().checked_add(1).ok_or_else(|| {
        PyMemoryError::new_err(format!("{method} dispatch argument count overflowed"))
    })?;
    let mut call_arguments = Vec::new();
    call_arguments
        .try_reserve_exact(argument_count)
        .map_err(|_| {
            PyMemoryError::new_err(format!("unable to allocate {method} dispatch arguments"))
        })?;
    call_arguments.push(tensor.clone().into_any());
    call_arguments.extend(args.iter());
    let call_args = PyTuple::new(py, call_arguments)?;

    // Disable the top mode for the complete attempt so explicit forwarding
    // through the TensorBase descriptor reaches the next mode.
    let active_mode = torch_function_mode_stack::pop();
    let Some(mode) = active_mode.get() else {
        return Ok(None);
    };
    validate_torch_function_mode_handler(mode.bind(py))?;
    let handler = mode.bind(py).getattr("__torch_function__")?;
    let result = call_torch_function_handler(py, &handler, &function, &types, &call_args, kwargs)?;
    if !is_not_implemented(py, &result) {
        return Ok(Some(result));
    }

    Err(torch_function_dispatch_error(
        py,
        qualified_method,
        Some(mode),
        None,
    )?)
}

fn torch_function_dispatch_error(
    py: Python<'_>,
    function: &str,
    mode: Option<&Py<PyAny>>,
    override_type: Option<&Py<PyAny>>,
) -> PyResult<PyErr> {
    let mut handlers = Vec::with_capacity(2);
    if let Some(mode) = mode {
        handlers.push(format!(
            "  - mode object {}",
            mode.bind(py).repr()?.to_str()?
        ));
    }
    if let Some(override_type) = override_type {
        handlers.push(format!(
            "  - tensor subclass {}",
            override_type.bind(py).repr()?.to_str()?
        ));
    }
    Ok(PyTypeError::new_err(format!(
        "Multiple dispatch failed for '{function}'; all __torch_function__ handlers returned NotImplemented:\n\n{}\n\nFor more information, try re-running with TORCH_LOGS=not_implemented",
        handlers.join("\n")
    )))
}

fn torch_function_dispatch_error_for_overrides(
    py: Python<'_>,
    function: &str,
    mode: Option<&Py<PyAny>>,
    overrides: &[ProbedTorchFunctionOverride<'_>],
) -> PyResult<PyErr> {
    let mut handlers = Vec::new();
    handlers
        .try_reserve_exact(overrides.len() + usize::from(mode.is_some()))
        .map_err(|_| PyMemoryError::new_err("unable to allocate torch-function diagnostics"))?;
    if let Some(mode) = mode {
        handlers.push(format!(
            "  - mode object {}",
            mode.bind(py).repr()?.to_str()?
        ));
    }
    for probed in overrides {
        handlers.push(format!(
            "  - tensor subclass {}",
            probed.dispatch_type.repr()?.to_str()?
        ));
    }
    Ok(PyTypeError::new_err(format!(
        "Multiple dispatch failed for '{function}'; all __torch_function__ handlers returned NotImplemented:\n\n{}\n\nFor more information, try re-running with TORCH_LOGS=not_implemented",
        handlers.join("\n")
    )))
}

fn dispatch_dtype_binary(
    operation: DTypeBinaryOperation,
    py: Python<'_>,
    first: &BoundDTypeOperand<'_>,
    second: &BoundDTypeOperand<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let overrides = ordered_dtype_overrides(operation, first, second)?;
    if torch_function_mode_stack::is_empty() && overrides.is_empty() {
        return apply_dtype_binary(operation, py, first, second);
    }

    let function = variable_function(py, operation.name())?;
    let dispatch_types = PyTuple::new(
        py,
        overrides.iter().map(|probed| probed.dispatch_type.clone()),
    )?;

    // Generated variable functions validate their schema before dispatch and
    // disable the top mode for the complete attempt. Explicit forwarding from
    // a mode therefore reaches the next mode, operand overrides, then the
    // native singleton path.
    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result =
            call_torch_function_handler(py, &handler, &function, &dispatch_types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    for probed in &overrides {
        let handler = resolve_torch_function_override(py, probed)?;
        let result =
            call_torch_function_handler(py, &handler, &function, &dispatch_types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    if active_mode.get().is_none() && overrides.is_empty() {
        return apply_dtype_binary(operation, py, first, second);
    }

    Err(torch_function_dispatch_error_for_overrides(
        py,
        operation.qualified_name(),
        active_mode.get(),
        &overrides,
    )?)
}

fn apply_dtype_binary(
    operation: DTypeBinaryOperation,
    py: Python<'_>,
    first: &BoundDTypeOperand<'_>,
    second: &BoundDTypeOperand<'_>,
) -> PyResult<Py<PyAny>> {
    let (BoundDTypeOperand::DType(first), BoundDTypeOperand::DType(second)) = (first, second)
    else {
        unreachable!("dtype overrides were dispatched before the native path")
    };

    match operation {
        DTypeBinaryOperation::CanCast => first.can_cast_to(*second).into_py_any(py),
        DTypeBinaryOperation::PromoteTypes => Ok(dtype_object(py, first.promote(*second))?
            .clone_ref(py)
            .into_any()),
    }
}

fn dispatch_single_tensor_override(
    operation: SingleTensorOverrideOperation,
    py: Python<'_>,
    input: &BoundTensorOrTorchFunction<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let function = variable_function(py, operation.name)?;
    let dispatch_types = match input {
        BoundTensorOrTorchFunction::Tensor(_) => PyTuple::empty(py),
        BoundTensorOrTorchFunction::Override(probed) => {
            PyTuple::new(py, [probed.dispatch_type.clone()])?
        }
    };

    // PyTorch disables the top mode for the complete dispatch attempt. A mode
    // can explicitly call `func(*args, **kwargs)` to reach the next mode.
    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result =
            call_torch_function_handler(py, &handler, &function, &dispatch_types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    match input {
        BoundTensorOrTorchFunction::Override(probed) => {
            let handler = resolve_torch_function_override(py, probed)?;
            let result = call_torch_function_handler(
                py,
                &handler,
                &function,
                &dispatch_types,
                args,
                kwargs,
            )?;
            if !is_not_implemented(py, &result) {
                return Ok(result);
            }
            Err(torch_function_dispatch_error(
                py,
                operation.qualified_name,
                active_mode.get(),
                Some(probed.dispatch_type.as_unbound()),
            )?)
        }
        BoundTensorOrTorchFunction::Tensor(tensor) => {
            if active_mode.get().is_some() {
                return Err(torch_function_dispatch_error(
                    py,
                    operation.qualified_name,
                    active_mode.get(),
                    None,
                )?);
            }
            (operation.apply_native)(py, tensor)
        }
    }
}

#[allow(
    clippy::unnecessary_wraps,
    reason = "single-tensor native callbacks share a fallible signature"
)]
fn apply_top_level_positive(_py: Python<'_>, tensor: &Bound<'_, PyTensor>) -> PyResult<Py<PyAny>> {
    Ok(tensor.clone().unbind().into_any())
}

fn apply_top_level_ravel(py: Python<'_>, tensor: &Bound<'_, PyTensor>) -> PyResult<Py<PyAny>> {
    let inner = tensor
        .try_borrow()?
        .inner
        .ravel()
        .map_err(|error| tensor_error(&error))?;
    Ok(Py::new(py, PyTensor::new(inner))?.into_any())
}

fn dispatch_top_level_reshape(
    py: Python<'_>,
    input: &BoundTensorOrTorchFunction<'_>,
    shape: &BoundTopLevelReshapeShape<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let overrides = ordered_top_level_reshape_overrides(input, shape)?;
    if torch_function_mode_stack::is_empty()
        && let BoundTensorOrTorchFunction::Tensor(tensor) = input
        && let BoundTopLevelReshapeShape::Native(shape) = shape
    {
        return apply_top_level_reshape(py, tensor, shape);
    }

    let function = variable_function(py, "reshape")?;
    let types = PyTuple::new(
        py,
        overrides.iter().map(|probed| probed.dispatch_type.clone()),
    )?;

    // Generated variable functions validate their schema before dispatch and
    // disable the top mode for the full dispatch attempt. Forwarding from a
    // mode therefore reaches the next mode, then input overrides, then native.
    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    for probed in &overrides {
        let handler = resolve_torch_function_override(py, probed)?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    Err(torch_function_dispatch_error_for_overrides(
        py,
        "torch.reshape",
        active_mode.get(),
        &overrides,
    )?)
}

fn apply_top_level_reshape(
    py: Python<'_>,
    tensor: &Bound<'_, PyTensor>,
    shape: &BoundTopLevelNativeReshapeShape<'_>,
) -> PyResult<Py<PyAny>> {
    let shape = parse_top_level_reshape_native_dimensions(shape)?;
    let inner = tensor
        .try_borrow()?
        .inner
        .reshape(&shape)
        .map_err(|error| tensor_error(&error))?;
    Ok(Py::new(py, PyTensor::new(inner))?.into_any())
}

fn apply_top_level_detach(py: Python<'_>, tensor: &Bound<'_, PyTensor>) -> PyResult<Py<PyAny>> {
    let inner = tensor
        .try_borrow()?
        .inner
        .detach()
        .map_err(|error| tensor_error(&error))?;
    Ok(Py::new(py, PyTensor::new(inner))?.into_any())
}

#[allow(
    clippy::unnecessary_wraps,
    reason = "single-tensor native callbacks share a fallible signature"
)]
fn apply_top_level_lazy_bit_identity(
    _py: Python<'_>,
    tensor: &Bound<'_, PyTensor>,
) -> PyResult<Py<PyAny>> {
    // Native tensors expose neither complex storage nor lazy view bits. Real
    // conjugation and resolving clear lazy bits are exact identities, without
    // touching storage, metadata, or autograd state.
    Ok(tensor.clone().unbind().into_any())
}

#[allow(
    clippy::unnecessary_wraps,
    reason = "single-tensor native callbacks share a fallible signature"
)]
fn apply_top_level_real(_py: Python<'_>, tensor: &Bound<'_, PyTensor>) -> PyResult<Py<PyAny>> {
    // Float32 is the only supported dtype, so every native Tensor is already
    // real. Return the wrapper directly instead of consulting Python-visible
    // Tensor.real, which users may shadow on the class.
    Ok(tensor.clone().unbind().into_any())
}

#[allow(
    clippy::unnecessary_wraps,
    reason = "Tensor.imag and top-level torch.imag share this fallible error path"
)]
fn apply_tensor_imag(_tensor: &Bound<'_, PyTensor>) -> PyResult<Py<PyAny>> {
    Err(imag_non_complex_error())
}

#[allow(
    clippy::unnecessary_wraps,
    reason = "single-tensor native callbacks share a fallible signature"
)]
fn apply_top_level_imag(_py: Python<'_>, tensor: &Bound<'_, PyTensor>) -> PyResult<Py<PyAny>> {
    // Native float32 tensors have no imaginary view. Share Tensor.imag's
    // real-dtype error path after top-level __torch_function__ dispatch.
    apply_tensor_imag(tensor)
}

fn ordered_unary_out_overrides<'py>(
    operation: UnaryOutOperation,
    call: &BoundUnaryOutCall<'py>,
) -> PyResult<Vec<ProbedTorchFunctionOverride<'py>>> {
    let input = match &call.input {
        BoundTensorOrTorchFunction::Override(probed) => Some(probed),
        BoundTensorOrTorchFunction::Tensor(_) => None,
    };
    let out = match &call.out {
        Some(BoundTensorOrTorchFunction::Override(probed)) => Some(probed),
        Some(BoundTensorOrTorchFunction::Tensor(_)) | None => None,
    };
    ordered_binary_overrides(input, out, operation.dispatch_allocation_error)
}

fn dispatch_top_level_unary_out(
    operation: UnaryOutOperation,
    py: Python<'_>,
    call: &BoundUnaryOutCall<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let overrides = ordered_unary_out_overrides(operation, call)?;
    if torch_function_mode_stack::is_empty() && overrides.is_empty() {
        return apply_top_level_unary_out(operation, py, call);
    }

    let function = variable_function(py, operation.name)?;
    let types = PyTuple::new(
        py,
        overrides.iter().map(|probed| probed.dispatch_type.clone()),
    )?;

    // Disable the top mode for the complete dispatch attempt. A mode can call
    // the public function explicitly to forward to the next mode.
    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    for probed in &overrides {
        let handler = resolve_torch_function_override(py, probed)?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    Err(torch_function_dispatch_error_for_overrides(
        py,
        operation.qualified_name,
        active_mode.get(),
        &overrides,
    )?)
}

fn apply_top_level_unary_out(
    operation: UnaryOutOperation,
    py: Python<'_>,
    call: &BoundUnaryOutCall<'_>,
) -> PyResult<Py<PyAny>> {
    if call.out.is_some() {
        return Err(PyRuntimeError::new_err(operation.out_unsupported_error));
    }

    let BoundTensorOrTorchFunction::Tensor(input) = &call.input else {
        unreachable!("unary-out overrides were dispatched before the native path")
    };
    let input = input.try_borrow()?;
    let output = (operation.apply)(&input.inner).map_err(|error| tensor_error(&error))?;
    Ok(Py::new(py, PyTensor::new(output))?.into_any())
}

fn ordered_top_level_sum_overrides<'py>(
    call: &BoundTopLevelSumCall<'py>,
) -> PyResult<Vec<ProbedTorchFunctionOverride<'py>>> {
    let mut overrides = Vec::new();
    overrides
        .try_reserve_exact(3)
        .map_err(|_| PyMemoryError::new_err("unable to allocate sum dispatch operands"))?;

    if let BoundTensorOrTorchFunction::Override(probed) = &call.input {
        insert_ordered_torch_function_override(&mut overrides, probed)?;
    }
    if let BoundTopLevelSumDType::Override(probed) = &call.dtype {
        insert_ordered_torch_function_override(&mut overrides, probed)?;
    }
    if let Some(BoundTensorOrTorchFunction::Override(probed)) = &call.out {
        insert_ordered_torch_function_override(&mut overrides, probed)?;
    }
    Ok(overrides)
}

fn dispatch_top_level_sum(
    py: Python<'_>,
    call: &BoundTopLevelSumCall<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let overrides = ordered_top_level_sum_overrides(call)?;
    if torch_function_mode_stack::is_empty() && overrides.is_empty() {
        return apply_top_level_sum(py, call);
    }

    let function = variable_function(py, "sum")?;
    let types = PyTuple::new(
        py,
        overrides.iter().map(|probed| probed.dispatch_type.clone()),
    )?;

    // Generated reduction bindings validate their schema before dispatch and
    // disable the top mode for the complete attempt. A forwarding mode therefore
    // reaches the next mode, then dtype/input/out overrides, then native code.
    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    for probed in &overrides {
        let handler = resolve_torch_function_override(py, probed)?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    if active_mode.get().is_none() && overrides.is_empty() {
        return apply_top_level_sum(py, call);
    }

    Err(torch_function_dispatch_error_for_overrides(
        py,
        "torch.sum",
        active_mode.get(),
        &overrides,
    )?)
}

fn apply_top_level_sum(py: Python<'_>, call: &BoundTopLevelSumCall<'_>) -> PyResult<Py<PyAny>> {
    if !call.full_reduction {
        return Err(PyNotImplementedError::new_err(
            "sum(): only full reductions with dim=None support keepdim; dim and out reductions are not supported",
        ));
    }

    let BoundTensorOrTorchFunction::Tensor(input) = &call.input else {
        unreachable!("sum overrides were dispatched before the native path")
    };
    let input = input.try_borrow()?;
    let mut output = input.inner.sum();
    if call.keepdim_full_reduction {
        output = output
            .reshape(full_reduction_keepdim_shape(&input.inner))
            .map_err(|error| tensor_error(&error))?;
    }
    Ok(Py::new(py, PyTensor::new(output))?.into_any())
}

fn ordered_top_level_mean_overrides<'py>(
    call: &BoundTopLevelMeanCall<'py>,
) -> PyResult<Vec<ProbedTorchFunctionOverride<'py>>> {
    let mut overrides = Vec::new();
    overrides
        .try_reserve_exact(3)
        .map_err(|_| PyMemoryError::new_err("unable to allocate mean dispatch operands"))?;

    if let BoundTensorOrTorchFunction::Override(probed) = &call.input {
        insert_ordered_torch_function_override(&mut overrides, probed)?;
    }
    if let BoundTopLevelMeanDType::Override(probed) = &call.dtype {
        insert_ordered_torch_function_override(&mut overrides, probed)?;
    }
    if let Some(BoundTensorOrTorchFunction::Override(probed)) = &call.out {
        insert_ordered_torch_function_override(&mut overrides, probed)?;
    }
    Ok(overrides)
}

fn dispatch_top_level_mean(
    py: Python<'_>,
    call: &BoundTopLevelMeanCall<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let overrides = ordered_top_level_mean_overrides(call)?;
    if torch_function_mode_stack::is_empty() && overrides.is_empty() {
        return apply_top_level_mean(py, call);
    }

    let function = variable_function(py, "mean")?;
    let types = PyTuple::new(
        py,
        overrides.iter().map(|probed| probed.dispatch_type.clone()),
    )?;

    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    for probed in &overrides {
        let handler = resolve_torch_function_override(py, probed)?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    if active_mode.get().is_none() && overrides.is_empty() {
        return apply_top_level_mean(py, call);
    }

    Err(torch_function_dispatch_error_for_overrides(
        py,
        "torch.mean",
        active_mode.get(),
        &overrides,
    )?)
}

fn apply_top_level_mean(py: Python<'_>, call: &BoundTopLevelMeanCall<'_>) -> PyResult<Py<PyAny>> {
    if !call.full_reduction {
        return Err(PyNotImplementedError::new_err(
            "mean(): only full reductions with dim=None support keepdim; dim, out, and dtype conversion reductions are not supported",
        ));
    }

    let BoundTensorOrTorchFunction::Tensor(input) = &call.input else {
        unreachable!("mean overrides were dispatched before the native path")
    };
    let input = input.try_borrow()?;
    let mut output = input.inner.mean().map_err(|error| tensor_error(&error))?;
    if call.keepdim_full_reduction {
        output = output
            .reshape(full_reduction_keepdim_shape(&input.inner))
            .map_err(|error| tensor_error(&error))?;
    }
    Ok(Py::new(py, PyTensor::new(output))?.into_any())
}

fn full_reduction_keepdim_shape(input: &CoreTensor) -> Vec<i64> {
    vec![1_i64; input.shape().len()]
}

fn dispatch_is_conj(
    py: Python<'_>,
    input: &BoundTensorOrTorchFunction<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let function = variable_function(py, "is_conj")?;
    let types = match input {
        BoundTensorOrTorchFunction::Tensor(_) => PyTuple::empty(py),
        BoundTensorOrTorchFunction::Override(resolved) => {
            PyTuple::new(py, [resolved.dispatch_type.clone()])?
        }
    };

    // PyTorch disables the top mode for the complete dispatch attempt. A mode
    // can explicitly call `func(*args, **kwargs)` to reach the next mode.
    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    match input {
        BoundTensorOrTorchFunction::Override(probed) => {
            let handler = resolve_torch_function_override(py, probed)?;
            let result =
                call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
            if !is_not_implemented(py, &result) {
                return Ok(result);
            }
            Err(torch_function_dispatch_error(
                py,
                "torch.is_conj",
                active_mode.get(),
                Some(probed.dispatch_type.as_unbound()),
            )?)
        }
        BoundTensorOrTorchFunction::Tensor(_) => {
            if active_mode.get().is_some() {
                return Err(torch_function_dispatch_error(
                    py,
                    "torch.is_conj",
                    active_mode.get(),
                    None,
                )?);
            }

            // Float32 is the only supported dtype and the native tensor model
            // has no conjugate views, so every reachable conjugate bit is clear.
            false.into_py_any(py)
        }
    }
}

fn dispatch_is_inference(
    py: Python<'_>,
    input: &BoundTensorOrTorchFunction<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let function = variable_function(py, "is_inference")?;
    let types = match input {
        BoundTensorOrTorchFunction::Tensor(_) => PyTuple::empty(py),
        BoundTensorOrTorchFunction::Override(resolved) => {
            PyTuple::new(py, [resolved.dispatch_type.clone()])?
        }
    };

    // PyTorch disables the top mode for the complete dispatch attempt. A mode
    // can explicitly call `func(*args, **kwargs)` to reach the next mode.
    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    match input {
        BoundTensorOrTorchFunction::Override(probed) => {
            let handler = resolve_torch_function_override(py, probed)?;
            let result =
                call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
            if !is_not_implemented(py, &result) {
                return Ok(result);
            }
            Err(torch_function_dispatch_error(
                py,
                "torch.is_inference",
                active_mode.get(),
                Some(probed.dispatch_type.as_unbound()),
            )?)
        }
        BoundTensorOrTorchFunction::Tensor(_) => {
            if active_mode.get().is_some() {
                return Err(torch_function_dispatch_error(
                    py,
                    "torch.is_inference",
                    active_mode.get(),
                    None,
                )?);
            }

            // The native engine does not expose inference mode, so every
            // reachable Tensor has ordinary autograd metadata. Report the
            // clear state without borrowing storage or touching its graph.
            false.into_py_any(py)
        }
    }
}

fn dispatch_adjoint(
    py: Python<'_>,
    input: &BoundTensorOrTorchFunction<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let function = variable_function(py, "adjoint")?;
    let types = match input {
        BoundTensorOrTorchFunction::Tensor(_) => PyTuple::empty(py),
        BoundTensorOrTorchFunction::Override(resolved) => {
            PyTuple::new(py, [resolved.dispatch_type.clone()])?
        }
    };

    // PyTorch disables the top mode for the complete dispatch attempt. A mode
    // can explicitly call `func(*args, **kwargs)` to reach the next mode.
    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    match input {
        BoundTensorOrTorchFunction::Override(probed) => {
            let handler = resolve_torch_function_override(py, probed)?;
            let result =
                call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
            if !is_not_implemented(py, &result) {
                return Ok(result);
            }
            Err(torch_function_dispatch_error(
                py,
                "torch.adjoint",
                active_mode.get(),
                Some(probed.dispatch_type.as_unbound()),
            )?)
        }
        BoundTensorOrTorchFunction::Tensor(tensor) => {
            if active_mode.get().is_some() {
                return Err(torch_function_dispatch_error(
                    py,
                    "torch.adjoint",
                    active_mode.get(),
                    None,
                )?);
            }
            matrix_adjoint(
                py,
                tensor,
                &ADJOINT_SCALAR_WARNING_EMITTED,
                ADJOINT_SCALAR_WARNING,
                "tensor.adjoint() is only supported on matrices or batches of matrices. Got 1-D tensor.",
            )
        }
    }
}

fn dispatch_top_level_movedim(
    operation: DimensionMoveOperation,
    py: Python<'_>,
    input: &BoundTensorOrTorchFunction<'_>,
    source: &ParsedCallArgument<'_>,
    destination: &ParsedCallArgument<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    if torch_function_mode_stack::is_empty()
        && let BoundTensorOrTorchFunction::Tensor(tensor) = input
    {
        return apply_top_level_movedim(operation, py, tensor, source, destination);
    }

    let function = variable_function(py, operation.name())?;
    let types = match input {
        BoundTensorOrTorchFunction::Tensor(_) => PyTuple::empty(py),
        BoundTensorOrTorchFunction::Override(probed) => {
            PyTuple::new(py, [probed.dispatch_type.clone()])?
        }
    };

    // Integer type matching is complete, but conversion and dimension range
    // checks remain deferred until every active override has had its chance.
    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    match input {
        BoundTensorOrTorchFunction::Override(probed) => {
            let handler = resolve_torch_function_override(py, probed)?;
            let result =
                call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
            if !is_not_implemented(py, &result) {
                return Ok(result);
            }
            Err(torch_function_dispatch_error(
                py,
                operation.qualified_name(),
                active_mode.get(),
                Some(probed.dispatch_type.as_unbound()),
            )?)
        }
        BoundTensorOrTorchFunction::Tensor(tensor) => {
            if active_mode.get().is_some() {
                return Err(torch_function_dispatch_error(
                    py,
                    operation.qualified_name(),
                    active_mode.get(),
                    None,
                )?);
            }
            apply_top_level_movedim(operation, py, tensor, source, destination)
        }
    }
}

fn apply_top_level_movedim(
    operation: DimensionMoveOperation,
    py: Python<'_>,
    input: &Bound<'_, PyTensor>,
    source: &ParsedCallArgument<'_>,
    destination: &ParsedCallArgument<'_>,
) -> PyResult<Py<PyAny>> {
    let [source, destination] = parse_dimension_swap_dimensions(
        operation.name(),
        ["source", "destination"],
        source,
        destination,
    )?;
    let inner = movedim_tensor(&input.try_borrow()?.inner, source, destination)?;
    Ok(Py::new(py, PyTensor::new(inner))?.into_any())
}

fn dispatch_view_as(
    py: Python<'_>,
    tensor: &Bound<'_, PyTensor>,
    other: &BoundTensorOrTorchFunction<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    if torch_function_mode_stack::is_empty()
        && let BoundTensorOrTorchFunction::Tensor(other) = other
    {
        return apply_view_as(py, tensor, other);
    }

    let function = py.get_type::<PyTensorBase>().getattr("view_as")?.unbind();
    let types = match other {
        BoundTensorOrTorchFunction::Tensor(_) => PyTuple::empty(py),
        BoundTensorOrTorchFunction::Override(probed) => {
            PyTuple::new(py, [probed.dispatch_type.clone()])?
        }
    };
    let argument_count = args
        .len()
        .checked_add(1)
        .ok_or_else(|| PyMemoryError::new_err("view_as dispatch argument count overflowed"))?;
    let mut call_arguments = Vec::new();
    call_arguments
        .try_reserve_exact(argument_count)
        .map_err(|_| PyMemoryError::new_err("unable to allocate view_as dispatch arguments"))?;
    call_arguments.push(tensor.clone().into_any());
    call_arguments.extend(args.iter());
    let call_args = PyTuple::new(py, call_arguments)?;

    // Generated tensor methods validate their schema before dispatch and
    // disable the top mode for the complete attempt. Explicit forwarding from
    // a mode therefore reaches the next mode, then the operand override or the
    // native view path.
    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result =
            call_torch_function_handler(py, &handler, &function, &types, &call_args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    match other {
        BoundTensorOrTorchFunction::Override(probed) => {
            // Resolve only after the mode has declined so mode side effects on
            // the operand's handler match PyTorch's deferred lookup.
            let handler = resolve_torch_function_override(py, probed)?;
            let result =
                call_torch_function_handler(py, &handler, &function, &types, &call_args, kwargs)?;
            if !is_not_implemented(py, &result) {
                return Ok(result);
            }
            Err(torch_function_dispatch_error(
                py,
                "torch.Tensor.view_as",
                active_mode.get(),
                Some(probed.dispatch_type.as_unbound()),
            )?)
        }
        BoundTensorOrTorchFunction::Tensor(other) => {
            if active_mode.get().is_some() {
                return Err(torch_function_dispatch_error(
                    py,
                    "torch.Tensor.view_as",
                    active_mode.get(),
                    None,
                )?);
            }
            apply_view_as(py, tensor, other)
        }
    }
}

fn apply_view_as(
    py: Python<'_>,
    tensor: &Bound<'_, PyTensor>,
    other: &Bound<'_, PyTensor>,
) -> PyResult<Py<PyAny>> {
    let shape = tensor_shape_as_i64(other)?;
    let inner = tensor
        .try_borrow()?
        .inner
        .view(shape)
        .map_err(|error| tensor_error(&error))?;
    Ok(Py::new(py, PyTensor::new(inner))?.into_any())
}

fn dispatch_matmul(
    py: Python<'_>,
    tensor: &Bound<'_, PyTensor>,
    other: &BoundTensorOrTorchFunction<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    if torch_function_mode_stack::is_empty()
        && let BoundTensorOrTorchFunction::Tensor(other) = other
    {
        let other = other.try_borrow()?;
        let result = tensor.try_borrow()?.matrix_multiply(&other)?;
        return Ok(Py::new(py, result)?.into_any());
    }

    let function = py.get_type::<PyTensorBase>().getattr("matmul")?.unbind();
    let types = match other {
        BoundTensorOrTorchFunction::Tensor(_) => PyTuple::empty(py),
        BoundTensorOrTorchFunction::Override(probed) => {
            PyTuple::new(py, [probed.dispatch_type.clone()])?
        }
    };
    let argument_count = args
        .len()
        .checked_add(1)
        .ok_or_else(|| PyMemoryError::new_err("matmul dispatch argument count overflowed"))?;
    let mut call_arguments = Vec::new();
    call_arguments
        .try_reserve_exact(argument_count)
        .map_err(|_| PyMemoryError::new_err("unable to allocate matmul dispatch arguments"))?;
    call_arguments.push(tensor.clone().into_any());
    call_arguments.extend(args.iter());
    let call_args = PyTuple::new(py, call_arguments)?;

    // Disable the top mode for the complete attempt so forwarding through the
    // TensorBase descriptor reaches the next mode, just as top-level dispatch does.
    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result =
            call_torch_function_handler(py, &handler, &function, &types, &call_args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    match other {
        BoundTensorOrTorchFunction::Override(probed) => {
            let handler = resolve_torch_function_override(py, probed)?;
            let result =
                call_torch_function_handler(py, &handler, &function, &types, &call_args, kwargs)?;
            if !is_not_implemented(py, &result) {
                return Ok(result);
            }
            Err(torch_function_dispatch_error(
                py,
                "torch.Tensor.matmul",
                active_mode.get(),
                Some(probed.dispatch_type.as_unbound()),
            )?)
        }
        BoundTensorOrTorchFunction::Tensor(other) => {
            if active_mode.get().is_some() {
                return Err(torch_function_dispatch_error(
                    py,
                    "torch.Tensor.matmul",
                    active_mode.get(),
                    None,
                )?);
            }
            let other = other.try_borrow()?;
            let result = tensor.try_borrow()?.matrix_multiply(&other)?;
            Ok(Py::new(py, result)?.into_any())
        }
    }
}

fn ordered_binary_overrides<'py>(
    first: Option<&ProbedTorchFunctionOverride<'py>>,
    second: Option<&ProbedTorchFunctionOverride<'py>>,
    allocation_error: &'static str,
) -> PyResult<Vec<ProbedTorchFunctionOverride<'py>>> {
    let mut overrides = Vec::new();
    overrides
        .try_reserve_exact(2)
        .map_err(|_| PyMemoryError::new_err(allocation_error))?;

    if let Some(probed) = first {
        insert_ordered_torch_function_override(&mut overrides, probed)?;
    }
    if let Some(probed) = second {
        insert_ordered_torch_function_override(&mut overrides, probed)?;
    }
    Ok(overrides)
}

fn insert_ordered_torch_function_override<'py>(
    overrides: &mut Vec<ProbedTorchFunctionOverride<'py>>,
    probed: &ProbedTorchFunctionOverride<'py>,
) -> PyResult<()> {
    // PyTorch reports a class-valued operand itself in the dispatch types, but
    // orders an incoming operand by its runtime type. Its metaclass is
    // therefore compared with reported classes, preserving class argument
    // order and repeated class identities without changing ordinary instance
    // subclass precedence.
    if overrides
        .iter()
        .any(|existing| existing.dispatch_type.is(probed.precedence_type.as_any()))
    {
        return Ok(());
    }

    let mut position = overrides.len();
    for (index, existing) in overrides.iter().enumerate() {
        let existing_type = existing
            .dispatch_type
            .cast::<PyType>()
            .expect("a torch-function dispatch type is a Python type");
        if probed.precedence_type.is_subclass(existing_type.as_any())? {
            position = index;
            break;
        }
    }
    overrides.insert(position, probed.clone());
    Ok(())
}

fn ordered_top_level_reshape_overrides<'py>(
    input: &BoundTensorOrTorchFunction<'py>,
    shape: &BoundTopLevelReshapeShape<'py>,
) -> PyResult<Vec<ProbedTorchFunctionOverride<'py>>> {
    let shape_overrides = match shape {
        BoundTopLevelReshapeShape::Native(_) => &[][..],
        BoundTopLevelReshapeShape::Override(overrides) => overrides,
    };
    let input_count = usize::from(matches!(input, BoundTensorOrTorchFunction::Override(_)));
    let mut overrides = Vec::new();
    overrides
        .try_reserve_exact(input_count + shape_overrides.len())
        .map_err(|_| PyMemoryError::new_err("unable to allocate reshape dispatch operands"))?;
    if let BoundTensorOrTorchFunction::Override(probed) = input {
        insert_ordered_torch_function_override(&mut overrides, probed)?;
    }
    for probed in shape_overrides {
        insert_ordered_torch_function_override(&mut overrides, probed)?;
    }
    Ok(overrides)
}

fn ordered_matmul_overrides<'py>(
    input: &BoundTensorOrTorchFunction<'py>,
    other: &BoundTensorOrTorchFunction<'py>,
) -> PyResult<Vec<ProbedTorchFunctionOverride<'py>>> {
    let input = match input {
        BoundTensorOrTorchFunction::Override(probed) => Some(probed),
        BoundTensorOrTorchFunction::Tensor(_) => None,
    };
    let other = match other {
        BoundTensorOrTorchFunction::Override(probed) => Some(probed),
        BoundTensorOrTorchFunction::Tensor(_) => None,
    };
    ordered_binary_overrides(input, other, "unable to allocate matmul dispatch operands")
}

fn ordered_dtype_overrides<'py>(
    operation: DTypeBinaryOperation,
    first: &BoundDTypeOperand<'py>,
    second: &BoundDTypeOperand<'py>,
) -> PyResult<Vec<ProbedTorchFunctionOverride<'py>>> {
    let first = match first {
        BoundDTypeOperand::Override(probed) => Some(probed),
        BoundDTypeOperand::DType(_) => None,
    };
    let second = match second {
        BoundDTypeOperand::Override(probed) => Some(probed),
        BoundDTypeOperand::DType(_) => None,
    };
    ordered_binary_overrides(first, second, operation.dispatch_allocation_error())
}

fn ordered_multiplication_overrides<'py>(
    operation: MultiplicationOperation,
    input: &BoundMulOperand<'py>,
    other: &BoundMulOperand<'py>,
) -> PyResult<Vec<ProbedTorchFunctionOverride<'py>>> {
    let input = match input {
        BoundMulOperand::Override(probed) => Some(probed),
        BoundMulOperand::Tensor(_) | BoundMulOperand::Scalar(_) => None,
    };
    let other = match other {
        BoundMulOperand::Override(probed) => Some(probed),
        BoundMulOperand::Tensor(_) | BoundMulOperand::Scalar(_) => None,
    };
    ordered_binary_overrides(input, other, operation.dispatch_allocation_error())
}

fn dispatch_top_level_matmul(
    py: Python<'_>,
    input: &BoundTensorOrTorchFunction<'_>,
    other: &BoundTensorOrTorchFunction<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let overrides = ordered_matmul_overrides(input, other)?;
    if torch_function_mode_stack::is_empty() && overrides.is_empty() {
        let (BoundTensorOrTorchFunction::Tensor(input), BoundTensorOrTorchFunction::Tensor(other)) =
            (input, other)
        else {
            unreachable!("matmul overrides were collected before the native fast path")
        };
        let other = other.try_borrow()?;
        let result = input.try_borrow()?.matrix_multiply(&other)?;
        return Ok(Py::new(py, result)?.into_any());
    }

    let function = variable_function(py, "matmul")?;
    let types = PyTuple::new(
        py,
        overrides.iter().map(|probed| probed.dispatch_type.clone()),
    )?;

    // Disable the top mode for the complete dispatch attempt. A mode can call
    // the public function explicitly to forward to the next mode.
    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    for probed in &overrides {
        let handler = resolve_torch_function_override(py, probed)?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    Err(torch_function_dispatch_error_for_overrides(
        py,
        "torch.matmul",
        active_mode.get(),
        &overrides,
    )?)
}

fn ordered_top_level_mm_overrides<'py>(
    call: &BoundTopLevelMmCall<'py>,
) -> PyResult<Vec<ProbedTorchFunctionOverride<'py>>> {
    let mut overrides = Vec::new();
    overrides
        .try_reserve_exact(3)
        .map_err(|_| PyMemoryError::new_err("unable to allocate mm dispatch operands"))?;

    if let BoundTensorOrTorchFunction::Override(probed) = &call.input {
        insert_ordered_torch_function_override(&mut overrides, probed)?;
    }
    if let BoundTensorOrTorchFunction::Override(probed) = &call.mat2 {
        insert_ordered_torch_function_override(&mut overrides, probed)?;
    }
    if let Some(BoundTensorOrTorchFunction::Override(probed)) = &call.out {
        insert_ordered_torch_function_override(&mut overrides, probed)?;
    }
    Ok(overrides)
}

fn dispatch_top_level_mm(
    py: Python<'_>,
    call: &BoundTopLevelMmCall<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let overrides = ordered_top_level_mm_overrides(call)?;
    if torch_function_mode_stack::is_empty() && overrides.is_empty() {
        return apply_top_level_mm(py, call);
    }

    let function = variable_function(py, "mm")?;
    let types = PyTuple::new(
        py,
        overrides.iter().map(|probed| probed.dispatch_type.clone()),
    )?;

    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    for probed in &overrides {
        let handler = resolve_torch_function_override(py, probed)?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    Err(torch_function_dispatch_error_for_overrides(
        py,
        "torch.mm",
        active_mode.get(),
        &overrides,
    )?)
}

fn apply_top_level_mm(py: Python<'_>, call: &BoundTopLevelMmCall<'_>) -> PyResult<Py<PyAny>> {
    if call.out.is_some() {
        return Err(PyRuntimeError::new_err(
            "mm(): the 'out' argument is not supported",
        ));
    }

    let BoundTensorOrTorchFunction::Tensor(input) = &call.input else {
        unreachable!("mm overrides were dispatched before the native path")
    };
    let BoundTensorOrTorchFunction::Tensor(mat2) = &call.mat2 else {
        unreachable!("mm overrides were dispatched before the native path")
    };
    let input = input.try_borrow()?;
    let mat2 = mat2.try_borrow()?;
    if input.inner.dtype() != DType::Float32
        || !input.inner.device().is_cpu()
        || mat2.inner.dtype() != DType::Float32
        || !mat2.inner.device().is_cpu()
    {
        return Err(mm_unsupported_native_input());
    }
    if input.inner.shape().len() != 2 {
        return Err(PyRuntimeError::new_err("self must be a matrix"));
    }
    if mat2.inner.shape().len() != 2 {
        return Err(PyRuntimeError::new_err("mat2 must be a matrix"));
    }
    let result = input.matrix_multiply(&mat2)?;
    Ok(Py::new(py, result)?.into_any())
}

fn dispatch_top_level_multiplication(
    operation: MultiplicationOperation,
    py: Python<'_>,
    input: &BoundMulOperand<'_>,
    other: &BoundMulOperand<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let overrides = ordered_multiplication_overrides(operation, input, other)?;
    if torch_function_mode_stack::is_empty() && overrides.is_empty() {
        return apply_top_level_multiplication(operation, py, input, other);
    }

    let function = variable_function(py, operation.name())?;
    let types = PyTuple::new(
        py,
        overrides.iter().map(|probed| probed.dispatch_type.clone()),
    )?;

    // Disable the top mode for the complete dispatch attempt. A mode can call
    // the public function explicitly to forward to the next mode.
    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    for probed in &overrides {
        let handler = resolve_torch_function_override(py, probed)?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    Err(torch_function_dispatch_error_for_overrides(
        py,
        operation.qualified_name(),
        active_mode.get(),
        &overrides,
    )?)
}

fn apply_top_level_multiplication(
    operation: MultiplicationOperation,
    py: Python<'_>,
    input: &BoundMulOperand<'_>,
    other: &BoundMulOperand<'_>,
) -> PyResult<Py<PyAny>> {
    let result = match (input, other) {
        (BoundMulOperand::Tensor(input), BoundMulOperand::Tensor(other)) => {
            let other = other.try_borrow()?;
            BinaryOperation::Multiply.apply_tensors(&input.try_borrow()?.inner, &other.inner)
        }
        (BoundMulOperand::Tensor(tensor), BoundMulOperand::Scalar(scalar))
        | (BoundMulOperand::Scalar(scalar), BoundMulOperand::Tensor(tensor)) => {
            let scalar = parse_top_level_mul_scalar(scalar)?;
            BinaryOperation::Multiply.apply_scalar(&tensor.try_borrow()?.inner, scalar, false)
        }
        (BoundMulOperand::Scalar(_), BoundMulOperand::Scalar(_)) => {
            return Err(PyTypeError::new_err(format!(
                "{}(): scalar-scalar multiplication is not supported; at least one operand must be Tensor",
                operation.name()
            )));
        }
        (BoundMulOperand::Override(_), _) | (_, BoundMulOperand::Override(_)) => {
            unreachable!("multiplication overrides were dispatched before the native path")
        }
    };
    Ok(Py::new(
        py,
        PyTensor::new(result.map_err(|error| tensor_error(&error))?),
    )?
    .into_any())
}

fn ordered_addition_overrides<'py>(
    call: &BoundTopLevelAdditionCall<'py>,
) -> PyResult<Vec<ProbedTorchFunctionOverride<'py>>> {
    let input = match &call.input {
        BoundAddOperand::Tensor(_) | BoundAddOperand::Scalar(_) => None,
        BoundAddOperand::Override(probed) => Some(probed),
    };
    let other = match &call.other {
        BoundAddOperand::Tensor(_) | BoundAddOperand::Scalar(_) => None,
        BoundAddOperand::Override(probed) => Some(probed),
    };
    let alpha = match &call.alpha {
        BoundSubAlpha::Default | BoundSubAlpha::PythonBool | BoundSubAlpha::NonDefault => None,
        BoundSubAlpha::Override(probed) => Some(probed),
    };
    let out = match &call.out {
        Some(BoundTensorOrTorchFunction::Override(probed)) => Some(probed),
        Some(BoundTensorOrTorchFunction::Tensor(_)) | None => None,
    };

    let mut overrides = Vec::new();
    overrides
        .try_reserve_exact(4)
        .map_err(|_| PyMemoryError::new_err("unable to allocate add dispatch operands"))?;
    for probed in [input, other, alpha, out].into_iter().flatten() {
        insert_ordered_torch_function_override(&mut overrides, probed)?;
    }
    Ok(overrides)
}

fn dispatch_top_level_addition(
    py: Python<'_>,
    call: &BoundTopLevelAdditionCall<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let overrides = ordered_addition_overrides(call)?;
    if torch_function_mode_stack::is_empty() && overrides.is_empty() {
        return apply_top_level_addition(py, call);
    }

    let function = variable_function(py, "add")?;
    let types = PyTuple::new(
        py,
        overrides.iter().map(|probed| probed.dispatch_type.clone()),
    )?;

    // Generated variable functions validate their schema before dispatch, but
    // delay native-only unsupported cases such as scalar-left operands,
    // concrete out tensors, and nondefault alpha until active handlers can override.
    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    for probed in &overrides {
        let handler = resolve_torch_function_override(py, probed)?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    if active_mode.get().is_none() && overrides.is_empty() {
        return apply_top_level_addition(py, call);
    }

    Err(torch_function_dispatch_error_for_overrides(
        py,
        "torch.add",
        active_mode.get(),
        &overrides,
    )?)
}

fn apply_top_level_addition(
    py: Python<'_>,
    call: &BoundTopLevelAdditionCall<'_>,
) -> PyResult<Py<PyAny>> {
    if call.out.is_some() {
        return Err(PyRuntimeError::new_err(
            "add(): the 'out' argument is not supported",
        ));
    }
    match &call.alpha {
        BoundSubAlpha::Default => {}
        BoundSubAlpha::PythonBool => {
            return Err(PyRuntimeError::new_err(
                "Boolean alpha only supported for Boolean results.",
            ));
        }
        BoundSubAlpha::NonDefault => {
            return Err(PyNotImplementedError::new_err(
                "add(): alpha values other than 1 are not supported",
            ));
        }
        BoundSubAlpha::Override(_) => {
            unreachable!("addition alpha overrides were dispatched before the native path")
        }
    }

    let result = match (&call.input, &call.other) {
        (BoundAddOperand::Tensor(input), BoundAddOperand::Tensor(other)) => {
            let input = input.try_borrow()?;
            let other = other.try_borrow()?;
            validate_top_level_add_tensor(&input)?;
            validate_top_level_add_tensor(&other)?;
            BinaryOperation::Add.apply_tensors(&input.inner, &other.inner)
        }
        (BoundAddOperand::Tensor(input), BoundAddOperand::Scalar(scalar)) => {
            let input = input.try_borrow()?;
            validate_top_level_add_tensor(&input)?;
            let scalar = parse_supported_arithmetic_scalar(scalar)?;
            BinaryOperation::Add.apply_scalar(&input.inner, scalar.into_f32(), false)
        }
        (BoundAddOperand::Scalar(_), BoundAddOperand::Tensor(_) | BoundAddOperand::Scalar(_)) => {
            return Err(addition_unsupported_native_input());
        }
        (BoundAddOperand::Override(_), _) | (_, BoundAddOperand::Override(_)) => {
            unreachable!("addition operand overrides were dispatched before the native path")
        }
    };
    Ok(Py::new(
        py,
        PyTensor::new(result.map_err(|error| tensor_error(&error))?),
    )?
    .into_any())
}

fn validate_top_level_add_tensor(tensor: &PyTensor) -> PyResult<()> {
    if tensor.inner.dtype() == DType::Float32 && tensor.inner.device().is_cpu() {
        Ok(())
    } else {
        Err(addition_unsupported_native_input())
    }
}

fn validate_top_level_division_tensor(
    operation: DivisionOperation,
    tensor: &PyTensor,
) -> PyResult<()> {
    if tensor.inner.dtype() == DType::Float32 && tensor.inner.device().is_cpu() {
        Ok(())
    } else {
        Err(division_unsupported_native_input(operation))
    }
}

fn ordered_top_level_division_overrides<'py>(
    operation: DivisionOperation,
    call: &BoundTopLevelDivisionCall<'py>,
) -> PyResult<Vec<ProbedTorchFunctionOverride<'py>>> {
    let input = match &call.input {
        BoundDivOperand::Override(probed) => Some(probed),
        BoundDivOperand::Tensor(_) | BoundDivOperand::Scalar(_) => None,
    };
    let other = match &call.other {
        BoundDivOperand::Override(probed) => Some(probed),
        BoundDivOperand::Tensor(_) | BoundDivOperand::Scalar(_) => None,
    };
    let rounding_mode = match &call.rounding_mode {
        BoundDivisionRoundingMode::Override(probed) => Some(probed),
        BoundDivisionRoundingMode::TrueDivision | BoundDivisionRoundingMode::Rounding => None,
    };
    let out = match &call.out {
        Some(BoundTensorOrTorchFunction::Override(probed)) => Some(probed),
        Some(BoundTensorOrTorchFunction::Tensor(_)) | None => None,
    };

    let mut overrides = Vec::new();
    overrides
        .try_reserve_exact(4)
        .map_err(|_| PyMemoryError::new_err(operation.dispatch_allocation_error()))?;
    for probed in [input, other, rounding_mode, out].into_iter().flatten() {
        insert_ordered_torch_function_override(&mut overrides, probed)?;
    }
    Ok(overrides)
}

fn dispatch_top_level_division(
    operation: DivisionOperation,
    py: Python<'_>,
    call: &BoundTopLevelDivisionCall<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let overrides = ordered_top_level_division_overrides(operation, call)?;
    if torch_function_mode_stack::is_empty() && overrides.is_empty() {
        return apply_top_level_division(operation, py, call);
    }

    let function = variable_function(py, operation.name())?;
    let types = PyTuple::new(
        py,
        overrides.iter().map(|probed| probed.dispatch_type.clone()),
    )?;

    // Generated variable functions validate their schema before dispatch, but
    // delay native-only unsupported cases such as concrete out tensors,
    // scalar-left operands, and non-None rounding modes until active handlers
    // can override.
    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    for probed in &overrides {
        let handler = resolve_torch_function_override(py, probed)?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    if active_mode.get().is_none() && overrides.is_empty() {
        return apply_top_level_division(operation, py, call);
    }

    Err(torch_function_dispatch_error_for_overrides(
        py,
        operation.qualified_name(),
        active_mode.get(),
        &overrides,
    )?)
}

fn apply_top_level_division(
    operation: DivisionOperation,
    py: Python<'_>,
    call: &BoundTopLevelDivisionCall<'_>,
) -> PyResult<Py<PyAny>> {
    if call.out.is_some() {
        return Err(PyRuntimeError::new_err(operation.out_unsupported_error()));
    }
    match &call.rounding_mode {
        BoundDivisionRoundingMode::TrueDivision => {}
        BoundDivisionRoundingMode::Rounding => {
            return Err(PyNotImplementedError::new_err(
                operation.rounding_mode_unsupported_error(),
            ));
        }
        BoundDivisionRoundingMode::Override(_) => {
            unreachable!("division rounding_mode overrides were dispatched before the native path")
        }
    }

    let result = match (&call.input, &call.other) {
        (BoundDivOperand::Tensor(input), BoundDivOperand::Tensor(other)) => {
            let input = input.try_borrow()?;
            let other = other.try_borrow()?;
            validate_top_level_division_tensor(operation, &input)?;
            validate_top_level_division_tensor(operation, &other)?;
            if is_grad_enabled() && (input.inner.requires_grad() || other.inner.requires_grad()) {
                return Err(PyRuntimeError::new_err(
                    operation.autograd_unsupported_error(),
                ));
            }
            BinaryOperation::Divide.apply_tensors(&input.inner, &other.inner)
        }
        (BoundDivOperand::Tensor(input), BoundDivOperand::Scalar(scalar)) => {
            let input = input.try_borrow()?;
            validate_top_level_division_tensor(operation, &input)?;
            if is_grad_enabled() && input.inner.requires_grad() {
                return Err(PyRuntimeError::new_err(
                    operation.autograd_unsupported_error(),
                ));
            }
            let scalar = parse_top_level_mul_scalar(scalar)?;
            BinaryOperation::Divide.apply_scalar(&input.inner, scalar, false)
        }
        (BoundDivOperand::Scalar(_), BoundDivOperand::Tensor(_)) => {
            return Err(division_unsupported_native_input(operation));
        }
        (BoundDivOperand::Scalar(_), BoundDivOperand::Scalar(_)) => {
            return Err(PyTypeError::new_err(format!(
                "{}(): scalar-scalar division is not supported; at least one operand must be Tensor",
                operation.name()
            )));
        }
        (BoundDivOperand::Override(_), _) | (_, BoundDivOperand::Override(_)) => {
            unreachable!("division operand overrides were dispatched before the native path")
        }
    };
    Ok(Py::new(
        py,
        PyTensor::new(result.map_err(|error| tensor_error(&error))?),
    )?
    .into_any())
}

fn tensor_division_method<'py>(
    operation: DivisionOperation,
    slf: &Bound<'py, PyTensorBase>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (other_argument, rounding_mode_argument, keyword_error) =
        bind_tensor_division_method_arguments(operation, args, kwargs)?;
    let other = parse_tensor_division_method_other(operation, &other_argument, args, kwargs)?;
    let rounding_mode = parse_tensor_division_rounding_mode(
        operation,
        rounding_mode_argument.as_ref(),
        args,
        kwargs,
    )?;
    if let Some(keyword_error) = keyword_error {
        return Err(keyword_error);
    }

    let input = parse_tensor_division_method_receiver(operation, slf.as_any())?;
    let call = BoundTensorMethodDivisionCall {
        input,
        other,
        rounding_mode,
    };
    dispatch_tensor_division_method(operation, slf.py(), slf.as_any(), &call, args, kwargs)
}

fn parse_tensor_division_method_receiver<'py>(
    operation: DivisionOperation,
    receiver: &Bound<'py, PyAny>,
) -> PyResult<BoundDivOperand<'py>> {
    if receiver.is_exact_instance_of::<PyTensor>() {
        return Ok(BoundDivOperand::Tensor(
            receiver.cast::<PyTensor>()?.clone(),
        ));
    }
    if let Some(probed) = probe_torch_function_override(receiver) {
        return Ok(BoundDivOperand::Override(probed));
    }
    if receiver.is_instance_of::<PyTensor>() {
        return Err(division_unsupported_native_input(operation));
    }
    Err(division_unsupported_native_input(operation))
}

fn parse_tensor_division_method_other<'py>(
    operation: DivisionOperation,
    value: &ParsedCallArgument<'py>,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundDivOperand<'py>> {
    if value.value.is_exact_instance_of::<PyTensor>() {
        return Ok(BoundDivOperand::Tensor(
            value.value.cast::<PyTensor>()?.clone(),
        ));
    }
    if let Some(probed) = probe_torch_function_override(&value.value) {
        return Ok(BoundDivOperand::Override(probed));
    }
    if value.value.is_instance_of::<PyTensor>() {
        return Err(division_unsupported_native_input(operation));
    }
    if is_real_arithmetic_scalar(&value.value)? {
        return Ok(BoundDivOperand::Scalar(value.value.clone()));
    }

    if matches!(operation, DivisionOperation::Divide) {
        return Err(tensor_division_method_binding_error(
            operation, positional, keywords,
        )?);
    }
    parse_tensor_argument(operation.name(), "other", value)?;
    unreachable!("unsupported Tensor.div operands were rejected by parse_tensor_argument")
}

fn parse_tensor_division_rounding_mode<'py>(
    operation: DivisionOperation,
    value: Option<&ParsedCallArgument<'py>>,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundDivisionRoundingMode<'py>> {
    let Some(value) = value else {
        return Ok(BoundDivisionRoundingMode::TrueDivision);
    };
    if value.value.is_none() {
        return Ok(BoundDivisionRoundingMode::TrueDivision);
    }
    if let Some(probed) = probe_torch_function_override(&value.value) {
        return Ok(BoundDivisionRoundingMode::Override(probed));
    }
    if value.value.cast::<PyString>().is_ok() {
        return Ok(BoundDivisionRoundingMode::Rounding);
    }
    Err(tensor_division_method_binding_error(
        operation, positional, keywords,
    )?)
}

fn parse_top_level_division_rounding_mode<'py>(
    operation: DivisionOperation,
    value: Option<&ParsedCallArgument<'py>>,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundDivisionRoundingMode<'py>> {
    let Some(value) = value else {
        return Ok(BoundDivisionRoundingMode::TrueDivision);
    };
    if value.value.is_none() {
        return Ok(BoundDivisionRoundingMode::TrueDivision);
    }
    if let Some(probed) = probe_torch_function_override(&value.value) {
        return Ok(BoundDivisionRoundingMode::Override(probed));
    }
    if value.value.cast::<PyString>().is_ok() {
        return Ok(BoundDivisionRoundingMode::Rounding);
    }
    Err(top_level_division_binding_error(
        operation, positional, keywords,
    )?)
}

fn ordered_tensor_division_method_overrides<'py>(
    operation: DivisionOperation,
    call: &BoundTensorMethodDivisionCall<'py>,
) -> PyResult<Vec<ProbedTorchFunctionOverride<'py>>> {
    let input = match &call.input {
        BoundDivOperand::Override(probed) => Some(probed),
        BoundDivOperand::Tensor(_) | BoundDivOperand::Scalar(_) => None,
    };
    let other = match &call.other {
        BoundDivOperand::Override(probed) => Some(probed),
        BoundDivOperand::Tensor(_) | BoundDivOperand::Scalar(_) => None,
    };
    let rounding_mode = match &call.rounding_mode {
        BoundDivisionRoundingMode::Override(probed) => Some(probed),
        BoundDivisionRoundingMode::TrueDivision | BoundDivisionRoundingMode::Rounding => None,
    };

    let mut overrides = Vec::new();
    overrides
        .try_reserve_exact(3)
        .map_err(|_| PyMemoryError::new_err(operation.dispatch_allocation_error()))?;
    for probed in [input, other, rounding_mode].into_iter().flatten() {
        insert_ordered_torch_function_override(&mut overrides, probed)?;
    }
    Ok(overrides)
}

fn dispatch_tensor_division_method(
    operation: DivisionOperation,
    py: Python<'_>,
    receiver: &Bound<'_, PyAny>,
    call: &BoundTensorMethodDivisionCall<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let overrides = ordered_tensor_division_method_overrides(operation, call)?;
    if torch_function_mode_stack::is_empty() && overrides.is_empty() {
        return apply_tensor_division_method(operation, py, call);
    }

    let function = py
        .get_type::<PyTensorBase>()
        .getattr(operation.name())?
        .unbind();
    let types = PyTuple::new(
        py,
        overrides.iter().map(|probed| probed.dispatch_type.clone()),
    )?;
    let argument_count = args.len().checked_add(1).ok_or_else(|| {
        PyMemoryError::new_err(format!(
            "{} dispatch argument count overflowed",
            operation.name()
        ))
    })?;
    let mut call_arguments = Vec::new();
    call_arguments
        .try_reserve_exact(argument_count)
        .map_err(|_| {
            PyMemoryError::new_err(format!(
                "unable to allocate {} dispatch arguments",
                operation.name()
            ))
        })?;
    call_arguments.push(receiver.clone());
    call_arguments.extend(args.iter());
    let call_args = PyTuple::new(py, call_arguments)?;

    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result =
            call_torch_function_handler(py, &handler, &function, &types, &call_args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    for probed in &overrides {
        let handler = resolve_torch_function_override(py, probed)?;
        let result =
            call_torch_function_handler(py, &handler, &function, &types, &call_args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    Err(torch_function_dispatch_error_for_overrides(
        py,
        operation.qualified_method_name(),
        active_mode.get(),
        &overrides,
    )?)
}

fn apply_tensor_division_method(
    operation: DivisionOperation,
    py: Python<'_>,
    call: &BoundTensorMethodDivisionCall<'_>,
) -> PyResult<Py<PyAny>> {
    match &call.rounding_mode {
        BoundDivisionRoundingMode::TrueDivision => {}
        BoundDivisionRoundingMode::Rounding => {
            return Err(PyNotImplementedError::new_err(
                operation.rounding_mode_unsupported_error(),
            ));
        }
        BoundDivisionRoundingMode::Override(_) => {
            unreachable!("division rounding_mode overrides were dispatched before the native path")
        }
    }

    let result = match (&call.input, &call.other) {
        (BoundDivOperand::Tensor(input), BoundDivOperand::Tensor(other)) => {
            let input = input.try_borrow()?;
            let other = other.try_borrow()?;
            if is_grad_enabled() && (input.inner.requires_grad() || other.inner.requires_grad()) {
                return Err(PyRuntimeError::new_err(
                    operation.autograd_unsupported_error(),
                ));
            }
            BinaryOperation::Divide.apply_tensors(&input.inner, &other.inner)
        }
        (BoundDivOperand::Tensor(tensor), BoundDivOperand::Scalar(scalar)) => {
            let tensor = tensor.try_borrow()?;
            if is_grad_enabled() && tensor.inner.requires_grad() {
                return Err(PyRuntimeError::new_err(
                    operation.autograd_unsupported_error(),
                ));
            }
            let scalar = parse_top_level_mul_scalar(scalar)?;
            BinaryOperation::Divide.apply_scalar(&tensor.inner, scalar, false)
        }
        (BoundDivOperand::Override(_), _) | (_, BoundDivOperand::Override(_)) => {
            unreachable!("division operand overrides were dispatched before the native path")
        }
        (BoundDivOperand::Scalar(_), _) => {
            unreachable!("Tensor.div receivers are never parsed as scalar operands")
        }
    };
    Ok(Py::new(
        py,
        PyTensor::new(result.map_err(|error| tensor_error(&error))?),
    )?
    .into_any())
}

fn ordered_subtraction_overrides<'py>(
    operation: SubtractionOperation,
    call: &BoundTopLevelSubtractionCall<'py>,
) -> PyResult<Vec<ProbedTorchFunctionOverride<'py>>> {
    let input = match &call.input {
        BoundSubOperand::Tensor(_) | BoundSubOperand::Scalar(_) => None,
        BoundSubOperand::Override(probed) => Some(probed),
    };
    let other = match &call.other {
        BoundSubOperand::Tensor(_) | BoundSubOperand::Scalar(_) => None,
        BoundSubOperand::Override(probed) => Some(probed),
    };
    let alpha = match &call.alpha {
        BoundSubAlpha::Default | BoundSubAlpha::PythonBool | BoundSubAlpha::NonDefault => None,
        BoundSubAlpha::Override(probed) => Some(probed),
    };
    let out = match &call.out {
        Some(BoundTensorOrTorchFunction::Override(probed)) => Some(probed),
        Some(BoundTensorOrTorchFunction::Tensor(_)) | None => None,
    };

    let mut overrides = Vec::new();
    overrides
        .try_reserve_exact(4)
        .map_err(|_| PyMemoryError::new_err(operation.dispatch_allocation_error()))?;
    for probed in [input, other, alpha, out].into_iter().flatten() {
        insert_ordered_torch_function_override(&mut overrides, probed)?;
    }
    Ok(overrides)
}

fn dispatch_top_level_subtraction(
    operation: SubtractionOperation,
    py: Python<'_>,
    call: &BoundTopLevelSubtractionCall<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let overrides = ordered_subtraction_overrides(operation, call)?;
    if torch_function_mode_stack::is_empty() && overrides.is_empty() {
        return apply_top_level_subtraction(operation, py, call, args, kwargs);
    }

    let function = variable_function(py, operation.name())?;
    let types = PyTuple::new(
        py,
        overrides.iter().map(|probed| probed.dispatch_type.clone()),
    )?;

    // Generated variable functions validate their schema before dispatch, but
    // delay native-only unsupported cases such as concrete out tensors and
    // nondefault alpha until active handlers have had a chance to override.
    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    for probed in &overrides {
        let handler = resolve_torch_function_override(py, probed)?;
        let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    if active_mode.get().is_none() && overrides.is_empty() {
        return apply_top_level_subtraction(operation, py, call, args, kwargs);
    }

    Err(torch_function_dispatch_error_for_overrides(
        py,
        operation.qualified_name(),
        active_mode.get(),
        &overrides,
    )?)
}

fn apply_top_level_subtraction(
    operation: SubtractionOperation,
    py: Python<'_>,
    call: &BoundTopLevelSubtractionCall<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    if call.alpha_is_positional
        && !matches!(
            (&call.input, &call.other),
            (BoundSubOperand::Tensor(_), BoundSubOperand::Scalar(_))
        )
    {
        return Err(top_level_subtract_binding_error(
            operation,
            args,
            kwargs,
            Some(&SubtractBindingMismatch::InvalidPositionalOverload),
        )?);
    }
    if call.out.is_some() {
        return Err(PyRuntimeError::new_err(operation.out_unsupported_error()));
    }
    match &call.alpha {
        BoundSubAlpha::Default => {}
        BoundSubAlpha::PythonBool => {
            return Err(PyRuntimeError::new_err(
                "Boolean alpha only supported for Boolean results.",
            ));
        }
        BoundSubAlpha::NonDefault => {
            return Err(PyNotImplementedError::new_err(
                operation.alpha_unsupported_error(),
            ));
        }
        BoundSubAlpha::Override(_) => {
            unreachable!("subtraction alpha overrides were dispatched before the native path")
        }
    }

    let result = match (&call.input, &call.other) {
        (BoundSubOperand::Tensor(input), BoundSubOperand::Tensor(other)) => {
            let other = other.try_borrow()?;
            BinaryOperation::Subtract.apply_tensors(&input.try_borrow()?.inner, &other.inner)
        }
        (BoundSubOperand::Tensor(tensor), BoundSubOperand::Scalar(scalar)) => {
            let scalar = parse_supported_arithmetic_scalar(scalar)?;
            if scalar.is_python_bool() {
                return Err(bool_subtraction_error());
            }
            BinaryOperation::Subtract.apply_scalar(
                &tensor.try_borrow()?.inner,
                scalar.into_f32(),
                false,
            )
        }
        (BoundSubOperand::Scalar(scalar), BoundSubOperand::Tensor(tensor)) => {
            let scalar = parse_supported_arithmetic_scalar(scalar)?;
            if scalar.is_python_bool() {
                return Err(bool_subtraction_error());
            }
            BinaryOperation::Subtract.apply_scalar(
                &tensor.try_borrow()?.inner,
                scalar.into_f32(),
                true,
            )
        }
        (BoundSubOperand::Scalar(_), BoundSubOperand::Scalar(_)) => {
            return Err(PyTypeError::new_err(format!(
                "{}(): scalar-scalar subtraction is not supported; at least one operand must be Tensor",
                operation.name()
            )));
        }
        (BoundSubOperand::Override(_), _) | (_, BoundSubOperand::Override(_)) => {
            unreachable!("subtraction operand overrides were dispatched before the native path")
        }
    };
    Ok(Py::new(
        py,
        PyTensor::new(result.map_err(|error| tensor_error(&error))?),
    )?
    .into_any())
}

fn tensor_add_sub_method<'py>(
    operation: AddSubMethodOperation,
    slf: &Bound<'py, PyTensorBase>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (other_argument, alpha_argument, keyword_error) =
        bind_tensor_add_sub_method_arguments(operation, args, kwargs)?;
    let other = parse_tensor_add_sub_method_other(operation, &other_argument, args, kwargs)?;
    let alpha =
        parse_tensor_add_sub_method_alpha(operation, alpha_argument.as_ref(), args, kwargs)?;
    if let Some(keyword_error) = keyword_error {
        return Err(keyword_error);
    }

    let input = parse_tensor_add_sub_method_receiver(operation, slf.as_any())?;
    let alpha_precedes_other =
        tensor_add_sub_method_alpha_precedes_other(alpha_argument.as_ref(), &other_argument);
    let call = BoundTensorMethodAddSubCall {
        input,
        other,
        alpha,
        alpha_precedes_other,
    };
    dispatch_tensor_add_sub_method(operation, slf.py(), slf.as_any(), &call, args, kwargs)
}

fn tensor_add_sub_method_alpha_precedes_other(
    alpha: Option<&ParsedCallArgument<'_>>,
    other: &ParsedCallArgument<'_>,
) -> bool {
    alpha
        .and_then(|alpha| alpha.position)
        .is_some_and(|alpha_position| {
            other
                .position
                .is_none_or(|other_position| alpha_position < other_position)
        })
}

fn parse_tensor_add_sub_method_receiver<'py>(
    operation: AddSubMethodOperation,
    receiver: &Bound<'py, PyAny>,
) -> PyResult<BoundSubOperand<'py>> {
    if receiver.is_exact_instance_of::<PyTensor>() {
        return Ok(BoundSubOperand::Tensor(
            receiver.cast::<PyTensor>()?.clone(),
        ));
    }
    if let Some(probed) = probe_torch_function_override(receiver) {
        return Ok(BoundSubOperand::Override(probed));
    }
    if receiver.is_instance_of::<PyTensor>() {
        return Err(add_sub_method_unsupported_native_input(operation));
    }
    Err(add_sub_method_unsupported_native_input(operation))
}

fn parse_tensor_add_sub_method_other<'py>(
    operation: AddSubMethodOperation,
    value: &ParsedCallArgument<'py>,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundSubOperand<'py>> {
    if value.value.is_exact_instance_of::<PyTensor>() {
        return Ok(BoundSubOperand::Tensor(
            value.value.cast::<PyTensor>()?.clone(),
        ));
    }
    if let Some(probed) = probe_torch_function_override(&value.value) {
        return Ok(BoundSubOperand::Override(probed));
    }
    if value.value.is_instance_of::<PyTensor>() {
        return Err(add_sub_method_unsupported_native_input(operation));
    }
    if is_real_arithmetic_scalar(&value.value)? {
        return Ok(BoundSubOperand::Scalar(value.value.clone()));
    }

    if matches!(operation, AddSubMethodOperation::Subtract) {
        return Err(tensor_subtract_binding_error(
            SubtractionOperation::Subtract,
            positional,
            keywords,
            None,
        )?);
    }
    parse_tensor_argument(operation.name(), "other", value)?;
    unreachable!("unsupported Tensor add/sub operands were rejected by parse_tensor_argument")
}

fn parse_tensor_add_sub_method_alpha<'py>(
    operation: AddSubMethodOperation,
    alpha: Option<&ParsedCallArgument<'py>>,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundSubAlpha<'py>> {
    let Some(alpha) = alpha else {
        return Ok(BoundSubAlpha::Default);
    };
    if let Some(probed) = probe_torch_function_override(&alpha.value) {
        return Ok(BoundSubAlpha::Override(probed));
    }
    let Some(scalar) = parse_arithmetic_scalar(&alpha.value)? else {
        if matches!(operation, AddSubMethodOperation::Subtract) {
            return Err(tensor_subtract_binding_error(
                SubtractionOperation::Subtract,
                positional,
                keywords,
                Some(&SubtractBindingMismatch::IncorrectKeyword("alpha")),
            )?);
        }
        let actual = python_type_name(&alpha.value)?;
        return Err(PyTypeError::new_err(format!(
            "{}(): argument 'alpha' must be Number, not {actual}",
            operation.name()
        )));
    };
    if scalar.is_python_bool() {
        return Ok(BoundSubAlpha::PythonBool);
    }
    if scalar.is_one() {
        Ok(BoundSubAlpha::Default)
    } else {
        Ok(BoundSubAlpha::NonDefault)
    }
}

fn ordered_tensor_add_sub_method_overrides<'py>(
    operation: AddSubMethodOperation,
    call: &BoundTensorMethodAddSubCall<'py>,
) -> PyResult<Vec<ProbedTorchFunctionOverride<'py>>> {
    let input = match &call.input {
        BoundSubOperand::Override(probed) => Some(probed),
        BoundSubOperand::Tensor(_) | BoundSubOperand::Scalar(_) => None,
    };
    let other = match &call.other {
        BoundSubOperand::Override(probed) => Some(probed),
        BoundSubOperand::Tensor(_) | BoundSubOperand::Scalar(_) => None,
    };
    let alpha = match &call.alpha {
        BoundSubAlpha::Override(probed) => Some(probed),
        BoundSubAlpha::Default | BoundSubAlpha::PythonBool | BoundSubAlpha::NonDefault => None,
    };

    let mut overrides = Vec::new();
    overrides
        .try_reserve_exact(3)
        .map_err(|_| PyMemoryError::new_err(operation.dispatch_allocation_error()))?;
    let ordered_operands = if call.alpha_precedes_other {
        [input, alpha, other]
    } else {
        [input, other, alpha]
    };
    for probed in ordered_operands.into_iter().flatten() {
        insert_ordered_torch_function_override(&mut overrides, probed)?;
    }
    Ok(overrides)
}

fn dispatch_tensor_add_sub_method(
    operation: AddSubMethodOperation,
    py: Python<'_>,
    receiver: &Bound<'_, PyAny>,
    call: &BoundTensorMethodAddSubCall<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let overrides = ordered_tensor_add_sub_method_overrides(operation, call)?;
    if torch_function_mode_stack::is_empty() && overrides.is_empty() {
        return apply_tensor_add_sub_method(operation, py, call);
    }

    let function = py
        .get_type::<PyTensorBase>()
        .getattr(operation.name())?
        .unbind();
    let types = PyTuple::new(
        py,
        overrides.iter().map(|probed| probed.dispatch_type.clone()),
    )?;
    let argument_count = args.len().checked_add(1).ok_or_else(|| {
        PyMemoryError::new_err(format!(
            "{} dispatch argument count overflowed",
            operation.name()
        ))
    })?;
    let mut call_arguments = Vec::new();
    call_arguments
        .try_reserve_exact(argument_count)
        .map_err(|_| {
            PyMemoryError::new_err(format!(
                "unable to allocate {} dispatch arguments",
                operation.name()
            ))
        })?;
    call_arguments.push(receiver.clone());
    call_arguments.extend(args.iter());
    let call_args = PyTuple::new(py, call_arguments)?;

    let active_mode = torch_function_mode_stack::pop();
    if let Some(mode) = active_mode.get() {
        validate_torch_function_mode_handler(mode.bind(py))?;
        let handler = mode.bind(py).getattr("__torch_function__")?;
        let result =
            call_torch_function_handler(py, &handler, &function, &types, &call_args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    for probed in &overrides {
        let handler = resolve_torch_function_override(py, probed)?;
        let result =
            call_torch_function_handler(py, &handler, &function, &types, &call_args, kwargs)?;
        if !is_not_implemented(py, &result) {
            return Ok(result);
        }
    }

    Err(torch_function_dispatch_error_for_overrides(
        py,
        operation.qualified_method_name(),
        active_mode.get(),
        &overrides,
    )?)
}

fn apply_tensor_add_sub_method(
    operation: AddSubMethodOperation,
    py: Python<'_>,
    call: &BoundTensorMethodAddSubCall<'_>,
) -> PyResult<Py<PyAny>> {
    match &call.alpha {
        BoundSubAlpha::Default => {}
        BoundSubAlpha::PythonBool => {
            return Err(PyRuntimeError::new_err(
                "Boolean alpha only supported for Boolean results.",
            ));
        }
        BoundSubAlpha::NonDefault => {
            return Err(PyNotImplementedError::new_err(
                operation.alpha_unsupported_error(),
            ));
        }
        BoundSubAlpha::Override(_) => {
            unreachable!("add/sub alpha overrides were dispatched before the native path")
        }
    }

    let result = match (&call.input, &call.other) {
        (BoundSubOperand::Tensor(input), BoundSubOperand::Tensor(other)) => {
            let other = other.try_borrow()?;
            operation
                .binary_operation()
                .apply_tensors(&input.try_borrow()?.inner, &other.inner)
        }
        (BoundSubOperand::Tensor(tensor), BoundSubOperand::Scalar(scalar)) => {
            let scalar = parse_supported_arithmetic_scalar(scalar)?;
            if matches!(
                operation,
                AddSubMethodOperation::Sub | AddSubMethodOperation::Subtract
            ) && scalar.is_python_bool()
            {
                return Err(bool_subtraction_error());
            }
            operation.binary_operation().apply_scalar(
                &tensor.try_borrow()?.inner,
                scalar.into_f32(),
                false,
            )
        }
        (BoundSubOperand::Override(_), _) | (_, BoundSubOperand::Override(_)) => {
            unreachable!("add/sub operand overrides were dispatched before the native path")
        }
        (BoundSubOperand::Scalar(_), _) => {
            unreachable!("Tensor add/sub receivers are never parsed as scalar operands")
        }
    };
    Ok(Py::new(
        py,
        PyTensor::new(result.map_err(|error| tensor_error(&error))?),
    )?
    .into_any())
}

fn parse_supported_arithmetic_scalar(value: &Bound<'_, PyAny>) -> PyResult<ParsedArithmeticScalar> {
    match parse_arithmetic_scalar(value) {
        Ok(Some(ParsedArithmeticScalar::WideNumpyUnsigned)) => {
            Err(PyTypeError::new_err("an integer is required"))
        }
        Ok(Some(scalar)) => Ok(scalar),
        Ok(None) => unreachable!("top-level sub scalar types were checked while binding"),
        Err(_) if value.is_instance_of::<PyInt>() => {
            let message = if python_integer_is_negative(value)? {
                "can't convert negative int to unsigned"
            } else {
                "int too big to convert"
            };
            Err(PyOverflowError::new_err(message))
        }
        Err(error) => Err(error),
    }
}

fn parse_top_level_mul_scalar(value: &Bound<'_, PyAny>) -> PyResult<f32> {
    match parse_arithmetic_scalar(value) {
        Ok(Some(ParsedArithmeticScalar::WideNumpyUnsigned)) => {
            Err(PyTypeError::new_err("an integer is required"))
        }
        Ok(Some(scalar)) => Ok(scalar.into_f32()),
        Ok(None) => unreachable!("top-level mul scalar types were checked while binding"),
        Err(_) if value.is_instance_of::<PyInt>() => {
            let message = if python_integer_is_negative(value)? {
                "can't convert negative int to unsigned"
            } else {
                "int too big to convert"
            };
            Err(PyOverflowError::new_err(message))
        }
        Err(error) => Err(error),
    }
}

struct AsTensorCallArguments<'py> {
    data: Option<ParsedCallArgument<'py>>,
    dtype: Option<Bound<'py, PyAny>>,
    device: Option<Bound<'py, PyAny>>,
    keyword_error: Option<PyErr>,
}

struct AsArrayCallArguments<'py> {
    obj: Option<ParsedCallArgument<'py>>,
    dtype: Option<Bound<'py, PyAny>>,
    device: Option<Bound<'py, PyAny>>,
    copy: Option<Bound<'py, PyAny>>,
    requires_grad: Option<Bound<'py, PyAny>>,
    keyword_error: Option<PyErr>,
}

struct ScalarTensorCallArguments<'py> {
    scalar: Option<ParsedCallArgument<'py>>,
    dtype: Option<Bound<'py, PyAny>>,
    layout: Option<Bound<'py, PyAny>>,
    device: Option<Bound<'py, PyAny>>,
    pin_memory: Option<Bound<'py, PyAny>>,
    requires_grad: Option<Bound<'py, PyAny>>,
    keyword_error: Option<PyErr>,
}

struct ArangeCallArguments<'py> {
    end: Option<ParsedCallArgument<'py>>,
    unsupported_overload: bool,
    out: Option<Bound<'py, PyAny>>,
    dtype: Option<Bound<'py, PyAny>>,
    layout: Option<Bound<'py, PyAny>>,
    device: Option<Bound<'py, PyAny>>,
    pin_memory: Option<Bound<'py, PyAny>>,
    requires_grad: Option<Bound<'py, PyAny>>,
    keyword_error: Option<PyErr>,
}

struct CreationCallArguments<'py> {
    size: Option<CreationSizeArgument<'py>>,
    shape: Option<Bound<'py, PyAny>>,
    out: Option<Bound<'py, PyAny>>,
    dtype: Option<Bound<'py, PyAny>>,
    layout: Option<Bound<'py, PyAny>>,
    device: Option<Bound<'py, PyAny>>,
    pin_memory: Option<Bound<'py, PyAny>>,
    requires_grad: Option<Bound<'py, PyAny>>,
    keyword_error: Option<PyErr>,
}

struct LikeFactoryCallArguments<'py> {
    input: Option<ParsedCallArgument<'py>>,
    dtype: Option<Bound<'py, PyAny>>,
    layout: Option<Bound<'py, PyAny>>,
    device: Option<Bound<'py, PyAny>>,
    requires_grad: Option<Bound<'py, PyAny>>,
    memory_format: Option<Bound<'py, PyAny>>,
    keyword_error: Option<PyErr>,
}

struct FullLikeCallArguments<'py> {
    input: Option<ParsedCallArgument<'py>>,
    fill_value: Option<Bound<'py, PyAny>>,
    dtype: Option<Bound<'py, PyAny>>,
    layout: Option<Bound<'py, PyAny>>,
    device: Option<Bound<'py, PyAny>>,
    requires_grad: Option<Bound<'py, PyAny>>,
    memory_format: Option<Bound<'py, PyAny>>,
    keyword_error: Option<PyErr>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CreationSizeOrigin {
    Positional,
    SizeKeyword,
    ShapeKeyword,
}

enum CreationSizeArgument<'py> {
    Single {
        value: Bound<'py, PyAny>,
        origin: CreationSizeOrigin,
    },
    Variadic(Bound<'py, PyTuple>),
}

enum PendingCreationSize<'py> {
    Dimensions(Vec<usize>),
    PositionalScalar(Bound<'py, PyAny>),
    Variadic(Bound<'py, PyTuple>),
}

struct ParsedCreationSize {
    dimensions: Vec<usize>,
    scalar_dimension: Option<usize>,
}

struct FullCallArguments<'py> {
    size: Option<Bound<'py, PyAny>>,
    fill_value: Option<Bound<'py, PyAny>>,
    out: Option<Bound<'py, PyAny>>,
    dtype: Option<Bound<'py, PyAny>>,
    layout: Option<Bound<'py, PyAny>>,
    device: Option<Bound<'py, PyAny>>,
    pin_memory: Option<Bound<'py, PyAny>>,
    requires_grad: Option<Bound<'py, PyAny>>,
    keyword_error: Option<PyErr>,
}

struct ParsedFullArguments {
    size: Vec<i64>,
    fill_value: ParsedFillValue,
    has_out: bool,
    dtype: DType,
    device: Device,
    pin_memory: bool,
    requires_grad: bool,
}

struct EyeCallArguments<'py> {
    n: Option<Bound<'py, PyAny>>,
    m: Option<Bound<'py, PyAny>>,
    dtype: Option<Bound<'py, PyAny>>,
    device: Option<Bound<'py, PyAny>>,
    requires_grad: Option<Bound<'py, PyAny>>,
    keyword_error: Option<PyErr>,
}

enum ParsedSqueezeDimensions {
    All,
    Single(i64),
    Multiple(Vec<i64>),
}

#[derive(Clone, Copy)]
enum BinaryOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DivisionOperation {
    Div,
    Divide,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MultiplicationOperation {
    Mul,
    Multiply,
}

#[derive(Clone, Copy)]
enum AddSubMethodOperation {
    Add,
    Sub,
    Subtract,
}

#[derive(Clone, Copy)]
enum SubtractionOperation {
    Sub,
    Subtract,
}

impl AddSubMethodOperation {
    const fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Subtract => "subtract",
        }
    }

    const fn qualified_method_name(self) -> &'static str {
        match self {
            Self::Add => "torch.Tensor.add",
            Self::Sub => "torch.Tensor.sub",
            Self::Subtract => "torch.Tensor.subtract",
        }
    }

    const fn dispatch_allocation_error(self) -> &'static str {
        match self {
            Self::Add => "unable to allocate add dispatch operands",
            Self::Sub => "unable to allocate sub dispatch operands",
            Self::Subtract => "unable to allocate subtract dispatch operands",
        }
    }

    const fn alpha_unsupported_error(self) -> &'static str {
        match self {
            Self::Add => "add(): alpha values other than 1 are not supported",
            Self::Sub => "sub(): alpha values other than 1 are not supported",
            Self::Subtract => "subtract(): alpha values other than 1 are not supported",
        }
    }

    const fn binary_operation(self) -> BinaryOperation {
        match self {
            Self::Add => BinaryOperation::Add,
            Self::Sub | Self::Subtract => BinaryOperation::Subtract,
        }
    }
}

impl SubtractionOperation {
    const fn name(self) -> &'static str {
        match self {
            Self::Sub => "sub",
            Self::Subtract => "subtract",
        }
    }

    const fn qualified_name(self) -> &'static str {
        match self {
            Self::Sub => "torch.sub",
            Self::Subtract => "torch.subtract",
        }
    }

    const fn dispatch_allocation_error(self) -> &'static str {
        match self {
            Self::Sub => "unable to allocate sub dispatch operands",
            Self::Subtract => "unable to allocate subtract dispatch operands",
        }
    }

    const fn out_unsupported_error(self) -> &'static str {
        match self {
            Self::Sub => "sub(): the 'out' argument is not supported",
            Self::Subtract => "subtract(): the 'out' argument is not supported",
        }
    }

    const fn alpha_unsupported_error(self) -> &'static str {
        match self {
            Self::Sub => "sub(): alpha values other than 1 are not supported",
            Self::Subtract => "subtract(): alpha values other than 1 are not supported",
        }
    }
}

impl MultiplicationOperation {
    const fn name(self) -> &'static str {
        match self {
            Self::Mul => "mul",
            Self::Multiply => "multiply",
        }
    }

    const fn qualified_name(self) -> &'static str {
        match self {
            Self::Mul => "torch.mul",
            Self::Multiply => "torch.multiply",
        }
    }

    const fn dispatch_allocation_error(self) -> &'static str {
        match self {
            Self::Mul => "unable to allocate mul dispatch operands",
            Self::Multiply => "unable to allocate multiply dispatch operands",
        }
    }
}

impl DivisionOperation {
    const fn name(self) -> &'static str {
        match self {
            Self::Div => "div",
            Self::Divide => "divide",
        }
    }

    const fn qualified_name(self) -> &'static str {
        match self {
            Self::Div => "torch.div",
            Self::Divide => "torch.divide",
        }
    }

    const fn qualified_method_name(self) -> &'static str {
        match self {
            Self::Div => "torch.Tensor.div",
            Self::Divide => "torch.Tensor.divide",
        }
    }

    const fn dispatch_allocation_error(self) -> &'static str {
        match self {
            Self::Div => "unable to allocate div dispatch operands",
            Self::Divide => "unable to allocate divide dispatch operands",
        }
    }

    const fn rounding_mode_unsupported_error(self) -> &'static str {
        match self {
            Self::Div => "div(): non-None rounding_mode is not supported",
            Self::Divide => "divide(): non-None rounding_mode is not supported",
        }
    }

    const fn out_unsupported_error(self) -> &'static str {
        match self {
            Self::Div => "div(): the 'out' argument is not supported",
            Self::Divide => "divide(): the 'out' argument is not supported",
        }
    }

    const fn autograd_unsupported_error(self) -> &'static str {
        match self {
            Self::Div => "div(): autograd recording is not supported",
            Self::Divide => "divide(): autograd recording is not supported",
        }
    }
}

#[pymethods]
impl PyTensor {
    #[classattr]
    fn __array_priority__() -> f64 {
        1000.0
    }

    #[doc = "\nReturns the number of bytes consumed by the \"view\" of elements of the Tensor\nif the Tensor does not use sparse storage layout.\nDefined to be :meth:`~Tensor.numel()` * :meth:`~Tensor.element_size()`\n"]
    #[getter]
    fn nbytes(&self) -> usize {
        self.inner.numel() * self.inner.element_size()
    }

    #[getter]
    fn device(&self) -> PyDevice {
        PyDevice::from_device(self.inner.device())
    }

    /// NumPy-style transpose view with every dimension reversed.
    #[getter(T)]
    fn numpy_transpose(&self, py: Python<'_>) -> PyResult<Self> {
        match self.inner.shape().len() {
            0 => warn_once(py, &T_SCALAR_WARNING_EMITTED, T_SCALAR_WARNING)?,
            2 => {}
            _ => warn_once(py, &T_NON_MATRIX_WARNING_EMITTED, T_NON_MATRIX_WARNING)?,
        }
        self.inner
            .reverse_dimensions()
            .map(Self::new)
            .map_err(|error| transpose_error(&error))
    }

    /// Matrix transpose view with the final two dimensions swapped.
    #[getter(mT)]
    fn matrix_transpose(slf: PyRef<'_, Self>) -> PyResult<Py<Self>> {
        let rank = slf.inner.shape().len();
        if rank == 0 {
            warn_once(slf.py(), &MT_SCALAR_WARNING_EMITTED, MT_SCALAR_WARNING)?;
            return Ok(slf.into());
        }
        let inner = slf
            .inner
            .matrix_transpose()
            .map_err(|error| transpose_error(&error))?;
        Py::new(slf.py(), Self::new(inner))
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nt() -> Tensor\n\nSee :func:`torch.t`\n"]
    #[pyo3(text_signature = None)]
    fn t(&self) -> PyResult<Self> {
        let rank = self.inner.shape().len();
        if rank > 2 {
            return Err(PyRuntimeError::new_err(format!(
                "t() expects a tensor with <= 2 dimensions, but self is {rank}D"
            )));
        }
        self.inner
            .t()
            .map(Self::new)
            .map_err(|error| transpose_error(&error))
    }

    #[pyo3(signature = (dim=None))]
    fn stride(&self, py: Python<'_>, dim: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        let Some(dim) = dim else {
            return Ok(PyTuple::new(py, self.inner.stride().iter().copied())?
                .into_any()
                .unbind());
        };
        let dim = parse_stride_dimension(dim)?;
        let axis = normalize_dimension(dim, self.inner.shape().len())?;
        self.inner.stride()[axis].into_py_any(py)
    }

    fn storage_offset(&self) -> usize {
        self.inner.storage_offset()
    }

    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, memory_format=torch.contiguous_format)")]
    fn is_contiguous(
        &self,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<bool> {
        if !args.is_empty() {
            return Err(PyTypeError::new_err(format!(
                "is_contiguous() takes 0 positional arguments but {} {} given",
                args.len(),
                if args.len() == 1 { "was" } else { "were" }
            )));
        }
        let mut memory_format = MemoryFormat::Contiguous;
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key = key.extract::<String>()?;
                if key != "memory_format" {
                    return Err(PyTypeError::new_err(format!(
                        "is_contiguous() got an unexpected keyword argument '{key}'"
                    )));
                }
                memory_format = parse_is_contiguous_memory_format(&value)?;
            }
        }
        Ok(self.inner.is_contiguous_with_memory_format(memory_format))
    }

    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, memory_format=torch.contiguous_format)")]
    fn contiguous(
        slf: PyRef<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<Self>> {
        if !args.is_empty() {
            return Err(PyTypeError::new_err(format!(
                "contiguous() takes 0 positional arguments but {} {} given",
                args.len(),
                if args.len() == 1 { "was" } else { "were" }
            )));
        }

        let mut memory_format = MemoryFormat::Contiguous;
        if let Some(kwargs) = kwargs {
            // PyTorch converts the recognized argument before reporting any
            // extra keywords, independent of keyword insertion order.
            if let Some(value) = kwargs.get_item("memory_format")? {
                memory_format = parse_contiguous_memory_format(&value)?;
            }
            for (key, _) in kwargs {
                let key = key.extract::<String>()?;
                if key != "memory_format" {
                    return Err(PyTypeError::new_err(format!(
                        "contiguous() got an unexpected keyword argument '{key}'"
                    )));
                }
            }
        }

        if slf.inner.is_contiguous_with_memory_format(memory_format) {
            return Ok(slf.into());
        }
        let inner = slf
            .inner
            .try_contiguous(memory_format)
            .map_err(|error| tensor_error(&error))?;
        Py::new(slf.py(), Self::new(inner))
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn transpose(
        &self,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let ([dim0, dim1], keyword_error) =
            bind_dimension_swap_arguments("transpose", args, kwargs, ["dim0", "dim1"])?;
        if let Some(keyword_error) = keyword_error {
            return Err(keyword_error);
        }
        let [dim0, dim1] =
            parse_dimension_swap_dimensions("transpose", ["dim0", "dim1"], &dim0, &dim1)?;
        self.inner
            .transpose(dim0, dim1)
            .map(Self::new)
            .map_err(|error| transpose_error(&error))
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nswapdims(dim0, dim1) -> Tensor\n\nSee :func:`torch.swapdims`\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn swapdims(
        &self,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let ([dim0, dim1], keyword_error) =
            bind_dimension_swap_arguments("swapdims", args, kwargs, ["dim0", "dim1"])?;
        if let Some(keyword_error) = keyword_error {
            return Err(keyword_error);
        }
        let [dim0, dim1] =
            parse_dimension_swap_dimensions("swapdims", ["dim0", "dim1"], &dim0, &dim1)?;
        self.inner
            .transpose(dim0, dim1)
            .map(Self::new)
            .map_err(|error| transpose_error(&error))
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nswapaxes(axis0, axis1) -> Tensor\n\nSee :func:`torch.swapaxes`\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn swapaxes(
        &self,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let ([axis0, axis1], keyword_error) =
            bind_dimension_swap_arguments("swapaxes", args, kwargs, ["axis0", "axis1"])?;
        if let Some(keyword_error) = keyword_error {
            return Err(keyword_error);
        }
        let [axis0, axis1] =
            parse_dimension_swap_dimensions("swapaxes", ["axis0", "axis1"], &axis0, &axis1)?;
        self.inner
            .transpose(axis0, axis1)
            .map(Self::new)
            .map_err(|error| transpose_error(&error))
    }

    #[pyo3(signature = (*args, **kwargs), text_signature = "(dim=None)")]
    fn squeeze(
        &self,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let dimensions = bind_method_squeeze_arguments(args, kwargs)?;
        apply_squeeze(&self.inner, dimensions)
            .map(Self::new)
            .map_err(|error| tensor_error(&error))
    }

    #[pyo3(signature = (*args, **kwargs), text_signature = "(start_dim=0, end_dim=-1)")]
    fn flatten(
        slf: PyRef<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<Self>> {
        let (start_dim, end_dim) = bind_method_flatten_arguments(args, kwargs)?;
        let inner = slf
            .inner
            .flatten(start_dim, end_dim)
            .map_err(|error| tensor_error(&error))?;
        if same_tensor_metadata(&slf.inner, &inner) {
            return Ok(slf.into());
        }
        Py::new(slf.py(), Self::new(inner))
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nravel() -> Tensor\n\nsee :func:`torch.ravel`\n"]
    #[pyo3(text_signature = None)]
    fn ravel(&self) -> PyResult<Self> {
        self.inner
            .ravel()
            .map(Self::new)
            .map_err(|error| tensor_error(&error))
    }

    fn __iter__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let dimension = py
            .get_type::<PyTensorBase>()
            .getattr("dim")?
            .call1((slf.clone(),))?;
        if dimension.eq(0_usize)? {
            return Err(PyTypeError::new_err("iteration over a 0-d tensor"));
        }
        let outputs = py
            .get_type::<PyTensorBase>()
            .getattr("unbind")?
            .call1((slf.clone(), 0_i64))?;
        Ok(outputs.try_iter()?.into_any().unbind())
    }

    #[pyo3(signature = (*shape_dimensions, shape=None))]
    fn reshape(
        &self,
        shape_dimensions: &Bound<'_, PyTuple>,
        shape: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let shape = parse_reshape_shape(shape_dimensions, shape)?;
        self.inner
            .reshape(shape)
            .map(Self::new)
            .map_err(|error| tensor_error(&error))
    }

    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__(
        &self,
        py: Python<'_>,
        dtype: Option<&Bound<'_, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        if copy == Some(false) {
            return Err(PyValueError::new_err(
                "cannot create a non-copying NumPy view of tensor storage",
            ));
        }
        self.numpy_array_copy(py, dtype)
    }

    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn is_same_size(
        &self,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<bool> {
        let (arguments, keyword_error) =
            bind_tensor_arguments("is_same_size", args, kwargs, ["other"])?;
        let other = parse_tensor_argument("is_same_size", "other", &arguments[0])?;
        if let Some(keyword_error) = keyword_error {
            return Err(keyword_error);
        }
        let other = other.try_borrow()?;
        Ok(self.inner.is_same_size(&other.inner))
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nis_set_to(tensor) -> bool\n\nReturns True if both tensors are pointing to the exact same memory (same\nstorage, offset, size and stride).\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn is_set_to(
        &self,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<bool> {
        let (arguments, keyword_error) =
            bind_tensor_arguments("is_set_to", args, kwargs, ["tensor"])?;
        let tensor = parse_tensor_argument("is_set_to", "tensor", &arguments[0])?;
        if let Some(keyword_error) = keyword_error {
            return Err(keyword_error);
        }
        let tensor = tensor.try_borrow()?;
        Ok(self.inner.is_set_to(&tensor.inner))
    }

    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let values = self
            .inner
            .try_to_vec()
            .map_err(|error| tensor_error(&error))?;
        nested_list(py, &values, self.inner.shape())
    }

    fn item(&self) -> PyResult<f32> {
        self.inner.item().map_err(|error| item_error(&error))
    }

    #[pyo3(text_signature = None)]
    fn is_nonzero(&self) -> PyResult<bool> {
        self.truth_value()
    }

    /// equal(other) -> bool
    ///
    /// See :func:`torch.equal`.
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn equal(
        &self,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<bool> {
        let (arguments, keyword_error) = bind_tensor_arguments("equal", args, kwargs, ["other"])?;
        let other = parse_tensor_argument("equal", "other", &arguments[0])?;
        if let Some(keyword_error) = keyword_error {
            return Err(keyword_error);
        }
        let other = other.try_borrow()?;
        Ok(self.inner == other.inner)
    }

    #[pyo3(signature = (*, memory_format=None))]
    fn clone(&self, memory_format: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let memory_format = parse_clone_memory_format(memory_format)?;
        self.inner
            .try_clone_with_memory_format(memory_format)
            .map(Self::new)
            .map_err(|error| tensor_error(&error))
    }

    fn detach(&self) -> PyResult<Self> {
        self.inner
            .detach()
            .map(Self::new)
            .map_err(|error| tensor_error(&error))
    }

    fn backward(&self) -> PyResult<()> {
        self.inner.backward().map_err(|error| tensor_error(&error))
    }

    fn relu(&self) -> PyResult<Self> {
        self.inner
            .relu()
            .map(Self::new)
            .map_err(|error| tensor_error(&error))
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nsum(dim=None, keepdim=False, dtype=None) -> Tensor\n\nSee :func:`torch.sum`\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn sum(&self, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let call = bind_method_sum_arguments(args, kwargs)?;
        let mut output = self.inner.sum();
        if call.keepdim_full_reduction {
            output = output
                .reshape(full_reduction_keepdim_shape(&self.inner))
                .map_err(|error| tensor_error(&error))?;
        }
        Ok(Self::new(output))
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nmean(dim=None, keepdim=False, dtype=None) -> Tensor\n\nSee :func:`torch.mean`\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn mean(
        &self,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let call = bind_method_mean_arguments(args, kwargs)?;
        let mut output = self.inner.mean().map_err(|error| tensor_error(&error))?;
        if call.keepdim_full_reduction {
            output = output
                .reshape(full_reduction_keepdim_shape(&self.inner))
                .map_err(|error| tensor_error(&error))?;
        }
        Ok(Self::new(output))
    }

    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary_operation(py, other, BinaryOperation::Add, false)
    }

    fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary_operation(py, other, BinaryOperation::Add, true)
    }

    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary_operation(py, other, BinaryOperation::Subtract, false)
    }

    fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary_operation(py, other, BinaryOperation::Subtract, true)
    }

    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary_operation(py, other, BinaryOperation::Multiply, false)
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nmul(value) -> Tensor\n\nSee :func:`torch.mul`.\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn mul(&self, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        self.multiplication_method(MultiplicationOperation::Mul, args, kwargs)
    }

    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary_operation(py, other, BinaryOperation::Multiply, true)
    }

    fn __neg__(&self) -> PyResult<Self> {
        self.negated()
    }

    fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary_operation(py, other, BinaryOperation::Divide, false)
    }

    fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary_operation(py, other, BinaryOperation::Divide, true)
    }

    fn __matmul__(&self, other: &Self) -> PyResult<Self> {
        self.matrix_multiply(other)
    }

    fn __bool__(&self) -> PyResult<bool> {
        self.truth_value()
    }

    fn __len__(&self) -> PyResult<usize> {
        self.inner
            .shape()
            .first()
            .copied()
            .ok_or_else(|| PyTypeError::new_err("len() of a 0-d tensor"))
    }

    fn __repr__(&self) -> PyResult<String> {
        let values = self
            .inner
            .try_to_vec()
            .map_err(|error| tensor_error(&error))?;
        Ok(format!(
            "tensor({:?}, shape={:?})",
            values,
            self.inner.shape()
        ))
    }
}

impl PyTensor {
    fn matrix_multiply(&self, other: &Self) -> PyResult<Self> {
        self.inner
            .matmul(&other.inner)
            .map(Self::new)
            .map_err(|error| tensor_error(&error))
    }

    fn negated(&self) -> PyResult<Self> {
        self.inner
            .negate()
            .map(Self::new)
            .map_err(|error| tensor_error(&error))
    }

    fn multiplication_method(
        &self,
        operation: MultiplicationOperation,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let (other, keyword_error) = bind_multiplication_argument(operation, args, kwargs)?;
        let other_tensor = other.value.cast::<Self>().ok();
        let scalar = other_tensor
            .is_none()
            .then(|| parse_arithmetic_scalar(&other.value));

        if scalar
            .as_ref()
            .is_some_and(|result| matches!(result, Ok(None)))
        {
            return match operation {
                MultiplicationOperation::Mul => {
                    let actual = python_type_name(&other.value)?;
                    Err(mul_argument_type_error(other.position, &actual))
                }
                MultiplicationOperation::Multiply => Err(multiply_binding_error(args, kwargs)?),
            };
        }
        if let Some(keyword_error) = keyword_error {
            return Err(keyword_error);
        }

        let result = if let Some(other_tensor) = other_tensor {
            let other_tensor = other_tensor.try_borrow()?;
            BinaryOperation::Multiply.apply_tensors(&self.inner, &other_tensor.inner)
        } else {
            let scalar = match scalar.expect("a non-tensor mul operand has a scalar parse result") {
                Ok(Some(scalar)) => scalar,
                Ok(None) => unreachable!("unsupported mul operand types were rejected above"),
                Err(_) if other.value.is_instance_of::<PyInt>() => {
                    let message = if python_integer_is_negative(&other.value)? {
                        "can't convert negative int to unsigned"
                    } else {
                        "int too big to convert"
                    };
                    return Err(PyOverflowError::new_err(message));
                }
                Err(error) => return Err(error),
            };
            if matches!(scalar, ParsedArithmeticScalar::WideNumpyUnsigned) {
                return Err(PyTypeError::new_err("an integer is required"));
            }
            BinaryOperation::Multiply.apply_scalar(&self.inner, scalar.into_f32(), false)
        };

        result.map(Self::new).map_err(|error| tensor_error(&error))
    }

    pub(crate) fn truth_value(&self) -> PyResult<bool> {
        match self.inner.numel() {
            0 => Err(PyRuntimeError::new_err(
                "Boolean value of Tensor with no values is ambiguous",
            )),
            1 => self
                .inner
                .item()
                .map(|value| value != 0.0)
                .map_err(|error| tensor_error(&error)),
            _ => Err(PyRuntimeError::new_err(
                "Boolean value of Tensor with more than one value is ambiguous",
            )),
        }
    }

    fn numpy_array_copy(
        &self,
        py: Python<'_>,
        dtype: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let numpy = PyModule::import(py, "numpy")?;
        let values = self
            .inner
            .try_to_vec()
            .map_err(|error| tensor_error(&error))?;
        let values = PyList::new(py, values)?;
        let arguments = PyDict::new(py);
        if let Some(dtype) = dtype {
            arguments.set_item("dtype", dtype)?;
        } else {
            arguments.set_item("dtype", numpy.getattr("float32")?)?;
        }
        let array = numpy.getattr("array")?.call((values,), Some(&arguments))?;
        let shape = PyTuple::new(py, self.inner.shape().iter().copied())?;
        let array = array.call_method1("reshape", (shape,))?;
        Ok(array.unbind())
    }

    fn numpy_reflected_divide(
        &self,
        py: Python<'_>,
        numerator: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let denominator = self.numpy_array_copy(py, None)?;
        let numpy = PyModule::import(py, "numpy")?;
        let result = numpy
            .getattr("true_divide")?
            .call1((numerator, denominator.bind(py)))?;
        Ok(result.unbind())
    }

    fn binary_operation(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        operation: BinaryOperation,
        reverse: bool,
    ) -> PyResult<Py<PyAny>> {
        let result = if let Ok(other) = other.cast::<Self>() {
            let other = other.try_borrow()?;
            if reverse {
                operation.apply_tensors(&other.inner, &self.inner)
            } else {
                operation.apply_tensors(&self.inner, &other.inner)
            }
        } else {
            let Some(scalar) = parse_arithmetic_scalar(other)? else {
                return Ok(py.NotImplemented());
            };
            let scalar = match scalar {
                ParsedArithmeticScalar::WideNumpyUnsigned => {
                    if reverse && matches!(operation, BinaryOperation::Divide) {
                        return self.numpy_reflected_divide(py, other);
                    }
                    return Ok(py.NotImplemented());
                }
                scalar => scalar,
            };
            if matches!(operation, BinaryOperation::Subtract) && scalar.is_python_bool() {
                return Err(bool_subtraction_error());
            }
            operation.apply_scalar(&self.inner, scalar.into_f32(), reverse)
        };

        Self::new(result.map_err(|error| tensor_error(&error))?).into_py_any(py)
    }
}

impl BinaryOperation {
    fn apply_tensors(
        self,
        left: &CoreTensor,
        right: &CoreTensor,
    ) -> Result<CoreTensor, TensorError> {
        match self {
            Self::Add => left.add(right),
            Self::Subtract => left.sub(right),
            Self::Multiply => left.mul(right),
            Self::Divide => left.div(right),
        }
    }

    fn apply_scalar(
        self,
        tensor: &CoreTensor,
        scalar: f32,
        reverse: bool,
    ) -> Result<CoreTensor, TensorError> {
        match (self, reverse) {
            (Self::Add, _) => tensor.add_scalar(scalar),
            (Self::Subtract, false) => tensor.sub_scalar(scalar),
            (Self::Subtract, true) => tensor.scalar_sub(scalar),
            (Self::Multiply, _) => tensor.mul_scalar(scalar),
            (Self::Divide, false) => tensor.div_scalar(scalar),
            (Self::Divide, true) => tensor.scalar_div(scalar),
        }
    }
}

#[pyfunction(
    signature = (data, *, dtype=None, device=None, requires_grad=StrictBool(false)),
    text_signature = "(data, *, dtype=None, device=None, requires_grad=False)"
)]
fn tensor(
    data: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
    requires_grad: StrictBool,
) -> PyResult<PyTensor> {
    let requires_grad = requires_grad.0;
    let dtype_was_explicit = dtype.is_some();
    let (dtype, device) = parse_metadata("tensor", dtype, device)?;
    let (flattened, shape) = if let Ok(scalar) = data.extract::<f32>() {
        (vec![scalar], Vec::new())
    } else if data.cast::<PyBytes>().is_ok() {
        return Err(PyTypeError::new_err("new(): invalid data type 'bytes'"));
    } else if data.cast::<PyMemoryView>().is_ok() {
        if let Some(buffer) = flatten_buffer(data, dtype_was_explicit)? {
            buffer
        } else {
            let mut flattened = Vec::new();
            let shape = flatten_rectangular(data, &mut flattened)?;
            (flattened, shape)
        }
    } else if is_sequence_input(data)? {
        let mut flattened = Vec::new();
        let shape = flatten_rectangular(data, &mut flattened)?;
        (flattened, shape)
    } else {
        return Err(unsupported_tensor_data_error(data, dtype_was_explicit)?);
    };
    CoreTensor::from_vec_with_metadata(flattened, shape, dtype, device)
        .map(|inner| PyTensor::new(inner.with_requires_grad(requires_grad)))
        .map_err(|error| tensor_error(&error))
}

const MIN_BACKWARD_LEAF_ROOTS: usize = 2;
const MAX_BACKWARD_LEAF_ROOTS: usize = 10;

#[pyfunction]
fn _backward_leaf_roots(roots: &Bound<'_, PyAny>) -> PyResult<()> {
    if !roots.is_exact_instance_of::<PyTuple>() && !roots.is_exact_instance_of::<PyList>() {
        return Err(PyTypeError::new_err(
            "_backward_leaf_roots expects an exact tuple or list",
        ));
    }

    let roots = roots.cast::<PySequence>()?;
    let root_count = roots.len()?;
    if !(MIN_BACKWARD_LEAF_ROOTS..=MAX_BACKWARD_LEAF_ROOTS).contains(&root_count) {
        return Err(PyTypeError::new_err(format!(
            "_backward_leaf_roots expects between {MIN_BACKWARD_LEAF_ROOTS} and {MAX_BACKWARD_LEAF_ROOTS} roots"
        )));
    }

    // Hold every Python borrow until all elements have been checked, then let
    // the native slice engine validate every root before committing gradients.
    let mut root_guards = Vec::with_capacity(root_count);
    for index in 0..root_count {
        let root = roots.get_item(index)?;
        if !root.is_exact_instance_of::<PyTensor>() {
            return Err(PyTypeError::new_err(format!(
                "_backward_leaf_roots expected an exact native Tensor at element {index}"
            )));
        }
        root_guards.push(root.extract::<PyRef<'_, PyTensor>>()?);
    }
    let native_roots = root_guards
        .iter()
        .map(|root| root.inner())
        .collect::<Vec<_>>();
    CoreTensor::backward_leaf_roots(&native_roots).map_err(|error| tensor_error(&error))
}

fn scalar_tensor_impl(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyTensor> {
    let arguments = bind_scalar_tensor_arguments(args, kwargs)?;
    let (value, dtype, device, pin_memory, requires_grad) =
        parse_scalar_tensor_arguments(arguments)?;
    if pin_memory {
        return Err(PyRuntimeError::new_err(
            "scalar_tensor(): pin_memory=True is not supported; only unpinned CPU storage is implemented",
        ));
    }
    rank_zero_scalar_tensor(value, dtype, device, requires_grad)
}

fn extract_exact_python_float_scalar(value: &Bound<'_, PyAny>) -> PyResult<Option<f32>> {
    if !value.is_exact_instance_of::<PyFloat>() {
        return Ok(None);
    }

    #[allow(clippy::cast_possible_truncation)]
    let value = value.extract::<f64>()? as f32;
    Ok(Some(value))
}

fn as_tensor_float_sequence(value: &Bound<'_, PyAny>) -> PyResult<Option<(Vec<f32>, Vec<usize>)>> {
    if !is_exact_list_or_tuple(value) {
        return Ok(None);
    }

    let mut flattened = Vec::new();
    let mut active_containers = Vec::new();
    let Some(shape) =
        flatten_as_tensor_float_sequence(value, &mut flattened, &mut active_containers, 0)?
    else {
        return Ok(None);
    };
    Ok(Some((flattened, shape)))
}

fn flatten_as_tensor_float_sequence(
    value: &Bound<'_, PyAny>,
    output: &mut Vec<f32>,
    active_containers: &mut Vec<*mut ffi::PyObject>,
    depth: usize,
) -> PyResult<Option<Vec<usize>>> {
    if let Some(scalar) = extract_exact_python_float_scalar(value)? {
        output.push(scalar);
        return Ok(Some(Vec::new()));
    }

    if !is_exact_list_or_tuple(value) {
        return Ok(None);
    }
    if depth >= AS_TENSOR_MAX_SEQUENCE_DIMENSIONS {
        return Err(as_tensor_too_many_dimensions_error(value));
    }
    let pointer = value.as_ptr();
    if active_containers
        .iter()
        .any(|seen| std::ptr::addr_eq(*seen, pointer))
    {
        return Err(as_tensor_too_many_dimensions_error(value));
    }
    active_containers.push(pointer);

    let length = value.len()?;
    if length == 0 {
        active_containers.pop();
        return Ok(Some(vec![0]));
    }

    let Some(first_shape) = flatten_as_tensor_float_sequence(
        &value.get_item(0)?,
        output,
        active_containers,
        depth + 1,
    )?
    else {
        active_containers.pop();
        return Ok(None);
    };
    for index in 1..length {
        let Some(shape) = flatten_as_tensor_float_sequence(
            &value.get_item(index)?,
            output,
            active_containers,
            depth + 1,
        )?
        else {
            active_containers.pop();
            return Ok(None);
        };
        if shape != first_shape {
            active_containers.pop();
            return Err(PyValueError::new_err(
                "expected a rectangular sequence, but nested shapes differ",
            ));
        }
    }

    let mut shape = Vec::with_capacity(first_shape.len() + 1);
    shape.push(length);
    shape.extend(first_shape);
    active_containers.pop();
    Ok(Some(shape))
}

fn is_exact_list_or_tuple(value: &Bound<'_, PyAny>) -> bool {
    value.is_exact_instance_of::<PyList>() || value.is_exact_instance_of::<PyTuple>()
}

fn as_tensor_too_many_dimensions_error(value: &Bound<'_, PyAny>) -> PyErr {
    let container = if value.is_exact_instance_of::<PyList>() {
        "list"
    } else {
        "tuple"
    };
    PyValueError::new_err(format!("too many dimensions '{container}'"))
}

fn rank_zero_scalar_tensor(
    value: f32,
    dtype: DType,
    device: Device,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    CoreTensor::full_with_metadata(Vec::new(), value, dtype, device)
        .map(|inner| PyTensor::new(inner.with_requires_grad(requires_grad)))
        .map_err(|error| tensor_error(&error))
}

fn arange_impl(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyTensor> {
    let (elements, requires_grad) = parse_arange_arguments(bind_arange_arguments(args, kwargs)?)?;
    CoreTensor::arange_float32(elements)
        .map(|inner| PyTensor::new(inner.with_requires_grad(requires_grad)))
        .map_err(|error| scalar_creation_error(&error, Some(elements)))
}

fn parse_requires_grad(function: &str, requires_grad: &Bound<'_, PyAny>) -> PyResult<bool> {
    if requires_grad.is_exact_instance_of::<PyBool>() {
        return requires_grad.is_truthy();
    }
    let type_name = python_type_name(requires_grad)?;
    Err(PyTypeError::new_err(format!(
        "{function}(): argument 'requires_grad' must be bool, not {type_name}"
    )))
}

fn parse_factory_requires_grad(
    function: &str,
    requires_grad: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    match requires_grad {
        None => Ok(false),
        Some(requires_grad) if requires_grad.is_none() => Ok(false),
        Some(requires_grad) => parse_requires_grad(function, requires_grad),
    }
}

#[pyfunction(signature = (input, *, memory_format=None))]
fn clone(input: &PyTensor, memory_format: Option<&Bound<'_, PyAny>>) -> PyResult<PyTensor> {
    let memory_format = parse_clone_memory_format(memory_format)?;
    input
        .inner
        .try_clone_with_memory_format(memory_format)
        .map(PyTensor::new)
        .map_err(|error| tensor_error(&error))
}

#[pyfunction(signature = (*args, **kwargs), text_signature = None)]
fn relu(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
    let input = bind_legacy_single_tensor_argument("relu", args, kwargs)?;
    let tensor = input
        .value
        .cast::<PyTensor>()
        .expect("the relu input type was checked while binding");
    tensor.try_borrow()?.relu()
}

#[pyfunction(signature = (*args, **kwargs), text_signature = None)]
fn is_same_size(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<bool> {
    let ([input, other], keyword_error) =
        bind_legacy_binary_arguments("is_same_size", args, kwargs, LegacyBinaryInputKind::Tensor)?;
    let input = parse_tensor_argument("is_same_size", "input", &input)?;
    let other = parse_tensor_argument("is_same_size", "other", &other)?;
    if let Some(keyword_error) = keyword_error {
        return Err(keyword_error);
    }
    let input = input.try_borrow()?;
    let other = other.try_borrow()?;
    Ok(input.inner.is_same_size(&other.inner))
}

/// equal(input, other) -> bool
///
/// Returns ``True`` if two tensors have the same size and elements, and
/// ``False`` otherwise. NaNs compare unequal, while tensor dtype is ignored.
#[pyfunction(signature = (*args, **kwargs), text_signature = None)]
fn equal(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<bool> {
    let (arguments, keyword_error) =
        bind_tensor_arguments("equal", args, kwargs, ["input", "other"])?;
    let input = parse_tensor_argument("equal", "input", &arguments[0])?;
    let other = parse_tensor_argument("equal", "other", &arguments[1])?;
    if let Some(keyword_error) = keyword_error {
        return Err(keyword_error);
    }
    let input = input.try_borrow()?;
    let other = other.try_borrow()?;
    Ok(input.inner == other.inner)
}

// Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
#[allow(clippy::doc_markdown)]
#[doc = "\nt(input) -> Tensor\n\nExpects :attr:`input` to be <= 2-D tensor and transposes dimensions 0\nand 1.\n\n0-D and 1-D tensors are returned as is. When input is a 2-D tensor this\nis equivalent to ``transpose(input, 0, 1)``.\n\nArgs:\n    input (Tensor): the input tensor.\n\nExample::\n\n    >>> x = torch.randn(())\n    >>> x\n    tensor(0.1995)\n    >>> torch.t(x)\n    tensor(0.1995)\n    >>> x = torch.randn(3)\n    >>> x\n    tensor([ 2.4320, -0.4608,  0.7702])\n    >>> torch.t(x)\n    tensor([ 2.4320, -0.4608,  0.7702])\n    >>> x = torch.randn(2, 3)\n    >>> x\n    tensor([[ 0.4875,  0.9158, -0.5872],\n            [ 0.3938, -0.6929,  0.6932]])\n    >>> torch.t(x)\n    tensor([[ 0.4875,  0.3938],\n            [ 0.9158, -0.6929],\n            [-0.5872,  0.6932]])\n\nSee also :func:`torch.transpose`.\n"]
#[pyfunction(signature = (*args, **kwargs), text_signature = None)]
fn t(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
    let input = bind_legacy_single_tensor_argument("t", args, kwargs)?;
    let tensor = input
        .value
        .cast::<PyTensor>()
        .expect("the t input type was checked while binding");
    tensor.try_borrow()?.t()
}

#[pyfunction(signature = (*args, **kwargs))]
fn transpose(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
    apply_top_level_dimension_swap("transpose", ["input", "dim0", "dim1"], args, kwargs)
}

// Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
#[allow(clippy::doc_markdown)]
#[doc = "\nswapdims(input, dim0, dim1) -> Tensor\n\nAlias for :func:`torch.transpose`.\n\nThis function is equivalent to NumPy's swapaxes function.\n\nExamples::\n\n    >>> x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])\n    >>> x\n    tensor([[[0, 1],\n            [2, 3]],\n\n            [[4, 5],\n            [6, 7]]])\n    >>> torch.swapdims(x, 0, 1)\n    tensor([[[0, 1],\n            [4, 5]],\n\n            [[2, 3],\n            [6, 7]]])\n    >>> torch.swapdims(x, 0, 2)\n    tensor([[[0, 4],\n            [2, 6]],\n\n            [[1, 5],\n            [3, 7]]])\n"]
#[pyfunction(signature = (*args, **kwargs), text_signature = None)]
fn swapdims(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
    apply_top_level_dimension_swap("swapdims", ["input", "dim0", "dim1"], args, kwargs)
}

// Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
#[allow(clippy::doc_markdown)]
#[doc = "\nswapaxes(input, axis0, axis1) -> Tensor\n\nAlias for :func:`torch.transpose`.\n\nThis function is equivalent to NumPy's swapaxes function.\n\nExamples::\n\n    >>> x = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])\n    >>> x\n    tensor([[[0, 1],\n            [2, 3]],\n\n            [[4, 5],\n            [6, 7]]])\n    >>> torch.swapaxes(x, 0, 1)\n    tensor([[[0, 1],\n            [4, 5]],\n\n            [[2, 3],\n            [6, 7]]])\n    >>> torch.swapaxes(x, 0, 2)\n    tensor([[[0, 4],\n            [2, 6]],\n\n            [[1, 5],\n            [3, 7]]])\n"]
#[pyfunction(signature = (*args, **kwargs), text_signature = None)]
fn swapaxes(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
    apply_top_level_dimension_swap("swapaxes", ["input", "axis0", "axis1"], args, kwargs)
}

fn apply_top_level_dimension_swap(
    operation: &str,
    argument_names: [&str; 3],
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyTensor> {
    let ([input, dim0, dim1], keyword_error) =
        bind_dimension_swap_arguments(operation, args, kwargs, argument_names)?;
    if let Some(keyword_error) = keyword_error {
        return Err(keyword_error);
    }
    let input_type = python_type_name(&input.value)?;
    let input_tensor = input.value.cast::<PyTensor>().map_err(|_| {
        dimension_swap_argument_type_error(
            operation,
            "input",
            input.position,
            "Tensor",
            &input_type,
        )
    })?;
    let input_tensor = input_tensor.try_borrow()?;
    let [dim0, dim1] = parse_dimension_swap_dimensions(
        operation,
        [argument_names[1], argument_names[2]],
        &dim0,
        &dim1,
    )?;
    input_tensor
        .inner
        .transpose(dim0, dim1)
        .map(PyTensor::new)
        .map_err(|error| transpose_error(&error))
}

#[pyfunction(signature = (*args, **kwargs), text_signature = "(input, dim=None)")]
fn squeeze(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
    let (input, input_position, dimension) = bind_top_level_squeeze_arguments(args, kwargs)?;
    let input_type = python_type_name(&input)?;
    let input = match input.cast::<PyTensor>() {
        Ok(input) => input,
        Err(_) if matches!(&dimension, ParsedSqueezeDimensions::All) => {
            return Err(squeeze_argument_type_error(
                "input",
                input_position,
                "Tensor",
                &input_type,
            ));
        }
        Err(_) => {
            return Err(squeeze_top_level_input_with_dimension_error(
                args, kwargs, &dimension,
            )?);
        }
    };
    let input = input.try_borrow()?;
    apply_squeeze(&input.inner, dimension)
        .map(PyTensor::new)
        .map_err(|error| tensor_error(&error))
}

#[pyfunction(signature = (*args, **kwargs), text_signature = "(input, start_dim=0, end_dim=-1)")]
fn flatten(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyTensor>> {
    let (input, start_dim, end_dim) = bind_top_level_flatten_arguments(args, kwargs)?;
    let tensor = input
        .cast::<PyTensor>()
        .expect("the flatten input type was checked while binding");
    let input_object = tensor.clone().unbind();
    let inner = {
        let tensor = tensor.try_borrow()?;
        tensor
            .inner
            .flatten(start_dim, end_dim)
            .map_err(|error| tensor_error(&error))?
    };
    let tensor = input_object.bind(args.py()).try_borrow()?;
    if same_tensor_metadata(&tensor.inner, &inner) {
        drop(tensor);
        return Ok(input_object);
    }
    drop(tensor);
    Py::new(args.py(), PyTensor::new(inner))
}

#[pyfunction(
    signature = (*args, **kwargs),
    text_signature = "(*size, shape=None, out=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False)"
)]
fn zeros(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
    let arguments = bind_creation_arguments("zeros", args, kwargs)?;
    let (size, dtype, device, requires_grad) = parse_creation_arguments("zeros", arguments)?;
    let ParsedCreationSize {
        dimensions,
        scalar_dimension,
    } = size;
    let shape = dimensions.clone();
    CoreTensor::zeros_with_metadata(dimensions, dtype, device)
        .map(|inner| PyTensor::new(inner.with_requires_grad(requires_grad)))
        .map_err(|error| creation_factory_error(&error, &shape, scalar_dimension))
}

#[pyfunction(
    signature = (*args, **kwargs),
    text_signature = "(*size, shape=None, out=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False)"
)]
fn ones(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
    let arguments = bind_creation_arguments("ones", args, kwargs)?;
    let (size, dtype, device, requires_grad) = parse_creation_arguments("ones", arguments)?;
    let ParsedCreationSize {
        dimensions,
        scalar_dimension,
    } = size;
    let shape = dimensions.clone();
    CoreTensor::ones_with_metadata(dimensions, dtype, device)
        .map(|inner| PyTensor::new(inner.with_requires_grad(requires_grad)))
        .map_err(|error| creation_factory_error(&error, &shape, scalar_dimension))
}

#[pyfunction(
    signature = (*args, **kwargs),
    text_signature = "(n, m=None, *, dtype=None, device=None, requires_grad=False)"
)]
fn eye(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
    let arguments = bind_eye_arguments(args, kwargs)?;
    let (n, m, dtype, device, requires_grad) = parse_eye_arguments(arguments)?;
    let shape = [n, m];

    CoreTensor::eye_with_metadata(n, m, dtype, device)
        .map(|inner| PyTensor::new(inner.with_requires_grad(requires_grad)))
        .map_err(|error| eye_shape_error(&error, &shape))
}

#[pyfunction(
    signature = (*args, **kwargs),
    text_signature = "(size, fill_value, *, out=None, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False)"
)]
fn full(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
    let arguments = bind_full_arguments(args, kwargs)?;
    let ParsedFullArguments {
        size,
        fill_value,
        has_out,
        dtype,
        device,
        pin_memory,
        requires_grad,
    } = parse_full_arguments(arguments)?;
    let shape = validate_size(size)?;
    CoreTensor::validate_full_shape(&shape)
        .map_err(|error| creation_shape_error(&error, &shape))?;
    if has_out {
        return Err(PyRuntimeError::new_err(
            "full(): the 'out' argument is not supported",
        ));
    }
    let fill_value = fill_value.into_f32()?;
    if pin_memory {
        return Err(PyRuntimeError::new_err(
            "full(): pin_memory=True is not supported; only unpinned CPU storage is implemented",
        ));
    }
    CoreTensor::full_with_metadata(shape, fill_value, dtype, device)
        .map(|inner| PyTensor::new(inner.with_requires_grad(requires_grad)))
        .map_err(|error| tensor_error(&error))
}

fn layout_objects(py: Python<'_>) -> PyResult<&'static PyLayoutObjects> {
    LAYOUT_OBJECTS.get_or_try_init(py, || create_layout_objects(py))
}

fn strided_object(py: Python<'_>) -> PyResult<&'static Py<PyAny>> {
    Ok(&layout_objects(py)?.strided)
}

pub(crate) fn warn_once(py: Python<'_>, emitted: &AtomicBool, message: &CStr) -> PyResult<()> {
    // Always-warning mode deliberately bypasses the per-site marker. An unused
    // marker must therefore remain available if the process later returns to
    // the default once-only policy, while a marker consumed before always mode
    // remains consumed after the policy is disabled again.
    if !WARN_ALWAYS_ENABLED.load(Ordering::SeqCst) && emitted.swap(true, Ordering::Relaxed) {
        return Ok(());
    }
    PyErr::warn(py, &py.get_type::<PyUserWarning>(), message, 1)
}

#[pyfunction(name = "_set_warnAlways", signature = (b, /), text_signature = None)]
fn set_warn_always_native(b: &Bound<'_, PyAny>) -> PyResult<()> {
    if !b.is_exact_instance_of::<PyBool>() {
        let type_name = python_type_name(b)?;
        return Err(PyRuntimeError::new_err(format!(
            "setWarnOnlyOnce expects a bool, but got {type_name}"
        )));
    }
    WARN_ALWAYS_ENABLED.store(b.is_truthy()?, Ordering::SeqCst);
    Ok(())
}

#[pyfunction(name = "_get_warnAlways", signature = (), text_signature = None)]
fn get_warn_always_native() -> bool {
    WARN_ALWAYS_ENABLED.load(Ordering::SeqCst)
}

fn add_warn_always_builtins(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(set_warn_always_native, module)?)?;
    module.add_function(wrap_pyfunction!(get_warn_always_native, module)?)?;
    let exports = module.getattr("__all__")?;
    exports.call_method1("remove", ("_set_warnAlways",))?;
    exports.call_method1("remove", ("_get_warnAlways",))?;
    Ok(())
}

#[pyfunction(name = "_set_cudnn_enabled", signature = (enabled, /), text_signature = None)]
fn set_cudnn_enabled_native(enabled: &Bound<'_, PyAny>) -> PyResult<()> {
    if !enabled.is_exact_instance_of::<PyBool>() {
        let type_name = python_type_name(enabled)?;
        return Err(PyRuntimeError::new_err(format!(
            "set_enabled_cudnn expects a bool, but got {type_name}"
        )));
    }
    CUDNN_ENABLED.store(enabled.is_truthy()?, Ordering::SeqCst);
    Ok(())
}

#[pyfunction(name = "_get_cudnn_enabled", signature = (), text_signature = None)]
fn get_cudnn_enabled_native() -> bool {
    CUDNN_ENABLED.load(Ordering::SeqCst)
}

#[pyfunction(
    name = "_set_cudnn_benchmark",
    signature = (benchmark, /),
    text_signature = None
)]
fn set_cudnn_benchmark_native(benchmark: &Bound<'_, PyAny>) -> PyResult<()> {
    if !benchmark.is_exact_instance_of::<PyBool>() {
        let type_name = python_type_name(benchmark)?;
        return Err(PyRuntimeError::new_err(format!(
            "set_benchmark_cudnn expects a bool, but got {type_name}"
        )));
    }
    CUDNN_BENCHMARK.store(benchmark.is_truthy()?, Ordering::SeqCst);
    Ok(())
}

#[pyfunction(name = "_get_cudnn_benchmark", signature = (), text_signature = None)]
fn get_cudnn_benchmark_native() -> bool {
    CUDNN_BENCHMARK.load(Ordering::SeqCst)
}

#[allow(
    unsafe_code,
    clippy::cast_possible_truncation,
    reason = "PyTorch accepts int64 values and deliberately narrows them to C int"
)]
fn set_cudnn_benchmark_limit_native(object: &Bound<'_, PyAny>) -> PyResult<()> {
    let is_integer = !object.is_instance_of::<PyBool>()
        && (object.is_instance_of::<PyInt>() || has_numpy_integer_ancestry(object)?);
    if !is_integer {
        let type_name = python_type_name(object)?;
        return Err(PyRuntimeError::new_err(format!(
            "set_benchmark_limit_cudnn expects an int, but got {type_name}"
        )));
    }

    let mut overflow = 0;
    // SAFETY: validation above accepts only Python int instances and NumPy
    // integer scalars. The object remains live and overflow is writable.
    let value = unsafe { ffi::PyLong_AsLongLongAndOverflow(object.as_ptr(), &raw mut overflow) };
    if PyErr::occurred(object.py()) {
        return Err(PyErr::fetch(object.py()));
    }
    if overflow != 0 {
        return Err(PyValueError::new_err("Overflow when unpacking long long"));
    }

    // PyTorch first unpacks to int64_t and then narrows with a C++ cast to int.
    CUDNN_BENCHMARK_LIMIT.store(value as i32, Ordering::SeqCst);
    Ok(())
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn set_cudnn_benchmark_limit_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    object: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: CPython supplies a live borrowed object to a METH_O callback.
    let object = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, object) };
    set_cudnn_benchmark_limit_native(&object)?;
    Ok(py.None().into_ptr())
}

#[pyfunction(
    name = "_cuda_get_cudnn_benchmark_limit",
    signature = (),
    text_signature = None
)]
#[pyo3(pass_module)]
fn get_cudnn_benchmark_limit_native(_module: &Bound<'_, PyModule>) -> i32 {
    CUDNN_BENCHMARK_LIMIT.load(Ordering::SeqCst)
}

// PyTorch publishes this private setter as a module-bound METH_O built-in.
// Keep validation separate from its CPython wrapper so assignment through the
// proxy and direct native calls share one atomic state transition.
fn set_cudnn_deterministic_native(object: &Bound<'_, PyAny>) -> PyResult<()> {
    if !object.is_exact_instance_of::<PyBool>() {
        let type_name = python_type_name(object)?;
        return Err(PyRuntimeError::new_err(format!(
            "set_deterministic_cudnn expects a bool, but got {type_name}"
        )));
    }
    CUDNN_DETERMINISTIC.store(object.is_truthy()?, Ordering::SeqCst);
    Ok(())
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn set_cudnn_deterministic_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    object: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: CPython supplies a live borrowed object to a METH_O callback.
    let object = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, object) };
    set_cudnn_deterministic_native(&object)?;
    Ok(py.None().into_ptr())
}

#[pyfunction(
    name = "_get_cudnn_deterministic",
    signature = (),
    text_signature = None
)]
#[pyo3(pass_module)]
fn get_cudnn_deterministic_native(_module: &Bound<'_, PyModule>) -> bool {
    CUDNN_DETERMINISTIC.load(Ordering::SeqCst)
}

// PyTorch publishes this private setter as a module-bound METH_O built-in.
// Its historical error message refers to cuBLAS even for the cuDNN preference.
fn set_cudnn_allow_tf32_native(object: &Bound<'_, PyAny>) -> PyResult<()> {
    if !object.is_exact_instance_of::<PyBool>() {
        let type_name = python_type_name(object)?;
        return Err(PyRuntimeError::new_err(format!(
            "set_allow_tf32_cublas expects a bool, but got {type_name}"
        )));
    }
    CUDNN_ALLOW_TF32.store(object.is_truthy()?, Ordering::SeqCst);
    Ok(())
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn set_cudnn_allow_tf32_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    object: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: CPython supplies a live borrowed object to a METH_O callback.
    let object = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, object) };
    set_cudnn_allow_tf32_native(&object)?;
    Ok(py.None().into_ptr())
}

#[pyfunction(
    name = "_get_cudnn_allow_tf32",
    signature = (),
    text_signature = None
)]
#[pyo3(pass_module)]
fn get_cudnn_allow_tf32_native(_module: &Bound<'_, PyModule>) -> bool {
    CUDNN_ALLOW_TF32.load(Ordering::SeqCst)
}

#[allow(
    unsafe_code,
    reason = "PyCFunction_NewEx requires an audited stable-ABI raw-pointer call"
)]
fn add_cudnn_deterministic_setter(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let mut definition = pyo3::impl_::pymethods::PyMethodDef::noargs(
        c"_set_cudnn_deterministic",
        pyo3::impl_::trampoline::get_trampoline_function!(
            binaryfunc,
            set_cudnn_deterministic_callback
        ),
        c"",
    )
    .into_raw();
    definition.ml_flags = ffi::METH_O;
    let definition = Box::leak(Box::new(definition));
    let module_name = module.name()?;
    // SAFETY: the leaked method definition, module, and module name all remain
    // live for the duration required by the newly owned built-in function.
    let function = unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(
            py,
            ffi::PyCFunction_NewEx(definition, module.as_ptr(), module_name.as_ptr()),
        )?
        .cast_into::<PyCFunction>()?
    };
    module.add_function(function)
}

#[allow(
    unsafe_code,
    reason = "PyCFunction_NewEx requires an audited stable-ABI raw-pointer call"
)]
fn add_cudnn_allow_tf32_setter(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let mut definition = pyo3::impl_::pymethods::PyMethodDef::noargs(
        c"_set_cudnn_allow_tf32",
        pyo3::impl_::trampoline::get_trampoline_function!(
            binaryfunc,
            set_cudnn_allow_tf32_callback
        ),
        c"",
    )
    .into_raw();
    definition.ml_flags = ffi::METH_O;
    let definition = Box::leak(Box::new(definition));
    let module_name = module.name()?;
    // SAFETY: the leaked method definition, module, and module name all remain
    // live for the duration required by the newly owned built-in function.
    let function = unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(
            py,
            ffi::PyCFunction_NewEx(definition, module.as_ptr(), module_name.as_ptr()),
        )?
        .cast_into::<PyCFunction>()?
    };
    module.add_function(function)
}

#[allow(
    unsafe_code,
    reason = "PyCFunction_NewEx requires an audited stable-ABI raw-pointer call"
)]
fn add_cudnn_benchmark_limit_setter(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let mut definition = pyo3::impl_::pymethods::PyMethodDef::noargs(
        c"_cuda_set_cudnn_benchmark_limit",
        pyo3::impl_::trampoline::get_trampoline_function!(
            binaryfunc,
            set_cudnn_benchmark_limit_callback
        ),
        c"",
    )
    .into_raw();
    definition.ml_flags = ffi::METH_O;
    let definition = Box::leak(Box::new(definition));
    let module_name = module.name()?;
    // SAFETY: the leaked method definition, module, and module name all remain
    // live for the duration required by the newly owned built-in function.
    let function = unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(
            py,
            ffi::PyCFunction_NewEx(definition, module.as_ptr(), module_name.as_ptr()),
        )?
        .cast_into::<PyCFunction>()?
    };
    module.add_function(function)
}

fn add_cudnn_builtins(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(set_cudnn_enabled_native, module)?)?;
    module.add_function(wrap_pyfunction!(get_cudnn_enabled_native, module)?)?;
    module.add_function(wrap_pyfunction!(set_cudnn_benchmark_native, module)?)?;
    module.add_function(wrap_pyfunction!(get_cudnn_benchmark_native, module)?)?;
    add_cudnn_benchmark_limit_setter(module)?;
    module.add_function(wrap_pyfunction!(get_cudnn_benchmark_limit_native, module)?)?;
    add_cudnn_deterministic_setter(module)?;
    module.add_function(wrap_pyfunction!(get_cudnn_deterministic_native, module)?)?;
    add_cudnn_allow_tf32_setter(module)?;
    module.add_function(wrap_pyfunction!(get_cudnn_allow_tf32_native, module)?)?;
    let exports = module.getattr("__all__")?;
    for name in [
        "_set_cudnn_enabled",
        "_get_cudnn_enabled",
        "_set_cudnn_benchmark",
        "_get_cudnn_benchmark",
        "_cuda_set_cudnn_benchmark_limit",
        "_cuda_get_cudnn_benchmark_limit",
        "_set_cudnn_deterministic",
        "_get_cudnn_deterministic",
        "_set_cudnn_allow_tf32",
        "_get_cudnn_allow_tf32",
    ] {
        exports.call_method1("remove", (name,))?;
    }
    Ok(())
}

// PyTorch exposes CUDA matmul preferences even when a build cannot execute
// CUDA tensors. Keep the state native and process-global, but do not use it to
// claim any CUDA runtime or cuBLAS execution support.
fn set_cublas_allow_tf32_native(object: &Bound<'_, PyAny>) -> PyResult<()> {
    if !object.is_exact_instance_of::<PyBool>() {
        let type_name = python_type_name(object)?;
        return Err(PyRuntimeError::new_err(format!(
            "set_allow_tf32_cublas expects a bool, but got {type_name}"
        )));
    }
    CUBLAS_ALLOW_TF32.store(object.is_truthy()?, Ordering::SeqCst);
    Ok(())
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn set_cublas_allow_tf32_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    object: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: CPython supplies a live borrowed object to a METH_O callback.
    let object = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, object) };
    set_cublas_allow_tf32_native(&object)?;
    Ok(py.None().into_ptr())
}

#[pyfunction(
    name = "_get_cublas_allow_tf32",
    signature = (),
    text_signature = None
)]
#[pyo3(pass_module)]
fn get_cublas_allow_tf32_native(_module: &Bound<'_, PyModule>) -> bool {
    CUBLAS_ALLOW_TF32.load(Ordering::SeqCst)
}

fn parse_cublas_reduced_precision_bool(
    value: &Bound<'_, PyAny>,
    native_name: &str,
    argument_name: &str,
) -> PyResult<bool> {
    if !value.is_exact_instance_of::<PyBool>() {
        let type_name = python_type_name(value)?;
        return Err(PyRuntimeError::new_err(format!(
            "{native_name} expects a bool for {argument_name}, but got {type_name}"
        )));
    }
    value.is_truthy()
}

fn set_cublas_reduced_precision_reduction_native(
    allow_reduced_precision: &Bound<'_, PyAny>,
    allow_splitk: Option<&Bound<'_, PyAny>>,
    allow_reduced_precision_state: &AtomicBool,
    allow_splitk_state: &AtomicBool,
    native_name: &str,
) -> PyResult<()> {
    let allow_reduced_precision = parse_cublas_reduced_precision_bool(
        allow_reduced_precision,
        native_name,
        "allow_reduced_precision",
    )?;
    let allow_splitk = match allow_splitk {
        Some(value) if !value.is_none() => {
            parse_cublas_reduced_precision_bool(value, native_name, "allow_splitk")?
        }
        _ => true,
    };
    if allow_reduced_precision && !allow_splitk {
        return Err(PyRuntimeError::new_err(
            "allow_splitk=False is not supported when reduced precision reductions are enabled",
        ));
    }
    allow_reduced_precision_state.store(allow_reduced_precision, Ordering::SeqCst);
    allow_splitk_state.store(allow_splitk, Ordering::SeqCst);
    Ok(())
}

fn set_cublas_reduced_precision_reduction_varargs(
    args: &Bound<'_, PyTuple>,
    allow_reduced_precision_state: &AtomicBool,
    allow_splitk_state: &AtomicBool,
    native_name: &str,
) -> PyResult<()> {
    match args.len() {
        0 => {
            return Err(PyTypeError::new_err(
                "function takes at least 1 argument (0 given)",
            ));
        }
        len if len > 2 => {
            return Err(PyTypeError::new_err(format!(
                "function takes at most 2 arguments ({len} given)"
            )));
        }
        _ => {}
    }

    let allow_reduced_precision = args.get_item(0)?;
    let allow_splitk = if args.len() == 2 {
        Some(args.get_item(1)?)
    } else {
        None
    };
    set_cublas_reduced_precision_reduction_native(
        &allow_reduced_precision,
        allow_splitk.as_ref(),
        allow_reduced_precision_state,
        allow_splitk_state,
        native_name,
    )
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn set_cublas_allow_fp16_reduced_precision_reduction_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: CPython supplies a live borrowed tuple to a METH_VARARGS callback.
    let args = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, args) }.cast_into::<PyTuple>()?;
    set_cublas_reduced_precision_reduction_varargs(
        &args,
        &CUBLAS_ALLOW_FP16_REDUCED_PRECISION_REDUCTION,
        &CUBLAS_ALLOW_FP16_REDUCED_PRECISION_REDUCTION_SPLIT_K,
        "set_allow_fp16_reduction_cublas",
    )?;
    Ok(py.None().into_ptr())
}

#[pyfunction(
    name = "_get_cublas_allow_fp16_reduced_precision_reduction",
    signature = (),
    text_signature = None
)]
#[pyo3(pass_module)]
fn get_cublas_allow_fp16_reduced_precision_reduction_native(
    _module: &Bound<'_, PyModule>,
) -> (bool, bool) {
    (
        CUBLAS_ALLOW_FP16_REDUCED_PRECISION_REDUCTION.load(Ordering::SeqCst),
        CUBLAS_ALLOW_FP16_REDUCED_PRECISION_REDUCTION_SPLIT_K.load(Ordering::SeqCst),
    )
}

#[allow(
    unsafe_code,
    reason = "the callback is entered through PyO3's panic-safe C trampoline"
)]
unsafe fn set_cublas_allow_bf16_reduced_precision_reduction_callback(
    py: Python<'_>,
    _module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
) -> PyResult<*mut ffi::PyObject> {
    // SAFETY: CPython supplies a live borrowed tuple to a METH_VARARGS callback.
    let args = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, args) }.cast_into::<PyTuple>()?;
    set_cublas_reduced_precision_reduction_varargs(
        &args,
        &CUBLAS_ALLOW_BF16_REDUCED_PRECISION_REDUCTION,
        &CUBLAS_ALLOW_BF16_REDUCED_PRECISION_REDUCTION_SPLIT_K,
        "set_allow_bf16_reduction_cublas",
    )?;
    Ok(py.None().into_ptr())
}

#[pyfunction(
    name = "_get_cublas_allow_bf16_reduced_precision_reduction",
    signature = (),
    text_signature = None
)]
#[pyo3(pass_module)]
fn get_cublas_allow_bf16_reduced_precision_reduction_native(
    _module: &Bound<'_, PyModule>,
) -> (bool, bool) {
    (
        CUBLAS_ALLOW_BF16_REDUCED_PRECISION_REDUCTION.load(Ordering::SeqCst),
        CUBLAS_ALLOW_BF16_REDUCED_PRECISION_REDUCTION_SPLIT_K.load(Ordering::SeqCst),
    )
}

#[allow(
    unsafe_code,
    reason = "PyCFunction_NewEx requires an audited stable-ABI raw-pointer call"
)]
fn add_cublas_allow_tf32_setter(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let mut definition = pyo3::impl_::pymethods::PyMethodDef::noargs(
        c"_set_cublas_allow_tf32",
        pyo3::impl_::trampoline::get_trampoline_function!(
            binaryfunc,
            set_cublas_allow_tf32_callback
        ),
        c"",
    )
    .into_raw();
    definition.ml_flags = ffi::METH_O;
    let definition = Box::leak(Box::new(definition));
    let module_name = module.name()?;
    // SAFETY: the leaked method definition, module, and module name all remain
    // live for the duration required by the newly owned built-in function.
    let function = unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(
            py,
            ffi::PyCFunction_NewEx(definition, module.as_ptr(), module_name.as_ptr()),
        )?
        .cast_into::<PyCFunction>()?
    };
    module.add_function(function)
}

#[allow(
    unsafe_code,
    reason = "PyCFunction_NewEx requires an audited stable-ABI raw-pointer call"
)]
fn add_cublas_reduced_precision_reduction_setter(
    module: &Bound<'_, PyModule>,
    name: &'static CStr,
    callback: ffi::PyCFunction,
) -> PyResult<()> {
    let py = module.py();
    let mut definition =
        pyo3::impl_::pymethods::PyMethodDef::noargs(name, callback, c"").into_raw();
    definition.ml_flags = ffi::METH_VARARGS;
    let definition = Box::leak(Box::new(definition));
    let module_name = module.name()?;
    // SAFETY: the leaked method definition, module, and module name all remain
    // live for the duration required by the newly owned built-in function.
    let function = unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(
            py,
            ffi::PyCFunction_NewEx(definition, module.as_ptr(), module_name.as_ptr()),
        )?
        .cast_into::<PyCFunction>()?
    };
    module.add_function(function)
}

fn add_cublas_matmul_builtins(module: &Bound<'_, PyModule>) -> PyResult<()> {
    add_cublas_allow_tf32_setter(module)?;
    module.add_function(wrap_pyfunction!(get_cublas_allow_tf32_native, module)?)?;
    add_cublas_reduced_precision_reduction_setter(
        module,
        c"_set_cublas_allow_fp16_reduced_precision_reduction",
        pyo3::impl_::trampoline::get_trampoline_function!(
            binaryfunc,
            set_cublas_allow_fp16_reduced_precision_reduction_callback
        ),
    )?;
    module.add_function(wrap_pyfunction!(
        get_cublas_allow_fp16_reduced_precision_reduction_native,
        module
    )?)?;
    add_cublas_reduced_precision_reduction_setter(
        module,
        c"_set_cublas_allow_bf16_reduced_precision_reduction",
        pyo3::impl_::trampoline::get_trampoline_function!(
            binaryfunc,
            set_cublas_allow_bf16_reduced_precision_reduction_callback
        ),
    )?;
    module.add_function(wrap_pyfunction!(
        get_cublas_allow_bf16_reduced_precision_reduction_native,
        module
    )?)?;
    let exports = module.getattr("__all__")?;
    for name in [
        "_set_cublas_allow_tf32",
        "_get_cublas_allow_tf32",
        "_set_cublas_allow_fp16_reduced_precision_reduction",
        "_get_cublas_allow_fp16_reduced_precision_reduction",
        "_set_cublas_allow_bf16_reduced_precision_reduction",
        "_get_cublas_allow_bf16_reduced_precision_reduction",
    ] {
        exports.call_method1("remove", (name,))?;
    }
    Ok(())
}

#[pyfunction(name = "_set_sdp_use_flash", signature = (enabled, /), text_signature = None)]
fn set_sdp_use_flash_native(enabled: &Bound<'_, PyAny>) -> PyResult<()> {
    if !enabled.is_exact_instance_of::<PyBool>() {
        let type_name = python_type_name(enabled)?;
        return Err(PyRuntimeError::new_err(format!(
            "set_sdp_use_math expects a bool, but got {type_name}"
        )));
    }
    FLASH_SDP_ENABLED.store(enabled.is_truthy()?, Ordering::SeqCst);
    Ok(())
}

#[pyfunction(name = "_get_flash_sdp_enabled", signature = (), text_signature = None)]
fn get_flash_sdp_enabled_native() -> bool {
    FLASH_SDP_ENABLED.load(Ordering::SeqCst)
}

fn add_flash_sdp_builtins(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(set_sdp_use_flash_native, module)?)?;
    module.add_function(wrap_pyfunction!(get_flash_sdp_enabled_native, module)?)?;
    let exports = module.getattr("__all__")?;
    exports.call_method1("remove", ("_set_sdp_use_flash",))?;
    exports.call_method1("remove", ("_get_flash_sdp_enabled",))?;
    Ok(())
}

#[pyfunction(
    name = "_set_sdp_use_mem_efficient",
    signature = (enabled, /),
    text_signature = None
)]
fn set_sdp_use_mem_efficient_native(enabled: &Bound<'_, PyAny>) -> PyResult<()> {
    if !enabled.is_exact_instance_of::<PyBool>() {
        let type_name = python_type_name(enabled)?;
        // PyTorch 2.13 uses this shared SDP validation diagnostic for the
        // memory-efficient setter as well as the math setter.
        return Err(PyRuntimeError::new_err(format!(
            "set_sdp_use_math expects a bool, but got {type_name}"
        )));
    }
    MEM_EFFICIENT_SDP_ENABLED.store(enabled.is_truthy()?, Ordering::SeqCst);
    Ok(())
}

#[pyfunction(
    name = "_get_mem_efficient_sdp_enabled",
    signature = (),
    text_signature = None
)]
fn get_mem_efficient_sdp_enabled_native() -> bool {
    MEM_EFFICIENT_SDP_ENABLED.load(Ordering::SeqCst)
}

fn add_mem_efficient_sdp_builtins(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(set_sdp_use_mem_efficient_native, module)?)?;
    module.add_function(wrap_pyfunction!(
        get_mem_efficient_sdp_enabled_native,
        module
    )?)?;
    let exports = module.getattr("__all__")?;
    exports.call_method1("remove", ("_set_sdp_use_mem_efficient",))?;
    exports.call_method1("remove", ("_get_mem_efficient_sdp_enabled",))?;
    Ok(())
}

#[pyfunction(name = "_set_sdp_use_math", signature = (enabled, /), text_signature = None)]
fn set_sdp_use_math_native(enabled: &Bound<'_, PyAny>) -> PyResult<()> {
    if !enabled.is_exact_instance_of::<PyBool>() {
        let type_name = python_type_name(enabled)?;
        return Err(PyRuntimeError::new_err(format!(
            "set_sdp_use_math expects a bool, but got {type_name}"
        )));
    }
    MATH_SDP_ENABLED.store(enabled.is_truthy()?, Ordering::SeqCst);
    Ok(())
}

#[pyfunction(name = "_get_math_sdp_enabled", signature = (), text_signature = None)]
fn get_math_sdp_enabled_native() -> bool {
    MATH_SDP_ENABLED.load(Ordering::SeqCst)
}

fn add_math_sdp_builtins(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(set_sdp_use_math_native, module)?)?;
    module.add_function(wrap_pyfunction!(get_math_sdp_enabled_native, module)?)?;
    let exports = module.getattr("__all__")?;
    exports.call_method1("remove", ("_set_sdp_use_math",))?;
    exports.call_method1("remove", ("_get_math_sdp_enabled",))?;
    Ok(())
}

#[pyfunction(name = "_set_sdp_use_cudnn", signature = (enabled, /), text_signature = None)]
fn set_sdp_use_cudnn_native(enabled: &Bound<'_, PyAny>) -> PyResult<()> {
    if !enabled.is_exact_instance_of::<PyBool>() {
        let type_name = python_type_name(enabled)?;
        return Err(PyRuntimeError::new_err(format!(
            "set_sdp_use_cudnn expects a bool, but got %s{type_name}"
        )));
    }
    CUDNN_SDP_ENABLED.store(enabled.is_truthy()?, Ordering::SeqCst);
    Ok(())
}

#[pyfunction(name = "_get_cudnn_sdp_enabled", signature = (), text_signature = None)]
fn get_cudnn_sdp_enabled_native() -> bool {
    CUDNN_SDP_ENABLED.load(Ordering::SeqCst)
}

fn add_cudnn_sdp_builtins(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(set_sdp_use_cudnn_native, module)?)?;
    module.add_function(wrap_pyfunction!(get_cudnn_sdp_enabled_native, module)?)?;
    let exports = module.getattr("__all__")?;
    exports.call_method1("remove", ("_set_sdp_use_cudnn",))?;
    exports.call_method1("remove", ("_get_cudnn_sdp_enabled",))?;
    Ok(())
}

#[pyfunction(
    name = "_set_math_sdp_allow_fp16_bf16_reduction",
    signature = (enabled, /),
    text_signature = None
)]
fn set_math_sdp_allow_fp16_bf16_reduction_native(enabled: &Bound<'_, PyAny>) -> PyResult<()> {
    if !enabled.is_exact_instance_of::<PyBool>() {
        let type_name = python_type_name(enabled)?;
        return Err(PyRuntimeError::new_err(format!(
            "set_sdp_use_math expects a bool, but got {type_name}"
        )));
    }
    MATH_SDP_ALLOW_FP16_BF16_REDUCTION.store(enabled.is_truthy()?, Ordering::SeqCst);
    Ok(())
}

#[pyfunction(
    name = "_get_math_sdp_allow_fp16_bf16_reduction",
    signature = (),
    text_signature = None
)]
fn get_math_sdp_allow_fp16_bf16_reduction_native() -> bool {
    MATH_SDP_ALLOW_FP16_BF16_REDUCTION.load(Ordering::SeqCst)
}

fn add_math_sdp_reduction_builtins(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        set_math_sdp_allow_fp16_bf16_reduction_native,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        get_math_sdp_allow_fp16_bf16_reduction_native,
        module
    )?)?;
    let exports = module.getattr("__all__")?;
    exports.call_method1("remove", ("_set_math_sdp_allow_fp16_bf16_reduction",))?;
    exports.call_method1("remove", ("_get_math_sdp_allow_fp16_bf16_reduction",))?;
    Ok(())
}

#[pyfunction(name = "_set_nnpack_enabled", signature = (enabled, /), text_signature = None)]
fn set_nnpack_enabled_native(enabled: &Bound<'_, PyAny>) -> PyResult<()> {
    if !enabled.is_exact_instance_of::<PyBool>() {
        let type_name = python_type_name(enabled)?;
        return Err(PyRuntimeError::new_err(format!(
            "set_enabled_NNPACK expects a bool, but got {type_name}"
        )));
    }
    NNPACK_ENABLED.store(enabled.is_truthy()?, Ordering::SeqCst);
    Ok(())
}

#[pyfunction(name = "_get_nnpack_enabled", signature = (), text_signature = None)]
fn get_nnpack_enabled_native() -> bool {
    NNPACK_ENABLED.load(Ordering::SeqCst)
}

fn add_nnpack_builtins(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(set_nnpack_enabled_native, module)?)?;
    module.add_function(wrap_pyfunction!(get_nnpack_enabled_native, module)?)?;
    let exports = module.getattr("__all__")?;
    exports.call_method1("remove", ("_set_nnpack_enabled",))?;
    exports.call_method1("remove", ("_get_nnpack_enabled",))?;
    Ok(())
}

#[pyfunction(
    name = "_exchange_enable_guard_collectives",
    signature = (enabled, /),
    text_signature = None
)]
fn exchange_enable_guard_collectives_native(enabled: bool) -> bool {
    GUARD_COLLECTIVES_ENABLED.swap(enabled, Ordering::SeqCst)
}

fn add_compiler_state_builtins(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        exchange_enable_guard_collectives_native,
        module
    )?)?;
    module
        .getattr("__all__")?
        .call_method1("remove", ("_exchange_enable_guard_collectives",))?;
    Ok(())
}

fn parse_clone_memory_format(memory_format: Option<&Bound<'_, PyAny>>) -> PyResult<MemoryFormat> {
    let Some(memory_format) = memory_format else {
        return Ok(MemoryFormat::Preserve);
    };
    if memory_format.is_none() {
        return Ok(MemoryFormat::Preserve);
    }
    if let Ok(memory_format) = memory_format.cast::<PyMemoryFormat>() {
        return Ok(memory_format.try_borrow()?.inner());
    }

    let type_name = memory_format.get_type().name()?;
    Err(PyTypeError::new_err(format!(
        "clone(): argument 'memory_format' must be torch.memory_format, not {type_name}"
    )))
}

fn parse_is_contiguous_memory_format(memory_format: &Bound<'_, PyAny>) -> PyResult<MemoryFormat> {
    if let Ok(memory_format) = memory_format.cast::<PyMemoryFormat>() {
        return Ok(memory_format.try_borrow()?.inner());
    }

    let type_name = memory_format.get_type().name()?;
    Err(PyTypeError::new_err(format!(
        "is_contiguous(): argument 'memory_format' must be torch.memory_format, not {type_name}"
    )))
}

fn parse_contiguous_memory_format(memory_format: &Bound<'_, PyAny>) -> PyResult<MemoryFormat> {
    if let Ok(memory_format) = memory_format.cast::<PyMemoryFormat>() {
        return Ok(memory_format.try_borrow()?.inner());
    }

    let type_name = memory_format.get_type().name()?;
    Err(PyTypeError::new_err(format!(
        "contiguous(): argument 'memory_format' must be torch.memory_format, not {type_name}"
    )))
}

fn parse_float_memory_format(memory_format: &Bound<'_, PyAny>) -> PyResult<MemoryFormat> {
    if memory_format.is_none() {
        return Ok(MemoryFormat::Preserve);
    }
    if let Ok(memory_format) = memory_format.cast::<PyMemoryFormat>() {
        return Ok(memory_format.try_borrow()?.inner());
    }

    let type_name = memory_format.get_type().name()?;
    Err(PyTypeError::new_err(format!(
        "float(): argument 'memory_format' must be torch.memory_format, not {type_name}"
    )))
}

fn parse_cpu_memory_format(memory_format: &Bound<'_, PyAny>) -> PyResult<MemoryFormat> {
    if memory_format.is_none() {
        return Ok(MemoryFormat::Preserve);
    }
    if let Ok(memory_format) = memory_format.cast::<PyMemoryFormat>() {
        return Ok(memory_format.try_borrow()?.inner());
    }

    let type_name = memory_format.get_type().name()?;
    Err(PyTypeError::new_err(format!(
        "cpu(): argument 'memory_format' must be torch.memory_format, not {type_name}"
    )))
}

fn bind_creation_arguments<'py>(
    function: &str,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<CreationCallArguments<'py>> {
    let mut size_was_provided = !positional.is_empty();
    let size = match positional.len() {
        0 => None,
        1 => optional_call_argument(positional.get_item(0)?).map(|value| {
            CreationSizeArgument::Single {
                value,
                origin: CreationSizeOrigin::Positional,
            }
        }),
        length => {
            let first = positional.get_item(0)?;
            if creation_variadic_size_start(function, &first)? {
                Some(CreationSizeArgument::Variadic(positional.clone()))
            } else {
                return Err(PyTypeError::new_err(format!(
                    "{function}() takes 1 positional argument but {length} were given"
                )));
            }
        }
    };
    let mut arguments = CreationCallArguments {
        size,
        shape: None,
        out: None,
        dtype: None,
        layout: None,
        device: None,
        pin_memory: None,
        requires_grad: None,
        keyword_error: None,
    };
    let Some(keywords) = keywords else {
        return Ok(arguments);
    };
    for (key, value) in keywords {
        let key = key.extract::<String>()?;
        match key.as_str() {
            "size" => {
                if size_was_provided {
                    arguments.keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err(format!(
                            "{function}() got multiple values for argument 'size'"
                        ))
                    });
                } else {
                    size_was_provided = true;
                    arguments.size =
                        optional_call_argument(value).map(|value| CreationSizeArgument::Single {
                            value,
                            origin: CreationSizeOrigin::SizeKeyword,
                        });
                }
            }
            "shape" => arguments.shape = optional_call_argument(value),
            "out" => arguments.out = optional_call_argument(value),
            "dtype" => arguments.dtype = optional_call_argument(value),
            "layout" => arguments.layout = optional_call_argument(value),
            "device" => arguments.device = optional_call_argument(value),
            "pin_memory" => arguments.pin_memory = optional_call_argument(value),
            "requires_grad" => arguments.requires_grad = optional_call_argument(value),
            _ => {
                arguments.keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "{function}() got an unexpected keyword argument '{key}'"
                    ))
                });
            }
        }
    }
    Ok(arguments)
}

fn creation_variadic_size_start(function: &str, dimension: &Bound<'_, PyAny>) -> PyResult<bool> {
    if dimension.is_instance_of::<PyBool>() {
        return Ok(false);
    }
    if python_number_index(dimension).is_err() {
        return Ok(false);
    }
    if dimension.is_instance_of::<PyTuple>() || dimension.is_instance_of::<PyList>() {
        return Ok(false);
    }

    // PyTorch's SymInt argument parser probes an arbitrary leading __index__
    // provider once more before the variadic dimensions are unpacked.
    if !dimension.is_instance_of::<PyInt>() && python_number_index(dimension).is_err() {
        return Err(creation_sequence_dimension_type_error(function, dimension)?);
    }
    Ok(true)
}

fn bind_like_factory_arguments<'py>(
    function: &str,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<LikeFactoryCallArguments<'py>> {
    if positional.len() > 1 {
        return Err(PyTypeError::new_err(format!(
            "{function}() takes 1 positional argument but {} were given",
            positional.len()
        )));
    }

    let mut arguments = LikeFactoryCallArguments {
        input: if positional.is_empty() {
            None
        } else {
            Some(ParsedCallArgument {
                value: positional.get_item(0)?,
                position: Some(1),
            })
        },
        dtype: None,
        layout: None,
        device: None,
        requires_grad: None,
        memory_format: None,
        keyword_error: None,
    };
    let Some(keywords) = keywords else {
        return Ok(arguments);
    };

    for (key, value) in keywords {
        let key = key.extract::<String>()?;
        match key.as_str() {
            "input" | "x" | "a" | "x1" => {
                if arguments.input.is_some() {
                    arguments.keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err(format!(
                            "{function}() got multiple values for argument 'input'"
                        ))
                    });
                } else {
                    arguments.input = Some(ParsedCallArgument {
                        value,
                        position: None,
                    });
                }
            }
            "dtype" => arguments.dtype = optional_call_argument(value),
            "layout" => arguments.layout = optional_call_argument(value),
            "device" => arguments.device = optional_call_argument(value),
            "requires_grad" => arguments.requires_grad = optional_call_argument(value),
            "memory_format" => arguments.memory_format = optional_call_argument(value),
            _ => {
                arguments.keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "{function}() got an unexpected keyword argument '{key}'"
                    ))
                });
            }
        }
    }
    Ok(arguments)
}

fn bind_full_like_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<FullLikeCallArguments<'py>> {
    if positional.len() > 2 {
        return Err(PyTypeError::new_err(format!(
            "full_like() takes 2 positional arguments but {} were given",
            positional.len()
        )));
    }

    let mut arguments = FullLikeCallArguments {
        input: if positional.is_empty() {
            None
        } else {
            Some(ParsedCallArgument {
                value: positional.get_item(0)?,
                position: Some(1),
            })
        },
        fill_value: if positional.len() < 2 {
            None
        } else {
            Some(positional.get_item(1)?)
        },
        dtype: None,
        layout: None,
        device: None,
        requires_grad: None,
        memory_format: None,
        keyword_error: None,
    };
    let Some(keywords) = keywords else {
        return Ok(arguments);
    };

    for (key, value) in keywords {
        let key = key.extract::<String>()?;
        match key.as_str() {
            "input" | "x" | "a" | "x1" => {
                if arguments.input.is_some() {
                    arguments.keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err("full_like() got multiple values for argument 'input'")
                    });
                } else {
                    arguments.input = Some(ParsedCallArgument {
                        value,
                        position: None,
                    });
                }
            }
            "fill_value" => {
                if arguments.fill_value.is_some() {
                    arguments.keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err(
                            "full_like() got multiple values for argument 'fill_value'",
                        )
                    });
                } else {
                    arguments.fill_value = Some(value);
                }
            }
            "dtype" => arguments.dtype = optional_call_argument(value),
            "layout" => arguments.layout = optional_call_argument(value),
            "device" => arguments.device = optional_call_argument(value),
            "requires_grad" => arguments.requires_grad = optional_call_argument(value),
            "memory_format" => arguments.memory_format = optional_call_argument(value),
            _ => {
                arguments.keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "full_like() got an unexpected keyword argument '{key}'"
                    ))
                });
            }
        }
    }
    Ok(arguments)
}

fn optional_call_argument(value: Bound<'_, PyAny>) -> Option<Bound<'_, PyAny>> {
    if value.is_none() { None } else { Some(value) }
}

fn validate_creation_out(function: &str, out: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
    let Some(out) = out else {
        return Ok(false);
    };
    if out.cast::<PyTensor>().is_ok() {
        return Ok(true);
    }

    let actual = python_type_name(out)?;
    Err(PyTypeError::new_err(format!(
        "{function}(): argument 'out' must be Tensor, not {actual}"
    )))
}

fn bind_arange_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<ArangeCallArguments<'py>> {
    if positional.len() > 1 {
        return Err(arange_overload_unsupported());
    }

    let mut arguments = ArangeCallArguments {
        end: if positional.is_empty() {
            None
        } else {
            Some(ParsedCallArgument {
                value: positional.get_item(0)?,
                position: Some(1),
            })
        },
        unsupported_overload: false,
        out: None,
        dtype: None,
        layout: None,
        device: None,
        pin_memory: None,
        requires_grad: None,
        keyword_error: None,
    };
    let Some(keywords) = keywords else {
        return Ok(arguments);
    };

    for (key, value) in keywords {
        let key = key.extract::<String>()?;
        match key.as_str() {
            "end" => {
                if arguments.end.is_some() {
                    // A positional value combined with `end=` selects
                    // PyTorch's two-bound overload rather than duplicating the
                    // one-bound argument. That overload remains unsupported.
                    arguments.unsupported_overload = true;
                } else {
                    arguments.end = optional_call_argument(value).map(|value| ParsedCallArgument {
                        value,
                        position: None,
                    });
                }
            }
            "start" | "step" => arguments.unsupported_overload = true,
            "out" => arguments.out = optional_call_argument(value),
            "dtype" => arguments.dtype = optional_call_argument(value),
            "layout" => arguments.layout = optional_call_argument(value),
            "device" => arguments.device = optional_call_argument(value),
            "pin_memory" => arguments.pin_memory = optional_call_argument(value),
            "requires_grad" => arguments.requires_grad = optional_call_argument(value),
            _ => {
                arguments.keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "arange() got an unexpected keyword argument '{key}'"
                    ))
                });
            }
        }
    }
    Ok(arguments)
}

fn parse_arange_arguments(arguments: ArangeCallArguments<'_>) -> PyResult<(usize, bool)> {
    let ArangeCallArguments {
        end,
        unsupported_overload,
        out,
        dtype,
        layout,
        device,
        pin_memory,
        requires_grad,
        keyword_error,
    } = arguments;

    if unsupported_overload {
        return Err(arange_overload_unsupported());
    }
    let Some(end) = end else {
        return Err(PyTypeError::new_err(
            "arange() missing required argument 'end'",
        ));
    };
    let exact_float_endpoint = end.value.is_exact_instance_of::<PyFloat>();
    let exact_integer_endpoint = end.value.is_exact_instance_of::<PyInt>();
    let numpy_float_endpoint = if exact_float_endpoint || exact_integer_endpoint {
        false
    } else {
        is_numpy_scalar_of_types(&end.value, &["floating"])?
    };
    // Peek only at a valid native dtype here so unsupported endpoint/dtype
    // combinations retain the endpoint-first error ordering below.
    let exact_integer_with_explicit_float32 = if exact_integer_endpoint {
        match dtype
            .as_ref()
            .and_then(|dtype| dtype.cast::<PyDType>().ok())
        {
            Some(dtype) => dtype.try_borrow()?.inner() == DType::Float32,
            None => false,
        }
    } else {
        false
    };
    if !exact_float_endpoint && !numpy_float_endpoint && !exact_integer_with_explicit_float32 {
        let position = end
            .position
            .map_or_else(String::new, |position| format!(" (position {position})"));
        let actual = python_type_name(&end.value)?;
        return Err(PyTypeError::new_err(format!(
            "arange(): argument 'end'{position} must be an exact Python float, not {actual}"
        )));
    }

    let dtype = parse_dtype("arange", dtype.as_ref())?;
    parse_factory_layout("arange", layout.as_ref())?;
    validate_device_argument_type("arange", device.as_ref())?;
    let pin_memory = parse_factory_bool("arange", "pin_memory", pin_memory.as_ref())?;
    let requires_grad = parse_factory_requires_grad("arange", requires_grad.as_ref())?;
    if let Some(error) = keyword_error {
        return Err(error);
    }
    let device = parse_device("arange", device.as_ref())?;

    if out.is_some() {
        return Err(PyRuntimeError::new_err(
            "arange(): the 'out' argument is not supported",
        ));
    }
    if dtype != DType::Float32 || device != Device::Cpu {
        return Err(PyRuntimeError::new_err(
            "arange(): only the default float32 CPU metadata is supported",
        ));
    }
    if pin_memory {
        return Err(PyRuntimeError::new_err(
            "arange(): pin_memory=True is not supported; only unpinned CPU storage is implemented",
        ));
    }
    let elements = arange_element_count(extract_arange_end(&end.value)?)?;
    Ok((elements, requires_grad))
}

#[allow(clippy::cast_precision_loss)]
fn extract_arange_end(end: &Bound<'_, PyAny>) -> PyResult<f64> {
    if end.is_exact_instance_of::<PyInt>() {
        // Match PyTorch's signed-then-unsigned scalar conversion, including
        // the distinct overflow errors outside the combined i64/u64 range.
        if let Ok(end) = end.extract::<i64>() {
            return Ok(end as f64);
        }
        return end.extract::<u64>().map(|end| end as f64);
    }
    end.extract::<f64>()
}

fn arange_element_count(end: f64) -> PyResult<usize> {
    if !end.is_finite() {
        let value = if end.is_nan() {
            if end.is_sign_negative() {
                "-nan"
            } else {
                "nan"
            }
        } else if end.is_sign_negative() {
            "-inf"
        } else {
            "inf"
        };
        return Err(PyRuntimeError::new_err(format!(
            "unsupported range: 0 -> {value}"
        )));
    }
    if end < 0.0 {
        return Err(PyRuntimeError::new_err(
            "upper bound and lower bound inconsistent with step sign",
        ));
    }

    let elements = end.ceil();
    let i64_exclusive_upper_bound: f64 = 9_223_372_036_854_775_808.0;
    if elements.to_bits() == i64_exclusive_upper_bound.to_bits() {
        return Err(PyRuntimeError::new_err(
            "IntArrayRef contains an int that cannot be represented as a SymInt: -9223372036854775808",
        ));
    }
    if elements > i64_exclusive_upper_bound {
        return Err(PyRuntimeError::new_err("invalid size, possible overflow?"));
    }
    #[cfg(target_pointer_width = "32")]
    if elements > f64::from(u32::MAX) {
        return Err(PyRuntimeError::new_err("invalid size, possible overflow?"));
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(elements as usize)
}

fn arange_overload_unsupported() -> PyErr {
    PyTypeError::new_err(
        "arange(): start and step overloads are not supported; pass one exact Python float endpoint",
    )
}

fn bind_as_tensor_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<AsTensorCallArguments<'py>> {
    if positional.len() > 1 {
        return Err(PyTypeError::new_err(format!(
            "as_tensor() takes 1 positional argument but {} were given",
            positional.len()
        )));
    }

    let mut arguments = AsTensorCallArguments {
        data: if positional.is_empty() {
            None
        } else {
            Some(ParsedCallArgument {
                value: positional.get_item(0)?,
                position: Some(1),
            })
        },
        dtype: None,
        device: None,
        keyword_error: None,
    };
    let Some(keywords) = keywords else {
        return Ok(arguments);
    };

    for (key, value) in keywords {
        let key = key.extract::<String>()?;
        match key.as_str() {
            "data" => {
                if arguments.data.is_some() {
                    arguments.keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err("as_tensor() got multiple values for argument 'data'")
                    });
                } else {
                    arguments.data = Some(ParsedCallArgument {
                        value,
                        position: None,
                    });
                }
            }
            "dtype" => arguments.dtype = optional_call_argument(value),
            "device" => arguments.device = optional_call_argument(value),
            _ => {
                arguments.keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "as_tensor() got an unexpected keyword argument '{key}'"
                    ))
                });
            }
        }
    }
    Ok(arguments)
}

fn bind_asarray_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<AsArrayCallArguments<'py>> {
    if positional.len() > 1 {
        return Err(PyTypeError::new_err(format!(
            "asarray() takes 1 positional argument but {} were given",
            positional.len()
        )));
    }

    let mut arguments = AsArrayCallArguments {
        obj: if positional.is_empty() {
            None
        } else {
            Some(ParsedCallArgument {
                value: positional.get_item(0)?,
                position: Some(1),
            })
        },
        dtype: None,
        device: None,
        copy: None,
        requires_grad: None,
        keyword_error: None,
    };
    let Some(keywords) = keywords else {
        return Ok(arguments);
    };

    for (key, value) in keywords {
        let key = key.extract::<String>()?;
        match key.as_str() {
            "obj" => {
                if arguments.obj.is_some() {
                    arguments.keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err("asarray() got multiple values for argument 'obj'")
                    });
                } else {
                    arguments.obj = Some(ParsedCallArgument {
                        value,
                        position: None,
                    });
                }
            }
            "dtype" => arguments.dtype = optional_call_argument(value),
            "device" => arguments.device = optional_call_argument(value),
            "copy" => arguments.copy = optional_call_argument(value),
            "requires_grad" => arguments.requires_grad = optional_call_argument(value),
            _ => {
                arguments.keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "asarray() got an unexpected keyword argument '{key}'"
                    ))
                });
            }
        }
    }
    Ok(arguments)
}

fn parse_as_tensor_dtype(dtype: Option<&Bound<'_, PyAny>>) -> PyResult<DType> {
    parse_identity_dtype("as_tensor", dtype)
}

fn validate_identity_dtype_type(function: &str, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let Some(dtype) = dtype else {
        return Ok(());
    };
    if dtype.cast::<PyDType>().is_ok() {
        return Ok(());
    }

    let type_name = dtype.get_type().name()?;
    Err(PyTypeError::new_err(format!(
        "{function}(): argument 'dtype' must be torch.dtype, not {type_name}"
    )))
}

fn parse_identity_dtype(function: &str, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<DType> {
    let dtype = parse_dtype(function, dtype)?;
    if dtype == DType::Float32 {
        return Ok(dtype);
    }
    Err(PyNotImplementedError::new_err(format!(
        "{function}(): dtype conversions are not supported; only torch.float32 identity is implemented"
    )))
}

fn validate_as_tensor_device_type(
    function: &str,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let Some(device) = device else {
        return Ok(());
    };
    if device.cast::<PyDevice>().is_ok() || device.cast::<PyString>().is_ok() {
        return Ok(());
    }
    let type_name = device.get_type().name()?;
    Err(PyTypeError::new_err(format!(
        "{function}(): argument 'device' must be torch.device, not {type_name}"
    )))
}

fn parse_as_tensor_device(function: &str, device: Option<&Bound<'_, PyAny>>) -> PyResult<Device> {
    let Some(device) = device else {
        return Ok(Device::Cpu);
    };
    if let Ok(device) = device.cast::<PyDevice>() {
        let device = device.try_borrow()?;
        if device.inner().is_cpu() && !device.has_index() {
            return Ok(Device::Cpu);
        }
        return Err(PyNotImplementedError::new_err(format!(
            "{function}(): indexed CPU devices require a copy and are not supported"
        )));
    }

    let specification = device.cast::<PyString>()?.to_str()?;
    if specification == "cpu" {
        return Ok(Device::Cpu);
    }
    validate_as_tensor_device_string(specification)?;
    parse_device_value(function, device)?;
    Err(PyNotImplementedError::new_err(format!(
        "{function}(): explicit indexed CPU devices require a copy and are not supported"
    )))
}

fn validate_as_tensor_device_string(specification: &str) -> PyResult<()> {
    if specification.is_empty() {
        return Err(PyRuntimeError::new_err("Device string must not be empty"));
    }
    let (device_type, index) = specification
        .split_once(':')
        .map_or((specification, None), |(device_type, index)| {
            (device_type, Some(index))
        });
    let known_type = matches!(
        device_type,
        "cpu"
            | "cuda"
            | "ipu"
            | "xpu"
            | "mkldnn"
            | "opengl"
            | "opencl"
            | "ideep"
            | "hip"
            | "ve"
            | "fpga"
            | "maia"
            | "xla"
            | "lazy"
            | "vulkan"
            | "mps"
            | "meta"
            | "hpu"
            | "mtia"
            | "privateuseone"
    );
    if !known_type {
        if !device_type.is_empty()
            && device_type
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte == b'_')
        {
            return Err(PyRuntimeError::new_err(format!(
                "Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone device type at start of device string: {device_type}"
            )));
        }
        return Err(PyRuntimeError::new_err(format!(
            "Invalid device string: '{specification}'"
        )));
    }
    if let Some(index) = index {
        let valid_digits = !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit());
        if !valid_digits || (index.len() > 1 && index.starts_with('0')) {
            return Err(PyRuntimeError::new_err(format!(
                "Invalid device string: '{specification}'"
            )));
        }
        if index.parse::<i32>().is_err() {
            return Err(PyRuntimeError::new_err(format!(
                "Could not parse device index '{index}' in device string '{specification}'"
            )));
        }
    }
    Ok(())
}

fn validate_asarray_bool(argument: &str, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.is_exact_instance_of::<PyBool>() {
        return Ok(());
    }
    let actual = python_type_name(value)?;
    Err(PyTypeError::new_err(format!(
        "asarray(): argument '{argument}' must be bool, not {actual}"
    )))
}

fn validate_asarray_copy(copy: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let Some(copy) = copy else {
        return Ok(());
    };
    if !copy.is_truthy()? {
        return Ok(());
    }
    Err(PyNotImplementedError::new_err(
        "asarray(): copy=True requires a copy and is not supported",
    ))
}

fn asarray_copy_requested(copy: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
    let Some(copy) = copy else {
        return Ok(false);
    };
    copy.is_truthy()
}

fn validate_asarray_scalar_copy(copy: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    if copy.is_none() {
        return Ok(());
    }
    Err(PyNotImplementedError::new_err(
        "asarray(): copy=False for Python float scalar inputs is not supported because scalar conversion requires fresh storage",
    ))
}

fn validate_asarray_sequence_copy(copy: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let Some(copy) = copy else {
        return Ok(());
    };
    if copy.is_truthy()? {
        return Ok(());
    }
    Err(PyValueError::new_err(
        "can't alias arbitrary sequence into a tensor.",
    ))
}

fn validate_asarray_device_string_syntax(device: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let Some(device) = device else {
        return Ok(());
    };
    if let Ok(device) = device.cast::<PyString>() {
        validate_as_tensor_device_string(device.to_str()?)?;
    }
    Ok(())
}

fn validate_asarray_requires_grad(requires_grad: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    if requires_grad.is_none() {
        return Ok(());
    }
    Err(PyNotImplementedError::new_err(
        "asarray(): explicit requires_grad changes are not supported; existing tensor autograd state is preserved",
    ))
}

fn dispatch_as_tensor_mode(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<Py<PyAny>>> {
    if torch_function_mode_stack::is_empty() {
        return Ok(None);
    }

    let function = variable_function(py, "as_tensor")?;
    let types = PyTuple::empty(py);
    let active_mode = torch_function_mode_stack::pop();
    let Some(mode) = active_mode.get() else {
        return Ok(None);
    };
    validate_torch_function_mode_handler(mode.bind(py))?;
    let handler = mode.bind(py).getattr("__torch_function__")?;
    let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
    if !is_not_implemented(py, &result) {
        return Ok(Some(result));
    }

    Err(torch_function_dispatch_error(
        py,
        "torch.as_tensor",
        Some(mode),
        None,
    )?)
}

fn dispatch_asarray_mode(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<Py<PyAny>>> {
    if torch_function_mode_stack::is_empty() {
        return Ok(None);
    }

    let function = variable_function(py, "asarray")?;
    let types = PyTuple::empty(py);
    let active_mode = torch_function_mode_stack::pop();
    let Some(mode) = active_mode.get() else {
        return Ok(None);
    };
    validate_torch_function_mode_handler(mode.bind(py))?;
    let handler = mode.bind(py).getattr("__torch_function__")?;
    let result = call_torch_function_handler(py, &handler, &function, &types, args, kwargs)?;
    if !is_not_implemented(py, &result) {
        return Ok(Some(result));
    }

    Err(torch_function_dispatch_error(
        py,
        "torch.asarray",
        Some(mode),
        None,
    )?)
}

fn bind_scalar_tensor_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<ScalarTensorCallArguments<'py>> {
    if positional.len() > 1 {
        return Err(PyTypeError::new_err(format!(
            "scalar_tensor() takes 1 positional argument but {} were given",
            positional.len()
        )));
    }

    let mut arguments = ScalarTensorCallArguments {
        scalar: if positional.is_empty() {
            None
        } else {
            Some(ParsedCallArgument {
                value: positional.get_item(0)?,
                position: Some(1),
            })
        },
        dtype: None,
        layout: None,
        device: None,
        pin_memory: None,
        requires_grad: None,
        keyword_error: None,
    };
    let Some(keywords) = keywords else {
        return Ok(arguments);
    };

    for (key, value) in keywords {
        let key = key.extract::<String>()?;
        match key.as_str() {
            "s" => {
                if arguments.scalar.is_some() {
                    arguments.keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err("scalar_tensor() got multiple values for argument 's'")
                    });
                } else {
                    arguments.scalar = Some(ParsedCallArgument {
                        value,
                        position: None,
                    });
                }
            }
            "dtype" => arguments.dtype = optional_call_argument(value),
            "layout" => arguments.layout = optional_call_argument(value),
            "device" => arguments.device = optional_call_argument(value),
            "pin_memory" => arguments.pin_memory = optional_call_argument(value),
            "requires_grad" => arguments.requires_grad = optional_call_argument(value),
            _ => {
                arguments.keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "scalar_tensor() got an unexpected keyword argument '{key}'"
                    ))
                });
            }
        }
    }
    Ok(arguments)
}

fn parse_scalar_tensor_arguments(
    arguments: ScalarTensorCallArguments<'_>,
) -> PyResult<(f32, DType, Device, bool, bool)> {
    let ScalarTensorCallArguments {
        scalar,
        dtype,
        layout,
        device,
        pin_memory,
        requires_grad,
        keyword_error,
    } = arguments;
    let Some(scalar) = scalar else {
        return Err(PyTypeError::new_err(
            "scalar_tensor() missing 1 required positional arguments: \"s\"",
        ));
    };

    validate_scalar_tensor_value(&scalar)?;
    let dtype = parse_dtype("scalar_tensor", dtype.as_ref())?;
    parse_scalar_tensor_layout(layout.as_ref())?;
    validate_scalar_tensor_device_type(device.as_ref())?;
    let pin_memory = parse_scalar_tensor_bool("pin_memory", pin_memory.as_ref())?;
    let requires_grad = parse_factory_requires_grad("scalar_tensor", requires_grad.as_ref())?;
    if let Some(error) = keyword_error {
        return Err(error);
    }
    let device = parse_scalar_tensor_device(device.as_ref())?;
    let value = convert_scalar_tensor_value(&scalar)?;
    Ok((value, dtype, device, pin_memory, requires_grad))
}

fn validate_scalar_tensor_value(scalar: &ParsedCallArgument<'_>) -> PyResult<()> {
    let value = &scalar.value;
    let valid = if let Ok(tensor) = value.cast::<PyTensor>() {
        let tensor = tensor.try_borrow()?;
        tensor.inner.shape().is_empty() && !tensor.inner.requires_grad()
    } else if value.is_instance_of::<PyInt>()
        || value.is_instance_of::<PyFloat>()
        || value.is_instance_of::<PyComplex>()
    {
        true
    } else {
        is_numpy_scalar_tensor_number(value)?
    };
    if valid {
        return Ok(());
    }

    let position = scalar
        .position
        .map_or_else(String::new, |position| format!(" (position {position})"));
    let actual = python_type_name(value)?;
    Err(PyTypeError::new_err(format!(
        "scalar_tensor(): argument 's'{position} must be Number, not {actual}"
    )))
}

fn is_numpy_scalar_tensor_number(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    is_numpy_scalar_of_types(value, &["bool_", "integer", "floating", "complexfloating"])
}

fn is_numpy_scalar_of_types(value: &Bound<'_, PyAny>, scalar_types: &[&str]) -> PyResult<bool> {
    let Ok(numpy) = PyModule::import(value.py(), "numpy") else {
        return Ok(false);
    };
    let value_type = value.get_type();
    let numpy_generic = numpy.getattr("generic")?.cast_into::<PyType>()?;
    if !value_type.is_subclass(numpy_generic.as_any())? {
        return Ok(false);
    }
    for scalar_type in scalar_types {
        let scalar_type = numpy.getattr(*scalar_type)?.cast_into::<PyType>()?;
        if value_type.is_subclass(scalar_type.as_any())? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn convert_scalar_tensor_value(scalar: &ParsedCallArgument<'_>) -> PyResult<f32> {
    let value = &scalar.value;
    let parsed = if let Ok(tensor) = value.cast::<PyTensor>() {
        tensor
            .try_borrow()?
            .inner
            .item()
            .map(ParsedFillValue::TensorScalar)
            .map_err(|error| tensor_error(&error))?
    } else if value.is_instance_of::<PyInt>() {
        if let Ok(value) = value.extract::<i64>() {
            ParsedFillValue::SignedInteger(value)
        } else {
            ParsedFillValue::UnsignedInteger(value.extract::<u64>()?)
        }
    } else if value.is_instance_of::<PyFloat>() {
        ParsedFillValue::Float(value.extract::<f64>()?)
    } else if value.is_instance_of::<PyComplex>() {
        return Err(scalar_tensor_overflow());
    } else {
        parse_numpy_scalar_tensor_value(value)?
    };
    parsed.into_scalar_tensor_f32()
}

fn parse_numpy_scalar_tensor_value(value: &Bound<'_, PyAny>) -> PyResult<ParsedFillValue> {
    let numpy = PyModule::import(value.py(), "numpy")?;
    if value.is_instance(&numpy.getattr("bool_")?)? {
        return value
            .is_truthy()
            .map(|value| ParsedFillValue::SignedInteger(i64::from(value)));
    }
    if value.is_instance(&numpy.getattr("integer")?)? {
        return value
            .extract::<i64>()
            .map(ParsedFillValue::SignedInteger)
            .map_err(|_| PyTypeError::new_err("an integer is required"));
    }
    if value.is_instance(&numpy.getattr("floating")?)?
        || value.is_instance(&numpy.getattr("complexfloating")?)?
    {
        return value.extract::<f64>().map(ParsedFillValue::Float);
    }
    unreachable!("scalar_tensor NumPy values were validated before conversion")
}

fn parse_scalar_tensor_layout(layout: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    parse_factory_layout("scalar_tensor", layout)
}

fn parse_factory_layout(function: &str, layout: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let Some(layout) = layout else {
        return Ok(());
    };
    if layout.is_instance(layout_objects(layout.py())?.layout.bind(layout.py()))? {
        return Ok(());
    }
    let actual = python_type_name(layout)?;
    Err(PyTypeError::new_err(format!(
        "{function}(): argument 'layout' must be torch.layout, not {actual}"
    )))
}

fn validate_scalar_tensor_device_type(device: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let Some(device) = device else {
        return Ok(());
    };
    if device.cast::<PyDevice>().is_ok() || device.cast::<PyString>().is_ok() {
        return Ok(());
    }
    let actual = python_type_name(device)?;
    Err(PyTypeError::new_err(format!(
        "scalar_tensor(): argument 'device' must be torch.device, not {actual}"
    )))
}

fn parse_scalar_tensor_device(device: Option<&Bound<'_, PyAny>>) -> PyResult<Device> {
    let Some(device) = device else {
        return Ok(Device::Cpu);
    };
    if let Ok(device) = device.cast::<PyDevice>() {
        return Ok(device.try_borrow()?.inner());
    }
    let specification = device.cast::<PyString>()?.to_str()?;
    if specification.is_empty() {
        return Err(PyRuntimeError::new_err("Device string must not be empty"));
    }
    let (device_type, index) = specification
        .split_once(':')
        .map_or((specification, None), |(device_type, index)| {
            (device_type, Some(index))
        });
    let known_type = matches!(
        device_type,
        "cpu"
            | "cuda"
            | "ipu"
            | "xpu"
            | "mkldnn"
            | "opengl"
            | "opencl"
            | "ideep"
            | "hip"
            | "ve"
            | "fpga"
            | "maia"
            | "xla"
            | "lazy"
            | "vulkan"
            | "mps"
            | "meta"
            | "hpu"
            | "mtia"
            | "privateuseone"
    );
    if !known_type {
        if !device_type.is_empty()
            && device_type
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte == b'_')
        {
            return Err(PyRuntimeError::new_err(format!(
                "Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone device type at start of device string: {device_type}"
            )));
        }
        return Err(PyRuntimeError::new_err(format!(
            "Invalid device string: '{specification}'"
        )));
    }
    if let Some(index) = index {
        let valid_digits = !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit());
        if !valid_digits || (index.len() > 1 && index.starts_with('0')) {
            return Err(PyRuntimeError::new_err(format!(
                "Invalid device string: '{specification}'"
            )));
        }
        if index.parse::<i32>().is_err() {
            return Err(PyRuntimeError::new_err(format!(
                "Could not parse device index '{index}' in device string '{specification}'"
            )));
        }
    }
    if device_type == "cpu" {
        return Ok(Device::Cpu);
    }
    Err(PyRuntimeError::new_err(format!(
        "scalar_tensor(): device '{specification}' is not supported; only 'cpu' is implemented"
    )))
}

fn parse_scalar_tensor_bool(argument: &str, value: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
    parse_factory_bool("scalar_tensor", argument, value)
}

fn parse_factory_bool(
    function: &str,
    argument: &str,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    let Some(value) = value else {
        return Ok(false);
    };
    if value.is_exact_instance_of::<PyBool>() {
        return value.is_truthy();
    }
    let actual = python_type_name(value)?;
    Err(PyTypeError::new_err(format!(
        "{function}(): argument '{argument}' must be bool, not {actual}"
    )))
}

fn bind_eye_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<EyeCallArguments<'py>> {
    // PyTorch rejects excess positional arguments before inspecting keywords.
    if positional.len() > 2 {
        return Err(PyTypeError::new_err(format!(
            "eye() takes from 1 to 2 positional arguments but {} were given",
            positional.len()
        )));
    }

    let mut arguments = EyeCallArguments {
        n: if positional.is_empty() {
            None
        } else {
            Some(positional.get_item(0)?)
        },
        m: if positional.len() < 2 {
            None
        } else {
            Some(positional.get_item(1)?)
        },
        dtype: None,
        device: None,
        requires_grad: None,
        keyword_error: None,
    };
    let Some(keywords) = keywords else {
        return Ok(arguments);
    };

    for (key, value) in keywords {
        let key = key.extract::<String>()?;
        match key.as_str() {
            "n" => {
                if arguments.n.is_some() {
                    arguments.keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err("eye() got multiple values for argument 'n'")
                    });
                } else {
                    arguments.n = Some(value);
                }
            }
            "m" => {
                if arguments.m.is_some() {
                    arguments.keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err("eye() got multiple values for argument 'm'")
                    });
                } else {
                    arguments.m = Some(value);
                }
            }
            "dtype" => arguments.dtype = optional_call_argument(value),
            "device" => arguments.device = optional_call_argument(value),
            "requires_grad" => arguments.requires_grad = optional_call_argument(value),
            _ => {
                arguments.keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "eye() got an unexpected keyword argument '{key}'"
                    ))
                });
            }
        }
    }
    Ok(arguments)
}

fn parse_eye_arguments(
    arguments: EyeCallArguments<'_>,
) -> PyResult<(usize, usize, DType, Device, bool)> {
    let EyeCallArguments {
        n,
        m,
        dtype,
        device,
        requires_grad,
        keyword_error,
    } = arguments;
    let Some(n) = n else {
        return Err(PyTypeError::new_err(
            "eye() missing 1 required positional argument: 'n'",
        ));
    };

    // Factory options are type-checked before dimension conversion. Device
    // resolution and shape validation happen only after all declared option
    // types and competing keywords have been checked.
    let dtype = parse_dtype("eye", dtype.as_ref())?;
    validate_device_argument_type("eye", device.as_ref())?;
    let requires_grad = parse_factory_requires_grad("eye", requires_grad.as_ref())?;
    if let Some(error) = keyword_error {
        return Err(error);
    }
    let device = parse_device("eye", device.as_ref())?;
    let n = parse_eye_dimension("n", &n)?;
    let m = m.map_or(Ok(n), |m| parse_eye_dimension("m", &m))?;
    let n = validate_eye_dimension("n", n)?;
    let m = validate_eye_dimension("m", m)?;
    Ok((n, m, dtype, device, requires_grad))
}

fn parse_creation_arguments(
    function: &str,
    arguments: CreationCallArguments<'_>,
) -> PyResult<(ParsedCreationSize, DType, Device, bool)> {
    let CreationCallArguments {
        size,
        shape,
        out,
        dtype,
        layout,
        device,
        pin_memory,
        requires_grad,
        keyword_error,
    } = arguments;

    // PyTorch validates declared argument types in signature order, reports
    // duplicate or unknown keywords, converts an accepted scalar dimension,
    // and only then resolves a valid device specification.
    let size = parse_creation_size(function, size.as_ref(), shape.as_ref())?;
    let has_out = validate_creation_out(function, out.as_ref())?;
    let dtype = parse_dtype(function, dtype.as_ref())?;
    parse_factory_layout(function, layout.as_ref())?;
    validate_device_argument_type(function, device.as_ref())?;
    let pin_memory = parse_factory_bool(function, "pin_memory", pin_memory.as_ref())?;
    let requires_grad = parse_factory_requires_grad(function, requires_grad.as_ref())?;
    if let Some(error) = keyword_error {
        return Err(error);
    }
    let size = finish_creation_size(function, size)?;
    let device = parse_device(function, device.as_ref())?;
    if has_out {
        return Err(PyRuntimeError::new_err(format!(
            "{function}(): the 'out' argument is not supported"
        )));
    }
    if pin_memory {
        return Err(PyRuntimeError::new_err(format!(
            "{function}(): pin_memory=True is not supported; only unpinned CPU storage is implemented"
        )));
    }
    Ok((size, dtype, device, requires_grad))
}

fn parse_like_factory_arguments<'py>(
    function: &str,
    arguments: LikeFactoryCallArguments<'py>,
) -> PyResult<(Bound<'py, PyTensor>, bool)> {
    let LikeFactoryCallArguments {
        input,
        dtype,
        layout,
        device,
        requires_grad,
        memory_format,
        keyword_error,
    } = arguments;

    let Some(input) = input else {
        return Err(PyTypeError::new_err(format!(
            "{function}() missing 1 required positional arguments: \"input\""
        )));
    };
    let input = parse_exact_native_like_factory_tensor_argument(function, "input", &input)?;
    let memory_format = parse_like_factory_memory_format(function, memory_format.as_ref())?;
    parse_identity_dtype(function, dtype.as_ref())?;
    parse_factory_layout(function, layout.as_ref())?;
    validate_as_tensor_device_type(function, device.as_ref())?;
    let requires_grad = parse_factory_requires_grad(function, requires_grad.as_ref())?;
    if let Some(error) = keyword_error {
        return Err(error);
    }
    parse_as_tensor_device(function, device.as_ref())?;

    if !matches!(
        memory_format,
        MemoryFormat::Preserve | MemoryFormat::Contiguous
    ) {
        return Err(PyNotImplementedError::new_err(format!(
            "{function}(): only default-equivalent memory_format is supported"
        )));
    }
    if !torch_function_mode_stack::is_empty() {
        return Err(PyNotImplementedError::new_err(format!(
            "{function}(): __torch_function__ modes are not supported"
        )));
    }
    Ok((input.clone(), requires_grad))
}

fn parse_full_like_arguments(
    arguments: FullLikeCallArguments<'_>,
) -> PyResult<(Bound<'_, PyTensor>, ParsedFillValue, bool)> {
    let FullLikeCallArguments {
        input,
        fill_value,
        dtype,
        layout,
        device,
        requires_grad,
        memory_format,
        keyword_error,
    } = arguments;

    let Some(input) = input else {
        if fill_value.is_none() {
            return Err(PyTypeError::new_err(
                "full_like() missing 2 required positional arguments: \"input\", \"fill_value\"",
            ));
        }
        return Err(PyTypeError::new_err(
            "full_like() missing 1 required positional arguments: \"input\"",
        ));
    };
    let Some(fill_value) = fill_value else {
        return Err(PyTypeError::new_err(
            "full_like() missing 1 required positional arguments: \"fill_value\"",
        ));
    };

    let input = parse_exact_native_like_factory_tensor_argument("full_like", "input", &input)?;
    let fill_value = parse_fill_value("full_like", &fill_value)?;
    let memory_format = parse_like_factory_memory_format("full_like", memory_format.as_ref())?;
    parse_identity_dtype("full_like", dtype.as_ref())?;
    parse_factory_layout("full_like", layout.as_ref())?;
    validate_as_tensor_device_type("full_like", device.as_ref())?;
    let requires_grad = parse_factory_requires_grad("full_like", requires_grad.as_ref())?;
    if let Some(error) = keyword_error {
        return Err(error);
    }
    parse_as_tensor_device("full_like", device.as_ref())?;

    if !matches!(
        memory_format,
        MemoryFormat::Preserve | MemoryFormat::Contiguous
    ) {
        return Err(PyNotImplementedError::new_err(
            "full_like(): only default-equivalent memory_format is supported",
        ));
    }
    if !torch_function_mode_stack::is_empty() {
        return Err(PyNotImplementedError::new_err(
            "full_like(): __torch_function__ modes are not supported",
        ));
    }
    Ok((input.clone(), fill_value, requires_grad))
}

fn parse_like_factory_memory_format(
    function: &str,
    memory_format: Option<&Bound<'_, PyAny>>,
) -> PyResult<MemoryFormat> {
    let Some(memory_format) = memory_format else {
        return Ok(MemoryFormat::Preserve);
    };
    if let Ok(memory_format) = memory_format.cast::<PyMemoryFormat>() {
        return Ok(memory_format.try_borrow()?.inner());
    }

    let actual = python_type_name(memory_format)?;
    Err(PyTypeError::new_err(format!(
        "{function}(): argument 'memory_format' must be torch.memory_format, not {actual}"
    )))
}

fn bind_full_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<FullCallArguments<'py>> {
    // PyTorch rejects excess positional arguments before inspecting keywords.
    if positional.len() > 2 {
        return Err(PyTypeError::new_err(format!(
            "full() takes 2 positional arguments but {} were given",
            positional.len()
        )));
    }

    let mut arguments = FullCallArguments {
        size: if positional.is_empty() {
            None
        } else {
            Some(positional.get_item(0)?)
        },
        fill_value: if positional.len() < 2 {
            None
        } else {
            Some(positional.get_item(1)?)
        },
        out: None,
        dtype: None,
        layout: None,
        device: None,
        pin_memory: None,
        requires_grad: None,
        keyword_error: None,
    };
    let Some(keywords) = keywords else {
        return Ok(arguments);
    };

    for (key, value) in keywords {
        let key = key.extract::<String>()?;
        match key.as_str() {
            "size" => {
                if arguments.size.is_some() {
                    arguments.keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err("full() got multiple values for argument 'size'")
                    });
                } else {
                    arguments.size = Some(value);
                }
            }
            "fill_value" => {
                if arguments.fill_value.is_some() {
                    arguments.keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err("full() got multiple values for argument 'fill_value'")
                    });
                } else {
                    arguments.fill_value = Some(value);
                }
            }
            "out" => arguments.out = optional_call_argument(value),
            "dtype" => arguments.dtype = optional_call_argument(value),
            "layout" => arguments.layout = optional_call_argument(value),
            "device" => arguments.device = optional_call_argument(value),
            "pin_memory" => arguments.pin_memory = optional_call_argument(value),
            "requires_grad" => arguments.requires_grad = optional_call_argument(value),
            _ => {
                arguments.keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "full() got an unexpected keyword argument '{key}'"
                    ))
                });
            }
        }
    }
    Ok(arguments)
}

fn parse_full_arguments(arguments: FullCallArguments<'_>) -> PyResult<ParsedFullArguments> {
    let FullCallArguments {
        size,
        fill_value,
        out,
        dtype,
        layout,
        device,
        pin_memory,
        requires_grad,
        keyword_error,
    } = arguments;

    // PyTorch reports required arguments before validating any supplied
    // optional arguments. It then validates declared types in signature
    // order, reports duplicate or unknown keywords, and finally resolves a
    // syntactically valid device specification.
    let Some(size) = size else {
        return Err(PyTypeError::new_err(
            "full() missing 2 required positional argument: \"size\", \"fill_value\"",
        ));
    };
    let Some(fill_value) = fill_value else {
        return Err(PyTypeError::new_err(
            "full() missing 1 required positional arguments: \"fill_value\"",
        ));
    };

    let size = parse_size(&size)?;
    let fill_value = parse_fill_value("full", &fill_value)?;
    let has_out = validate_creation_out("full", out.as_ref())?;
    let dtype = parse_dtype("full", dtype.as_ref())?;
    parse_factory_layout("full", layout.as_ref())?;
    validate_device_argument_type("full", device.as_ref())?;
    let pin_memory = parse_factory_bool("full", "pin_memory", pin_memory.as_ref())?;
    let requires_grad = parse_factory_requires_grad("full", requires_grad.as_ref())?;
    if let Some(error) = keyword_error {
        return Err(error);
    }
    let device = parse_device("full", device.as_ref())?;
    Ok(ParsedFullArguments {
        size,
        fill_value,
        has_out,
        dtype,
        device,
        pin_memory,
        requires_grad,
    })
}

fn parse_creation_size<'py>(
    function: &str,
    size: Option<&CreationSizeArgument<'py>>,
    shape: Option<&Bound<'py, PyAny>>,
) -> PyResult<PendingCreationSize<'py>> {
    let (value, origin) = match (size, shape) {
        (Some(_), Some(_)) => {
            return Err(PyTypeError::new_err(format!(
                "{function}() received both 'size' and its compatibility alias 'shape'"
            )));
        }
        (Some(CreationSizeArgument::Variadic(dimensions)), None) => {
            return Ok(PendingCreationSize::Variadic(dimensions.clone()));
        }
        (Some(CreationSizeArgument::Single { value, origin }), None) => (value, *origin),
        (None, Some(value)) => (value, CreationSizeOrigin::ShapeKeyword),
        (None, None) => {
            return Err(PyTypeError::new_err(format!(
                "{function}() missing 1 required positional arguments: \"size\""
            )));
        }
    };

    let sequence_error = match value.extract::<Vec<usize>>() {
        Ok(dimensions) => return Ok(PendingCreationSize::Dimensions(dimensions)),
        Err(error) => error,
    };
    if !matches!(function, "zeros" | "ones") || origin != CreationSizeOrigin::Positional {
        return Err(sequence_error);
    }

    bind_creation_positional_dimension(function, value, sequence_error)
}

fn bind_creation_positional_dimension<'py>(
    function: &str,
    dimension: &Bound<'py, PyAny>,
    sequence_error: PyErr,
) -> PyResult<PendingCreationSize<'py>> {
    if dimension.is_instance_of::<PyBool>() {
        return Err(creation_dimension_type_error(function, dimension)?);
    }

    let indexed = if dimension.is_instance_of::<PyInt>() {
        dimension.clone()
    } else {
        let indexed = PyModule::import(dimension.py(), "operator")
            .and_then(|operator| operator.getattr("index"))
            .and_then(|index| index.call1((dimension,)));
        let Ok(indexed) = indexed else {
            if dimension.cast::<PySequence>().is_ok() {
                return Err(sequence_error);
            }
            return Err(creation_dimension_type_error(function, dimension)?);
        };
        indexed
    };
    Ok(PendingCreationSize::PositionalScalar(indexed))
}

fn finish_creation_size(
    function: &str,
    size: PendingCreationSize<'_>,
) -> PyResult<ParsedCreationSize> {
    let dimension = match size {
        PendingCreationSize::Dimensions(dimensions) => {
            return Ok(ParsedCreationSize {
                dimensions,
                scalar_dimension: None,
            });
        }
        PendingCreationSize::Variadic(dimensions) => {
            let dimensions = parse_variadic_creation_dimensions(function, &dimensions)?;
            return Ok(ParsedCreationSize {
                dimensions,
                scalar_dimension: None,
            });
        }
        PendingCreationSize::PositionalScalar(dimension) => dimension,
    };
    let dimension = extract_creation_dimension(function, &dimension)?;
    if dimension < 0 {
        return Err(creation_negative_dimension_error(
            function,
            dimension,
            &[dimension],
        ));
    }
    let dimension =
        usize::try_from(dimension).map_err(|_| creation_dimension_overflow(function))?;
    let mut dimensions = try_size_vector(1)?;
    try_push_size(&mut dimensions, dimension)?;
    Ok(ParsedCreationSize {
        dimensions,
        scalar_dimension: Some(dimension),
    })
}

fn parse_variadic_creation_dimensions(
    function: &str,
    dimensions: &Bound<'_, PyTuple>,
) -> PyResult<Vec<usize>> {
    let mut parsed = try_size_vector(dimensions.len())?;
    let mut signed = try_size_vector(dimensions.len())?;
    for (index, dimension) in dimensions.iter().enumerate() {
        let position = index + 1;
        let dimension = extract_variadic_creation_dimension(function, position, &dimension)?;
        try_push_size(&mut signed, dimension)?;
    }
    if let Some(dimension) = signed.iter().find(|dimension| **dimension < 0) {
        return Err(creation_negative_dimension_error(
            function, *dimension, &signed,
        ));
    }
    for dimension in signed {
        try_push_size(
            &mut parsed,
            usize::try_from(dimension).map_err(|_| creation_dimension_overflow(function))?,
        )?;
    }
    Ok(parsed)
}

fn extract_creation_dimension(function: &str, dimension: &Bound<'_, PyAny>) -> PyResult<i64> {
    dimension
        .extract::<i64>()
        .map_err(|_| creation_dimension_overflow(function))
}

fn extract_variadic_creation_dimension(
    function: &str,
    position: usize,
    dimension: &Bound<'_, PyAny>,
) -> PyResult<i64> {
    let indexed = if dimension.is_instance_of::<PyInt>() {
        dimension.clone()
    } else {
        let indexed = PyModule::import(dimension.py(), "operator")
            .and_then(|operator| operator.getattr("index"))
            .and_then(|index| index.call1((dimension,)));
        let Ok(indexed) = indexed else {
            return Err(creation_dimension_unpack_type_error(
                function, position, dimension,
            )?);
        };
        indexed
    };
    indexed
        .extract::<i64>()
        .map_err(|_| creation_dimension_overflow_at(function, position))
}

fn creation_dimension_type_error(function: &str, dimension: &Bound<'_, PyAny>) -> PyResult<PyErr> {
    let type_name = python_type_name(dimension)?;
    Ok(PyTypeError::new_err(format!(
        "{function}(): argument 'size' (position 1) must be tuple of ints, not {type_name}"
    )))
}

fn creation_sequence_dimension_type_error(
    function: &str,
    dimension: &Bound<'_, PyAny>,
) -> PyResult<PyErr> {
    let type_name = python_type_name(dimension)?;
    Ok(PyTypeError::new_err(format!(
        "{function}(): argument 'size' (position 1) must be tuple of ints, but found element of type {type_name} at pos 0"
    )))
}

fn creation_dimension_overflow(function: &str) -> PyErr {
    creation_dimension_overflow_at(function, 1)
}

fn creation_dimension_overflow_at(function: &str, position: usize) -> PyErr {
    PyTypeError::new_err(format!(
        "{function}(): argument 'size' failed to unpack the object at pos {position} with error \"Overflow when unpacking long long\""
    ))
}

fn creation_dimension_unpack_type_error(
    function: &str,
    position: usize,
    dimension: &Bound<'_, PyAny>,
) -> PyResult<PyErr> {
    let actual = python_type_name(dimension)?;
    Ok(PyTypeError::new_err(format!(
        "{function}(): argument 'size' failed to unpack the object at pos {position} with error \"type must be tuple of ints,but got {actual}\""
    )))
}

fn creation_negative_dimension_error(function: &str, dimension: i64, shape: &[i64]) -> PyErr {
    if function == "zeros" {
        PyRuntimeError::new_err("zeros: Dimension size must be non-negative.")
    } else {
        debug_assert_eq!(function, "ones");
        PyRuntimeError::new_err(format!(
            "Trying to create tensor with negative dimension {dimension}: {shape:?}"
        ))
    }
}

fn parse_metadata(
    function: &str,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<(DType, Device)> {
    Ok((
        parse_dtype(function, dtype)?,
        parse_device(function, device)?,
    ))
}

fn parse_dtype(function: &str, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<DType> {
    let Some(dtype) = dtype else {
        return Ok(DType::Float32);
    };
    if let Ok(dtype) = dtype.cast::<PyDType>() {
        return Ok(dtype.try_borrow()?.inner());
    }

    let type_name = dtype.get_type().name()?;
    Err(PyTypeError::new_err(format!(
        "{function}(): argument 'dtype' must be torch.dtype, not {type_name}"
    )))
}

fn parse_device(function: &str, device: Option<&Bound<'_, PyAny>>) -> PyResult<Device> {
    device.map_or(Ok(Device::Cpu), |device| {
        parse_device_value(function, device)
    })
}

fn validate_device_argument_type(
    function: &str,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let Some(device) = device else {
        return Ok(());
    };
    if device.cast::<PyDevice>().is_ok() || device.cast::<PyString>().is_ok() {
        return Ok(());
    }
    let error = device_argument_type_error(function, device)?;
    Err(error)
}

fn parse_eye_dimension(argument: &str, dimension: &Bound<'_, PyAny>) -> PyResult<i64> {
    if dimension.is_instance_of::<PyBool>() {
        return Err(eye_dimension_type_error(argument, "bool"));
    }

    if dimension.is_instance_of::<PyInt>() {
        return dimension
            .extract::<i64>()
            .map_err(|_| eye_dimension_overflow());
    }

    let type_name = dimension.get_type().name()?.to_str()?.to_owned();
    let indexed = PyModule::import(dimension.py(), "operator")
        .and_then(|operator| operator.getattr("index"))
        .and_then(|index| index.call1((dimension,)))
        .map_err(|_| eye_dimension_type_error(argument, &type_name))?;
    indexed
        .extract::<i64>()
        .map_err(|_| eye_dimension_overflow())
}

fn eye_dimension_type_error(argument: &str, type_name: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "eye(): argument '{argument}' must be int, not {type_name}"
    ))
}

fn eye_dimension_overflow() -> PyErr {
    PyValueError::new_err("Overflow when unpacking long long")
}

fn validate_eye_dimension(argument: &str, dimension: i64) -> PyResult<usize> {
    if dimension < 0 {
        return Err(PyRuntimeError::new_err(format!(
            "{argument} must be greater or equal to 0, got {dimension}"
        )));
    }
    usize::try_from(dimension).map_err(|_| {
        PyRuntimeError::new_err(format!(
            "eye(): argument '{argument}' exceeds the platform size limit"
        ))
    })
}

fn parse_size(size: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    if let Ok(size) = size.cast::<PyList>() {
        parse_size_dimensions(size.len(), size.iter())
    } else if let Ok(size) = size.cast::<PyTuple>() {
        parse_size_dimensions(size.len(), size.iter())
    } else {
        Err(PyTypeError::new_err(
            "full(): argument 'size' must be a tuple or list of integers",
        ))
    }
}

fn parse_stride_dimension(dimension: &Bound<'_, PyAny>) -> PyResult<i64> {
    if !dimension.is_instance_of::<PyBool>() && dimension.is_instance_of::<PyInt>() {
        return dimension
            .extract::<i64>()
            .map_err(|_| PyValueError::new_err("Overflow when unpacking long long"));
    }

    if let Ok(numpy) = PyModule::import(dimension.py(), "numpy") {
        let numpy_integer = numpy.getattr("integer")?;
        if dimension.is_instance(&numpy_integer)? {
            return dimension
                .extract::<i64>()
                .map_err(|_| PyValueError::new_err("Overflow when unpacking long long"));
        }
    }

    let type_name = dimension.get_type().name()?;
    Err(PyTypeError::new_err(format!(
        "stride(): argument 'dim' must be int, not {type_name}"
    )))
}

fn bind_unbind_dimension<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<Option<ParsedCallArgument<'py>>> {
    if positional.len() > 1 {
        return Err(PyTypeError::new_err(format!(
            "unbind() takes from 0 to 1 positional arguments but {} were given",
            positional.len()
        )));
    }

    let dimension = if positional.is_empty() {
        if let Some(keywords) = keywords {
            keywords.get_item("dim")?.map(|value| ParsedCallArgument {
                value,
                position: None,
            })
        } else {
            None
        }
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(0)?,
            position: Some(1),
        })
    };

    // PyTorch validates the recognized argument type before diagnosing extra
    // or duplicate keywords. Integer conversion remains deferred until after
    // TorchFunctionMode dispatch so a mode can observe the original object.
    if let Some(dimension) = &dimension {
        validate_dimension_swap_dimension("unbind", "dim", dimension.position, &dimension.value)?;
    }

    if let Some(keywords) = keywords {
        for key in keywords.keys() {
            let key = key.extract::<String>()?;
            if key != "dim" {
                return Err(PyTypeError::new_err(format!(
                    "unbind() got an unexpected keyword argument '{key}'"
                )));
            }
            if !positional.is_empty() {
                return Err(PyTypeError::new_err(
                    "unbind() got multiple values for argument 'dim'",
                ));
            }
        }
    }

    Ok(dimension)
}

fn bind_top_level_unbind_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<(
    BoundTensorOrTorchFunction<'py>,
    Option<ParsedCallArgument<'py>>,
)> {
    if positional.len() > 2 {
        return Err(PyTypeError::new_err(format!(
            "unbind() takes from 1 to 2 positional arguments but {} were given",
            positional.len()
        )));
    }

    let keyword_argument = |names: &[&str]| -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(keywords) = keywords else {
            return Ok(None);
        };
        for name in names {
            if let Some(value) = keywords.get_item(*name)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    };

    let input = if positional.is_empty() {
        keyword_argument(&["input", "x", "a", "x1"])?.map(|value| ParsedCallArgument {
            value,
            position: None,
        })
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(0)?,
            position: Some(1),
        })
    };
    let dimension = if positional.len() < 2 {
        keyword_argument(&["dim"])?.map(|value| ParsedCallArgument {
            value,
            position: None,
        })
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(1)?,
            position: Some(2),
        })
    };

    let Some(input) = input else {
        return Err(PyTypeError::new_err(
            "unbind() missing 1 required positional arguments: \"input\"",
        ));
    };
    let bound_input = parse_tensor_or_torch_function_argument("unbind", "input", &input)?;
    if let Some(dimension) = &dimension {
        validate_dimension_swap_dimension("unbind", "dim", dimension.position, &dimension.value)?;
    }

    if let Some(keywords) = keywords {
        let bound_keyword_count = usize::from(input.position.is_none())
            + usize::from(
                dimension
                    .as_ref()
                    .is_some_and(|dimension| dimension.position.is_none()),
            );
        if keywords.len() > bound_keyword_count {
            for key in keywords.keys() {
                let key = key.extract::<String>()?;
                let position = match key.as_str() {
                    "input" => 0,
                    "dim" => 1,
                    _ => {
                        return Err(PyTypeError::new_err(format!(
                            "unbind() got an unexpected keyword argument '{key}'"
                        )));
                    }
                };
                if position < positional.len() {
                    return Err(PyTypeError::new_err(format!(
                        "unbind() got multiple values for argument '{key}'"
                    )));
                }
            }
        }
    }

    Ok((bound_input, dimension))
}

fn bind_top_level_unsqueeze_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<([ParsedCallArgument<'py>; 2], Option<PyErr>)> {
    const NAMES: [&str; 2] = ["input", "dim"];

    if positional.len() > NAMES.len() {
        return Err(PyTypeError::new_err(format!(
            "unsqueeze() takes 2 positional arguments but {} were given",
            positional.len()
        )));
    }

    let keyword_argument = |names: &[&str]| -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(keywords) = keywords else {
            return Ok(None);
        };
        for name in names {
            if let Some(value) = keywords.get_item(*name)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    };

    let input = if positional.is_empty() {
        keyword_argument(&["input", "x", "a", "x1"])?.map(|value| ParsedCallArgument {
            value,
            position: None,
        })
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(0)?,
            position: Some(1),
        })
    };
    let dimension = if positional.len() < 2 {
        keyword_argument(&["dim"])?.map(|value| ParsedCallArgument {
            value,
            position: None,
        })
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(1)?,
            position: Some(2),
        })
    };

    let Some(input) = input else {
        return Err(PyTypeError::new_err(
            "unsqueeze() missing 2 required positional argument: \"input\", \"dim\"",
        ));
    };
    let Some(dimension) = dimension else {
        parse_tensor_argument("unsqueeze", "input", &input)?;
        return Err(PyTypeError::new_err(
            "unsqueeze() missing 1 required positional arguments: \"dim\"",
        ));
    };

    let mut keyword_error = None;
    if let Some(keywords) = keywords {
        let keyword_arguments =
            usize::from(input.position.is_none()) + usize::from(dimension.position.is_none());
        if keywords.len() > keyword_arguments {
            for key in keywords.keys() {
                let key = key.extract::<String>()?;
                let position = match key.as_str() {
                    "input" => 0,
                    "dim" => 1,
                    _ => {
                        keyword_error.get_or_insert_with(|| {
                            PyTypeError::new_err(format!(
                                "unsqueeze() got an unexpected keyword argument '{key}'"
                            ))
                        });
                        break;
                    }
                };
                if position < positional.len() {
                    keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err(format!(
                            "unsqueeze() got multiple values for argument '{key}'"
                        ))
                    });
                    break;
                }
            }
        }
    }

    Ok(([input, dimension], keyword_error))
}

fn parse_exact_native_unsqueeze_tensor_argument<'a, 'py>(
    argument: &str,
    value: &'a ParsedCallArgument<'py>,
) -> PyResult<&'a Bound<'py, PyTensor>> {
    if !value.value.is_exact_instance_of::<PyTensor>() {
        if value.value.is_instance_of::<PyTensor>() {
            return Err(unsqueeze_unsupported_native_input());
        }
        return parse_tensor_argument("unsqueeze", argument, value);
    }
    Ok(value.value.cast::<PyTensor>()?)
}

fn bind_select_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<([ParsedCallArgument<'py>; 2], Option<PyErr>)> {
    const NAMES: [&str; 2] = ["dim", "index"];

    if positional.len() > NAMES.len() {
        return Err(PyTypeError::new_err(format!(
            "select() takes 2 positional arguments but {} were given",
            positional.len()
        )));
    }

    let mut arguments: [Option<ParsedCallArgument<'py>>; 2] = std::array::from_fn(|_| None);
    for (argument_index, value) in positional.iter().enumerate() {
        arguments[argument_index] = Some(ParsedCallArgument {
            value,
            position: Some(argument_index + 1),
        });
    }

    let mut keyword_error = None;
    if let Some(keywords) = keywords {
        for (key, value) in keywords {
            let key = key.extract::<String>()?;
            let Some(argument_index) = NAMES.iter().position(|name| *name == key) else {
                keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "select() got an unexpected keyword argument '{key}'"
                    ))
                });
                continue;
            };
            if arguments[argument_index].is_some() {
                keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "select() got multiple values for argument '{}'",
                        NAMES[argument_index]
                    ))
                });
                continue;
            }
            arguments[argument_index] = Some(ParsedCallArgument {
                value,
                position: None,
            });
        }
    }

    if let Some(first_missing) = arguments.iter().position(Option::is_none) {
        validate_select_argument_prefix(&arguments, first_missing)?;
        return Err(select_missing_arguments_error(&NAMES[first_missing..]));
    }

    validate_select_argument_prefix(&arguments, NAMES.len())?;
    Ok((
        arguments.map(|argument| argument.expect("all required select arguments were bound")),
        keyword_error,
    ))
}

fn bind_top_level_select_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<(
    BoundTensorOrTorchFunction<'py>,
    [ParsedCallArgument<'py>; 2],
    Option<PyErr>,
)> {
    const NAMES: [&str; 3] = ["input", "dim", "index"];

    if positional.len() > NAMES.len() {
        return Err(PyTypeError::new_err(format!(
            "select() takes 3 positional arguments but {} were given",
            positional.len()
        )));
    }

    let mut arguments: [Option<ParsedCallArgument<'py>>; 3] = std::array::from_fn(|_| None);
    for (argument_index, value) in positional.iter().enumerate() {
        arguments[argument_index] = Some(ParsedCallArgument {
            value,
            position: Some(argument_index + 1),
        });
    }

    if let Some(keywords) = keywords {
        if arguments[0].is_none() {
            for name in ["input", "x", "a", "x1"] {
                if let Some(value) = keywords.get_item(name)? {
                    arguments[0] = Some(ParsedCallArgument {
                        value,
                        position: None,
                    });
                    break;
                }
            }
        }
        for (argument_index, name) in NAMES.iter().enumerate().skip(1) {
            if arguments[argument_index].is_none()
                && let Some(value) = keywords.get_item(*name)?
            {
                arguments[argument_index] = Some(ParsedCallArgument {
                    value,
                    position: None,
                });
            }
        }
    }

    if let Some(first_missing) = arguments.iter().position(Option::is_none) {
        if first_missing >= 1 {
            let input = arguments[0]
                .as_ref()
                .expect("the select input preceding a binding gap is present");
            parse_tensor_or_torch_function_argument("select", "input", input)?;
        }
        if first_missing >= 2 {
            let dimension = arguments[1]
                .as_ref()
                .expect("the select dimension preceding a binding gap is present");
            validate_dimension_swap_dimension(
                "select",
                "dim",
                dimension.position,
                &dimension.value,
            )?;
        }

        return Err(select_missing_arguments_error(&NAMES[first_missing..]));
    }

    let [input, dimension, index] =
        arguments.map(|argument| argument.expect("all required select arguments were bound"));
    let input_was_keyword = input.position.is_none();
    let input = parse_tensor_or_torch_function_argument("select", "input", &input)?;
    validate_dimension_swap_dimension("select", "dim", dimension.position, &dimension.value)?;
    validate_select_index(&index)?;

    let mut keyword_error = None;
    if let Some(keywords) = keywords {
        let bound_keyword_count = usize::from(input_was_keyword)
            + usize::from(dimension.position.is_none())
            + usize::from(index.position.is_none());
        if keywords.len() > bound_keyword_count {
            for key in keywords.keys() {
                let key = key.extract::<String>()?;
                let Some(position) = NAMES.iter().position(|name| *name == key) else {
                    keyword_error = Some(PyTypeError::new_err(format!(
                        "select() got an unexpected keyword argument '{key}'"
                    )));
                    break;
                };
                if position < positional.len() {
                    keyword_error = Some(PyTypeError::new_err(format!(
                        "select() got multiple values for argument '{key}'"
                    )));
                    break;
                }
            }
        }
    }

    Ok((input, [dimension, index], keyword_error))
}

fn select_missing_arguments_error(missing: &[&str]) -> PyErr {
    let quoted_names = missing
        .iter()
        .map(|name| format!("\"{name}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let argument = if missing.len() == 1 {
        "arguments"
    } else {
        "argument"
    };
    PyTypeError::new_err(format!(
        "select() missing {} required positional {argument}: {quoted_names}",
        missing.len()
    ))
}

fn validate_select_argument_prefix(
    arguments: &[Option<ParsedCallArgument<'_>>; 2],
    length: usize,
) -> PyResult<()> {
    if length >= 1 {
        let dimension = arguments[0]
            .as_ref()
            .expect("the select dimension preceding a binding gap is present");
        validate_dimension_swap_dimension("select", "dim", dimension.position, &dimension.value)?;
    }
    if length >= 2 {
        let index = arguments[1]
            .as_ref()
            .expect("the select index preceding a binding gap is present");
        validate_select_index(index)?;
    }
    Ok(())
}

fn validate_select_index(index: &ParsedCallArgument<'_>) -> PyResult<()> {
    if is_dimension_swap_integer(&index.value)? || probe_select_index(&index.value) {
        return Ok(());
    }

    let actual = python_type_name(&index.value)?;
    Err(dimension_swap_argument_type_error(
        "select",
        "index",
        index.position,
        "int",
        &actual,
    ))
}

fn probe_select_index(index: &Bound<'_, PyAny>) -> bool {
    if index.is_instance_of::<PyBool>() {
        return false;
    }
    call_python_index(index).is_ok()
}

fn call_python_index<'py>(index: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    PyModule::import(index.py(), "operator")?
        .getattr("index")?
        .call1((index,))
}

fn extract_select_index(index: &Bound<'_, PyAny>) -> PyResult<i64> {
    if is_dimension_swap_integer(index)? {
        return extract_dimension_swap_dimension(index);
    }

    // SymInt conversion probes an arbitrary __index__ provider once more
    // before obtaining the concrete value. The first result is intentionally
    // ignored, so stateful providers observe the same three calls as PyTorch:
    // binding validation, conversion validation, and extraction.
    if call_python_index(index).is_err() {
        let index_type = index.get_type().repr()?.to_str()?.to_owned();
        return Err(PyRuntimeError::new_err(format!(
            "Unable to cast Python instance of type {index_type} to C++ type '?' (#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)"
        )));
    }
    let concrete = call_python_index(index)?;
    extract_dimension_swap_dimension(&concrete)
}

pub(crate) fn bind_size_dimension<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<Option<ParsedCallArgument<'py>>> {
    if positional.len() > 1 {
        return Err(PyTypeError::new_err(format!(
            "size() takes from 0 to 1 positional arguments but {} were given",
            positional.len()
        )));
    }

    let dimension = if positional.is_empty() {
        if let Some(keywords) = keywords {
            keywords.get_item("dim")?.map(|value| ParsedCallArgument {
                value,
                position: None,
            })
        } else {
            None
        }
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(0)?,
            position: Some(1),
        })
    };

    // PyTorch validates the recognized argument type before diagnosing extra
    // keywords, but treats None as the optional no-argument overload only
    // after binding has otherwise completed.
    if let Some(dimension) = &dimension
        && !dimension.value.is_none()
    {
        validate_size_dimension(dimension)?;
    }

    if let Some(keywords) = keywords {
        let bound_keyword_count = usize::from(positional.is_empty() && dimension.is_some());
        if keywords.len() > bound_keyword_count {
            for key in keywords.keys() {
                let key = key.extract::<String>()?;
                if key != "dim" {
                    return Err(PyTypeError::new_err(format!(
                        "size() got an unexpected keyword argument '{key}'"
                    )));
                }
                if !positional.is_empty() {
                    return Err(PyTypeError::new_err(
                        "size() got multiple values for argument 'dim'",
                    ));
                }
            }
        }
    }

    match dimension {
        Some(dimension) if !dimension.value.is_none() => Ok(Some(dimension)),
        _ => Ok(None),
    }
}

fn validate_size_dimension(dimension: &ParsedCallArgument<'_>) -> PyResult<()> {
    if is_dimension_swap_integer(&dimension.value)? {
        return Ok(());
    }

    let actual = python_type_name(&dimension.value)?;
    Err(size_dimension_type_error(dimension, &actual))
}

fn size_dimension_type_error(dimension: &ParsedCallArgument<'_>, actual: &str) -> PyErr {
    let position = dimension
        .position
        .map_or_else(String::new, |position| format!(" (position {position})"));
    PyTypeError::new_err(format!(
        "size(): argument 'dim'{position} must be int, not {actual}"
    ))
}

fn is_dimension_swap_integer(dimension: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !dimension.is_instance_of::<PyBool>() && dimension.is_instance_of::<PyInt>() {
        return Ok(true);
    }

    if let Ok(numpy) = PyModule::import(dimension.py(), "numpy") {
        let numpy_integer = numpy.getattr("integer")?.cast_into::<PyType>()?;
        if dimension.get_type().is_subclass(numpy_integer.as_any())? {
            return Ok(true);
        }
    }

    Ok(false)
}

fn validate_dimension_swap_dimension(
    operation: &str,
    argument: &str,
    position: Option<usize>,
    dimension: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if is_dimension_swap_integer(dimension)? {
        return Ok(());
    }

    let type_name = python_type_name(dimension)?;
    Err(dimension_swap_argument_type_error(
        operation, argument, position, "int", &type_name,
    ))
}

fn parse_dimension_swap_dimensions(
    operation: &str,
    argument_names: [&str; 2],
    dim0: &ParsedCallArgument<'_>,
    dim1: &ParsedCallArgument<'_>,
) -> PyResult<[i64; 2]> {
    // PyTorch validates the declared types in signature order before it
    // converts either integer. This lets a later type mismatch take
    // precedence over an earlier integer that overflows during conversion.
    validate_dimension_swap_dimension(operation, argument_names[0], dim0.position, &dim0.value)?;
    validate_dimension_swap_dimension(operation, argument_names[1], dim1.position, &dim1.value)?;
    // TensorOptions-style generated bindings convert dimensions in reverse
    // declaration order after type checking. Keep the values in declaration
    // order for the transpose engine after reproducing that observable order.
    let dim1 = extract_dimension_swap_dimension(&dim1.value)?;
    let dim0 = extract_dimension_swap_dimension(&dim0.value)?;
    Ok([dim0, dim1])
}

pub(crate) fn extract_dimension_swap_dimension(dimension: &Bound<'_, PyAny>) -> PyResult<i64> {
    dimension.extract::<i64>().map_err(|error| {
        let py = dimension.py();
        // PyLong_AsLongLong reports a traceback-free range error with this
        // CPython message. An accepted integer object's __index__ can raise
        // its own OverflowError; preserve that exception and traceback.
        let message = error.value(py).to_string();
        let is_range_overflow = error.is_instance_of::<PyOverflowError>(py)
            && error.traceback(py).is_none()
            && matches!(
                message.as_str(),
                "int too big to convert" | "Python int too large to convert to C long"
            );
        if is_range_overflow {
            PyValueError::new_err("Overflow when unpacking long long")
        } else {
            error
        }
    })
}

fn parse_flatten_dimension(
    argument: &str,
    position: Option<usize>,
    dimension: &Bound<'_, PyAny>,
) -> PyResult<i64> {
    if !dimension.is_instance_of::<PyBool>() && dimension.is_instance_of::<PyInt>() {
        return dimension
            .extract::<i64>()
            .map_err(|_| PyValueError::new_err("Overflow when unpacking long long"));
    }

    if let Ok(numpy) = PyModule::import(dimension.py(), "numpy") {
        let numpy_integer = numpy.getattr("integer")?;
        if dimension.is_instance(&numpy_integer)? {
            return dimension
                .extract::<i64>()
                .map_err(|_| PyValueError::new_err("Overflow when unpacking long long"));
        }
    }

    let actual = python_type_name(dimension)?;
    let position = position.map_or_else(String::new, |position| format!(" (position {position})"));
    Err(PyTypeError::new_err(format!(
        "flatten(): argument '{argument}'{position} must be int, not {actual}"
    )))
}

fn bind_top_level_sum_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundTopLevelSumCall<'py>> {
    if positional.len() > 3 {
        return Err(PyTypeError::new_err(format!(
            "sum() takes from 2 to 3 positional arguments but {} were given",
            positional.len()
        )));
    }

    let keyword_input = top_level_sum_keyword_input(keywords)?;
    let input_argument = top_level_sum_input_argument(positional, keyword_input.as_ref())?;
    let Some(input_argument) = input_argument else {
        if positional.is_empty() && keywords.is_none_or(PyDictMethods::is_empty) {
            return Err(top_level_sum_invalid_combination(positional, keywords)?);
        }
        return Err(PyTypeError::new_err(
            "sum() missing 1 required positional arguments: \"input\"",
        ));
    };

    let keyword_dim = top_level_sum_keyword(keywords, "dim")?;
    let keyword_keepdim = top_level_sum_keyword(keywords, "keepdim")?;
    let keyword_dtype = top_level_sum_keyword(keywords, "dtype")?;
    let keyword_out = top_level_sum_keyword(keywords, "out")?;
    let keyword_input_duplicated = keyword_input.as_ref().is_some_and(|input| input.duplicated);
    let input_argument_duplicated = positional.len() >= 1 && keyword_input.is_some()
        || positional.is_empty() && keyword_input_duplicated;
    let dim_argument_duplicated = positional.len() >= 2 && keyword_dim.is_some();
    let keepdim_argument_duplicated = positional.len() >= 3 && keyword_keepdim.is_some();

    if top_level_sum_has_unexpected_keyword(keywords)?
        || input_argument_duplicated
        || dim_argument_duplicated
        || keepdim_argument_duplicated
    {
        return Err(top_level_sum_invalid_combination(positional, keywords)?);
    }

    let has_dimension = positional.len() >= 2 || keyword_dim.is_some();
    let has_keepdim = positional.len() >= 3 || keyword_keepdim.is_some();
    let has_out = keyword_out.is_some();
    if !has_dimension && (has_keepdim || has_out) {
        return Err(top_level_sum_invalid_combination(positional, keywords)?);
    }

    let input = parse_top_level_sum_input(
        &input_argument,
        has_dimension || has_keepdim || has_out || keyword_dtype.is_some(),
        positional,
        keywords,
    )?;

    let dimension = top_level_sum_dimension_argument(positional, keyword_dim)?;
    if let Some(dimension) = &dimension
        && !is_sum_dimension_argument(&dimension.value)?
    {
        return Err(top_level_sum_invalid_combination(positional, keywords)?);
    }

    let keepdim = top_level_sum_keepdim_argument(positional, keyword_keepdim)?;
    if let Some(keepdim) = &keepdim
        && !keepdim.value.is_exact_instance_of::<PyBool>()
    {
        return Err(sum_argument_type_error(
            "keepdim",
            keepdim.position,
            "bool",
            &keepdim.value,
        )?);
    }

    let dtype = match keyword_dtype {
        Some(dtype) => bind_top_level_sum_dtype(&dtype.value, has_dimension, positional, keywords)?,
        None => BoundTopLevelSumDType::Native,
    };

    let out = parse_top_level_sum_out(keyword_out)?;
    let explicit_full_dimension = dimension
        .as_ref()
        .is_some_and(|dimension| dimension.value.is_none());
    let keepdim_is_true = keepdim
        .as_ref()
        .is_some_and(|keepdim| keepdim.value.extract::<bool>().is_ok_and(|keepdim| keepdim));
    let full_reduction = (!has_dimension || explicit_full_dimension) && out.is_none();
    let keepdim_full_reduction = explicit_full_dimension && keepdim_is_true && out.is_none();

    Ok(BoundTopLevelSumCall {
        input,
        dtype,
        out,
        full_reduction,
        keepdim_full_reduction,
    })
}

struct TopLevelSumKeywordInput<'py> {
    value: Bound<'py, PyAny>,
    duplicated: bool,
}

fn top_level_sum_input_argument<'py>(
    positional: &Bound<'py, PyTuple>,
    keyword_input: Option<&TopLevelSumKeywordInput<'py>>,
) -> PyResult<Option<ParsedCallArgument<'py>>> {
    if positional.is_empty() {
        return Ok(keyword_input.map(|input| ParsedCallArgument {
            value: input.value.clone(),
            position: None,
        }));
    }

    Ok(Some(ParsedCallArgument {
        value: positional.get_item(0)?,
        position: Some(1),
    }))
}

struct TopLevelSumKeyword<'py> {
    value: Bound<'py, PyAny>,
}

fn top_level_sum_keyword_input<'py>(
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<Option<TopLevelSumKeywordInput<'py>>> {
    let Some(keywords) = keywords else {
        return Ok(None);
    };

    let mut value = None;
    let mut seen = 0_usize;
    for name in ["input", "x", "a", "x1"] {
        if let Some(candidate) = keywords.get_item(name)? {
            seen += 1;
            if value.is_none() {
                value = Some(candidate);
            }
        }
    }
    Ok(value.map(|value| TopLevelSumKeywordInput {
        value,
        duplicated: seen > 1,
    }))
}

fn top_level_sum_keyword<'py>(
    keywords: Option<&Bound<'py, PyDict>>,
    name: &str,
) -> PyResult<Option<TopLevelSumKeyword<'py>>> {
    Ok(keywords
        .map(|keywords| keywords.get_item(name))
        .transpose()?
        .flatten()
        .map(|value| TopLevelSumKeyword { value }))
}

fn top_level_sum_has_unexpected_keyword(keywords: Option<&Bound<'_, PyDict>>) -> PyResult<bool> {
    let Some(keywords) = keywords else {
        return Ok(false);
    };
    for key in keywords.keys() {
        let key = key.extract::<String>()?;
        if !matches!(
            key.as_str(),
            "input" | "x" | "a" | "x1" | "dim" | "keepdim" | "dtype" | "out"
        ) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn parse_top_level_sum_input<'py>(
    input: &ParsedCallArgument<'py>,
    overload_mismatch_on_non_tensor: bool,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundTensorOrTorchFunction<'py>> {
    if let Ok(tensor) = input.value.cast::<PyTensor>() {
        return Ok(BoundTensorOrTorchFunction::Tensor(tensor.clone()));
    }
    if let Some(probed) = probe_torch_function_override(&input.value) {
        return Ok(BoundTensorOrTorchFunction::Override(probed));
    }
    if overload_mismatch_on_non_tensor {
        return Err(top_level_sum_invalid_combination(positional, keywords)?);
    }
    Err(legacy_single_tensor_type_error("sum", input)?)
}

fn bind_top_level_sum_dtype<'py>(
    dtype: &Bound<'py, PyAny>,
    has_dimension: bool,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundTopLevelSumDType<'py>> {
    if dtype.is_none() {
        return Ok(BoundTopLevelSumDType::Native);
    }
    if let Ok(dtype) = dtype.cast::<PyDType>()
        && dtype.try_borrow()?.inner() == DType::Float32
    {
        return Ok(BoundTopLevelSumDType::Native);
    }
    if let Some(probed) = probe_dtype_torch_function_override(dtype) {
        return Ok(BoundTopLevelSumDType::Override(probed));
    }
    if has_dimension {
        return Err(sum_argument_type_error(
            "dtype",
            None,
            "torch.dtype",
            dtype,
        )?);
    }
    Err(top_level_sum_invalid_combination(positional, keywords)?)
}

fn top_level_sum_dimension_argument<'py>(
    positional: &Bound<'py, PyTuple>,
    keyword_dim: Option<TopLevelSumKeyword<'py>>,
) -> PyResult<Option<ParsedCallArgument<'py>>> {
    if positional.len() >= 2 {
        return Ok(Some(ParsedCallArgument {
            value: positional.get_item(1)?,
            position: Some(2),
        }));
    }

    Ok(keyword_dim.map(|dimension| ParsedCallArgument {
        value: dimension.value,
        position: None,
    }))
}

fn top_level_sum_keepdim_argument<'py>(
    positional: &Bound<'py, PyTuple>,
    keyword_keepdim: Option<TopLevelSumKeyword<'py>>,
) -> PyResult<Option<ParsedCallArgument<'py>>> {
    if positional.len() >= 3 {
        return Ok(Some(ParsedCallArgument {
            value: positional.get_item(2)?,
            position: Some(3),
        }));
    }

    Ok(keyword_keepdim.map(|keepdim| ParsedCallArgument {
        value: keepdim.value,
        position: None,
    }))
}

fn parse_top_level_sum_out(
    keyword_out: Option<TopLevelSumKeyword<'_>>,
) -> PyResult<Option<BoundTensorOrTorchFunction<'_>>> {
    let Some(out) = keyword_out else {
        return Ok(None);
    };
    if out.value.is_none() {
        return Ok(None);
    }
    let out = ParsedCallArgument {
        value: out.value,
        position: None,
    };
    parse_tensor_or_torch_function_argument("sum", "out", &out).map(Some)
}

fn is_sum_dimension_argument(dimension: &Bound<'_, PyAny>) -> PyResult<bool> {
    if dimension.is_none() || is_dimension_swap_integer(dimension)? {
        return Ok(true);
    }
    if let Ok(dimensions) = dimension.cast::<PyTuple>() {
        return dimensions.iter().try_fold(true, |valid, item| {
            Ok(valid && is_dimension_swap_integer(&item)?)
        });
    }
    if let Ok(dimensions) = dimension.cast::<PyList>() {
        return dimensions.iter().try_fold(true, |valid, item| {
            Ok(valid && is_dimension_swap_integer(&item)?)
        });
    }
    Ok(false)
}

fn sum_argument_type_error(
    argument: &str,
    position: Option<usize>,
    expected: &str,
    actual: &Bound<'_, PyAny>,
) -> PyResult<PyErr> {
    let position = position.map_or_else(String::new, |position| format!(" (position {position})"));
    let actual = python_type_name(actual)?;
    Ok(PyTypeError::new_err(format!(
        "sum(): argument '{argument}'{position} must be {expected}, not {actual}"
    )))
}

fn top_level_sum_invalid_combination(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyErr> {
    let summary = call_type_summary(positional, keywords, CallKeywordOrder::PyTorchUnorderedMap)?;
    Ok(PyTypeError::new_err(format!(
        "sum() received an invalid combination of arguments - got ({summary}), but expected one of:\n \
* (Tensor input, *, torch.dtype dtype = None)\n \
* (Tensor input, tuple of ints dim, bool keepdim = False, *, torch.dtype dtype = None, Tensor out = None)\n"
    )))
}

fn bind_top_level_mean_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundTopLevelMeanCall<'py>> {
    if positional.len() > 3 {
        return Err(PyTypeError::new_err(format!(
            "mean() takes from 2 to 3 positional arguments but {} were given",
            positional.len()
        )));
    }

    let keyword_input = top_level_sum_keyword_input(keywords)?;
    let input_argument = top_level_sum_input_argument(positional, keyword_input.as_ref())?;
    let Some(input_argument) = input_argument else {
        if positional.is_empty() && keywords.is_none_or(PyDictMethods::is_empty) {
            return Err(top_level_mean_invalid_combination(positional, keywords)?);
        }
        return Err(PyTypeError::new_err(
            "mean() missing 1 required positional arguments: \"input\"",
        ));
    };

    let keyword_dim = top_level_sum_keyword(keywords, "dim")?;
    let keyword_keepdim = top_level_sum_keyword(keywords, "keepdim")?;
    let keyword_dtype = top_level_sum_keyword(keywords, "dtype")?;
    let keyword_out = top_level_sum_keyword(keywords, "out")?;
    let keyword_input_duplicated = keyword_input.as_ref().is_some_and(|input| input.duplicated);
    let input_argument_duplicated = positional.len() >= 1 && keyword_input.is_some()
        || positional.is_empty() && keyword_input_duplicated;
    let dim_argument_duplicated = positional.len() >= 2 && keyword_dim.is_some();
    let keepdim_argument_duplicated = positional.len() >= 3 && keyword_keepdim.is_some();

    if top_level_mean_has_unexpected_keyword(keywords)?
        || input_argument_duplicated
        || dim_argument_duplicated
        || keepdim_argument_duplicated
    {
        return Err(top_level_mean_invalid_combination(positional, keywords)?);
    }

    let has_dimension = positional.len() >= 2 || keyword_dim.is_some();
    let has_keepdim = positional.len() >= 3 || keyword_keepdim.is_some();
    let has_out = keyword_out.is_some();
    if !has_dimension && has_keepdim {
        return Err(top_level_mean_invalid_combination(positional, keywords)?);
    }

    let input = parse_top_level_mean_input(
        &input_argument,
        has_dimension || has_keepdim || has_out || keyword_dtype.is_some(),
        positional,
        keywords,
    )?;

    let dimension = top_level_sum_dimension_argument(positional, keyword_dim)?;
    if let Some(dimension) = &dimension
        && !is_sum_dimension_argument(&dimension.value)?
    {
        return Err(top_level_mean_invalid_combination(positional, keywords)?);
    }

    let keepdim = top_level_sum_keepdim_argument(positional, keyword_keepdim)?;
    if let Some(keepdim) = &keepdim
        && !keepdim.value.is_exact_instance_of::<PyBool>()
    {
        return Err(mean_argument_type_error(
            "keepdim",
            keepdim.position,
            "bool",
            &keepdim.value,
        )?);
    }

    let dtype = match keyword_dtype {
        Some(dtype) => {
            bind_top_level_mean_dtype(&dtype.value, has_dimension, positional, keywords)?
        }
        None => BoundTopLevelMeanDType::Native,
    };

    let out = parse_top_level_mean_out(keyword_out)?;
    let explicit_full_dimension = dimension
        .as_ref()
        .is_some_and(|dimension| dimension.value.is_none());
    let keepdim_is_true = keepdim
        .as_ref()
        .is_some_and(|keepdim| keepdim.value.extract::<bool>().is_ok_and(|keepdim| keepdim));
    let full_reduction = (!has_dimension || explicit_full_dimension) && out.is_none();
    let keepdim_full_reduction = explicit_full_dimension && keepdim_is_true && out.is_none();

    Ok(BoundTopLevelMeanCall {
        input,
        dtype,
        out,
        full_reduction,
        keepdim_full_reduction,
    })
}

fn top_level_mean_has_unexpected_keyword(keywords: Option<&Bound<'_, PyDict>>) -> PyResult<bool> {
    let Some(keywords) = keywords else {
        return Ok(false);
    };
    for key in keywords.keys() {
        let key = key.extract::<String>()?;
        if !matches!(
            key.as_str(),
            "input" | "x" | "a" | "x1" | "dim" | "keepdim" | "dtype" | "out"
        ) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn parse_top_level_mean_input<'py>(
    input: &ParsedCallArgument<'py>,
    overload_mismatch_on_non_tensor: bool,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundTensorOrTorchFunction<'py>> {
    if input.value.is_exact_instance_of::<PyTensor>() {
        return Ok(BoundTensorOrTorchFunction::Tensor(
            input.value.cast::<PyTensor>()?.clone(),
        ));
    }
    if let Some(probed) = probe_torch_function_override(&input.value) {
        return Ok(BoundTensorOrTorchFunction::Override(probed));
    }
    if input.value.is_instance_of::<PyTensor>() {
        return Err(mean_unsupported_native_input());
    }
    if overload_mismatch_on_non_tensor {
        return Err(top_level_mean_invalid_combination(positional, keywords)?);
    }
    Err(legacy_single_tensor_type_error("mean", input)?)
}

fn bind_top_level_mean_dtype<'py>(
    dtype: &Bound<'py, PyAny>,
    has_dimension: bool,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundTopLevelMeanDType<'py>> {
    if dtype.is_none() {
        return Ok(BoundTopLevelMeanDType::Native);
    }
    if let Ok(dtype) = dtype.cast::<PyDType>() {
        if dtype.try_borrow()?.inner() == DType::Float32 {
            return Ok(BoundTopLevelMeanDType::Native);
        }
        return Err(mean_unsupported_dtype_conversion());
    }
    if let Some(probed) = probe_dtype_torch_function_override(dtype) {
        return Ok(BoundTopLevelMeanDType::Override(probed));
    }
    if has_dimension {
        return Err(mean_argument_type_error(
            "dtype",
            None,
            "torch.dtype",
            dtype,
        )?);
    }
    Err(top_level_mean_invalid_combination(positional, keywords)?)
}

fn parse_top_level_mean_out(
    keyword_out: Option<TopLevelSumKeyword<'_>>,
) -> PyResult<Option<BoundTensorOrTorchFunction<'_>>> {
    let Some(out) = keyword_out else {
        return Ok(None);
    };
    if out.value.is_none() {
        return Ok(None);
    }
    let out = ParsedCallArgument {
        value: out.value,
        position: None,
    };
    parse_tensor_or_torch_function_argument("mean", "out", &out).map(Some)
}

fn mean_argument_type_error(
    argument: &str,
    position: Option<usize>,
    expected: &str,
    actual: &Bound<'_, PyAny>,
) -> PyResult<PyErr> {
    let position = position.map_or_else(String::new, |position| format!(" (position {position})"));
    let actual = python_type_name(actual)?;
    Ok(PyTypeError::new_err(format!(
        "mean(): argument '{argument}'{position} must be {expected}, not {actual}"
    )))
}

fn top_level_mean_invalid_combination(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyErr> {
    let summary = call_type_summary(positional, keywords, CallKeywordOrder::PyTorchUnorderedMap)?;
    Ok(PyTypeError::new_err(format!(
        "mean() received an invalid combination of arguments - got ({summary}), but expected one of:\n \
* (Tensor input, *, torch.dtype dtype = None, Tensor out = None)\n \
* (Tensor input, tuple of ints dim, bool keepdim = False, *, torch.dtype dtype = None, Tensor out = None)\n"
    )))
}

fn mean_unsupported_native_input() -> PyErr {
    PyNotImplementedError::new_err(
        "mean(): only exact native CPU float32 Tensor inputs are supported",
    )
}

fn mean_unsupported_dtype_conversion() -> PyErr {
    PyNotImplementedError::new_err(
        "mean(): dtype conversions are not supported; only torch.float32 identity is implemented",
    )
}

fn bind_method_sum_arguments(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<BoundMethodReductionCall> {
    if positional.len() > 2 {
        if positional.len() == 3 && keywords.is_none_or(PyDictMethods::is_empty) {
            return Err(PyTypeError::new_err(
                "sum() takes from 1 to 2 positional arguments but 3 were given",
            ));
        }
        return Err(sum_method_invalid_combination(positional, keywords)?);
    }

    let keyword_dim = top_level_sum_keyword(keywords, "dim")?;
    let keyword_keepdim = top_level_sum_keyword(keywords, "keepdim")?;
    let keyword_dtype = top_level_sum_keyword(keywords, "dtype")?;
    if method_sum_has_unexpected_keyword(keywords)?
        || positional.len() >= 1 && keyword_dim.is_some()
        || positional.len() >= 2 && keyword_keepdim.is_some()
    {
        return Err(sum_method_invalid_combination(positional, keywords)?);
    }

    let has_dimension = positional.len() >= 1 || keyword_dim.is_some();
    let has_keepdim = positional.len() >= 2 || keyword_keepdim.is_some();
    if !has_dimension {
        if has_keepdim {
            return Err(sum_method_invalid_combination(positional, keywords)?);
        }
        if let Some(dtype) = keyword_dtype {
            bind_method_sum_dtype(&dtype.value, false, positional, keywords)?;
        }
        return Ok(BoundMethodReductionCall {
            keepdim_full_reduction: false,
        });
    }

    let dimension = method_sum_dimension_argument(positional, keyword_dim)?;
    let Some(dimension) = dimension else {
        unreachable!("sum method dimension is present when it has a positional or keyword value")
    };
    if !dimension.value.is_none() || !is_sum_dimension_argument(&dimension.value)? {
        return Err(sum_method_invalid_combination(positional, keywords)?);
    }

    let keepdim = method_sum_keepdim_argument(positional, keyword_keepdim)?;
    if let Some(keepdim) = &keepdim
        && !keepdim.value.is_exact_instance_of::<PyBool>()
    {
        return Err(sum_argument_type_error(
            "keepdim",
            keepdim.position,
            "bool",
            &keepdim.value,
        )?);
    }
    let keepdim_is_true = keepdim
        .as_ref()
        .is_some_and(|keepdim| keepdim.value.extract::<bool>().is_ok_and(|keepdim| keepdim));

    if let Some(dtype) = keyword_dtype {
        bind_method_sum_dtype(&dtype.value, true, positional, keywords)?;
    }
    Ok(BoundMethodReductionCall {
        keepdim_full_reduction: keepdim_is_true,
    })
}

fn method_sum_has_unexpected_keyword(keywords: Option<&Bound<'_, PyDict>>) -> PyResult<bool> {
    let Some(keywords) = keywords else {
        return Ok(false);
    };
    for key in keywords.keys() {
        let key = key.extract::<String>()?;
        if !matches!(key.as_str(), "dim" | "keepdim" | "dtype") {
            return Ok(true);
        }
    }
    Ok(false)
}

fn method_sum_dimension_argument<'py>(
    positional: &Bound<'py, PyTuple>,
    keyword_dim: Option<TopLevelSumKeyword<'py>>,
) -> PyResult<Option<ParsedCallArgument<'py>>> {
    if !positional.is_empty() {
        return Ok(Some(ParsedCallArgument {
            value: positional.get_item(0)?,
            position: Some(1),
        }));
    }

    Ok(keyword_dim.map(|dimension| ParsedCallArgument {
        value: dimension.value,
        position: None,
    }))
}

fn method_sum_keepdim_argument<'py>(
    positional: &Bound<'py, PyTuple>,
    keyword_keepdim: Option<TopLevelSumKeyword<'py>>,
) -> PyResult<Option<ParsedCallArgument<'py>>> {
    if positional.len() >= 2 {
        return Ok(Some(ParsedCallArgument {
            value: positional.get_item(1)?,
            position: Some(2),
        }));
    }

    Ok(keyword_keepdim.map(|keepdim| ParsedCallArgument {
        value: keepdim.value,
        position: None,
    }))
}

fn bind_method_sum_dtype(
    dtype: &Bound<'_, PyAny>,
    has_dimension: bool,
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    if dtype.is_none() {
        return Ok(());
    }
    if let Ok(dtype) = dtype.cast::<PyDType>()
        && dtype.try_borrow()?.inner() == DType::Float32
    {
        return Ok(());
    }
    if has_dimension {
        return Err(sum_argument_type_error(
            "dtype",
            None,
            "torch.dtype",
            dtype,
        )?);
    }
    Err(sum_method_invalid_combination(positional, keywords)?)
}

fn sum_method_invalid_combination(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyErr> {
    let summary = call_type_summary(positional, keywords, CallKeywordOrder::PyTorchUnorderedMap)?;
    Ok(PyTypeError::new_err(format!(
        "sum() received an invalid combination of arguments - got ({summary}), but expected one of:\n \
* (*, torch.dtype dtype = None)\n \
* (tuple of ints dim, bool keepdim = False, *, torch.dtype dtype = None)\n"
    )))
}

fn bind_method_mean_arguments(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<BoundMethodReductionCall> {
    if positional.len() > 2 {
        if positional.len() == 3 && keywords.is_none_or(PyDictMethods::is_empty) {
            return Err(PyTypeError::new_err(
                "mean() takes from 1 to 2 positional arguments but 3 were given",
            ));
        }
        return Err(mean_method_invalid_combination(positional, keywords)?);
    }

    let keyword_dim = top_level_sum_keyword(keywords, "dim")?;
    let keyword_keepdim = top_level_sum_keyword(keywords, "keepdim")?;
    let keyword_dtype = top_level_sum_keyword(keywords, "dtype")?;
    if method_mean_has_unexpected_keyword(keywords)?
        || !positional.is_empty() && keyword_dim.is_some()
        || positional.len() >= 2 && keyword_keepdim.is_some()
    {
        return Err(mean_method_invalid_combination(positional, keywords)?);
    }

    let has_dimension = !positional.is_empty() || keyword_dim.is_some();
    let has_keepdim = positional.len() >= 2 || keyword_keepdim.is_some();
    if !has_dimension {
        if has_keepdim {
            return Err(mean_method_invalid_combination(positional, keywords)?);
        }
        if let Some(dtype) = keyword_dtype {
            bind_method_mean_dtype(&dtype.value, false, positional, keywords)?;
        }
        return Ok(BoundMethodReductionCall {
            keepdim_full_reduction: false,
        });
    }

    let dimension = method_sum_dimension_argument(positional, keyword_dim)?;
    let Some(dimension) = dimension else {
        unreachable!("mean method dimension is present when it has a positional or keyword value")
    };
    if !is_sum_dimension_argument(&dimension.value)? {
        return Err(mean_method_invalid_combination(positional, keywords)?);
    }
    if !dimension.value.is_none() {
        return Err(mean_unsupported_reduction());
    }

    let keepdim = method_sum_keepdim_argument(positional, keyword_keepdim)?;
    if let Some(keepdim) = &keepdim
        && !keepdim.value.is_exact_instance_of::<PyBool>()
    {
        return Err(mean_argument_type_error(
            "keepdim",
            keepdim.position,
            "bool",
            &keepdim.value,
        )?);
    }
    let keepdim_is_true = keepdim
        .as_ref()
        .is_some_and(|keepdim| keepdim.value.extract::<bool>().is_ok_and(|keepdim| keepdim));

    if let Some(dtype) = keyword_dtype {
        bind_method_mean_dtype(&dtype.value, has_dimension, positional, keywords)?;
    }
    Ok(BoundMethodReductionCall {
        keepdim_full_reduction: keepdim_is_true,
    })
}

fn method_mean_has_unexpected_keyword(keywords: Option<&Bound<'_, PyDict>>) -> PyResult<bool> {
    let Some(keywords) = keywords else {
        return Ok(false);
    };
    for key in keywords.keys() {
        let key = key.extract::<String>()?;
        if !matches!(key.as_str(), "dim" | "keepdim" | "dtype") {
            return Ok(true);
        }
    }
    Ok(false)
}

fn bind_method_mean_dtype(
    dtype: &Bound<'_, PyAny>,
    has_dimension: bool,
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    if dtype.is_none() {
        return Ok(());
    }
    if let Ok(dtype) = dtype.cast::<PyDType>() {
        if dtype.try_borrow()?.inner() == DType::Float32 {
            return Ok(());
        }
        return Err(mean_unsupported_dtype_conversion());
    }
    if has_dimension {
        return Err(mean_argument_type_error(
            "dtype",
            None,
            "torch.dtype",
            dtype,
        )?);
    }
    Err(mean_method_invalid_combination(positional, keywords)?)
}

fn mean_method_invalid_combination(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyErr> {
    let summary = call_type_summary(positional, keywords, CallKeywordOrder::PyTorchUnorderedMap)?;
    Ok(PyTypeError::new_err(format!(
        "mean() received an invalid combination of arguments - got ({summary}), but expected one of:\n \
* (*, torch.dtype dtype = None)\n \
* (tuple of ints dim, bool keepdim = False, *, torch.dtype dtype = None)\n"
    )))
}

fn mean_unsupported_reduction() -> PyErr {
    PyNotImplementedError::new_err(
        "mean(): only full reductions with dim=None support keepdim; dim, out, and dtype conversion reductions are not supported",
    )
}

fn bind_method_flatten_arguments(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<(i64, i64)> {
    if positional.len() > 2 {
        return Err(PyTypeError::new_err(format!(
            "flatten() takes from 0 to 2 positional arguments but {} were given",
            positional.len()
        )));
    }

    let keyword_start = match keywords {
        Some(values) => values.get_item("start_dim")?,
        None => None,
    };
    let keyword_end = match keywords {
        Some(values) => values.get_item("end_dim")?,
        None => None,
    };
    let unexpected = first_unexpected_flatten_keyword(keywords, false)?;

    let start_dim =
        bind_flatten_dimension(positional, 0, keyword_start.as_ref(), "start_dim", 1, 0)?;
    let end_dim = bind_flatten_dimension(positional, 1, keyword_end.as_ref(), "end_dim", 2, -1)?;
    if start_dim.duplicated {
        return Err(multiple_flatten_argument("start_dim"));
    }
    if end_dim.duplicated {
        return Err(multiple_flatten_argument("end_dim"));
    }
    if let Some(unexpected) = unexpected {
        return Err(unexpected_flatten_keyword(&unexpected));
    }
    Ok((start_dim.value, end_dim.value))
}

fn bind_top_level_flatten_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<(Bound<'py, PyAny>, i64, i64)> {
    if positional.len() > 3 {
        return Err(PyTypeError::new_err(format!(
            "flatten() takes from 1 to 3 positional arguments but {} were given",
            positional.len()
        )));
    }

    let keyword_input = match keywords {
        Some(values) => values.get_item("input")?,
        None => None,
    };
    let keyword_start = match keywords {
        Some(values) => values.get_item("start_dim")?,
        None => None,
    };
    let keyword_end = match keywords {
        Some(values) => values.get_item("end_dim")?,
        None => None,
    };
    if positional.is_empty() && keyword_input.is_none() {
        return Err(PyTypeError::new_err(
            "flatten() missing 1 required positional arguments: \"input\"",
        ));
    }
    let unexpected = first_unexpected_flatten_keyword(keywords, true)?;

    let (input, input_duplicated) = if positional.is_empty() {
        (
            keyword_input.expect("the required keyword input was checked above"),
            false,
        )
    } else {
        let input = positional.get_item(0)?;
        validate_flatten_input(&input, Some(1))?;
        (input, keyword_input.is_some())
    };
    if positional.is_empty() {
        validate_flatten_input(&input, None)?;
    }

    let start_dim =
        bind_flatten_dimension(positional, 1, keyword_start.as_ref(), "start_dim", 2, 0)?;
    let end_dim = bind_flatten_dimension(positional, 2, keyword_end.as_ref(), "end_dim", 3, -1)?;
    if input_duplicated {
        return Err(multiple_flatten_argument("input"));
    }
    if start_dim.duplicated {
        return Err(multiple_flatten_argument("start_dim"));
    }
    if end_dim.duplicated {
        return Err(multiple_flatten_argument("end_dim"));
    }
    if let Some(unexpected) = unexpected {
        return Err(unexpected_flatten_keyword(&unexpected));
    }
    Ok((input, start_dim.value, end_dim.value))
}

struct ParsedFlattenDimension {
    value: i64,
    duplicated: bool,
}

fn bind_flatten_dimension(
    positional: &Bound<'_, PyTuple>,
    index: usize,
    keyword: Option<&Bound<'_, PyAny>>,
    name: &str,
    position: usize,
    default: i64,
) -> PyResult<ParsedFlattenDimension> {
    if positional.len() > index {
        let value = positional.get_item(index)?;
        return Ok(ParsedFlattenDimension {
            value: parse_flatten_dimension(name, Some(position), &value)?,
            duplicated: keyword.is_some(),
        });
    }
    Ok(ParsedFlattenDimension {
        value: keyword.map_or(Ok(default), |value| {
            parse_flatten_dimension(name, None, value)
        })?,
        duplicated: false,
    })
}

fn validate_flatten_input(input: &Bound<'_, PyAny>, position: Option<usize>) -> PyResult<()> {
    if input.cast::<PyTensor>().is_ok() {
        return Ok(());
    }
    let actual = python_type_name(input)?;
    let position = position.map_or_else(String::new, |position| format!(" (position {position})"));
    Err(PyTypeError::new_err(format!(
        "flatten(): argument 'input'{position} must be Tensor, not {actual}"
    )))
}

fn first_unexpected_flatten_keyword(
    keywords: Option<&Bound<'_, PyDict>>,
    allow_input: bool,
) -> PyResult<Option<String>> {
    let Some(keywords) = keywords else {
        return Ok(None);
    };
    for (key, _) in keywords {
        let key = key.extract::<String>()?;
        if !(matches!(key.as_str(), "start_dim" | "end_dim") || allow_input && key == "input") {
            return Ok(Some(key));
        }
    }
    Ok(None)
}

fn multiple_flatten_argument(argument: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "flatten() got multiple values for argument '{argument}'"
    ))
}

fn unexpected_flatten_keyword(keyword: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "flatten() got an unexpected keyword argument '{keyword}'"
    ))
}

fn same_tensor_metadata(left: &CoreTensor, right: &CoreTensor) -> bool {
    left.shape() == right.shape()
        && left.stride() == right.stride()
        && left.storage_offset() == right.storage_offset()
        && left.shares_storage_with(right)
}

fn apply_squeeze(
    input: &CoreTensor,
    dimensions: ParsedSqueezeDimensions,
) -> Result<CoreTensor, TensorError> {
    match dimensions {
        ParsedSqueezeDimensions::All => input.squeeze(),
        ParsedSqueezeDimensions::Single(dimension) => input.squeeze_dim(dimension),
        ParsedSqueezeDimensions::Multiple(dimensions) => input.squeeze_dims(dimensions),
    }
}

fn bind_method_squeeze_arguments(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<ParsedSqueezeDimensions> {
    let mut keyword_dimension = None;
    if let Some(keywords) = keywords {
        for (key, value) in keywords {
            let key = key.extract::<String>()?;
            if !matches!(key.as_str(), "dim" | "axis") {
                return Err(squeeze_method_binding_error(
                    positional,
                    Some(keywords),
                    Some(&key),
                )?);
            }
            if keyword_dimension.is_some() {
                return Err(squeeze_method_binding_error(
                    positional,
                    Some(keywords),
                    None,
                )?);
            }
            keyword_dimension = Some((key, value));
        }
    }

    if let Some((keyword, dimension)) = keyword_dimension {
        if !positional.is_empty() {
            return Err(squeeze_method_binding_error(positional, keywords, None)?);
        }
        return parse_squeeze_argument(&dimension, false, Some(&keyword), false, false);
    }

    match positional.len() {
        0 => Ok(ParsedSqueezeDimensions::All),
        1 => parse_squeeze_argument(&positional.get_item(0)?, true, None, false, false),
        length => {
            let mut dimensions = try_size_vector(length)?;
            for dimension in positional.iter() {
                let actual = python_type_name(&dimension)?;
                let Some(dimension) = parse_squeeze_integer(&dimension, true)? else {
                    return Err(squeeze_method_invalid_positional(&actual));
                };
                dimensions.push(dimension);
            }
            Ok(ParsedSqueezeDimensions::Multiple(dimensions))
        }
    }
}

fn bind_top_level_squeeze_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<(Bound<'py, PyAny>, Option<usize>, ParsedSqueezeDimensions)> {
    if positional.len() > 2 {
        return Err(squeeze_top_level_binding_error(positional, keywords, None)?);
    }

    let mut input = positional
        .get_item(0)
        .ok()
        .map(|value| (value, Some(1_usize)));
    let mut dimension = positional
        .get_item(1)
        .ok()
        .map(|value| (value, None::<String>, true));

    if let Some(keywords) = keywords {
        for (key, value) in keywords {
            let key = key.extract::<String>()?;
            match key.as_str() {
                "input" if input.is_none() => input = Some((value, None)),
                "dim" | "axis" if dimension.is_none() => {
                    dimension = Some((value, Some(key), false));
                }
                "input" | "dim" | "axis" => {
                    return Err(squeeze_top_level_binding_error(
                        positional,
                        Some(keywords),
                        None,
                    )?);
                }
                _ => {
                    return Err(squeeze_top_level_binding_error(
                        positional,
                        Some(keywords),
                        Some(&key),
                    )?);
                }
            }
        }
    }

    let Some((input, input_position)) = input else {
        if dimension.is_some() {
            return Err(PyTypeError::new_err(
                "squeeze() missing 1 required positional arguments: \"input\"",
            ));
        }
        return Err(squeeze_top_level_binding_error(positional, keywords, None)?);
    };

    let dimension = match dimension {
        None => ParsedSqueezeDimensions::All,
        Some((value, keyword, _)) => parse_squeeze_argument(
            &value,
            false,
            keyword.as_deref(),
            true,
            input_position.is_none(),
        )?,
    };
    Ok((input, input_position, dimension))
}

fn parse_squeeze_argument(
    argument: &Bound<'_, PyAny>,
    allow_index_protocol: bool,
    keyword: Option<&str>,
    top_level: bool,
    top_input_is_keyword: bool,
) -> PyResult<ParsedSqueezeDimensions> {
    if let Ok(dimensions) = argument.cast::<PyTuple>() {
        return parse_squeeze_sequence(
            argument,
            dimensions.len(),
            dimensions.iter(),
            keyword,
            top_level,
            top_input_is_keyword,
        );
    }
    if let Ok(dimensions) = argument.cast::<PyList>() {
        return parse_squeeze_sequence(
            argument,
            dimensions.len(),
            dimensions.iter(),
            keyword,
            top_level,
            top_input_is_keyword,
        );
    }

    let actual = python_type_name(argument)?;
    let Some(dimension) = parse_squeeze_integer(argument, allow_index_protocol)? else {
        return Err(match (top_level, keyword) {
            (true, Some(keyword)) => {
                squeeze_top_level_invalid_keyword(keyword, &actual, &actual, top_input_is_keyword)
            }
            (true, None) => squeeze_top_level_invalid_positional(&actual),
            (false, Some(keyword)) => squeeze_method_invalid_keyword(keyword, &actual, &actual),
            (false, None) => squeeze_method_invalid_positional(&actual),
        });
    };
    Ok(ParsedSqueezeDimensions::Single(dimension))
}

fn parse_squeeze_sequence<'py>(
    sequence: &Bound<'_, PyAny>,
    length: usize,
    dimensions: impl Iterator<Item = Bound<'py, PyAny>>,
    keyword: Option<&str>,
    top_level: bool,
    top_input_is_keyword: bool,
) -> PyResult<ParsedSqueezeDimensions> {
    let mut parsed = try_size_vector(length)?;
    for (index, dimension) in dimensions.enumerate() {
        let actual = python_type_name(&dimension)?;
        let parsed_dimension = parse_squeeze_integer(&dimension, true).map_err(|_| {
            PyTypeError::new_err(format!(
                "squeeze(): argument 'dim' failed to unpack the object at pos {} with error \"Overflow when unpacking long long\"",
                index + 1
            ))
        })?;
        let Some(dimension) = parsed_dimension else {
            if index == 0 {
                let sequence_type = python_type_name(sequence)?;
                let detail = call_argument_type_description(sequence)?;
                return Err(match (top_level, keyword) {
                    (true, Some(keyword)) => squeeze_top_level_invalid_keyword(
                        keyword,
                        &sequence_type,
                        &detail,
                        top_input_is_keyword,
                    ),
                    (true, None) => {
                        squeeze_top_level_invalid_positional_details(&sequence_type, &detail)
                    }
                    (false, Some(keyword)) => {
                        squeeze_method_invalid_keyword(keyword, &sequence_type, &detail)
                    }
                    (false, None) => {
                        squeeze_method_invalid_positional_details(&sequence_type, &detail)
                    }
                });
            }
            if dimension.is_instance_of::<PyBool>() {
                parsed.push(dimension.extract::<i64>()?);
                continue;
            }
            return Err(PyTypeError::new_err(format!(
                "squeeze(): argument 'dim' failed to unpack the object at pos {} with error \"type must be tuple of ints,but got {actual}\"",
                index + 1
            )));
        };
        parsed.push(dimension);
    }
    Ok(ParsedSqueezeDimensions::Multiple(parsed))
}

fn parse_squeeze_integer(
    dimension: &Bound<'_, PyAny>,
    allow_index_protocol: bool,
) -> PyResult<Option<i64>> {
    if dimension.is_instance_of::<PyBool>() {
        return Ok(None);
    }
    if dimension.is_instance_of::<PyInt>() {
        return dimension
            .extract::<i64>()
            .map(Some)
            .map_err(|_| PyValueError::new_err("Overflow when unpacking long long"));
    }

    let mut accepts_index = allow_index_protocol;
    if !accepts_index
        && let Ok(numpy) = PyModule::import(dimension.py(), "numpy")
        && let Ok(numpy_integer) = numpy.getattr("integer")
    {
        accepts_index = dimension.is_instance(&numpy_integer)?;
    }
    if !accepts_index {
        return Ok(None);
    }

    let Ok(indexed) = PyModule::import(dimension.py(), "operator")
        .and_then(|operator| operator.getattr("index"))
        .and_then(|index| index.call1((dimension,)))
    else {
        return Ok(None);
    };
    indexed
        .extract::<i64>()
        .map(Some)
        .map_err(|_| PyValueError::new_err("Overflow when unpacking long long"))
}

fn call_argument_type_description(value: &Bound<'_, PyAny>) -> PyResult<String> {
    let allocation = PythonAllocationFallback::new(value.py());
    call_argument_type_description_with(value, &allocation)
}

fn call_argument_type_description_with(
    value: &Bound<'_, PyAny>,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<String> {
    if !value.is_instance_of::<PyTuple>() && !value.is_instance_of::<PyList>() {
        return python_type_name_with(value, allocation);
    }

    let kind = python_type_name_with(value, allocation)?;
    let tuple = value.is_instance_of::<PyTuple>();
    let (opening, closing) = if tuple { ("(", ")") } else { ("[", "]") };
    let sequence = value.cast::<PySequence>()?;
    let length = sequence.len().unwrap_or(0);
    let mut description = kind;
    try_push_string_with(&mut description, " of ", allocation)?;
    try_push_string_with(&mut description, opening, allocation)?;
    for index in 0..length {
        if index != 0 {
            try_push_string_with(&mut description, ", ", allocation)?;
        }
        let name = python_type_name_with(&sequence.get_item(index)?, allocation)?;
        try_push_string_with(&mut description, &name, allocation)?;
    }
    if tuple && length == 1 {
        try_push_string_with(&mut description, ",", allocation)?;
    }
    try_push_string_with(&mut description, closing, allocation)?;
    Ok(description)
}

#[derive(Clone, Copy)]
enum CallKeywordOrder {
    Sorted,
    PyTorchUnorderedMap,
}

fn call_type_summary(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
    keyword_order: CallKeywordOrder,
) -> PyResult<String> {
    let allocation = PythonAllocationFallback::new(positional.py());
    call_type_summary_with(positional, keywords, keyword_order, &allocation)
}

fn pytorch_keyword_name<'a>(key: &'a Bound<'_, PyAny>) -> PyResult<&'a str> {
    key.cast::<PyString>()?
        .to_str()
        .map_err(|_| PyRuntimeError::new_err("error unpacking string as utf-8"))
}

fn call_type_summary_with(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
    keyword_order: CallKeywordOrder,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<String> {
    let mut summary = String::new();
    for (index, value) in positional.iter().enumerate() {
        if index != 0 {
            try_push_string_with(&mut summary, ", ", allocation)?;
        }
        let name = python_type_name_with(&value, allocation)?;
        try_push_string_with(&mut summary, &name, allocation)?;
    }

    let keyword_length = keywords.map_or(0, PyDictMethods::len);
    let mut keyword_names = try_size_vector_with(keyword_length, allocation)?;
    if let Some(keywords) = keywords {
        for (key, value) in keywords {
            let key = pytorch_keyword_name(&key)?;
            try_push_size_with(
                &mut keyword_names,
                (
                    try_string_from_str_with(key, allocation)?,
                    python_type_name_with(&value, allocation)?,
                ),
                allocation,
            )?;
        }
        match keyword_order {
            CallKeywordOrder::Sorted => {
                keyword_names.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            }
            CallKeywordOrder::PyTorchUnorderedMap => {
                keyword_names = pytorch_unordered_keyword_order(keyword_names, allocation)?;
            }
        }
    }

    if keyword_names.is_empty() {
        return Ok(summary);
    }
    let positional_empty = summary.is_empty();
    if !positional_empty {
        try_push_string_with(&mut summary, ", ", allocation)?;
    }
    for (index, (key, value)) in keyword_names.into_iter().enumerate() {
        if index != 0 {
            try_push_string_with(&mut summary, ", ", allocation)?;
        }
        try_push_string_with(&mut summary, &key, allocation)?;
        try_push_string_with(&mut summary, "=", allocation)?;
        try_push_string_with(&mut summary, &value, allocation)?;
    }
    if positional_empty {
        try_push_string_with(&mut summary, ", ", allocation)?;
    }
    Ok(summary)
}

const PYTORCH_UNORDERED_BUCKET_COUNTS: &[u64] = &[
    13,
    29,
    59,
    127,
    257,
    541,
    1_109,
    2_357,
    5_087,
    10_273,
    20_753,
    42_043,
    85_229,
    172_933,
    351_061,
    712_697,
    1_447_153,
    2_938_679,
    5_967_347,
    12_117_689,
    24_607_243,
    49_969_847,
    101_473_717,
    206_062_531,
    418_451_333,
    849_749_479,
    1_725_587_117,
    3_504_151_727,
    8_589_934_583,
    25_769_803_693,
    68_719_476_731,
    206_158_430_123,
    412_316_860_387,
    1_099_511_627_689,
    2_199_023_255_531,
    4_398_046_511_093,
    13_194_139_533_241,
    26_388_279_066_581,
    52_776_558_133_177,
    105_553_116_266_399,
    211_106_232_532_861,
    562_949_953_421_231,
    1_125_899_906_842_597,
    4_503_599_627_370_449,
    18_014_398_509_481_951,
    36_028_797_018_963_913,
    72_057_594_037_927_931,
    288_230_376_151_711_717,
    1_152_921_504_606_846_883,
    2_305_843_009_213_693_951,
    9_223_372_036_854_775_783,
    18_446_744_073_709_551_557,
];

fn pytorch_unordered_keyword_order<T>(
    keywords: Vec<(String, T)>,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<Vec<(String, T)>> {
    if cfg!(target_os = "windows") {
        pytorch_msvc_keyword_order(keywords, allocation)
    } else if cfg!(target_os = "macos") {
        pytorch_libcxx_keyword_order(keywords, allocation)
    } else {
        pytorch_libstdcxx_keyword_order(keywords, allocation)
    }
}

fn pytorch_msvc_keyword_order<T>(
    keywords: Vec<(String, T)>,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<Vec<(String, T)>> {
    // MSVC's unordered map stores elements in a linked list, appending new
    // buckets and inserting collisions at the front of an existing bucket.
    let capacity = keywords.len();
    if capacity == 0 {
        return Ok(keywords);
    }

    let mut bucket_count = 8_usize;
    let mut buckets = empty_keyword_buckets(bucket_count, allocation)?;
    let mut nodes = try_size_vector_with(capacity, allocation)?;
    let mut head = None;
    let mut tail = None;

    for (key, value) in keywords {
        if nodes.len() == bucket_count {
            bucket_count = if bucket_count < 512 {
                bucket_count.checked_mul(8)
            } else {
                bucket_count.checked_mul(2)
            }
            .ok_or_else(|| allocation.error())?;
            buckets = empty_keyword_buckets(bucket_count, allocation)?;
            (head, tail) =
                rehash_msvc_unordered_keywords(&mut nodes, &mut buckets, bucket_count, head);
        }

        let index = nodes.len();
        try_push_size_with(
            &mut nodes,
            UnorderedKeywordNode {
                hash: msvc_string_hash(&key),
                key,
                value: Some(value),
                previous: None,
                next: None,
            },
            allocation,
        )?;
        link_msvc_unordered_keyword(
            &mut nodes,
            &mut buckets,
            bucket_count,
            &mut head,
            &mut tail,
            index,
        );
    }

    take_unordered_keywords(nodes, head, capacity, allocation)
}

fn pytorch_libstdcxx_keyword_order<T>(
    keywords: Vec<(String, T)>,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<Vec<(String, T)>> {
    // PyTorch 2.13's overload formatter copies keyword arguments into
    // libstdc++'s `std::unordered_map`. Reproduce its MurmurHash64A buckets,
    // prime rehash policy, and bucket-local insertion order.

    let capacity = keywords.len();
    if capacity == 0 {
        return Ok(keywords);
    }

    let mut bucket_counts = PYTORCH_UNORDERED_BUCKET_COUNTS.iter().copied();
    let mut bucket_count = usize::try_from(
        bucket_counts
            .next()
            .expect("the libstdc++ bucket sequence is nonempty"),
    )
    .map_err(|_| allocation.error())?;
    let mut buckets = empty_keyword_buckets(bucket_count, allocation)?;
    let mut nodes = try_size_vector_with(capacity, allocation)?;
    let mut head = None;

    for (key, value) in keywords {
        if nodes.len() == bucket_count {
            bucket_count = usize::try_from(bucket_counts.next().ok_or_else(|| allocation.error())?)
                .map_err(|_| allocation.error())?;
            buckets = empty_keyword_buckets(bucket_count, allocation)?;
            head = rehash_unordered_keywords(&mut nodes, &mut buckets, bucket_count, head);
        }

        let index = nodes.len();
        try_push_size_with(
            &mut nodes,
            UnorderedKeywordNode {
                hash: pytorch_string_hash(&key),
                key,
                value: Some(value),
                previous: None,
                next: None,
            },
            allocation,
        )?;
        link_unordered_keyword(&mut nodes, &mut buckets, bucket_count, &mut head, index);
    }

    take_unordered_keywords(nodes, head, capacity, allocation)
}

fn pytorch_libcxx_keyword_order<T>(
    keywords: Vec<(String, T)>,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<Vec<(String, T)>> {
    // The macOS build of PyTorch uses libc++, whose default unordered map
    // starts with two buckets and then follows its prime rehash policy.
    let capacity = keywords.len();
    if capacity == 0 {
        return Ok(keywords);
    }

    let mut bucket_count = 0_usize;
    let mut buckets = Vec::new();
    let mut nodes = try_size_vector_with(capacity, allocation)?;
    let mut head = None;

    for (key, value) in keywords {
        if nodes.len() == bucket_count {
            bucket_count = if bucket_count == 0 {
                2
            } else {
                let candidate = bucket_count
                    .checked_mul(2)
                    .and_then(|count| count.checked_add(1))
                    .ok_or_else(|| allocation.error())?;
                libcxx_next_prime(candidate, allocation)?
            };
            buckets = empty_keyword_buckets(bucket_count, allocation)?;
            head = rehash_libcxx_unordered_keywords(&mut nodes, &mut buckets, bucket_count, head);
        }

        let index = nodes.len();
        try_push_size_with(
            &mut nodes,
            UnorderedKeywordNode {
                hash: libcxx_string_hash(&key),
                key,
                value: Some(value),
                previous: None,
                next: None,
            },
            allocation,
        )?;
        link_unordered_keyword(&mut nodes, &mut buckets, bucket_count, &mut head, index);
    }

    take_unordered_keywords(nodes, head, capacity, allocation)
}

fn take_unordered_keywords<T>(
    mut nodes: Vec<UnorderedKeywordNode<T>>,
    head: Option<usize>,
    capacity: usize,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<Vec<(String, T)>> {
    let mut ordered = try_size_vector_with(capacity, allocation)?;
    let mut current = head;
    while let Some(index) = current {
        let node = &mut nodes[index];
        current = node.next;
        try_push_size_with(
            &mut ordered,
            (
                std::mem::take(&mut node.key),
                node.value
                    .take()
                    .expect("an unordered keyword value is taken exactly once"),
            ),
            allocation,
        )?;
    }
    Ok(ordered)
}

struct UnorderedKeywordNode<T> {
    hash: u64,
    key: String,
    value: Option<T>,
    previous: Option<usize>,
    next: Option<usize>,
}

fn empty_keyword_buckets(
    bucket_count: usize,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<Vec<Option<usize>>> {
    let mut buckets = try_size_vector_with(bucket_count, allocation)?;
    buckets.resize(bucket_count, None);
    Ok(buckets)
}

fn rehash_unordered_keywords<T>(
    nodes: &mut [UnorderedKeywordNode<T>],
    buckets: &mut [Option<usize>],
    bucket_count: usize,
    old_head: Option<usize>,
) -> Option<usize> {
    let mut head = None;
    let mut current = old_head;
    while let Some(index) = current {
        current = nodes[index].next;
        nodes[index].previous = None;
        nodes[index].next = None;
        link_unordered_keyword(nodes, buckets, bucket_count, &mut head, index);
    }
    head
}

fn rehash_libcxx_unordered_keywords<T>(
    nodes: &mut [UnorderedKeywordNode<T>],
    buckets: &mut [Option<usize>],
    bucket_count: usize,
    mut head: Option<usize>,
) -> Option<usize> {
    let first = head?;
    let first_bucket = unordered_keyword_bucket(nodes[first].hash, bucket_count);
    buckets[first_bucket] = Some(first);

    let mut previous = first;
    let mut previous_bucket = first_bucket;
    while let Some(current) = nodes[previous].next {
        let bucket = unordered_keyword_bucket(nodes[current].hash, bucket_count);
        if bucket == previous_bucket {
            previous = current;
        } else if buckets[bucket].is_none() {
            buckets[bucket] = Some(current);
            previous = current;
            previous_bucket = bucket;
        } else {
            let next = nodes[current].next;
            nodes[previous].next = next;
            if let Some(next) = next {
                nodes[next].previous = Some(previous);
            }

            let bucket_first = buckets[bucket].expect("the bucket was checked as populated");
            let before_bucket = nodes[bucket_first].previous;
            nodes[current].previous = before_bucket;
            nodes[current].next = Some(bucket_first);
            nodes[bucket_first].previous = Some(current);
            if let Some(before_bucket) = before_bucket {
                nodes[before_bucket].next = Some(current);
            } else {
                head = Some(current);
            }
            buckets[bucket] = Some(current);
        }
    }
    head
}

fn rehash_msvc_unordered_keywords<T>(
    nodes: &mut [UnorderedKeywordNode<T>],
    buckets: &mut [Option<usize>],
    bucket_count: usize,
    mut head: Option<usize>,
) -> (Option<usize>, Option<usize>) {
    let Some(first) = head else {
        return (None, None);
    };
    let first_bucket = unordered_keyword_bucket(nodes[first].hash, bucket_count);
    buckets[first_bucket] = Some(first);

    let mut previous = first;
    while let Some(current) = nodes[previous].next {
        let bucket = unordered_keyword_bucket(nodes[current].hash, bucket_count);
        if buckets[bucket].is_none() {
            buckets[bucket] = Some(current);
            previous = current;
            continue;
        }

        let next = nodes[current].next;
        nodes[previous].next = next;
        if let Some(next) = next {
            nodes[next].previous = Some(previous);
        }

        let bucket_first = buckets[bucket].expect("the bucket was checked as populated");
        let before_bucket = nodes[bucket_first].previous;
        nodes[current].previous = before_bucket;
        nodes[current].next = Some(bucket_first);
        nodes[bucket_first].previous = Some(current);
        if let Some(before_bucket) = before_bucket {
            nodes[before_bucket].next = Some(current);
        } else {
            head = Some(current);
        }
        buckets[bucket] = Some(current);
    }

    let mut tail = head;
    while let Some(current) = tail
        && nodes[current].next.is_some()
    {
        tail = nodes[current].next;
    }
    (head, tail)
}

fn link_unordered_keyword<T>(
    nodes: &mut [UnorderedKeywordNode<T>],
    buckets: &mut [Option<usize>],
    bucket_count: usize,
    head: &mut Option<usize>,
    index: usize,
) {
    let bucket = unordered_keyword_bucket(nodes[index].hash, bucket_count);
    if let Some(next) = buckets[bucket] {
        let previous = nodes[next].previous;
        nodes[index].previous = previous;
        nodes[index].next = Some(next);
        nodes[next].previous = Some(index);
        if let Some(previous) = previous {
            nodes[previous].next = Some(index);
        } else {
            *head = Some(index);
        }
    } else {
        nodes[index].next = *head;
        if let Some(old_head) = *head {
            nodes[old_head].previous = Some(index);
        }
        *head = Some(index);
    }
    buckets[bucket] = Some(index);
}

fn link_msvc_unordered_keyword<T>(
    nodes: &mut [UnorderedKeywordNode<T>],
    buckets: &mut [Option<usize>],
    bucket_count: usize,
    head: &mut Option<usize>,
    tail: &mut Option<usize>,
    index: usize,
) {
    let bucket = unordered_keyword_bucket(nodes[index].hash, bucket_count);
    if let Some(next) = buckets[bucket] {
        let previous = nodes[next].previous;
        nodes[index].previous = previous;
        nodes[index].next = Some(next);
        nodes[next].previous = Some(index);
        if let Some(previous) = previous {
            nodes[previous].next = Some(index);
        } else {
            *head = Some(index);
        }
        buckets[bucket] = Some(index);
        return;
    }

    nodes[index].previous = *tail;
    if let Some(old_tail) = *tail {
        nodes[old_tail].next = Some(index);
    } else {
        *head = Some(index);
    }
    *tail = Some(index);
    buckets[bucket] = Some(index);
}

fn unordered_keyword_bucket(hash: u64, bucket_count: usize) -> usize {
    let bucket_count_u64 =
        u64::try_from(bucket_count).expect("the bucket count fits the 64-bit host ABI");
    usize::try_from(hash % bucket_count_u64).expect("a bucket index fits usize")
}

fn msvc_string_hash(value: &str) -> u64 {
    #[cfg(target_pointer_width = "64")]
    const OFFSET_BASIS: usize = 14_695_981_039_346_656_037;
    #[cfg(target_pointer_width = "64")]
    const PRIME: usize = 1_099_511_628_211;
    #[cfg(target_pointer_width = "32")]
    const OFFSET_BASIS: usize = 2_166_136_261;
    #[cfg(target_pointer_width = "32")]
    const PRIME: usize = 16_777_619;

    let mut hash = OFFSET_BASIS;
    for byte in value.as_bytes() {
        hash ^= usize::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash as u64
}

fn libcxx_next_prime(
    mut candidate: usize,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<usize> {
    candidate |= 1;
    while !libcxx_is_prime(candidate) {
        candidate = candidate.checked_add(2).ok_or_else(|| allocation.error())?;
    }
    Ok(candidate)
}

fn libcxx_is_prime(candidate: usize) -> bool {
    if candidate < 2 || candidate.is_multiple_of(2) {
        return candidate == 2;
    }
    let mut divisor = 3;
    while divisor <= candidate / divisor {
        if candidate.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

fn pytorch_string_hash(value: &str) -> u64 {
    const SEED: u64 = 0xC70F_6907;
    const MULTIPLIER: u64 = 0xC6A4_A793_5BD1_E995;

    let bytes = value.as_bytes();
    let length = u64::try_from(bytes.len()).expect("string length fits the 64-bit host ABI");
    let mut hash = SEED ^ length.wrapping_mul(MULTIPLIER);
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let mut word = u64::from_le_bytes(
            chunk
                .try_into()
                .expect("chunks_exact(8) yields eight-byte chunks"),
        );
        word = word.wrapping_mul(MULTIPLIER);
        word ^= word >> 47;
        word = word.wrapping_mul(MULTIPLIER);
        hash ^= word;
        hash = hash.wrapping_mul(MULTIPLIER);
    }

    let remainder = chunks.remainder();
    for (index, byte) in remainder.iter().enumerate() {
        hash ^= u64::from(*byte) << (index * 8);
    }
    if !remainder.is_empty() {
        hash = hash.wrapping_mul(MULTIPLIER);
    }
    hash ^= hash >> 47;
    hash = hash.wrapping_mul(MULTIPLIER);
    hash ^ (hash >> 47)
}

const LIBCXX_HASH_K0: u64 = 0xC3A5_C85C_97CB_3127;
const LIBCXX_HASH_K1: u64 = 0xB492_B66F_BE98_F273;
const LIBCXX_HASH_K2: u64 = 0x9AE1_6A3B_2F90_404F;
const LIBCXX_HASH_K3: u64 = 0xC949_D7C7_509E_6557;

fn libcxx_string_hash(value: &str) -> u64 {
    libcxx_city_hash(value.as_bytes())
}

fn libcxx_load_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_ne_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("a CityHash word contains eight bytes"),
    )
}

fn libcxx_load_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_ne_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("a CityHash word contains four bytes"),
    )
}

fn libcxx_shift_mix(value: u64) -> u64 {
    value ^ (value >> 47)
}

fn libcxx_hash_len_16(left: u64, right: u64) -> u64 {
    const MULTIPLIER: u64 = 0x9DDF_EA08_EB38_2D69;

    let mut first = (left ^ right).wrapping_mul(MULTIPLIER);
    first ^= first >> 47;
    let mut second = (right ^ first).wrapping_mul(MULTIPLIER);
    second ^= second >> 47;
    second.wrapping_mul(MULTIPLIER)
}

fn libcxx_hash_len_0_to_16(bytes: &[u8]) -> u64 {
    let length = bytes.len();
    let length_u64 = u64::try_from(length).expect("string length fits the 64-bit host ABI");
    if length > 8 {
        let first = libcxx_load_u64(bytes, 0);
        let last = libcxx_load_u64(bytes, length - 8);
        return libcxx_hash_len_16(
            first,
            last.wrapping_add(length_u64)
                .rotate_right(u32::try_from(length).expect("the short length fits u32")),
        ) ^ last;
    }
    if length >= 4 {
        let first = libcxx_load_u32(bytes, 0);
        let last = u64::from(libcxx_load_u32(bytes, length - 4));
        // PyTorch's macOS wheel uses system libc++ ABI v1. Its historical
        // CityHash expression shifts in uint32_t before widening to size_t.
        let shifted_first = u64::from(first.wrapping_shl(3));
        return libcxx_hash_len_16(length_u64.wrapping_add(shifted_first), last);
    }
    if length > 0 {
        let first = u32::from(bytes[0]);
        let middle = u32::from(bytes[length >> 1]);
        let last = u32::from(bytes[length - 1]);
        let left = u64::from(first + (middle << 8));
        let right =
            u64::from(u32::try_from(length).expect("the short length fits u32") + (last << 2));
        return libcxx_shift_mix(
            left.wrapping_mul(LIBCXX_HASH_K2) ^ right.wrapping_mul(LIBCXX_HASH_K3),
        )
        .wrapping_mul(LIBCXX_HASH_K2);
    }
    LIBCXX_HASH_K2
}

fn libcxx_hash_len_17_to_32(bytes: &[u8]) -> u64 {
    let length = bytes.len();
    let length_u64 = u64::try_from(length).expect("string length fits the 64-bit host ABI");
    let first = libcxx_load_u64(bytes, 0).wrapping_mul(LIBCXX_HASH_K1);
    let second = libcxx_load_u64(bytes, 8);
    let third = libcxx_load_u64(bytes, length - 8).wrapping_mul(LIBCXX_HASH_K2);
    let fourth = libcxx_load_u64(bytes, length - 16).wrapping_mul(LIBCXX_HASH_K0);
    libcxx_hash_len_16(
        first
            .wrapping_sub(second)
            .rotate_right(43)
            .wrapping_add(third.rotate_right(30))
            .wrapping_add(fourth),
        first
            .wrapping_add((second ^ LIBCXX_HASH_K3).rotate_right(20))
            .wrapping_sub(third)
            .wrapping_add(length_u64),
    )
}

fn libcxx_weak_hash_len_32_with_seeds(
    bytes: &[u8],
    offset: usize,
    mut first_seed: u64,
    mut second_seed: u64,
) -> (u64, u64) {
    let first = libcxx_load_u64(bytes, offset);
    let second = libcxx_load_u64(bytes, offset + 8);
    let third = libcxx_load_u64(bytes, offset + 16);
    let fourth = libcxx_load_u64(bytes, offset + 24);

    first_seed = first_seed.wrapping_add(first);
    second_seed = second_seed
        .wrapping_add(first_seed)
        .wrapping_add(fourth)
        .rotate_right(21);
    let saved_first_seed = first_seed;
    first_seed = first_seed.wrapping_add(second).wrapping_add(third);
    second_seed = second_seed.wrapping_add(first_seed.rotate_right(44));
    (
        first_seed.wrapping_add(fourth),
        second_seed.wrapping_add(saved_first_seed),
    )
}

#[allow(clippy::many_single_char_names)]
fn libcxx_hash_len_33_to_64(bytes: &[u8]) -> u64 {
    let length = bytes.len();
    let length_u64 = u64::try_from(length).expect("string length fits the 64-bit host ABI");
    let mut z = libcxx_load_u64(bytes, 24);
    let mut a = libcxx_load_u64(bytes, 0).wrapping_add(
        length_u64
            .wrapping_add(libcxx_load_u64(bytes, length - 16))
            .wrapping_mul(LIBCXX_HASH_K0),
    );
    let mut b = a.wrapping_add(z).rotate_right(52);
    let mut c = a.rotate_right(37);
    a = a.wrapping_add(libcxx_load_u64(bytes, 8));
    c = c.wrapping_add(a.rotate_right(7));
    a = a.wrapping_add(libcxx_load_u64(bytes, 16));
    let first_value = a.wrapping_add(z);
    let second_value = b.wrapping_add(a.rotate_right(31)).wrapping_add(c);
    a = libcxx_load_u64(bytes, 16).wrapping_add(libcxx_load_u64(bytes, length - 32));
    z = z.wrapping_add(libcxx_load_u64(bytes, length - 8));
    b = a.wrapping_add(z).rotate_right(52);
    c = a.rotate_right(37);
    a = a.wrapping_add(libcxx_load_u64(bytes, length - 24));
    c = c.wrapping_add(a.rotate_right(7));
    a = a.wrapping_add(libcxx_load_u64(bytes, length - 16));
    let third_value = a.wrapping_add(z);
    let fourth_value = b.wrapping_add(a.rotate_right(31)).wrapping_add(c);
    let result = libcxx_shift_mix(
        first_value
            .wrapping_add(fourth_value)
            .wrapping_mul(LIBCXX_HASH_K2)
            .wrapping_add(
                third_value
                    .wrapping_add(second_value)
                    .wrapping_mul(LIBCXX_HASH_K0),
            ),
    );
    libcxx_shift_mix(
        result
            .wrapping_mul(LIBCXX_HASH_K0)
            .wrapping_add(second_value),
    )
    .wrapping_mul(LIBCXX_HASH_K2)
}

#[allow(clippy::many_single_char_names)]
fn libcxx_city_hash(bytes: &[u8]) -> u64 {
    let length = bytes.len();
    if length <= 16 {
        return libcxx_hash_len_0_to_16(bytes);
    }
    if length <= 32 {
        return libcxx_hash_len_17_to_32(bytes);
    }
    if length <= 64 {
        return libcxx_hash_len_33_to_64(bytes);
    }

    let length_u64 = u64::try_from(length).expect("string length fits the 64-bit host ABI");
    let mut x = libcxx_load_u64(bytes, length - 40);
    let mut y =
        libcxx_load_u64(bytes, length - 16).wrapping_add(libcxx_load_u64(bytes, length - 56));
    let mut z = libcxx_hash_len_16(
        libcxx_load_u64(bytes, length - 48).wrapping_add(length_u64),
        libcxx_load_u64(bytes, length - 24),
    );
    let mut v = libcxx_weak_hash_len_32_with_seeds(bytes, length - 64, length_u64, z);
    let mut w =
        libcxx_weak_hash_len_32_with_seeds(bytes, length - 32, y.wrapping_add(LIBCXX_HASH_K1), x);
    x = x
        .wrapping_mul(LIBCXX_HASH_K1)
        .wrapping_add(libcxx_load_u64(bytes, 0));

    let mut offset = 0;
    let mut remaining = (length - 1) & !63;
    while remaining != 0 {
        x = x
            .wrapping_add(y)
            .wrapping_add(v.0)
            .wrapping_add(libcxx_load_u64(bytes, offset + 8))
            .rotate_right(37)
            .wrapping_mul(LIBCXX_HASH_K1);
        y = y
            .wrapping_add(v.1)
            .wrapping_add(libcxx_load_u64(bytes, offset + 48))
            .rotate_right(42)
            .wrapping_mul(LIBCXX_HASH_K1);
        x ^= w.1;
        y = y
            .wrapping_add(v.0)
            .wrapping_add(libcxx_load_u64(bytes, offset + 40));
        z = z
            .wrapping_add(w.0)
            .rotate_right(33)
            .wrapping_mul(LIBCXX_HASH_K1);
        v = libcxx_weak_hash_len_32_with_seeds(
            bytes,
            offset,
            v.1.wrapping_mul(LIBCXX_HASH_K1),
            x.wrapping_add(w.0),
        );
        w = libcxx_weak_hash_len_32_with_seeds(
            bytes,
            offset + 32,
            z.wrapping_add(w.1),
            y.wrapping_add(libcxx_load_u64(bytes, offset + 16)),
        );
        std::mem::swap(&mut z, &mut x);
        offset += 64;
        remaining -= 64;
    }
    libcxx_hash_len_16(
        libcxx_hash_len_16(v.0, w.0)
            .wrapping_add(libcxx_shift_mix(y).wrapping_mul(LIBCXX_HASH_K1))
            .wrapping_add(z),
        libcxx_hash_len_16(v.1, w.1).wrapping_add(x),
    )
}

fn squeeze_method_binding_error(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
    unknown_keyword: Option<&str>,
) -> PyResult<PyErr> {
    let summary = call_type_summary(positional, keywords, CallKeywordOrder::Sorted)?;
    let mismatch = unknown_keyword.map_or_else(String::new, |keyword| {
        format!("\n      didn't match because some of the keywords were incorrect: {keyword}")
    });
    Ok(PyTypeError::new_err(format!(
        "squeeze() received an invalid combination of arguments - got ({summary}), but expected one of:\n * (){mismatch}\n * (int dim){mismatch}\n * (tuple of ints dim){mismatch}\n"
    )))
}

fn squeeze_top_level_binding_error(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
    unknown_keyword: Option<&str>,
) -> PyResult<PyErr> {
    let summary = call_type_summary(positional, keywords, CallKeywordOrder::Sorted)?;
    let mismatch = unknown_keyword.map_or_else(String::new, |keyword| {
        format!("\n      didn't match because some of the keywords were incorrect: {keyword}")
    });
    Ok(PyTypeError::new_err(format!(
        "squeeze() received an invalid combination of arguments - got ({summary}), but expected one of:\n * (Tensor input)\n * (Tensor input, int dim){mismatch}\n * (Tensor input, tuple of ints dim){mismatch}\n"
    )))
}

fn squeeze_top_level_input_with_dimension_error(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
    dimension: &ParsedSqueezeDimensions,
) -> PyResult<PyErr> {
    let summary = call_type_summary(positional, keywords, CallKeywordOrder::Sorted)?;
    let input_is_keyword = positional.is_empty();
    let input = if input_is_keyword {
        let keywords = keywords.expect("a bound keyword input must have a keyword dictionary");
        keywords
            .get_item("input")?
            .expect("a successfully bound keyword input must remain present")
    } else {
        positional.get_item(0)?
    };
    let input_type = python_type_name(&input)?;

    let (dimension_value, dimension_keyword) = if positional.len() > 1 {
        (positional.get_item(1)?, None)
    } else {
        let keywords = keywords.expect("a bound keyword dimension must have keyword arguments");
        let mut found = None;
        for (key, value) in keywords {
            let key = key.extract::<String>()?;
            if matches!(key.as_str(), "dim" | "axis") {
                found = Some((value, key));
                break;
            }
        }
        found
            .map(|(value, keyword)| (value, Some(keyword)))
            .expect("a non-omitted bound dimension must remain present")
    };
    let dimension_type = python_type_name(&dimension_value)?;
    let dimension_detail_type = if dimension_value.is_instance_of::<PyTuple>()
        || dimension_value.is_instance_of::<PyList>()
    {
        call_argument_type_description(&dimension_value)?
    } else {
        dimension_type.clone()
    };
    let integer_compatible =
        !dimension_value.is_instance_of::<PyBool>() && dimension_value.is_instance_of::<PyInt>();
    let tuple_compatible = dimension_value.is_instance_of::<PyTuple>()
        && matches!(dimension, &ParsedSqueezeDimensions::Multiple(_));

    let input_detail = if input_is_keyword {
        format!("!input={input_type}!")
    } else {
        format!("!{input_type}!")
    };
    let dimension_detail = |invalid: bool| {
        let detail = dimension_keyword.as_ref().map_or_else(
            || {
                if invalid {
                    dimension_detail_type.clone()
                } else {
                    dimension_type.clone()
                }
            },
            |keyword| {
                if invalid {
                    format!("{keyword}={dimension_detail_type}")
                } else {
                    format!("{keyword}={dimension_type}")
                }
            },
        );
        if invalid {
            format!("!{detail}!")
        } else {
            detail
        }
    };
    let trailing = if input_is_keyword { ", " } else { "" };
    let integer_detail = format!(
        "{input_detail}, {}{trailing}",
        dimension_detail(!integer_compatible)
    );
    let tuple_detail = format!(
        "{input_detail}, {}{trailing}",
        dimension_detail(!tuple_compatible)
    );
    Ok(PyTypeError::new_err(format!(
        "squeeze() received an invalid combination of arguments - got ({summary}), but expected one of:\n * (Tensor input)\n * (Tensor input, int dim)\n      didn't match because some of the arguments have invalid types: ({integer_detail})\n * (Tensor input, tuple of ints dim)\n      didn't match because some of the arguments have invalid types: ({tuple_detail})\n"
    )))
}

fn squeeze_method_invalid_positional(actual: &str) -> PyErr {
    squeeze_method_invalid_positional_details(actual, actual)
}

fn squeeze_method_invalid_positional_details(actual: &str, detail: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "squeeze() received an invalid combination of arguments - got ({actual}), but expected one of:\n * ()\n      didn't match because some of the arguments have invalid types: (!{detail}!)\n * (int dim)\n      didn't match because some of the arguments have invalid types: (!{detail}!)\n * (tuple of ints dim)\n      didn't match because some of the arguments have invalid types: (!{detail}!)\n"
    ))
}

fn squeeze_method_invalid_keyword(keyword: &str, actual: &str, detail: &str) -> PyErr {
    if keyword == "axis" {
        return PyTypeError::new_err(format!(
            "squeeze() received an invalid combination of arguments - got ({keyword}={actual}, ), but expected one of:\n * ()\n      didn't match because some of the keywords were incorrect: {keyword}\n * (int dim)\n      didn't match because some of the keywords were incorrect: {keyword}\n * (tuple of ints dim)\n      didn't match because some of the keywords were incorrect: {keyword}\n"
        ));
    }
    PyTypeError::new_err(format!(
        "squeeze() received an invalid combination of arguments - got ({keyword}={actual}, ), but expected one of:\n * ()\n      didn't match because some of the keywords were incorrect: {keyword}\n * (int dim)\n      didn't match because some of the arguments have invalid types: (!{keyword}={detail}!, )\n * (tuple of ints dim)\n      didn't match because some of the arguments have invalid types: (!{keyword}={detail}!, )\n"
    ))
}

fn squeeze_top_level_invalid_positional(actual: &str) -> PyErr {
    squeeze_top_level_invalid_positional_details(actual, actual)
}

fn squeeze_top_level_invalid_positional_details(actual: &str, detail: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "squeeze() received an invalid combination of arguments - got (Tensor, {actual}), but expected one of:\n * (Tensor input)\n * (Tensor input, int dim)\n      didn't match because some of the arguments have invalid types: (Tensor, !{detail}!)\n * (Tensor input, tuple of ints dim)\n      didn't match because some of the arguments have invalid types: (Tensor, !{detail}!)\n"
    ))
}

fn squeeze_top_level_invalid_keyword(
    keyword: &str,
    actual: &str,
    detail: &str,
    input_is_keyword: bool,
) -> PyErr {
    if input_is_keyword {
        let mismatch = if keyword == "axis" {
            format!("some of the keywords were incorrect: {keyword}")
        } else {
            format!(
                "some of the arguments have invalid types: (input=Tensor, !{keyword}={detail}!, )"
            )
        };
        return PyTypeError::new_err(format!(
            "squeeze() received an invalid combination of arguments - got ({keyword}={actual}, input=Tensor, ), but expected one of:\n * (Tensor input)\n * (Tensor input, int dim)\n      didn't match because {mismatch}\n * (Tensor input, tuple of ints dim)\n      didn't match because {mismatch}\n"
        ));
    }
    if keyword == "axis" {
        return PyTypeError::new_err(format!(
            "squeeze() received an invalid combination of arguments - got (Tensor, {keyword}={actual}), but expected one of:\n * (Tensor input)\n * (Tensor input, int dim)\n      didn't match because some of the keywords were incorrect: {keyword}\n * (Tensor input, tuple of ints dim)\n      didn't match because some of the keywords were incorrect: {keyword}\n"
        ));
    }
    PyTypeError::new_err(format!(
        "squeeze() received an invalid combination of arguments - got (Tensor, {keyword}={actual}), but expected one of:\n * (Tensor input)\n * (Tensor input, int dim)\n      didn't match because some of the arguments have invalid types: (Tensor, !{keyword}={detail}!)\n * (Tensor input, tuple of ints dim)\n      didn't match because some of the arguments have invalid types: (Tensor, !{keyword}={detail}!)\n"
    ))
}

fn squeeze_argument_type_error(
    argument: &str,
    position: Option<usize>,
    expected: &str,
    actual: &str,
) -> PyErr {
    let position = position.map_or_else(String::new, |position| format!(" (position {position})"));
    PyTypeError::new_err(format!(
        "squeeze(): argument '{argument}'{position} must be {expected}, not {actual}"
    ))
}

pub(crate) fn bind_legacy_single_tensor_argument<'py>(
    function: &str,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<ParsedCallArgument<'py>> {
    let selection = select_legacy_single_argument(function, positional, keywords)?;
    if selection.input.value.cast::<PyTensor>().is_err() {
        return Err(legacy_single_tensor_type_error(function, &selection.input)?);
    }
    validate_legacy_single_keywords(function, &selection, keywords)?;
    Ok(selection.input)
}

fn bind_legacy_single_tensor_or_override_argument<'py>(
    function: &str,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundTensorOrTorchFunction<'py>> {
    let selection = select_legacy_single_argument(function, positional, keywords)?;
    let bound = if let Ok(tensor) = selection.input.value.cast::<PyTensor>() {
        BoundTensorOrTorchFunction::Tensor(tensor.clone())
    } else if let Some(probed) = probe_torch_function_override(&selection.input.value) {
        BoundTensorOrTorchFunction::Override(probed)
    } else {
        return Err(legacy_single_tensor_type_error(function, &selection.input)?);
    };
    validate_legacy_single_keywords(function, &selection, keywords)?;
    Ok(bound)
}

fn bind_unary_out_arguments<'py>(
    operation: UnaryOutOperation,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundUnaryOutCall<'py>> {
    let selection = select_legacy_single_argument(operation.name, positional, keywords)?;
    let input = parse_tensor_or_torch_function_argument(operation.name, "input", &selection.input)?;
    let out = match keywords
        .map(|values| values.get_item("out"))
        .transpose()?
        .flatten()
    {
        Some(out) if !out.is_none() => Some(parse_tensor_or_torch_function_argument(
            operation.name,
            "out",
            &ParsedCallArgument {
                value: out,
                position: None,
            },
        )?),
        Some(_) | None => None,
    };
    validate_unary_out_keywords(operation, &selection, keywords)?;
    Ok(BoundUnaryOutCall { input, out })
}

fn validate_unary_out_keywords(
    operation: UnaryOutOperation,
    selection: &LegacySingleArgumentSelection<'_>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let Some(keywords) = keywords else {
        return Ok(());
    };
    let has_out = keywords.get_item("out")?.is_some();
    // Legacy input aliases remain valid alongside the generated keyword-only
    // out parameter, but no other keyword may accompany them.
    let sole_alias =
        if selection.input.position.is_none() && keywords.len() == 1 + usize::from(has_out) {
            selection.keyword_alias
        } else {
            None
        };
    for key in keywords.keys() {
        let key = key.extract::<String>()?;
        if key == "out" || sole_alias == Some(key.as_str()) {
            continue;
        }
        if key == "input" {
            if selection.input.position.is_some() {
                return Err(PyTypeError::new_err(format!(
                    "{}() got multiple values for argument 'input'",
                    operation.name
                )));
            }
            continue;
        }
        return Err(PyTypeError::new_err(format!(
            "{}() got an unexpected keyword argument '{key}'",
            operation.name
        )));
    }
    Ok(())
}

struct LegacySingleArgumentSelection<'py> {
    input: ParsedCallArgument<'py>,
    keyword_alias: Option<&'static str>,
}

fn select_legacy_single_argument<'py>(
    function: &str,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<LegacySingleArgumentSelection<'py>> {
    if positional.len() > 1 {
        return Err(PyTypeError::new_err(format!(
            "{function}() takes 1 positional argument but {} were given",
            positional.len()
        )));
    }

    // PyTorch's legacy parser resolves `input`, `x`, `a`, then `x1` for type checking.
    let (keyword_input, keyword_alias) = match keywords {
        Some(values) => {
            if let Some(input) = values.get_item("input")? {
                (Some(input), None)
            } else if let Some(input) = values.get_item("x")? {
                (Some(input), Some("x"))
            } else if let Some(input) = values.get_item("a")? {
                (Some(input), Some("a"))
            } else if let Some(input) = values.get_item("x1")? {
                (Some(input), Some("x1"))
            } else {
                (None, None)
            }
        }
        None => (None, None),
    };
    if positional.is_empty() && keyword_input.is_none() {
        return Err(PyTypeError::new_err(format!(
            "{function}() missing 1 required positional arguments: \"input\""
        )));
    }

    let input = if positional.is_empty() {
        ParsedCallArgument {
            value: keyword_input.expect("the required keyword input was checked above"),
            position: None,
        }
    } else {
        ParsedCallArgument {
            value: positional.get_item(0)?,
            position: Some(1),
        }
    };
    Ok(LegacySingleArgumentSelection {
        input,
        keyword_alias,
    })
}

fn validate_legacy_single_keywords(
    function: &str,
    selection: &LegacySingleArgumentSelection<'_>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    if let Some(keywords) = keywords {
        // The legacy aliases are accepted only as the sole keyword. Mixed calls
        // validate their original keyword order and report an alias as unexpected.
        let sole_alias = if selection.input.position.is_none() && keywords.len() == 1 {
            selection.keyword_alias
        } else {
            None
        };
        for key in keywords.keys() {
            let key = key.extract::<String>()?;
            if sole_alias == Some(key.as_str()) {
                continue;
            }
            if key == "input" {
                if selection.input.position.is_some() {
                    return Err(PyTypeError::new_err(format!(
                        "{function}() got multiple values for argument 'input'"
                    )));
                }
                continue;
            }
            return Err(PyTypeError::new_err(format!(
                "{function}() got an unexpected keyword argument '{key}'"
            )));
        }
    }
    Ok(())
}

fn legacy_single_tensor_type_error(
    function: &str,
    input: &ParsedCallArgument<'_>,
) -> PyResult<PyErr> {
    let position = input
        .position
        .map_or_else(String::new, |position| format!(" (position {position})"));
    let input_type = python_type_name(&input.value)?;
    Ok(PyTypeError::new_err(format!(
        "{function}(): argument 'input'{position} must be Tensor, not {input_type}"
    )))
}

fn bind_tensor_arguments<'py, const N: usize>(
    function: &str,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
    names: [&str; N],
) -> PyResult<([ParsedCallArgument<'py>; N], Option<PyErr>)> {
    if positional.len() > N {
        return Err(PyTypeError::new_err(format!(
            "{function}() takes {N} positional {} but {} were given",
            if N == 1 { "argument" } else { "arguments" },
            positional.len()
        )));
    }

    let mut arguments: [Option<ParsedCallArgument<'py>>; N] = std::array::from_fn(|_| None);
    for (index, value) in positional.iter().enumerate() {
        arguments[index] = Some(ParsedCallArgument {
            value,
            position: Some(index + 1),
        });
    }

    let mut keyword_error = None;
    if let Some(keywords) = keywords {
        for (key, value) in keywords {
            let key = key.extract::<String>()?;
            let Some(index) = names.iter().position(|name| *name == key) else {
                keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "{function}() got an unexpected keyword argument '{key}'"
                    ))
                });
                continue;
            };
            if arguments[index].is_some() {
                keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "{function}() got multiple values for argument '{}'",
                        names[index]
                    ))
                });
                continue;
            }
            arguments[index] = Some(ParsedCallArgument {
                value,
                position: None,
            });
        }
    }

    if let Some(first_missing) = arguments.iter().position(Option::is_none) {
        // Supplied arguments earlier in the schema are converted before a
        // missing later argument is reported.
        for (name, argument) in names.iter().zip(arguments.iter()).take(first_missing) {
            parse_tensor_argument(
                function,
                name,
                argument
                    .as_ref()
                    .expect("arguments preceding the first gap are present"),
            )?;
        }
        // PyTorch reports the complete remaining schema suffix even when a
        // later argument in that suffix was supplied by keyword.
        let missing = &names[first_missing..];
        let quoted_names = missing
            .iter()
            .map(|name| format!("\"{name}\""))
            .collect::<Vec<_>>()
            .join(", ");
        let argument = if missing.len() == 1 {
            "arguments"
        } else {
            "argument"
        };
        return Err(PyTypeError::new_err(format!(
            "{function}() missing {} required positional {argument}: {quoted_names}",
            missing.len()
        )));
    }

    Ok((
        arguments.map(|argument| argument.expect("all required tensor arguments were checked")),
        keyword_error,
    ))
}

fn bind_dtype_binary_arguments<'py>(
    operation: DTypeBinaryOperation,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<([BoundDTypeOperand<'py>; 2], Vec<ConsumedDTypeKeyword<'py>>)> {
    let names = operation.argument_names();
    let function = operation.name();

    if positional.len() > names.len() {
        return Err(PyTypeError::new_err(format!(
            "{function}() takes 2 positional arguments but {} were given",
            positional.len()
        )));
    }

    // Keep the original kwargs intact for mode dispatch. Popping from a
    // shallow copy uses the dictionary's stored hashes and identifies the
    // exact string-subclass entry consumed by each canonical parameter.
    let remaining_keywords = keywords.map(PyDictMethods::copy).transpose()?;
    let mut consumed_keywords = Vec::new();
    let sentinel = PyDict::new(positional.py()).into_any();
    let mut operands: [Option<BoundDTypeOperand<'py>>; 2] = std::array::from_fn(|_| None);

    // PyTorch binds and validates each schema slot before looking up the next
    // keyword. In particular, the second key must not run Python equality code
    // before the first operand has been accepted as a dtype or override.
    for (index, name) in names.iter().copied().enumerate() {
        let argument = if index < positional.len() {
            Some(ParsedCallArgument {
                value: positional.get_item(index)?,
                position: Some(index + 1),
            })
        } else if let Some(remaining_keywords) = remaining_keywords.as_ref() {
            let keys_before = remaining_keywords.keys();
            // PyTorch's generated argument parser suppresses lookup failures
            // here; a miss is reported as the complete remaining schema suffix.
            let value = pop_dtype_keyword(remaining_keywords, name, &sentinel)
                .ok()
                .flatten();
            if let Some(value) = value {
                let keys_after = remaining_keywords.keys();
                let key = keys_before
                    .iter()
                    .find(|key| !keys_after.iter().any(|remaining| remaining.is(key)))
                    .ok_or_else(|| {
                        PyRuntimeError::new_err(format!(
                            "{function}() could not identify a consumed keyword"
                        ))
                    })?;
                consumed_keywords.push(ConsumedDTypeKeyword {
                    key,
                    position: index,
                });
                Some(ParsedCallArgument {
                    value,
                    position: None,
                })
            } else {
                None
            }
        } else {
            None
        };

        let Some(argument) = argument else {
            // PyTorch reports the complete remaining schema suffix even when
            // a later argument in that suffix was supplied by keyword.
            let missing = &names[index..];
            let quoted_names = missing
                .iter()
                .map(|name| format!("\"{name}\""))
                .collect::<Vec<_>>()
                .join(", ");
            let argument = if missing.len() == 1 {
                "arguments"
            } else {
                "argument"
            };
            return Err(PyTypeError::new_err(format!(
                "{function}() missing {} required positional {argument}: {quoted_names}",
                missing.len()
            )));
        };
        operands[index] = Some(parse_dtype_operand(operation, name, &argument)?);
    }

    Ok((
        operands.map(|operand| {
            operand.expect("all required dtype operands were bound and parsed above")
        }),
        consumed_keywords,
    ))
}

fn pop_dtype_keyword<'py>(
    keywords: &Bound<'py, PyDict>,
    name: &str,
    sentinel: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let value = keywords.call_method1("pop", (name, sentinel))?;
    Ok((!value.is(sentinel)).then_some(value))
}

fn validate_dtype_binary_keywords(
    operation: DTypeBinaryOperation,
    positional_count: usize,
    keywords: Option<&Bound<'_, PyDict>>,
    consumed_keywords: &[ConsumedDTypeKeyword<'_>],
) -> PyResult<()> {
    let Some(keywords) = keywords else {
        return Ok(());
    };
    if keywords.len() <= consumed_keywords.len() {
        return Ok(());
    }

    let names = operation.argument_names();
    let function = operation.name();
    let mut invalid_keyword_arguments = false;
    for key in keywords.keys().iter() {
        let matched_position = if key.eq(names[0])? {
            Some(0)
        } else if key.eq(names[1])? {
            Some(1)
        } else {
            None
        };
        let consumed_position = consumed_keywords
            .iter()
            .find(|consumed| consumed.key.is(&key))
            .map(|consumed| consumed.position);
        if matched_position.is_some() && matched_position == consumed_position {
            continue;
        }

        if matched_position.is_some_and(|position| position >= positional_count) {
            // PyTorch defers this generic overload mismatch while it checks
            // later original keys for a positional duplicate or a more
            // specific unexpected-key error.
            invalid_keyword_arguments = true;
            continue;
        }

        let key = key.extract::<String>()?;
        let mut message = matched_position.map_or_else(
            || format!("{function}() got an unexpected keyword argument '{key}'"),
            |_| format!("{function}() got multiple values for argument '{key}'"),
        );
        if let Some(nul) = message.find('\0') {
            message.truncate(nul);
        }
        return Err(PyTypeError::new_err(message));
    }

    if invalid_keyword_arguments {
        Err(PyTypeError::new_err("invalid keyword arguments"))
    } else {
        Ok(())
    }
}

fn parse_dtype_operand<'py>(
    operation: DTypeBinaryOperation,
    name: &str,
    argument: &ParsedCallArgument<'py>,
) -> PyResult<BoundDTypeOperand<'py>> {
    if let Ok(dtype) = argument.value.cast::<PyDType>() {
        return Ok(BoundDTypeOperand::DType(dtype.try_borrow()?.inner()));
    }
    if let Some(probed) = probe_dtype_torch_function_override(&argument.value) {
        return Ok(BoundDTypeOperand::Override(probed));
    }

    let position = argument
        .position
        .map_or_else(String::new, |position| format!(" (position {position})"));
    let actual = python_type_name(&argument.value)?;
    let function = operation.name();
    Err(PyTypeError::new_err(format!(
        "{function}(): argument '{name}'{position} must be torch.dtype, not {actual}"
    )))
}

#[derive(Clone, Copy)]
enum LegacyBinaryInputKind {
    Tensor,
    TensorOrTorchFunction,
    Multiplication(MultiplicationOperation),
}

impl LegacyBinaryInputKind {
    const fn uses_multiply_overload_binding(self, argument_count: usize) -> bool {
        matches!(
            self,
            Self::Multiplication(MultiplicationOperation::Multiply)
        ) && argument_count <= 2
    }
}

fn bind_legacy_binary_arguments<'py>(
    function: &str,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
    input_kind: LegacyBinaryInputKind,
) -> PyResult<([ParsedCallArgument<'py>; 2], Option<PyErr>)> {
    if positional.len() > 2 {
        return Err(PyTypeError::new_err(format!(
            "{function}() takes 2 positional arguments but {} were given",
            positional.len()
        )));
    }
    let argument_count = positional
        .len()
        .saturating_add(keywords.map_or(0, PyDictMethods::len));
    let multiply_overload_binding = input_kind.uses_multiply_overload_binding(argument_count);

    let keyword_argument = |names: &[&str]| -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(keywords) = keywords else {
            return Ok(None);
        };
        for name in names {
            if let Some(value) = keywords.get_item(*name)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    };

    let input = if positional.is_empty() {
        keyword_argument(&["input", "x", "a", "x1"])?.map(|value| ParsedCallArgument {
            value,
            position: None,
        })
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(0)?,
            position: Some(1),
        })
    };
    let other = if positional.len() < 2 {
        keyword_argument(&["other", "x2"])?.map(|value| ParsedCallArgument {
            value,
            position: None,
        })
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(1)?,
            position: Some(2),
        })
    };

    let Some(input) = input else {
        if multiply_overload_binding {
            return Err(top_level_multiply_binding_error(positional, keywords)?);
        }
        return Err(PyTypeError::new_err(format!(
            "{function}() missing 2 required positional argument: \"input\", \"other\""
        )));
    };
    let Some(other) = other else {
        match input_kind {
            LegacyBinaryInputKind::Tensor => {
                parse_tensor_argument(function, "input", &input)?;
            }
            LegacyBinaryInputKind::TensorOrTorchFunction => {
                parse_tensor_or_torch_function_argument(function, "input", &input)?;
            }
            LegacyBinaryInputKind::Multiplication(operation) => {
                if multiply_overload_binding {
                    return Err(top_level_multiply_binding_error(positional, keywords)?);
                }
                parse_top_level_multiplication_operand(
                    operation, "input", &input, positional, keywords,
                )?;
            }
        }
        return Err(PyTypeError::new_err(format!(
            "{function}() missing 1 required positional arguments: \"other\""
        )));
    };

    let mut keyword_error = None;
    if let Some(keywords) = keywords {
        let keyword_arguments =
            usize::from(input.position.is_none()) + usize::from(other.position.is_none());
        if keywords.len() > keyword_arguments {
            for key in keywords.keys() {
                let key = key.extract::<String>()?;
                let position = match key.as_str() {
                    "input" => 0,
                    "other" => 1,
                    _ => {
                        keyword_error = Some(PyTypeError::new_err(format!(
                            "{function}() got an unexpected keyword argument '{key}'"
                        )));
                        break;
                    }
                };
                if position < positional.len() {
                    keyword_error = Some(PyTypeError::new_err(format!(
                        "{function}() got multiple values for argument '{key}'"
                    )));
                    break;
                }
            }
        }
    }

    Ok(([input, other], keyword_error))
}

fn bind_tensor_add_sub_method_arguments<'py>(
    operation: AddSubMethodOperation,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundTensorMethodAddSubArguments<'py>> {
    match operation {
        AddSubMethodOperation::Add => bind_tensor_add_method_arguments(positional, keywords),
        AddSubMethodOperation::Sub => bind_tensor_sub_method_arguments(positional, keywords),
        AddSubMethodOperation::Subtract => bind_tensor_subtract_method_arguments(
            SubtractionOperation::Subtract,
            positional,
            keywords,
        ),
    }
}

fn bind_tensor_add_method_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundTensorMethodAddSubArguments<'py>> {
    let (mut other, mut alpha) = bind_tensor_add_method_positional_arguments(positional)?;
    let mut x2_fallback = None;
    let mut keyword_error = None;

    if alpha.is_none()
        && tensor_add_method_keywords_include_other_alias(keywords)?
        && tensor_add_method_positional_other_can_be_alpha(other.as_ref())?
    {
        alpha = other.take();
    }

    if let Some(keywords) = keywords {
        for (key, value) in keywords {
            let key = key.extract::<String>()?;
            match key.as_str() {
                "other" if other.is_none() => {
                    if x2_fallback.is_some() {
                        keyword_error.get_or_insert_with(|| {
                            PyTypeError::new_err("add() got an unexpected keyword argument 'x2'")
                        });
                    }
                    other = Some(ParsedCallArgument {
                        value,
                        position: None,
                    });
                }
                "other" => {
                    keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err("add() got multiple values for argument 'other'")
                    });
                }
                "x2" if other.is_none() && x2_fallback.is_none() => {
                    x2_fallback = Some(ParsedCallArgument {
                        value,
                        position: None,
                    });
                }
                "x2" => {
                    keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err("add() got an unexpected keyword argument 'x2'")
                    });
                }
                "alpha" if alpha.is_none() => {
                    alpha = Some(ParsedCallArgument {
                        value,
                        position: None,
                    });
                }
                "alpha" => {
                    keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err("add() got multiple values for argument 'alpha'")
                    });
                }
                _ => {
                    keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err(format!(
                            "add() got an unexpected keyword argument '{key}'"
                        ))
                    });
                }
            }
        }
    }

    let Some(other) = other.or(x2_fallback) else {
        if positional.is_empty() && keywords.is_none_or(PyDictMethods::is_empty) {
            return Err(tensor_add_method_binding_error(positional, keywords)?);
        }
        return Err(PyTypeError::new_err(
            "add() missing 1 required positional arguments: \"other\"",
        ));
    };
    Ok((other, alpha, keyword_error))
}

fn bind_tensor_add_method_positional_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
) -> PyResult<(
    Option<ParsedCallArgument<'py>>,
    Option<ParsedCallArgument<'py>>,
)> {
    match positional.len() {
        0 => Ok((None, None)),
        1 => Ok((
            Some(ParsedCallArgument {
                value: positional.get_item(0)?,
                position: Some(1),
            }),
            None,
        )),
        2 => {
            let first = positional.get_item(0)?;
            let second = positional.get_item(1)?;
            let first_is_scalar = is_real_arithmetic_scalar(&first)?;
            let first_has_override =
                !first_is_scalar && probe_torch_function_override(&first).is_some();
            let second_is_supported_other = second.is_instance_of::<PyTensor>()
                || is_real_arithmetic_scalar(&second)?
                || probe_torch_function_override(&second).is_some();
            if (first_is_scalar || first_has_override) && second_is_supported_other {
                return Ok((
                    Some(ParsedCallArgument {
                        value: second,
                        position: Some(2),
                    }),
                    Some(ParsedCallArgument {
                        value: first,
                        position: Some(1),
                    }),
                ));
            }
            Err(PyTypeError::new_err(format!(
                "add() takes 1 positional argument but {} were given",
                positional.len()
            )))
        }
        _ => Err(PyTypeError::new_err(format!(
            "add() takes 1 positional argument but {} were given",
            positional.len()
        ))),
    }
}

fn tensor_add_method_keywords_include_other_alias(
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<bool> {
    let Some(keywords) = keywords else {
        return Ok(false);
    };
    for key in keywords.keys() {
        let key = key.extract::<String>()?;
        if matches!(key.as_str(), "other" | "x2") {
            return Ok(true);
        }
    }
    Ok(false)
}

fn tensor_add_method_positional_other_can_be_alpha(
    other: Option<&ParsedCallArgument<'_>>,
) -> PyResult<bool> {
    let Some(other) = other else {
        return Ok(false);
    };
    if other.position != Some(1) {
        return Ok(false);
    }
    if is_real_arithmetic_scalar(&other.value)? {
        return Ok(true);
    }
    Ok(probe_torch_function_override(&other.value).is_some())
}

fn bind_tensor_sub_method_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundTensorMethodSubtractionArguments<'py>> {
    let (mut other, mut alpha) = bind_tensor_sub_method_positional_arguments(positional)?;
    let mut x2_fallback = None;
    let mut keyword_error = None;

    if let Some(keywords) = keywords {
        for (key, value) in keywords {
            let key = key.extract::<String>()?;
            match key.as_str() {
                "other" if other.is_none() => {
                    if x2_fallback.is_some() {
                        keyword_error.get_or_insert_with(|| {
                            PyTypeError::new_err("sub() got an unexpected keyword argument 'x2'")
                        });
                    }
                    other = Some(ParsedCallArgument {
                        value,
                        position: None,
                    });
                }
                "other" => {
                    keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err("sub() got multiple values for argument 'other'")
                    });
                }
                "x2" if other.is_none() && x2_fallback.is_none() => {
                    x2_fallback = Some(ParsedCallArgument {
                        value,
                        position: None,
                    });
                }
                "x2" => {
                    keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err("sub() got an unexpected keyword argument 'x2'")
                    });
                }
                "alpha" if alpha.is_none() => {
                    alpha = Some(ParsedCallArgument {
                        value,
                        position: None,
                    });
                }
                "alpha" => {
                    keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err("sub() got multiple values for argument 'alpha'")
                    });
                }
                _ => {
                    keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err(format!(
                            "sub() got an unexpected keyword argument '{key}'"
                        ))
                    });
                }
            }
        }
    }

    let Some(other) = other.or(x2_fallback) else {
        if positional.is_empty() && keywords.is_none_or(PyDictMethods::is_empty) {
            return Err(tensor_sub_method_binding_error(positional, keywords)?);
        }
        return Err(PyTypeError::new_err(
            "sub() missing 1 required positional arguments: \"other\"",
        ));
    };
    Ok((other, alpha, keyword_error))
}

fn bind_tensor_sub_method_positional_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
) -> PyResult<(
    Option<ParsedCallArgument<'py>>,
    Option<ParsedCallArgument<'py>>,
)> {
    match positional.len() {
        0 => Ok((None, None)),
        1 => Ok((
            Some(ParsedCallArgument {
                value: positional.get_item(0)?,
                position: Some(1),
            }),
            None,
        )),
        2 => {
            let first = positional.get_item(0)?;
            let second = positional.get_item(1)?;
            let first_is_scalar = is_real_arithmetic_scalar(&first)?;
            let first_has_override =
                !first_is_scalar && probe_torch_function_override(&first).is_some();
            let second_is_tensor_like = second.is_instance_of::<PyTensor>()
                || probe_torch_function_override(&second).is_some();
            if (first_is_scalar || first_has_override) && second_is_tensor_like {
                return Ok((
                    Some(ParsedCallArgument {
                        value: second,
                        position: Some(2),
                    }),
                    Some(ParsedCallArgument {
                        value: first,
                        position: Some(1),
                    }),
                ));
            }
            if (first_is_scalar || first_has_override) && is_real_arithmetic_scalar(&second)? {
                return Ok((
                    Some(ParsedCallArgument {
                        value: first,
                        position: Some(1),
                    }),
                    Some(ParsedCallArgument {
                        value: second,
                        position: Some(2),
                    }),
                ));
            }
            Err(PyTypeError::new_err(format!(
                "sub() takes 1 positional argument but {} were given",
                positional.len()
            )))
        }
        _ => Err(PyTypeError::new_err(format!(
            "sub() takes 1 positional argument but {} were given",
            positional.len()
        ))),
    }
}

fn bind_tensor_subtract_method_arguments<'py>(
    operation: SubtractionOperation,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundTensorMethodSubtractionArguments<'py>> {
    let (mut other, mut alpha) =
        bind_tensor_subtract_method_positional_arguments(operation, positional, keywords)?;
    let mut x2_fallback = None;
    let mut keyword_error = None;

    if let Some(keywords) = keywords {
        for (key, value) in keywords {
            let key = key.extract::<String>()?;
            match key.as_str() {
                "other" if other.is_none() => {
                    if x2_fallback.is_some() {
                        keyword_error.get_or_insert(tensor_subtract_binding_error(
                            operation,
                            positional,
                            Some(keywords),
                            Some(&SubtractBindingMismatch::IncorrectKeyword("x2")),
                        )?);
                    }
                    other = Some(ParsedCallArgument {
                        value,
                        position: None,
                    });
                }
                "other" => {
                    keyword_error.get_or_insert(tensor_subtract_binding_error(
                        operation,
                        positional,
                        Some(keywords),
                        Some(&SubtractBindingMismatch::IncorrectKeyword("other")),
                    )?);
                }
                "x2" if other.is_none() && x2_fallback.is_none() => {
                    x2_fallback = Some(ParsedCallArgument {
                        value,
                        position: None,
                    });
                }
                "x2" => {
                    keyword_error.get_or_insert(tensor_subtract_binding_error(
                        operation,
                        positional,
                        Some(keywords),
                        Some(&SubtractBindingMismatch::IncorrectKeyword("x2")),
                    )?);
                }
                "alpha" if alpha.is_none() => {
                    alpha = Some(ParsedCallArgument {
                        value,
                        position: None,
                    });
                }
                "alpha" => {
                    keyword_error.get_or_insert(tensor_subtract_binding_error(
                        operation,
                        positional,
                        Some(keywords),
                        Some(&SubtractBindingMismatch::IncorrectKeyword("alpha")),
                    )?);
                }
                _ => {
                    keyword_error.get_or_insert(tensor_subtract_binding_error(
                        operation,
                        positional,
                        Some(keywords),
                        Some(&SubtractBindingMismatch::IncorrectKeyword(&key)),
                    )?);
                }
            }
        }
    }

    let Some(other) = other.or(x2_fallback) else {
        return Err(tensor_subtract_binding_error(
            operation, positional, keywords, None,
        )?);
    };
    Ok((other, alpha, keyword_error))
}

fn bind_tensor_subtract_method_positional_arguments<'py>(
    operation: SubtractionOperation,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<(
    Option<ParsedCallArgument<'py>>,
    Option<ParsedCallArgument<'py>>,
)> {
    match positional.len() {
        0 => Ok((None, None)),
        1 => Ok((
            Some(ParsedCallArgument {
                value: positional.get_item(0)?,
                position: Some(1),
            }),
            None,
        )),
        2 => {
            let first = positional.get_item(0)?;
            let second = positional.get_item(1)?;
            if (is_real_arithmetic_scalar(&first)?
                || probe_torch_function_override(&first).is_some())
                && (is_real_arithmetic_scalar(&second)?
                    || probe_torch_function_override(&second).is_some())
            {
                return Ok((
                    Some(ParsedCallArgument {
                        value: first,
                        position: Some(1),
                    }),
                    Some(ParsedCallArgument {
                        value: second,
                        position: Some(2),
                    }),
                ));
            }
            Err(tensor_subtract_binding_error(
                operation,
                positional,
                keywords,
                Some(&SubtractBindingMismatch::InvalidPositionalOverload),
            )?)
        }
        _ => Err(tensor_subtract_binding_error(
            operation,
            positional,
            keywords,
            Some(&SubtractBindingMismatch::InvalidPositionalOverload),
        )?),
    }
}

fn bind_top_level_add_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundTopLevelAdditionArguments<'py>> {
    if positional.len() > 2 {
        return Err(PyTypeError::new_err(format!(
            "add() takes 2 positional arguments but {} were given",
            positional.len()
        )));
    }

    let keyword_argument = |names: &[&str]| -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(keywords) = keywords else {
            return Ok(None);
        };
        for name in names {
            if let Some(value) = keywords.get_item(*name)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    };

    let input = if positional.is_empty() {
        keyword_argument(&["input", "x", "a", "x1"])?.map(|value| ParsedCallArgument {
            value,
            position: None,
        })
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(0)?,
            position: Some(1),
        })
    };
    let other = if positional.len() < 2 {
        keyword_argument(&["other", "x2"])?.map(|value| ParsedCallArgument {
            value,
            position: None,
        })
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(1)?,
            position: Some(2),
        })
    };
    let alpha = keyword_argument(&["alpha"])?.map(|value| ParsedCallArgument {
        value,
        position: None,
    });
    let out = keyword_argument(&["out"])?.map(|value| ParsedCallArgument {
        value,
        position: None,
    });

    let Some(input) = input else {
        return Err(top_level_add_binding_error(positional, keywords)?);
    };
    let Some(other) = other else {
        parse_top_level_add_operand("input", &input)?;
        return Err(top_level_add_binding_error(positional, keywords)?);
    };

    let bound_keyword_count = usize::from(input.position.is_none())
        + usize::from(other.position.is_none())
        + usize::from(alpha.is_some())
        + usize::from(out.is_some());
    let keyword_error =
        bind_top_level_add_keyword_error(positional, keywords, bound_keyword_count)?;

    Ok(([input, other], alpha, out, keyword_error))
}

fn bind_top_level_add_keyword_error(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
    bound_keyword_count: usize,
) -> PyResult<Option<PyErr>> {
    let Some(keywords) = keywords else {
        return Ok(None);
    };
    if keywords.len() <= bound_keyword_count {
        return Ok(None);
    }

    for key in keywords.keys() {
        let key = key.extract::<String>()?;
        let position = match key.as_str() {
            "input" => 0,
            "other" => 1,
            "alpha" | "out" => usize::MAX,
            _ => {
                return Ok(Some(PyTypeError::new_err(format!(
                    "add() got an unexpected keyword argument '{key}'"
                ))));
            }
        };
        if position < positional.len() {
            return Ok(Some(PyTypeError::new_err(format!(
                "add() got multiple values for argument '{key}'"
            ))));
        }
    }
    Ok(None)
}

fn bind_top_level_mm_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundTopLevelMmArguments<'py>> {
    if positional.len() > 2 {
        return Err(top_level_mm_binding_error(positional, keywords)?);
    }

    let keyword_argument = |names: &[&str]| -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(keywords) = keywords else {
            return Ok(None);
        };
        for name in names {
            if let Some(value) = keywords.get_item(*name)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    };

    let input = if positional.is_empty() {
        keyword_argument(&["input", "x", "a", "x1"])?.map(|value| ParsedCallArgument {
            value,
            position: None,
        })
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(0)?,
            position: Some(1),
        })
    };
    let mat2 = if positional.len() < 2 {
        keyword_argument(&["mat2"])?.map(|value| ParsedCallArgument {
            value,
            position: None,
        })
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(1)?,
            position: Some(2),
        })
    };
    let out = keyword_argument(&["out"])?.map(|value| ParsedCallArgument {
        value,
        position: None,
    });

    let Some(input) = input else {
        return Err(PyTypeError::new_err(
            "mm() missing 2 required positional argument: \"input\", \"mat2\"",
        ));
    };
    let Some(mat2) = mat2 else {
        parse_exact_native_mm_tensor_or_torch_function_argument("input", &input)?;
        return Err(PyTypeError::new_err(
            "mm() missing 1 required positional arguments: \"mat2\"",
        ));
    };

    let bound_keyword_count = usize::from(input.position.is_none())
        + usize::from(mat2.position.is_none())
        + usize::from(out.is_some());
    let keyword_error = bind_top_level_mm_keyword_error(positional, keywords, bound_keyword_count)?;

    Ok(([input, mat2], out, keyword_error))
}

fn bind_top_level_mm_keyword_error(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
    bound_keyword_count: usize,
) -> PyResult<Option<PyErr>> {
    let Some(keywords) = keywords else {
        return Ok(None);
    };
    if keywords.len() <= bound_keyword_count {
        return Ok(None);
    }

    for key in keywords.keys() {
        let key = key.extract::<String>()?;
        let position = match key.as_str() {
            "input" | "x" | "a" | "x1" => 0,
            "mat2" => 1,
            "out" => usize::MAX,
            _ => {
                return Ok(Some(top_level_mm_binding_error(
                    positional,
                    Some(keywords),
                )?));
            }
        };
        if position < positional.len() {
            return Ok(Some(top_level_mm_binding_error(
                positional,
                Some(keywords),
            )?));
        }
    }
    Ok(None)
}

fn bind_top_level_subtraction_arguments<'py>(
    operation: SubtractionOperation,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundTopLevelSubtractionArguments<'py>> {
    let function = operation.name();
    if matches!(operation, SubtractionOperation::Subtract) {
        if positional.len() > 3
            || (positional.len() == 3 && keywords.is_some_and(|keywords| !keywords.is_empty()))
        {
            return Err(PyTypeError::new_err(format!(
                "{function}() takes 2 positional arguments but {} were given",
                positional.len()
            )));
        }
    } else if positional.len() > 2 {
        return Err(PyTypeError::new_err(format!(
            "{function}() takes 2 positional arguments but {} were given",
            positional.len()
        )));
    }

    if positional.len() == 3 {
        return bind_top_level_subtract_positional_scalar_overload(positional, keywords);
    }

    let keyword_argument = |names: &[&str]| -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(keywords) = keywords else {
            return Ok(None);
        };
        for name in names {
            if let Some(value) = keywords.get_item(*name)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    };

    let input = if positional.is_empty() {
        keyword_argument(&["input", "x", "a", "x1"])?.map(|value| ParsedCallArgument {
            value,
            position: None,
        })
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(0)?,
            position: Some(1),
        })
    };
    let other = if positional.len() < 2 {
        keyword_argument(&["other", "x2"])?.map(|value| ParsedCallArgument {
            value,
            position: None,
        })
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(1)?,
            position: Some(2),
        })
    };
    let alpha = keyword_argument(&["alpha"])?.map(|value| ParsedCallArgument {
        value,
        position: None,
    });
    let out = keyword_argument(&["out"])?.map(|value| ParsedCallArgument {
        value,
        position: None,
    });

    let Some(input) = input else {
        if matches!(operation, SubtractionOperation::Subtract) {
            return Err(top_level_subtract_binding_error(
                operation, positional, keywords, None,
            )?);
        }
        return Err(top_level_subtraction_binding_error(
            operation, positional, keywords,
        )?);
    };
    let Some(other) = other else {
        parse_top_level_subtraction_input(operation, &input, positional, keywords)?;
        if matches!(operation, SubtractionOperation::Subtract) {
            return Err(top_level_subtract_binding_error(
                operation, positional, keywords, None,
            )?);
        }
        return Err(top_level_subtraction_binding_error(
            operation, positional, keywords,
        )?);
    };

    let bound_keyword_count = usize::from(input.position.is_none())
        + usize::from(other.position.is_none())
        + usize::from(alpha.is_some())
        + usize::from(out.is_some());
    let keyword_error = bind_top_level_subtraction_keyword_error(
        operation,
        positional,
        keywords,
        bound_keyword_count,
    )?;

    Ok(([input, other], alpha, out, keyword_error))
}

fn bind_top_level_subtract_positional_scalar_overload<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundTopLevelSubtractionArguments<'py>> {
    let input = ParsedCallArgument {
        value: positional.get_item(0)?,
        position: Some(1),
    };
    let other = ParsedCallArgument {
        value: positional.get_item(1)?,
        position: Some(2),
    };
    let alpha = ParsedCallArgument {
        value: positional.get_item(2)?,
        position: Some(3),
    };
    if probe_torch_function_override(&input.value).is_none()
        && probe_torch_function_override(&other.value).is_none()
        && probe_torch_function_override(&alpha.value).is_none()
        && (!input.value.is_exact_instance_of::<PyTensor>()
            || !is_real_arithmetic_scalar(&other.value)?
            || !is_real_arithmetic_scalar(&alpha.value)?)
    {
        return Err(top_level_subtract_binding_error(
            SubtractionOperation::Subtract,
            positional,
            keywords,
            Some(&SubtractBindingMismatch::InvalidPositionalOverload),
        )?);
    }
    Ok(([input, other], Some(alpha), None, None))
}

fn bind_top_level_subtraction_keyword_error(
    operation: SubtractionOperation,
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
    bound_keyword_count: usize,
) -> PyResult<Option<PyErr>> {
    let Some(keywords) = keywords else {
        return Ok(None);
    };
    if keywords.len() <= bound_keyword_count {
        return Ok(None);
    }

    let function = operation.name();
    for key in keywords.keys() {
        let key = key.extract::<String>()?;
        let position = match key.as_str() {
            "input" => 0,
            "other" => 1,
            "alpha" | "out" => usize::MAX,
            _ => {
                return if matches!(operation, SubtractionOperation::Subtract) {
                    top_level_subtract_binding_error(
                        operation,
                        positional,
                        Some(keywords),
                        Some(&SubtractBindingMismatch::IncorrectKeyword(&key)),
                    )
                    .map(Some)
                } else {
                    Ok(Some(PyTypeError::new_err(format!(
                        "{function}() got an unexpected keyword argument '{key}'"
                    ))))
                };
            }
        };
        if position < positional.len() {
            return if matches!(operation, SubtractionOperation::Subtract) {
                top_level_subtract_binding_error(
                    operation,
                    positional,
                    Some(keywords),
                    Some(&SubtractBindingMismatch::IncorrectKeyword(&key)),
                )
                .map(Some)
            } else {
                Ok(Some(PyTypeError::new_err(format!(
                    "{function}() got multiple values for argument '{key}'"
                ))))
            };
        }
    }
    Ok(None)
}

fn bind_top_level_division_arguments<'py>(
    operation: DivisionOperation,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundTopLevelDivisionArguments<'py>> {
    if positional.len() > 2 {
        return Err(top_level_division_binding_error(
            operation, positional, keywords,
        )?);
    }

    let keyword_argument = |names: &[&str]| -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(keywords) = keywords else {
            return Ok(None);
        };
        for name in names {
            if let Some(value) = keywords.get_item(*name)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    };

    let input = if positional.is_empty() {
        keyword_argument(&["input", "x", "a", "x1"])?.map(|value| ParsedCallArgument {
            value,
            position: None,
        })
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(0)?,
            position: Some(1),
        })
    };
    let other = if positional.len() < 2 {
        keyword_argument(&["other", "x2"])?.map(|value| ParsedCallArgument {
            value,
            position: None,
        })
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(1)?,
            position: Some(2),
        })
    };
    let rounding_mode = keyword_argument(&["rounding_mode"])?.map(|value| ParsedCallArgument {
        value,
        position: None,
    });
    let out = keyword_argument(&["out"])?.map(|value| ParsedCallArgument {
        value,
        position: None,
    });

    let Some(input) = input else {
        return Err(top_level_division_binding_error(
            operation, positional, keywords,
        )?);
    };
    let Some(other) = other else {
        parse_top_level_division_input(operation, &input, positional, keywords)?;
        return Err(top_level_division_binding_error(
            operation, positional, keywords,
        )?);
    };

    let bound_keyword_count = usize::from(input.position.is_none())
        + usize::from(other.position.is_none())
        + usize::from(rounding_mode.is_some())
        + usize::from(out.is_some());
    let keyword_error = bind_top_level_division_keyword_error(
        operation,
        positional,
        keywords,
        bound_keyword_count,
    )?;

    Ok(([input, other], rounding_mode, out, keyword_error))
}

fn bind_top_level_division_keyword_error(
    operation: DivisionOperation,
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
    bound_keyword_count: usize,
) -> PyResult<Option<PyErr>> {
    let Some(keywords) = keywords else {
        return Ok(None);
    };
    if keywords.len() <= bound_keyword_count {
        return Ok(None);
    }

    let repeated_input_alias = usize::from(keywords.contains("input")?)
        + usize::from(keywords.contains("x")?)
        + usize::from(keywords.contains("a")?)
        + usize::from(keywords.contains("x1")?)
        > 1;
    let repeated_other_alias =
        usize::from(keywords.contains("other")?) + usize::from(keywords.contains("x2")?) > 1;
    for key in keywords.keys() {
        let key = key.extract::<String>()?;
        let position = match key.as_str() {
            "input" | "x" | "a" | "x1" => 0,
            "other" | "x2" => 1,
            "rounding_mode" | "out" => usize::MAX,
            _ => {
                return top_level_division_binding_error(operation, positional, Some(keywords))
                    .map(Some);
            }
        };
        if repeated_input_alias || repeated_other_alias {
            return top_level_division_binding_error(operation, positional, Some(keywords))
                .map(Some);
        }
        if position < positional.len() {
            return top_level_division_binding_error(operation, positional, Some(keywords))
                .map(Some);
        }
    }
    Ok(None)
}

fn bind_top_level_reshape_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<([ParsedCallArgument<'py>; 2], Option<PyErr>)> {
    if positional.len() > 2 {
        return Err(PyTypeError::new_err(format!(
            "reshape() takes 2 positional arguments but {} were given",
            positional.len()
        )));
    }

    let keyword_argument = |names: &[&str]| -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(keywords) = keywords else {
            return Ok(None);
        };
        for name in names {
            if let Some(value) = keywords.get_item(*name)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    };

    let input = if positional.is_empty() {
        keyword_argument(&["input", "x", "a", "x1"])?.map(|value| ParsedCallArgument {
            value,
            position: None,
        })
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(0)?,
            position: Some(1),
        })
    };
    let shape = if positional.len() < 2 {
        keyword_argument(&["shape"])?.map(|value| ParsedCallArgument {
            value,
            position: None,
        })
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(1)?,
            position: Some(2),
        })
    };

    let Some(input) = input else {
        return Err(PyTypeError::new_err(
            "reshape() missing 2 required positional argument: \"input\", \"shape\"",
        ));
    };
    let Some(shape) = shape else {
        parse_tensor_or_torch_function_argument("reshape", "input", &input)?;
        return Err(PyTypeError::new_err(
            "reshape() missing 1 required positional arguments: \"shape\"",
        ));
    };

    let mut keyword_error = None;
    if let Some(keywords) = keywords {
        let keyword_arguments =
            usize::from(input.position.is_none()) + usize::from(shape.position.is_none());
        if keywords.len() > keyword_arguments {
            for key in keywords.keys() {
                let key = key.extract::<String>()?;
                let position = match key.as_str() {
                    "input" => 0,
                    "shape" => 1,
                    _ => {
                        keyword_error = Some(PyTypeError::new_err(format!(
                            "reshape() got an unexpected keyword argument '{key}'"
                        )));
                        break;
                    }
                };
                if position < positional.len() {
                    keyword_error = Some(PyTypeError::new_err(format!(
                        "reshape() got multiple values for argument '{key}'"
                    )));
                    break;
                }
            }
        }
    }

    Ok(([input, shape], keyword_error))
}

fn bind_top_level_permute_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<([ParsedCallArgument<'py>; 2], Option<PyErr>)> {
    if positional.len() > 2 {
        return Err(PyTypeError::new_err(format!(
            "permute() takes 2 positional arguments but {} were given",
            positional.len()
        )));
    }

    let keyword_argument = |names: &[&str]| -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(keywords) = keywords else {
            return Ok(None);
        };
        for name in names {
            if let Some(value) = keywords.get_item(*name)? {
                return Ok(Some(value));
            }
        }
        Ok(None)
    };

    let input = if positional.is_empty() {
        keyword_argument(&["input", "x", "a", "x1"])?.map(|value| ParsedCallArgument {
            value,
            position: None,
        })
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(0)?,
            position: Some(1),
        })
    };
    let dimensions = if positional.len() < 2 {
        keyword_argument(&["dims"])?.map(|value| ParsedCallArgument {
            value,
            position: None,
        })
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(1)?,
            position: Some(2),
        })
    };

    let Some(input) = input else {
        return Err(PyTypeError::new_err(
            "permute() missing 2 required positional argument: \"input\", \"dims\"",
        ));
    };
    let Some(dimensions) = dimensions else {
        parse_tensor_argument("permute", "input", &input)?;
        return Err(PyTypeError::new_err(
            "permute() missing 1 required positional arguments: \"dims\"",
        ));
    };

    let mut keyword_error = None;
    if let Some(keywords) = keywords {
        let keyword_arguments =
            usize::from(input.position.is_none()) + usize::from(dimensions.position.is_none());
        if keywords.len() > keyword_arguments {
            for key in keywords.keys() {
                let key = key.extract::<String>()?;
                let position = match key.as_str() {
                    "input" => 0,
                    "dims" => 1,
                    _ => {
                        keyword_error = Some(PyTypeError::new_err(format!(
                            "permute() got an unexpected keyword argument '{key}'"
                        )));
                        break;
                    }
                };
                if position < positional.len() {
                    keyword_error = Some(PyTypeError::new_err(format!(
                        "permute() got multiple values for argument '{key}'"
                    )));
                    break;
                }
            }
        }
    }

    Ok(([input, dimensions], keyword_error))
}

fn parse_tensor_argument<'a, 'py>(
    function: &str,
    argument: &str,
    value: &'a ParsedCallArgument<'py>,
) -> PyResult<&'a Bound<'py, PyTensor>> {
    let Ok(tensor) = value.value.cast::<PyTensor>() else {
        let position = value
            .position
            .map_or_else(String::new, |position| format!(" (position {position})"));
        let actual = python_type_name(&value.value)?;
        return Err(PyTypeError::new_err(format!(
            "{function}(): argument '{argument}'{position} must be Tensor, not {actual}"
        )));
    };
    Ok(tensor)
}

fn parse_exact_native_like_factory_tensor_argument<'a, 'py>(
    function: &str,
    argument: &str,
    value: &'a ParsedCallArgument<'py>,
) -> PyResult<&'a Bound<'py, PyTensor>> {
    if !value.value.is_exact_instance_of::<PyTensor>() {
        if value.value.is_instance_of::<PyTensor>() {
            return Err(like_factory_unsupported_native_input(function));
        }
        return parse_tensor_argument(function, argument, value);
    }
    Ok(value.value.cast::<PyTensor>()?)
}

fn like_factory_input_shape(function: &str, input: &Bound<'_, PyTensor>) -> PyResult<Vec<usize>> {
    let tensor = input.try_borrow()?;
    validate_like_factory_native_input(function, &tensor.inner)?;
    let mut shape = try_size_vector(tensor.inner.shape().len())?;
    shape.extend_from_slice(tensor.inner.shape());
    Ok(shape)
}

fn validate_like_factory_native_input(function: &str, input: &CoreTensor) -> PyResult<()> {
    if input.dtype() == DType::Float32
        && input.device() == Device::Cpu
        && input.is_contiguous_with_memory_format(MemoryFormat::Contiguous)
        && like_factory_has_canonical_row_major_strides(input)
        && input.suggested_memory_format() == MemoryFormat::Contiguous
    {
        return Ok(());
    }
    Err(like_factory_unsupported_native_input(function))
}

fn like_factory_has_canonical_row_major_strides(input: &CoreTensor) -> bool {
    let shape = input.shape();
    let strides = input.stride();
    if shape.len() != strides.len() {
        return false;
    }

    let mut expected_stride = 1_usize;
    for axis in (0..shape.len()).rev() {
        if strides[axis] != expected_stride {
            return false;
        }
        if axis > 0 {
            let Some(next_stride) = expected_stride
                .checked_mul(shape[axis].max(1))
                .filter(|product| *product <= isize::MAX.unsigned_abs())
            else {
                return false;
            };
            expected_stride = next_stride;
        }
    }
    true
}

fn like_factory_unsupported_native_input(function: &str) -> PyErr {
    PyNotImplementedError::new_err(format!(
        "{function}(): only exact native CPU float32 row-major contiguous Tensor inputs are supported"
    ))
}

fn parse_tensor_or_torch_function_argument<'py>(
    function: &str,
    argument: &str,
    value: &ParsedCallArgument<'py>,
) -> PyResult<BoundTensorOrTorchFunction<'py>> {
    if let Ok(tensor) = value.value.cast::<PyTensor>() {
        return Ok(BoundTensorOrTorchFunction::Tensor(tensor.clone()));
    }
    if let Some(probed) = probe_torch_function_override(&value.value) {
        return Ok(BoundTensorOrTorchFunction::Override(probed));
    }
    parse_tensor_argument(function, argument, value)
        .map(|tensor| BoundTensorOrTorchFunction::Tensor(tensor.clone()))
}

fn parse_exact_native_mm_tensor_or_torch_function_argument<'py>(
    argument: &str,
    value: &ParsedCallArgument<'py>,
) -> PyResult<BoundTensorOrTorchFunction<'py>> {
    if value.value.is_exact_instance_of::<PyTensor>() {
        return Ok(BoundTensorOrTorchFunction::Tensor(
            value.value.cast::<PyTensor>()?.clone(),
        ));
    }
    if let Some(probed) = probe_torch_function_override(&value.value) {
        return Ok(BoundTensorOrTorchFunction::Override(probed));
    }
    if value.value.is_instance_of::<PyTensor>() {
        return Err(mm_unsupported_native_input());
    }
    parse_tensor_argument("mm", argument, value)
        .map(|tensor| BoundTensorOrTorchFunction::Tensor(tensor.clone()))
}

fn parse_top_level_mm_out(
    out: Option<ParsedCallArgument<'_>>,
) -> PyResult<Option<BoundTensorOrTorchFunction<'_>>> {
    let Some(out) = out else {
        return Ok(None);
    };
    if out.value.is_none() {
        return Ok(None);
    }
    parse_exact_native_mm_tensor_or_torch_function_argument("out", &out).map(Some)
}

fn parse_exact_native_add_tensor_or_torch_function_argument<'py>(
    argument: &str,
    value: &ParsedCallArgument<'py>,
) -> PyResult<BoundTensorOrTorchFunction<'py>> {
    if value.value.is_exact_instance_of::<PyTensor>() {
        return Ok(BoundTensorOrTorchFunction::Tensor(
            value.value.cast::<PyTensor>()?.clone(),
        ));
    }
    if let Some(probed) = probe_torch_function_override(&value.value) {
        return Ok(BoundTensorOrTorchFunction::Override(probed));
    }
    if value.value.is_instance_of::<PyTensor>() {
        return Err(addition_unsupported_native_input());
    }
    parse_tensor_argument("add", argument, value)
        .map(|tensor| BoundTensorOrTorchFunction::Tensor(tensor.clone()))
}

fn parse_exact_native_tensor_or_torch_function_argument<'py>(
    operation: SubtractionOperation,
    argument: &str,
    value: &ParsedCallArgument<'py>,
) -> PyResult<BoundTensorOrTorchFunction<'py>> {
    if value.value.is_exact_instance_of::<PyTensor>() {
        return Ok(BoundTensorOrTorchFunction::Tensor(
            value.value.cast::<PyTensor>()?.clone(),
        ));
    }
    if let Some(probed) = probe_torch_function_override(&value.value) {
        return Ok(BoundTensorOrTorchFunction::Override(probed));
    }
    if value.value.is_instance_of::<PyTensor>() {
        return Err(subtraction_unsupported_native_input(operation));
    }
    parse_tensor_argument(operation.name(), argument, value)
        .map(|tensor| BoundTensorOrTorchFunction::Tensor(tensor.clone()))
}

fn parse_top_level_add_operand<'py>(
    argument: &str,
    value: &ParsedCallArgument<'py>,
) -> PyResult<BoundAddOperand<'py>> {
    if value.value.is_exact_instance_of::<PyTensor>() {
        return Ok(BoundAddOperand::Tensor(
            value.value.cast::<PyTensor>()?.clone(),
        ));
    }
    if let Some(probed) = probe_torch_function_override(&value.value) {
        return Ok(BoundAddOperand::Override(probed));
    }
    if value.value.is_instance_of::<PyTensor>() {
        return Err(addition_unsupported_native_input());
    }
    if is_real_arithmetic_scalar(&value.value)? {
        return Ok(BoundAddOperand::Scalar(value.value.clone()));
    }
    parse_tensor_argument("add", argument, value)?;
    unreachable!("unsupported addition operands were rejected by parse_tensor_argument")
}

fn parse_top_level_add_alpha<'py>(
    alpha: Option<&ParsedCallArgument<'py>>,
) -> PyResult<BoundSubAlpha<'py>> {
    let Some(alpha) = alpha else {
        return Ok(BoundSubAlpha::Default);
    };
    if let Some(probed) = probe_torch_function_override(&alpha.value) {
        return Ok(BoundSubAlpha::Override(probed));
    }
    if is_boolean_arithmetic_scalar(&alpha.value)? {
        return Ok(BoundSubAlpha::PythonBool);
    }
    let Some(scalar) = parse_arithmetic_scalar(&alpha.value)? else {
        let actual = python_type_name(&alpha.value)?;
        return Err(PyTypeError::new_err(format!(
            "add(): argument 'alpha' must be Number, not {actual}",
        )));
    };
    if scalar.is_one() {
        Ok(BoundSubAlpha::Default)
    } else {
        Ok(BoundSubAlpha::NonDefault)
    }
}

fn parse_top_level_add_out(
    out: Option<ParsedCallArgument<'_>>,
) -> PyResult<Option<BoundTensorOrTorchFunction<'_>>> {
    let Some(out) = out else {
        return Ok(None);
    };
    if out.value.is_none() {
        return Ok(None);
    }
    parse_exact_native_add_tensor_or_torch_function_argument("out", &out).map(Some)
}

fn parse_top_level_subtraction_input<'py>(
    operation: SubtractionOperation,
    value: &ParsedCallArgument<'py>,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundSubOperand<'py>> {
    parse_top_level_subtraction_operand(operation, "input", value, positional, keywords)
}

fn parse_top_level_subtraction_other<'py>(
    operation: SubtractionOperation,
    value: &ParsedCallArgument<'py>,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundSubOperand<'py>> {
    parse_top_level_subtraction_operand(operation, "other", value, positional, keywords)
}

fn parse_top_level_subtraction_operand<'py>(
    operation: SubtractionOperation,
    argument: &str,
    value: &ParsedCallArgument<'py>,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundSubOperand<'py>> {
    if value.value.is_exact_instance_of::<PyTensor>() {
        return Ok(BoundSubOperand::Tensor(
            value.value.cast::<PyTensor>()?.clone(),
        ));
    }
    if let Some(probed) = probe_torch_function_override(&value.value) {
        return Ok(BoundSubOperand::Override(probed));
    }
    if value.value.is_instance_of::<PyTensor>() {
        return Err(subtraction_unsupported_native_input(operation));
    }
    if is_real_arithmetic_scalar(&value.value)? {
        return Ok(BoundSubOperand::Scalar(value.value.clone()));
    }

    if matches!(operation, SubtractionOperation::Subtract) {
        return Err(top_level_subtract_binding_error(
            operation, positional, keywords, None,
        )?);
    }
    parse_tensor_argument(operation.name(), argument, value)?;
    unreachable!("unsupported subtraction operands were rejected by parse_tensor_argument")
}

fn parse_top_level_subtraction_alpha<'py>(
    operation: SubtractionOperation,
    alpha: Option<&ParsedCallArgument<'py>>,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundSubAlpha<'py>> {
    let Some(alpha) = alpha else {
        return Ok(BoundSubAlpha::Default);
    };
    if let Some(probed) = probe_torch_function_override(&alpha.value) {
        return Ok(BoundSubAlpha::Override(probed));
    }
    let Some(scalar) = parse_arithmetic_scalar(&alpha.value)? else {
        if matches!(operation, SubtractionOperation::Subtract) {
            return Err(top_level_subtract_binding_error(
                operation,
                positional,
                keywords,
                Some(&SubtractBindingMismatch::IncorrectKeyword("alpha")),
            )?);
        }
        let actual = python_type_name(&alpha.value)?;
        return Err(PyTypeError::new_err(format!(
            "{}(): argument 'alpha' must be Number, not {actual}",
            operation.name()
        )));
    };
    if scalar.is_python_bool() {
        return Ok(BoundSubAlpha::PythonBool);
    }
    if scalar.is_one() {
        Ok(BoundSubAlpha::Default)
    } else {
        Ok(BoundSubAlpha::NonDefault)
    }
}

fn parse_top_level_subtraction_out<'py>(
    operation: SubtractionOperation,
    out: Option<ParsedCallArgument<'py>>,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<Option<BoundTensorOrTorchFunction<'py>>> {
    let Some(out) = out else {
        return Ok(None);
    };
    if out.value.is_none() {
        return Ok(None);
    }
    if matches!(operation, SubtractionOperation::Subtract)
        && !out.value.is_exact_instance_of::<PyTensor>()
        && probe_torch_function_override(&out.value).is_none()
    {
        return Err(top_level_subtract_binding_error(
            operation,
            positional,
            keywords,
            Some(&SubtractBindingMismatch::IncorrectKeyword("out")),
        )?);
    }
    parse_exact_native_tensor_or_torch_function_argument(operation, "out", &out).map(Some)
}

fn parse_top_level_division_input<'py>(
    operation: DivisionOperation,
    value: &ParsedCallArgument<'py>,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundDivOperand<'py>> {
    parse_top_level_division_operand(operation, "input", value, positional, keywords)
}

fn parse_top_level_division_other<'py>(
    operation: DivisionOperation,
    value: &ParsedCallArgument<'py>,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundDivOperand<'py>> {
    parse_top_level_division_operand(operation, "other", value, positional, keywords)
}

fn parse_top_level_division_operand<'py>(
    operation: DivisionOperation,
    argument: &str,
    value: &ParsedCallArgument<'py>,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundDivOperand<'py>> {
    if value.value.is_exact_instance_of::<PyTensor>() {
        return Ok(BoundDivOperand::Tensor(
            value.value.cast::<PyTensor>()?.clone(),
        ));
    }
    if let Some(probed) = probe_torch_function_override(&value.value) {
        return Ok(BoundDivOperand::Override(probed));
    }
    if value.value.is_instance_of::<PyTensor>() {
        return Err(division_unsupported_native_input(operation));
    }
    if is_real_arithmetic_scalar(&value.value)? {
        return Ok(BoundDivOperand::Scalar(value.value.clone()));
    }

    if matches!(operation, DivisionOperation::Divide) {
        return Err(top_level_division_binding_error(
            operation, positional, keywords,
        )?);
    }
    parse_tensor_argument(operation.name(), argument, value)?;
    unreachable!("unsupported top-level division operands were rejected by parse_tensor_argument")
}

fn parse_top_level_division_out<'py>(
    operation: DivisionOperation,
    out: Option<ParsedCallArgument<'py>>,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<Option<BoundTensorOrTorchFunction<'py>>> {
    let Some(out) = out else {
        return Ok(None);
    };
    if out.value.is_none() {
        return Ok(None);
    }
    if out.value.is_exact_instance_of::<PyTensor>() {
        return Ok(Some(BoundTensorOrTorchFunction::Tensor(
            out.value.cast::<PyTensor>()?.clone(),
        )));
    }
    if let Some(probed) = probe_torch_function_override(&out.value) {
        return Ok(Some(BoundTensorOrTorchFunction::Override(probed)));
    }
    if matches!(operation, DivisionOperation::Divide) {
        return Err(top_level_division_binding_error(
            operation, positional, keywords,
        )?);
    }
    parse_tensor_argument(operation.name(), "out", &out)
        .map(|tensor| Some(BoundTensorOrTorchFunction::Tensor(tensor.clone())))
}

fn subtraction_unsupported_native_input(operation: SubtractionOperation) -> PyErr {
    PyNotImplementedError::new_err(format!(
        "{}(): only exact native CPU float32 Tensor input and Tensor or real-number other operands are supported",
        operation.name()
    ))
}

fn addition_unsupported_native_input() -> PyErr {
    PyNotImplementedError::new_err(
        "add(): only exact native CPU float32 Tensor input with Tensor or real-number other operands is supported",
    )
}

fn mm_unsupported_native_input() -> PyErr {
    PyNotImplementedError::new_err(
        "mm(): only exact native CPU float32 rank-2 Tensor input and mat2 operands are supported",
    )
}

fn add_sub_method_unsupported_native_input(operation: AddSubMethodOperation) -> PyErr {
    PyNotImplementedError::new_err(format!(
        "{}(): only exact native CPU float32 Tensor input and Tensor or real-number other operands are supported",
        operation.name()
    ))
}

fn is_boolean_arithmetic_scalar(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if value.is_exact_instance_of::<PyBool>() {
        return Ok(true);
    }
    let Ok(numpy) = PyModule::import(value.py(), "numpy") else {
        return Ok(false);
    };
    let numpy_bool = numpy.getattr("bool_")?;
    value.is_instance(&numpy_bool)
}

fn division_unsupported_native_input(operation: DivisionOperation) -> PyErr {
    PyNotImplementedError::new_err(format!(
        "{}(): only exact native CPU float32 Tensor input and Tensor or real-number other operands with rounding_mode=None are supported",
        operation.name()
    ))
}

fn is_real_arithmetic_scalar(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if value.is_exact_instance_of::<PyBool>()
        || value.is_instance_of::<PyInt>()
        || value.is_instance_of::<PyFloat>()
    {
        return Ok(true);
    }

    let Ok(numpy) = PyModule::import(value.py(), "numpy") else {
        return Ok(false);
    };
    let generic = numpy.getattr("generic")?;
    if !value.is_instance(&generic)? {
        return Ok(false);
    }

    Ok(value.is_instance(&numpy.getattr("bool_")?)?
        || value.is_instance(&numpy.getattr("integer")?)?
        || value.is_instance(&numpy.getattr("floating")?)?)
}

fn parse_top_level_multiplication_operand<'py>(
    operation: MultiplicationOperation,
    argument: &str,
    value: &ParsedCallArgument<'py>,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundMulOperand<'py>> {
    if let Ok(tensor) = value.value.cast::<PyTensor>() {
        return Ok(BoundMulOperand::Tensor(tensor.clone()));
    }
    if let Some(probed) = probe_torch_function_override(&value.value) {
        return Ok(BoundMulOperand::Override(probed));
    }
    if is_real_arithmetic_scalar(&value.value)? {
        return Ok(BoundMulOperand::Scalar(value.value.clone()));
    }

    let argument_count = positional
        .len()
        .saturating_add(keywords.map_or(0, PyDictMethods::len));
    if operation == MultiplicationOperation::Multiply && argument_count <= 2 {
        return Err(top_level_multiply_binding_error(positional, keywords)?);
    }

    parse_tensor_argument(operation.name(), argument, value)?;
    unreachable!("unsupported multiplication operands were rejected by parse_tensor_argument")
}

fn bind_matmul_argument<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<(ParsedCallArgument<'py>, Option<PyErr>)> {
    if positional.is_empty()
        && let Some(keywords) = keywords
        && keywords.len() == 1
        && let Some(other) = keywords.get_item("x2")?
    {
        return Ok((
            ParsedCallArgument {
                value: other,
                position: None,
            },
            None,
        ));
    }
    bind_other_argument_with_x2_fallback("matmul", positional, keywords)
}

fn bind_multiplication_argument<'py>(
    operation: MultiplicationOperation,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<(ParsedCallArgument<'py>, Option<PyErr>)> {
    if positional.is_empty()
        && let Some(keywords) = keywords
        && keywords.len() == 1
        && let Some(other) = keywords.get_item("x2")?
    {
        return Ok((
            ParsedCallArgument {
                value: other,
                position: None,
            },
            None,
        ));
    }

    if matches!(operation, MultiplicationOperation::Multiply) {
        if positional.len() == 1 && keywords.is_none_or(PyDictMethods::is_empty) {
            return Ok((
                ParsedCallArgument {
                    value: positional.get_item(0)?,
                    position: Some(1),
                },
                None,
            ));
        }
        if positional.is_empty()
            && let Some(keywords) = keywords
            && keywords.len() == 1
            && let Some(other) = keywords.get_item("other")?
        {
            return Ok((
                ParsedCallArgument {
                    value: other,
                    position: None,
                },
                None,
            ));
        }
        return Err(multiply_binding_error(positional, keywords)?);
    }

    bind_other_argument_with_x2_fallback(operation.name(), positional, keywords)
}

fn bind_tensor_division_method_arguments<'py>(
    operation: DivisionOperation,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<BoundTensorMethodDivisionArguments<'py>> {
    if positional.len() > 1 {
        return Err(tensor_division_method_binding_error(
            operation, positional, keywords,
        )?);
    }

    let mut other = if positional.is_empty() {
        None
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(0)?,
            position: Some(1),
        })
    };
    let mut x2_fallback = None;
    let mut rounding_mode = None;
    let mut keyword_error = None;
    if let Some(keywords) = keywords {
        for (key, value) in keywords {
            let key = key.extract::<String>()?;
            match key.as_str() {
                "other" if other.is_none() => {
                    if x2_fallback.is_some() {
                        keyword_error.get_or_insert(tensor_division_method_binding_error(
                            operation,
                            positional,
                            Some(keywords),
                        )?);
                    }
                    other = Some(ParsedCallArgument {
                        value,
                        position: None,
                    });
                }
                "x2" if other.is_none() && x2_fallback.is_none() => {
                    x2_fallback = Some(ParsedCallArgument {
                        value,
                        position: None,
                    });
                }
                "rounding_mode" if rounding_mode.is_none() => {
                    rounding_mode = Some(ParsedCallArgument {
                        value,
                        position: None,
                    });
                }
                _ => {
                    keyword_error.get_or_insert(tensor_division_method_binding_error(
                        operation,
                        positional,
                        Some(keywords),
                    )?);
                }
            }
        }
    }

    let Some(other) = other.or(x2_fallback) else {
        if matches!(operation, DivisionOperation::Div)
            && keywords.is_some_and(|keywords| !keywords.is_empty())
        {
            return Err(PyTypeError::new_err(
                "div() missing 1 required positional arguments: \"other\"",
            ));
        }
        return Err(tensor_division_method_binding_error(
            operation, positional, keywords,
        )?);
    };
    Ok((other, rounding_mode, keyword_error))
}

fn bind_other_argument_with_x2_fallback<'py>(
    function: &str,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<(ParsedCallArgument<'py>, Option<PyErr>)> {
    if positional.len() > 1 {
        return Err(PyTypeError::new_err(format!(
            "{function}() takes 1 positional argument but {} were given",
            positional.len()
        )));
    }

    let mut other = if positional.is_empty() {
        None
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(0)?,
            position: Some(1),
        })
    };
    let mut x2_fallback = None;
    let mut keyword_error = None;
    if let Some(keywords) = keywords {
        for (key, value) in keywords {
            let key = key.extract::<String>()?;
            if key == "x2" {
                keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "{function}() got an unexpected keyword argument '{key}'"
                    ))
                });
                if x2_fallback.is_none() {
                    x2_fallback = Some(ParsedCallArgument {
                        value,
                        position: None,
                    });
                }
            } else if key != "other" {
                keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "{function}() got an unexpected keyword argument '{key}'"
                    ))
                });
            } else if other.is_some() {
                keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "{function}() got multiple values for argument 'other'"
                    ))
                });
            } else {
                other = Some(ParsedCallArgument {
                    value,
                    position: None,
                });
            }
        }
    }

    let other = other.or(x2_fallback).ok_or_else(|| {
        PyTypeError::new_err(format!(
            "{function}() missing 1 required positional arguments: \"other\""
        ))
    })?;
    Ok((other, keyword_error))
}

fn mul_argument_type_error(position: Option<usize>, actual: &str) -> PyErr {
    let position = position.map_or_else(String::new, |position| format!(" (position {position})"));
    PyTypeError::new_err(format!(
        "mul(): argument 'other'{position} must be Tensor, not {actual}"
    ))
}

#[allow(
    unsafe_code,
    reason = "PyLong_AsLongLongAndOverflow reads an int subclass without dispatching overrides"
)]
fn python_integer_is_negative(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let mut overflow = 0;
    // SAFETY: the caller has verified that value is a Python int instance, the
    // object remains live for the call, and overflow points to writable storage.
    let converted = unsafe { ffi::PyLong_AsLongLongAndOverflow(value.as_ptr(), &raw mut overflow) };
    if PyErr::occurred(value.py()) {
        return Err(PyErr::fetch(value.py()));
    }
    Ok(if overflow == 0 {
        converted < 0
    } else {
        overflow < 0
    })
}

fn top_level_multiply_binding_error(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyErr> {
    let allocation = PythonAllocationFallback::new(positional.py());
    let summary = call_type_summary_with(
        positional,
        keywords,
        CallKeywordOrder::PyTorchUnorderedMap,
        &allocation,
    )?;
    let argument_count = positional
        .len()
        .saturating_add(keywords.map_or(0, PyDictMethods::len));
    let mismatch = if argument_count == 2 {
        top_level_multiply_binding_mismatch(positional, keywords, &allocation)?
    } else {
        String::new()
    };

    let mut message = try_string_from_str_with(
        "multiply() received an invalid combination of arguments - got (",
        &allocation,
    )?;
    try_push_string_with(&mut message, &summary, &allocation)?;
    try_push_string_with(
        &mut message,
        "), but expected one of:\n * (Tensor input, Tensor other, *, Tensor out = None)\n * (Tensor input, Number other)",
        &allocation,
    )?;
    try_push_string_with(&mut message, &mismatch, &allocation)?;
    try_push_string_with(&mut message, "\n", &allocation)?;
    if let Some(nul) = message.find('\0') {
        message.truncate(nul);
    }
    let py = positional.py();
    let message = PyString::from_bytes(py, message.as_bytes()).map_err(|_| allocation.error())?;
    let exception = py
        .get_type::<PyTypeError>()
        .call1((message,))
        .map_err(|_| allocation.error())?;
    Ok(PyErr::from_value(exception))
}

fn top_level_mm_binding_error(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyErr> {
    let allocation = PythonAllocationFallback::new(positional.py());
    let summary = call_type_summary_with(
        positional,
        keywords,
        CallKeywordOrder::PyTorchUnorderedMap,
        &allocation,
    )?;

    let mut message = try_string_from_str_with(
        "mm() received an invalid combination of arguments - got (",
        &allocation,
    )?;
    try_push_string_with(&mut message, &summary, &allocation)?;
    try_push_string_with(
        &mut message,
        "), but expected one of:\n * (Tensor input, Tensor mat2, *, Tensor out = None)\n * (Tensor input, Tensor mat2, torch.dtype out_dtype, *, Tensor out = None)\n",
        &allocation,
    )?;
    if let Some(nul) = message.find('\0') {
        message.truncate(nul);
    }
    let py = positional.py();
    let message = PyString::from_bytes(py, message.as_bytes()).map_err(|_| allocation.error())?;
    let exception = py
        .get_type::<PyTypeError>()
        .call1((message,))
        .map_err(|_| allocation.error())?;
    Ok(PyErr::from_value(exception))
}

fn tensor_add_method_binding_error(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyErr> {
    tensor_add_sub_method_binding_error("add", positional, keywords)
}

fn tensor_sub_method_binding_error(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyErr> {
    tensor_add_sub_method_binding_error("sub", positional, keywords)
}

fn tensor_add_sub_method_binding_error(
    function: &'static str,
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyErr> {
    let allocation = PythonAllocationFallback::new(positional.py());
    let summary = call_type_summary_with(
        positional,
        keywords,
        CallKeywordOrder::PyTorchUnorderedMap,
        &allocation,
    )?;

    let mut message = try_string_from_str_with(function, &allocation)?;
    try_push_string_with(
        &mut message,
        "() received an invalid combination of arguments - got (",
        &allocation,
    )?;
    try_push_string_with(&mut message, &summary, &allocation)?;
    try_push_string_with(
        &mut message,
        "), but expected (Tensor other, *, Number alpha = 1)",
        &allocation,
    )?;
    if let Some(nul) = message.find('\0') {
        message.truncate(nul);
    }
    let py = positional.py();
    let message = PyString::from_bytes(py, message.as_bytes()).map_err(|_| allocation.error())?;
    let exception = py
        .get_type::<PyTypeError>()
        .call1((message,))
        .map_err(|_| allocation.error())?;
    Ok(PyErr::from_value(exception))
}

fn tensor_subtract_binding_error(
    operation: SubtractionOperation,
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
    mismatch: Option<&SubtractBindingMismatch<'_>>,
) -> PyResult<PyErr> {
    let allocation = PythonAllocationFallback::new(positional.py());
    let summary = call_type_summary_with(
        positional,
        keywords,
        CallKeywordOrder::PyTorchUnorderedMap,
        &allocation,
    )?;

    let mut message = try_string_from_str_with(operation.name(), &allocation)?;
    try_push_string_with(
        &mut message,
        "() received an invalid combination of arguments - got (",
        &allocation,
    )?;
    try_push_string_with(&mut message, &summary, &allocation)?;
    try_push_string_with(
        &mut message,
        "), but expected one of:\n * (Tensor other, *, Number alpha = 1)\n * (Number other, Number alpha = 1)",
        &allocation,
    )?;
    match mismatch {
        Some(SubtractBindingMismatch::InvalidPositionalOverload) => {
            let mut details = try_string_from_str_with("(", &allocation)?;
            for (index, value) in positional.iter().enumerate() {
                if index != 0 {
                    try_push_string_with(&mut details, ", ", &allocation)?;
                }
                push_multiply_mismatched_argument(
                    &mut details,
                    &value,
                    "Number",
                    None,
                    &allocation,
                )?;
            }
            try_push_string_with(&mut details, ")", &allocation)?;
            try_push_string_with(
                &mut message,
                "\n      didn't match because some of the arguments have invalid types: ",
                &allocation,
            )?;
            try_push_string_with(&mut message, &details, &allocation)?;
        }
        Some(SubtractBindingMismatch::IncorrectKeyword(keyword)) => {
            try_push_string_with(
                &mut message,
                "\n      didn't match because some of the keywords were incorrect: ",
                &allocation,
            )?;
            try_push_string_with(&mut message, keyword, &allocation)?;
        }
        None => {}
    }
    try_push_string_with(&mut message, "\n", &allocation)?;
    if let Some(nul) = message.find('\0') {
        message.truncate(nul);
    }
    let py = positional.py();
    let message = PyString::from_bytes(py, message.as_bytes()).map_err(|_| allocation.error())?;
    let exception = py
        .get_type::<PyTypeError>()
        .call1((message,))
        .map_err(|_| allocation.error())?;
    Ok(PyErr::from_value(exception))
}

fn top_level_subtraction_binding_error(
    operation: SubtractionOperation,
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyErr> {
    let allocation = PythonAllocationFallback::new(positional.py());
    let summary = call_type_summary_with(
        positional,
        keywords,
        CallKeywordOrder::PyTorchUnorderedMap,
        &allocation,
    )?;

    let mut message = try_string_from_str_with(operation.name(), &allocation)?;
    try_push_string_with(
        &mut message,
        "() received an invalid combination of arguments - got (",
        &allocation,
    )?;
    try_push_string_with(&mut message, &summary, &allocation)?;
    try_push_string_with(
        &mut message,
        "), but expected (Tensor input, Tensor other, *, Number alpha = 1, Tensor out = None)",
        &allocation,
    )?;
    if let Some(nul) = message.find('\0') {
        message.truncate(nul);
    }
    let py = positional.py();
    let message = PyString::from_bytes(py, message.as_bytes()).map_err(|_| allocation.error())?;
    let exception = py
        .get_type::<PyTypeError>()
        .call1((message,))
        .map_err(|_| allocation.error())?;
    Ok(PyErr::from_value(exception))
}

enum SubtractBindingMismatch<'a> {
    InvalidPositionalOverload,
    IncorrectKeyword(&'a str),
}

fn top_level_subtract_binding_error(
    operation: SubtractionOperation,
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
    mismatch: Option<&SubtractBindingMismatch<'_>>,
) -> PyResult<PyErr> {
    let allocation = PythonAllocationFallback::new(positional.py());
    let summary = call_type_summary_with(
        positional,
        keywords,
        CallKeywordOrder::PyTorchUnorderedMap,
        &allocation,
    )?;

    let mut message = try_string_from_str_with(operation.name(), &allocation)?;
    try_push_string_with(
        &mut message,
        "() received an invalid combination of arguments - got (",
        &allocation,
    )?;
    try_push_string_with(&mut message, &summary, &allocation)?;
    try_push_string_with(
        &mut message,
        "), but expected one of:\n * (Tensor input, Tensor other, *, Number alpha = 1, Tensor out = None)\n * (Tensor input, Number other, Number alpha = 1)",
        &allocation,
    )?;
    match mismatch {
        Some(SubtractBindingMismatch::InvalidPositionalOverload) => {
            let mut details = try_string_from_str_with("(", &allocation)?;
            for (index, value) in positional.iter().enumerate() {
                if index != 0 {
                    try_push_string_with(&mut details, ", ", &allocation)?;
                }
                let expected = match index {
                    0 => "Tensor",
                    1 | 2 => "Number",
                    _ => "",
                };
                push_multiply_mismatched_argument(
                    &mut details,
                    &value,
                    expected,
                    None,
                    &allocation,
                )?;
            }
            try_push_string_with(&mut details, ")", &allocation)?;
            try_push_string_with(
                &mut message,
                "\n      didn't match because some of the arguments have invalid types: ",
                &allocation,
            )?;
            try_push_string_with(&mut message, &details, &allocation)?;
        }
        Some(SubtractBindingMismatch::IncorrectKeyword(keyword)) => {
            try_push_string_with(
                &mut message,
                "\n      didn't match because some of the keywords were incorrect: ",
                &allocation,
            )?;
            try_push_string_with(&mut message, keyword, &allocation)?;
        }
        None => {}
    }
    try_push_string_with(&mut message, "\n", &allocation)?;
    if let Some(nul) = message.find('\0') {
        message.truncate(nul);
    }
    let py = positional.py();
    let message = PyString::from_bytes(py, message.as_bytes()).map_err(|_| allocation.error())?;
    let exception = py
        .get_type::<PyTypeError>()
        .call1((message,))
        .map_err(|_| allocation.error())?;
    Ok(PyErr::from_value(exception))
}

fn top_level_multiply_binding_mismatch(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<String> {
    let keyword_length = keywords.map_or(0, PyDictMethods::len);
    let mut keyword_names = try_size_vector_with(keyword_length, allocation)?;
    if let Some(keywords) = keywords {
        for (key, _) in keywords {
            let key = pytorch_keyword_name(&key)?;
            try_push_size_with(
                &mut keyword_names,
                (try_string_from_str_with(key, allocation)?, ()),
                allocation,
            )?;
        }
        keyword_names = pytorch_unordered_keyword_order(keyword_names, allocation)?;
    }

    let mut incorrect_keywords = try_size_vector_with(keyword_length, allocation)?;
    for (keyword, ()) in keyword_names {
        let fills_unbound_schema_position = match keyword.as_str() {
            "input" => positional.is_empty(),
            "other" => positional.len() < 2,
            _ => false,
        };
        if !fills_unbound_schema_position {
            try_push_size_with(&mut incorrect_keywords, keyword, allocation)?;
        }
    }
    if !incorrect_keywords.is_empty() {
        let mut mismatch = try_string_from_str_with(
            "\n      didn't match because some of the keywords were incorrect: ",
            allocation,
        )?;
        for (index, keyword) in incorrect_keywords.into_iter().enumerate() {
            if index != 0 {
                try_push_string_with(&mut mismatch, ", ", allocation)?;
            }
            try_push_string_with(&mut mismatch, &keyword, allocation)?;
        }
        return Ok(mismatch);
    }

    let mut mismatch = try_string_from_str_with(
        "\n      didn't match because some of the arguments have invalid types: (",
        allocation,
    )?;
    let mut argument_index = 0_usize;
    for (index, value) in positional.iter().enumerate() {
        if argument_index != 0 {
            try_push_string_with(&mut mismatch, ", ", allocation)?;
        }
        let expected = if index == 0 { "Tensor" } else { "Number" };
        push_multiply_mismatched_argument(&mut mismatch, &value, expected, None, allocation)?;
        argument_index += 1;
    }
    if let Some(keywords) = keywords {
        for (index, (keyword, expected)) in [("input", "Tensor"), ("other", "Number")]
            .into_iter()
            .enumerate()
        {
            if index < positional.len() {
                continue;
            }
            let Some(value) = keywords.get_item(keyword)? else {
                continue;
            };
            if argument_index != 0 {
                try_push_string_with(&mut mismatch, ", ", allocation)?;
            }
            push_multiply_mismatched_argument(
                &mut mismatch,
                &value,
                expected,
                Some(keyword),
                allocation,
            )?;
            argument_index += 1;
        }
    }
    if positional.is_empty() && argument_index != 0 {
        try_push_string_with(&mut mismatch, ", ", allocation)?;
    }
    try_push_string_with(&mut mismatch, ")", allocation)?;
    Ok(mismatch)
}

fn top_level_add_binding_error(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyErr> {
    let allocation = PythonAllocationFallback::new(positional.py());
    let summary = call_type_summary_with(
        positional,
        keywords,
        CallKeywordOrder::PyTorchUnorderedMap,
        &allocation,
    )?;

    let mut message = try_string_from_str_with(
        "add() received an invalid combination of arguments - got (",
        &allocation,
    )?;
    try_push_string_with(&mut message, &summary, &allocation)?;
    try_push_string_with(
        &mut message,
        "), but expected (Tensor input, Tensor or Number other, *, Number alpha = 1, Tensor out = None)",
        &allocation,
    )?;
    if let Some(nul) = message.find('\0') {
        message.truncate(nul);
    }
    let py = positional.py();
    let message = PyString::from_bytes(py, message.as_bytes()).map_err(|_| allocation.error())?;
    let exception = py
        .get_type::<PyTypeError>()
        .call1((message,))
        .map_err(|_| allocation.error())?;
    Ok(PyErr::from_value(exception))
}

fn push_multiply_mismatched_argument(
    mismatch: &mut String,
    value: &Bound<'_, PyAny>,
    expected_type: &str,
    keyword: Option<&str>,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<()> {
    let actual_type = python_type_name_with(value, allocation)?;
    let detail = call_argument_type_description_with(value, allocation)?;
    let invalid_type = actual_type != expected_type;
    if invalid_type {
        try_push_string_with(mismatch, "!", allocation)?;
    }
    if let Some(keyword) = keyword {
        try_push_string_with(mismatch, keyword, allocation)?;
        try_push_string_with(mismatch, "=", allocation)?;
    }
    try_push_string_with(mismatch, &detail, allocation)?;
    if invalid_type {
        try_push_string_with(mismatch, "!", allocation)?;
    }
    Ok(())
}

fn multiply_binding_error(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyErr> {
    let allocation = PythonAllocationFallback::new(positional.py());
    let summary = call_type_summary_with(
        positional,
        keywords,
        CallKeywordOrder::PyTorchUnorderedMap,
        &allocation,
    )?;
    let keyword_length = keywords.map_or(0, PyDictMethods::len);
    let (tensor_mismatch, number_mismatch) = if positional.len() + keyword_length == 1 {
        if positional.len() == 1 {
            let value = positional.get_item(0)?;
            let actual_type = python_type_name_with(&value, &allocation)?;
            let tensor_detail = call_argument_type_description_with(&value, &allocation)?;
            let number_detail = call_argument_type_description_with(&value, &allocation)?;
            (
                multiply_invalid_type_mismatch(
                    &tensor_detail,
                    &actual_type,
                    "Tensor",
                    None,
                    &allocation,
                )?,
                multiply_invalid_type_mismatch(
                    &number_detail,
                    &actual_type,
                    "Number",
                    None,
                    &allocation,
                )?,
            )
        } else {
            let keywords = keywords.expect("a single keyword argument is present");
            let (key, value) = keywords
                .iter()
                .next()
                .expect("a single keyword argument remains present");
            let key = pytorch_keyword_name(&key)?;
            if key == "other" {
                let actual_type = python_type_name_with(&value, &allocation)?;
                let tensor_detail = call_argument_type_description_with(&value, &allocation)?;
                let number_detail = call_argument_type_description_with(&value, &allocation)?;
                (
                    multiply_invalid_type_mismatch(
                        &tensor_detail,
                        &actual_type,
                        "Tensor",
                        Some("other"),
                        &allocation,
                    )?,
                    multiply_invalid_type_mismatch(
                        &number_detail,
                        &actual_type,
                        "Number",
                        Some("other"),
                        &allocation,
                    )?,
                )
            } else {
                let mismatch = multiply_invalid_keyword_mismatch(key, &allocation)?;
                (try_string_from_str_with(&mismatch, &allocation)?, mismatch)
            }
        }
    } else {
        (String::new(), String::new())
    };

    let mut message = try_string_from_str_with(
        "multiply() received an invalid combination of arguments - got (",
        &allocation,
    )?;
    try_push_string_with(&mut message, &summary, &allocation)?;
    try_push_string_with(
        &mut message,
        "), but expected one of:\n * (Tensor other)",
        &allocation,
    )?;
    try_push_string_with(&mut message, &tensor_mismatch, &allocation)?;
    try_push_string_with(&mut message, "\n * (Number other)", &allocation)?;
    try_push_string_with(&mut message, &number_mismatch, &allocation)?;
    try_push_string_with(&mut message, "\n", &allocation)?;
    if let Some(nul) = message.find('\0') {
        message.truncate(nul);
    }
    let py = positional.py();
    let message = PyString::from_bytes(py, message.as_bytes()).map_err(|_| allocation.error())?;
    let exception = py
        .get_type::<PyTypeError>()
        .call1((message,))
        .map_err(|_| allocation.error())?;
    Ok(PyErr::from_value(exception))
}

fn tensor_division_method_binding_error(
    operation: DivisionOperation,
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyErr> {
    let allocation = PythonAllocationFallback::new(positional.py());
    let summary = call_type_summary_with(
        positional,
        keywords,
        CallKeywordOrder::PyTorchUnorderedMap,
        &allocation,
    )?;

    let mut message = try_string_from_str_with(operation.name(), &allocation)?;
    try_push_string_with(
        &mut message,
        "() received an invalid combination of arguments - got (",
        &allocation,
    )?;
    try_push_string_with(&mut message, &summary, &allocation)?;
    try_push_string_with(
        &mut message,
        "), but expected one of:\n * (Tensor other)\n * (Tensor other, *, str rounding_mode)",
        &allocation,
    )?;
    if matches!(operation, DivisionOperation::Divide) {
        try_push_string_with(&mut message, "\n * (Number other)", &allocation)?;
    }
    try_push_string_with(
        &mut message,
        "\n * (Number other, *, str rounding_mode)\n",
        &allocation,
    )?;
    if let Some(nul) = message.find('\0') {
        message.truncate(nul);
    }
    let py = positional.py();
    let message = PyString::from_bytes(py, message.as_bytes()).map_err(|_| allocation.error())?;
    let exception = py
        .get_type::<PyTypeError>()
        .call1((message,))
        .map_err(|_| allocation.error())?;
    Ok(PyErr::from_value(exception))
}

fn top_level_division_binding_error(
    operation: DivisionOperation,
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyErr> {
    let allocation = PythonAllocationFallback::new(positional.py());
    let summary = call_type_summary_with(
        positional,
        keywords,
        CallKeywordOrder::PyTorchUnorderedMap,
        &allocation,
    )?;

    let mut message = try_string_from_str_with(operation.name(), &allocation)?;
    try_push_string_with(
        &mut message,
        "() received an invalid combination of arguments - got (",
        &allocation,
    )?;
    try_push_string_with(&mut message, &summary, &allocation)?;
    try_push_string_with(
        &mut message,
        "), but expected one of:\n * (Tensor input, Tensor other, *, Tensor out = None)\n * (Tensor input, Tensor other, *, str rounding_mode, Tensor out = None)",
        &allocation,
    )?;
    if matches!(operation, DivisionOperation::Divide) {
        try_push_string_with(
            &mut message,
            "\n * (Tensor input, Number other)",
            &allocation,
        )?;
    }
    try_push_string_with(
        &mut message,
        "\n * (Tensor input, Number other, *, str rounding_mode)\n",
        &allocation,
    )?;
    if let Some(nul) = message.find('\0') {
        message.truncate(nul);
    }
    let py = positional.py();
    let message = PyString::from_bytes(py, message.as_bytes()).map_err(|_| allocation.error())?;
    let exception = py
        .get_type::<PyTypeError>()
        .call1((message,))
        .map_err(|_| allocation.error())?;
    Ok(PyErr::from_value(exception))
}

fn multiply_invalid_type_mismatch(
    detail: &str,
    actual_type: &str,
    expected_type: &str,
    keyword: Option<&str>,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<String> {
    let mut mismatch = try_string_from_str_with(
        "\n      didn't match because some of the arguments have invalid types: (",
        allocation,
    )?;
    let invalid_type = actual_type != expected_type;
    if invalid_type {
        try_push_string_with(&mut mismatch, "!", allocation)?;
    }
    if let Some(keyword) = keyword {
        try_push_string_with(&mut mismatch, keyword, allocation)?;
        try_push_string_with(&mut mismatch, "=", allocation)?;
    }
    try_push_string_with(&mut mismatch, detail, allocation)?;
    if invalid_type {
        try_push_string_with(&mut mismatch, "!", allocation)?;
    }
    if keyword.is_some() {
        try_push_string_with(&mut mismatch, ", ", allocation)?;
    }
    try_push_string_with(&mut mismatch, ")", allocation)?;
    Ok(mismatch)
}

fn multiply_invalid_keyword_mismatch(
    keyword: &str,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<String> {
    let mut mismatch = try_string_from_str_with(
        "\n      didn't match because some of the keywords were incorrect: ",
        allocation,
    )?;
    try_push_string_with(&mut mismatch, keyword, allocation)?;
    Ok(mismatch)
}

#[derive(Clone, Copy)]
enum DimensionMoveOperation {
    Movedim,
    Moveaxis,
}

impl DimensionMoveOperation {
    const fn name(self) -> &'static str {
        match self {
            Self::Movedim => "movedim",
            Self::Moveaxis => "moveaxis",
        }
    }

    const fn qualified_name(self) -> &'static str {
        match self {
            Self::Movedim => "torch.movedim",
            Self::Moveaxis => "torch.moveaxis",
        }
    }

    const fn tensor_qualified_name(self) -> &'static str {
        match self {
            Self::Movedim => "torch.Tensor.movedim",
            Self::Moveaxis => "torch.Tensor.moveaxis",
        }
    }
}

#[derive(Clone, Copy)]
enum MovedimCallKind {
    TensorMethod(DimensionMoveOperation),
    VariableFunction(DimensionMoveOperation),
}

impl MovedimCallKind {
    const fn operation(self) -> DimensionMoveOperation {
        match self {
            Self::TensorMethod(operation) | Self::VariableFunction(operation) => operation,
        }
    }

    const fn integer_signature(self) -> &'static str {
        match self {
            Self::TensorMethod(_) => "(int source, int destination)",
            Self::VariableFunction(_) => "(Tensor input, int source, int destination)",
        }
    }

    const fn sequence_signature(self) -> &'static str {
        match self {
            Self::TensorMethod(_) => "(tuple of ints source, tuple of ints destination)",
            Self::VariableFunction(_) => {
                "(Tensor input, tuple of ints source, tuple of ints destination)"
            }
        }
    }
}

fn bind_movedim_arguments<'py>(
    operation: DimensionMoveOperation,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<[ParsedCallArgument<'py>; 2]> {
    let kind = MovedimCallKind::TensorMethod(operation);
    let names = ["source", "destination"];
    let arguments = bind_movedim_call_arguments(
        positional,
        keywords,
        kind,
        [c"source", c"destination"],
        false,
    )?;
    if !is_dimension_swap_integer(&arguments[0].value)?
        || !is_dimension_swap_integer(&arguments[1].value)?
    {
        return Err(movedim_binding_error(positional, keywords, kind, &names)?);
    }
    Ok(arguments)
}

fn bind_top_level_movedim_arguments<'py>(
    operation: DimensionMoveOperation,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<(
    BoundTensorOrTorchFunction<'py>,
    [ParsedCallArgument<'py>; 2],
)> {
    let kind = MovedimCallKind::VariableFunction(operation);
    let names = ["input", "source", "destination"];
    let [input, source, destination] = bind_movedim_call_arguments(
        positional,
        keywords,
        kind,
        [c"input", c"source", c"destination"],
        true,
    )?;
    let input = if let Ok(tensor) = input.value.cast::<PyTensor>() {
        BoundTensorOrTorchFunction::Tensor(tensor.clone())
    } else if let Some(probed) = probe_torch_function_override(&input.value) {
        BoundTensorOrTorchFunction::Override(probed)
    } else {
        return Err(movedim_binding_error(positional, keywords, kind, &names)?);
    };
    if !is_dimension_swap_integer(&source.value)? || !is_dimension_swap_integer(&destination.value)?
    {
        return Err(movedim_binding_error(positional, keywords, kind, &names)?);
    }
    Ok((input, [source, destination]))
}

fn bind_movedim_call_arguments<'py, const N: usize>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
    kind: MovedimCallKind,
    names: [&CStr; N],
    allow_input_aliases: bool,
) -> PyResult<[ParsedCallArgument<'py>; N]> {
    let argument_count = positional
        .len()
        .checked_add(keywords.map_or(0, PyDictMethods::len))
        .ok_or_else(|| {
            PyMemoryError::new_err(format!(
                "{} argument count overflowed",
                kind.operation().name()
            ))
        })?;
    let error_names = match kind {
        MovedimCallKind::TensorMethod(_) => &["source", "destination"][..],
        MovedimCallKind::VariableFunction(_) => &["input", "source", "destination"][..],
    };
    if positional.len() > N || argument_count != N {
        return Err(movedim_binding_error(
            positional,
            keywords,
            kind,
            error_names,
        )?);
    }

    let mut arguments: [Option<ParsedCallArgument<'py>>; N] = std::array::from_fn(|_| None);
    for (index, value) in positional.iter().enumerate() {
        arguments[index] = Some(ParsedCallArgument {
            value,
            position: Some(index + 1),
        });
    }

    if let Some(keywords) = keywords {
        for (index, name) in names.into_iter().enumerate() {
            let mut value = legacy_dict_get_item_string(keywords, name);
            if value.is_none() && allow_input_aliases && index == 0 {
                for alias in [c"x", c"a", c"x1"] {
                    if let Some(alias_value) = legacy_dict_get_item_string(keywords, alias) {
                        value = Some(alias_value);
                        break;
                    }
                }
            }
            let Some(value) = value else {
                continue;
            };
            if arguments[index].is_some() {
                return Err(movedim_binding_error(
                    positional,
                    Some(keywords),
                    kind,
                    error_names,
                )?);
            }
            arguments[index] = Some(ParsedCallArgument {
                value,
                position: None,
            });
        }
    }

    if arguments.iter().any(Option::is_none) {
        return Err(movedim_binding_error(
            positional,
            keywords,
            kind,
            error_names,
        )?);
    }
    Ok(arguments.map(|argument| argument.expect("all movedim arguments were bound above")))
}

#[allow(
    unsafe_code,
    reason = "PyTorch's generated parser uses exception-suppressing legacy dictionary lookup"
)]
pub(crate) fn legacy_dict_get_item_string<'py>(
    dictionary: &Bound<'py, PyDict>,
    name: &CStr,
) -> Option<Bound<'py, PyAny>> {
    // SAFETY: dictionary is a live exact dict, name is NUL-terminated, and
    // PyDict_GetItemString returns a borrowed value kept alive by dictionary.
    let value = unsafe { ffi::PyDict_GetItemString(dictionary.as_ptr(), name.as_ptr()) };
    if value.is_null() {
        // Legacy lookup suppresses comparison and hashing exceptions. Clear
        // defensively so a hostile str subclass cannot leak an error state.
        // SAFETY: the GIL is held and clearing no pending exception is valid.
        unsafe { ffi::PyErr_Clear() };
        return None;
    }
    // SAFETY: value is a live borrowed reference owned by dictionary for the
    // lifetime of this Bound handle and the attached interpreter.
    Some(unsafe { Bound::<PyAny>::from_borrowed_ptr(dictionary.py(), value) })
}

fn movedim_binding_error(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
    kind: MovedimCallKind,
    names: &[&str],
) -> PyResult<PyErr> {
    let allocation = PythonAllocationFallback::new(positional.py());
    let summary = call_type_summary_with(
        positional,
        keywords,
        CallKeywordOrder::PyTorchUnorderedMap,
        &allocation,
    )?;
    let argument_count = positional
        .len()
        .checked_add(keywords.map_or(0, PyDictMethods::len))
        .ok_or_else(|| allocation.error())?;

    let (integer_mismatch, sequence_mismatch) = if argument_count == names.len() {
        let (arguments, incorrect_keywords) =
            movedim_error_arguments(positional, keywords, names, &allocation)?;
        if incorrect_keywords.is_empty() {
            if arguments.iter().all(Option::is_some) {
                let mut complete = try_size_vector_with(arguments.len(), &allocation)?;
                for argument in arguments {
                    try_push_size_with(
                        &mut complete,
                        argument.expect("complete movedim diagnostics were checked"),
                        &allocation,
                    )?;
                }
                (
                    movedim_invalid_type_mismatch(&complete, kind, false, &allocation)?,
                    movedim_invalid_type_mismatch(&complete, kind, true, &allocation)?,
                )
            } else {
                (String::new(), String::new())
            }
        } else {
            (
                movedim_invalid_keyword_mismatch(&incorrect_keywords, &allocation)?,
                movedim_invalid_keyword_mismatch(&incorrect_keywords, &allocation)?,
            )
        }
    } else {
        (String::new(), String::new())
    };

    let mut message = try_string_from_str_with(kind.operation().name(), &allocation)?;
    try_push_string_with(
        &mut message,
        "() received an invalid combination of arguments - got (",
        &allocation,
    )?;
    try_push_string_with(&mut message, &summary, &allocation)?;
    try_push_string_with(&mut message, "), but expected one of:\n * ", &allocation)?;
    try_push_string_with(&mut message, kind.integer_signature(), &allocation)?;
    try_push_string_with(&mut message, &integer_mismatch, &allocation)?;
    try_push_string_with(&mut message, "\n * ", &allocation)?;
    try_push_string_with(&mut message, kind.sequence_signature(), &allocation)?;
    try_push_string_with(&mut message, &sequence_mismatch, &allocation)?;
    try_push_string_with(&mut message, "\n", &allocation)?;
    if let Some(nul) = message.find('\0') {
        message.truncate(nul);
    }
    let py = positional.py();
    let message = PyString::from_bytes(py, message.as_bytes()).map_err(|_| allocation.error())?;
    let exception = py
        .get_type::<PyTypeError>()
        .call1((message,))
        .map_err(|_| allocation.error())?;
    Ok(PyErr::from_value(exception))
}

fn movedim_error_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
    names: &[&str],
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<(Vec<Option<ParsedCallArgument<'py>>>, Vec<String>)> {
    let mut arguments = try_size_vector_with(names.len(), allocation)?;
    arguments.resize_with(names.len(), || None);
    for (index, value) in positional.iter().take(names.len()).enumerate() {
        arguments[index] = Some(ParsedCallArgument {
            value,
            position: Some(index + 1),
        });
    }

    let keyword_count = keywords.map_or(0, PyDictMethods::len);
    let mut incorrect = try_size_vector_with(keyword_count, allocation)?;
    if let Some(keywords) = keywords {
        for (key, value) in pytorch_ordered_keyword_entries_with(keywords, allocation)? {
            let index = names.iter().position(|name| *name == key);
            if let Some(index) = index
                && arguments[index].is_none()
            {
                arguments[index] = Some(ParsedCallArgument {
                    value,
                    position: None,
                });
            } else {
                try_push_size_with(&mut incorrect, key, allocation)?;
            }
        }
    }
    Ok((arguments, incorrect))
}

pub(crate) fn pytorch_ordered_keyword_entries<'py>(
    keywords: &Bound<'py, PyDict>,
) -> PyResult<Vec<(String, Bound<'py, PyAny>)>> {
    let allocation = PythonAllocationFallback::new(keywords.py());
    pytorch_ordered_keyword_entries_with(keywords, &allocation)
}

fn pytorch_ordered_keyword_entries_with<'py>(
    keywords: &Bound<'py, PyDict>,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<Vec<(String, Bound<'py, PyAny>)>> {
    let mut entries = try_size_vector_with(keywords.len(), allocation)?;
    for (key, value) in keywords {
        let key = pytorch_keyword_name(&key)?;
        try_push_size_with(
            &mut entries,
            (try_string_from_str_with(key, allocation)?, value),
            allocation,
        )?;
    }
    pytorch_unordered_keyword_order(entries, allocation)
}

fn movedim_invalid_type_mismatch(
    arguments: &[ParsedCallArgument<'_>],
    kind: MovedimCallKind,
    sequence_overload: bool,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<String> {
    let names = match kind {
        MovedimCallKind::TensorMethod(_) => &["source", "destination"][..],
        MovedimCallKind::VariableFunction(_) => &["input", "source", "destination"][..],
    };
    let mut mismatch = try_string_from_str_with(
        "\n      didn't match because some of the arguments have invalid types: (",
        allocation,
    )?;
    for (index, (name, argument)) in names.iter().copied().zip(arguments).enumerate() {
        if index != 0 {
            try_push_string_with(&mut mismatch, ", ", allocation)?;
        }
        let actual = python_type_name_with(&argument.value, allocation)?;
        let detail = call_argument_type_description_with(&argument.value, allocation)?;
        let input = matches!(kind, MovedimCallKind::VariableFunction(_)) && index == 0;
        let invalid = if input {
            actual != "Tensor"
        } else {
            sequence_overload || actual != "int"
        };
        if invalid {
            try_push_string_with(&mut mismatch, "!", allocation)?;
        }
        if argument.position.is_none() {
            try_push_string_with(&mut mismatch, name, allocation)?;
            try_push_string_with(&mut mismatch, "=", allocation)?;
        }
        try_push_string_with(&mut mismatch, &detail, allocation)?;
        if invalid {
            try_push_string_with(&mut mismatch, "!", allocation)?;
        }
    }
    if arguments[0].position.is_none() {
        try_push_string_with(&mut mismatch, ", ", allocation)?;
    }
    try_push_string_with(&mut mismatch, ")", allocation)?;
    Ok(mismatch)
}

fn movedim_invalid_keyword_mismatch(
    keywords: &[String],
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<String> {
    let mut mismatch = try_string_from_str_with(
        "\n      didn't match because some of the keywords were incorrect: ",
        allocation,
    )?;
    for (index, keyword) in keywords.iter().enumerate() {
        if index != 0 {
            try_push_string_with(&mut mismatch, ", ", allocation)?;
        }
        try_push_string_with(&mut mismatch, keyword, allocation)?;
    }
    Ok(mismatch)
}

fn bind_dimension_swap_arguments<'py, const N: usize>(
    operation: &str,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
    names: [&str; N],
) -> PyResult<([ParsedCallArgument<'py>; N], Option<PyErr>)> {
    if positional.len() > N {
        return Err(PyTypeError::new_err(format!(
            "{operation}() takes {N} positional arguments but {} were given",
            positional.len()
        )));
    }

    let mut arguments: [Option<ParsedCallArgument<'py>>; N] = std::array::from_fn(|_| None);
    for (index, value) in positional.iter().enumerate() {
        arguments[index] = Some(ParsedCallArgument {
            value,
            position: Some(index + 1),
        });
    }

    let mut keyword_error = None;
    if let Some(keywords) = keywords {
        for (key, value) in keywords {
            let key = key.extract::<String>()?;
            let Some(index) = names.iter().position(|name| *name == key) else {
                keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "{operation}() got an unexpected keyword argument '{key}'"
                    ))
                });
                continue;
            };
            if arguments[index].is_some() {
                keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "{operation}() got multiple values for argument '{}'",
                        names[index]
                    ))
                });
                continue;
            }
            arguments[index] = Some(ParsedCallArgument {
                value,
                position: None,
            });
        }
    }

    if let Some(first_missing) = arguments.iter().position(Option::is_none) {
        // PyTorch validates the recognized types before reporting a later
        // missing argument, but leaves integer conversion (and therefore
        // overflow) until the complete signature has bound successfully.
        validate_dimension_swap_argument_prefix(operation, &names, &arguments, first_missing)?;
        let missing = &names[first_missing..];
        let quoted_names = missing
            .iter()
            .map(|name| format!("\"{name}\""))
            .collect::<Vec<_>>()
            .join(", ");
        let argument = if missing.len() == 1 {
            "arguments"
        } else {
            "argument"
        };
        return Err(PyTypeError::new_err(format!(
            "{operation}() missing {} required positional {argument}: {quoted_names}",
            missing.len()
        )));
    }

    if keyword_error.is_some() {
        // Type errors take precedence over duplicate and unexpected keyword
        // errors. Successful calls defer validation to argument parsing so
        // each dimension is inspected only once on the hot path.
        validate_dimension_swap_argument_prefix(operation, &names, &arguments, N)?;
    }

    Ok((
        arguments.map(|argument| {
            argument.expect("all required dimension-swap arguments were bound above")
        }),
        keyword_error,
    ))
}

enum ViewShapeArgument<'py> {
    Dimension(Bound<'py, PyAny>),
    Tuple(Bound<'py, PyTuple>),
    List(Bound<'py, PyList>),
}

enum ViewArgument<'py> {
    Shape(ViewShapeArgument<'py>),
    DType(Bound<'py, PyDType>),
}

struct ViewKeywordArguments<'py> {
    shape: Option<Bound<'py, PyAny>>,
    dtype: Option<Bound<'py, PyAny>>,
    error: Option<PyErr>,
    saw_dtype: bool,
}

fn bind_view_keyword_arguments<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<ViewKeywordArguments<'py>> {
    let mut arguments = ViewKeywordArguments {
        shape: None,
        dtype: None,
        error: None,
        saw_dtype: false,
    };
    let Some(keywords) = keywords else {
        return Ok(arguments);
    };

    for (key, value) in keywords {
        let key = key.extract::<String>()?;
        match key.as_str() {
            "size" if positional.is_empty() && arguments.shape.is_none() => {
                arguments.shape = Some(value);
            }
            "size" => {
                arguments.error.get_or_insert_with(|| {
                    PyTypeError::new_err("view() got multiple values for argument 'size'")
                });
            }
            "dtype" => {
                arguments.saw_dtype = true;
                if positional.is_empty() && arguments.shape.is_none() && arguments.dtype.is_none() {
                    arguments.dtype = Some(value);
                } else {
                    arguments.error.get_or_insert_with(|| {
                        PyTypeError::new_err("view() got multiple values for argument 'dtype'")
                    });
                }
            }
            _ => {
                arguments.error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "view() got an unexpected keyword argument '{key}'"
                    ))
                });
            }
        }
    }
    Ok(arguments)
}

fn bind_view_argument<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<ViewArgument<'py>> {
    let ViewKeywordArguments {
        shape: keyword_shape,
        dtype: keyword_dtype,
        error: keyword_error,
        saw_dtype: saw_dtype_keyword,
    } = bind_view_keyword_arguments(positional, keywords)?;

    if saw_dtype_keyword && positional.is_empty() {
        if keyword_shape.is_none()
            && keyword_error.is_none()
            && keywords.is_some_and(|keywords| keywords.len() == 1)
            && let Some(dtype) = keyword_dtype
            && let Ok(dtype) = dtype.cast_into::<PyDType>()
        {
            return Ok(ViewArgument::DType(dtype));
        }
        return Err(unsupported_view_call_error(positional, keywords)?);
    }

    if positional.len() >= 2 {
        let first = positional.get_item(0)?;
        if !is_view_shape_dimension(&first) {
            return Err(unsupported_view_call_error(positional, keywords)?);
        }
        let sequence_shape = if let Ok(shape) = first.cast::<PyTuple>() {
            Some(ViewShapeArgument::Tuple(shape.clone()))
        } else if let Ok(shape) = first.cast::<PyList>() {
            Some(ViewShapeArgument::List(shape.clone()))
        } else {
            None
        };
        if let Some(shape) = sequence_shape
            && validate_view_shape_first(&shape).is_ok()
        {
            if keyword_error.is_some() {
                return Err(unsupported_view_call_error(positional, keywords)?);
            }
            return Ok(ViewArgument::Shape(shape));
        }
        // The two public overloads each probe the first variadic dimension
        // before mode dispatch. The remaining conversions happen afterward.
        if !is_view_shape_dimension(&first) {
            return Err(unsupported_view_call_error(positional, keywords)?);
        }
        if keyword_error.is_some() {
            return Err(unsupported_view_call_error(positional, keywords)?);
        }
        return Ok(ViewArgument::Shape(ViewShapeArgument::Tuple(
            positional.clone(),
        )));
    }

    let (value, positional_dimension) = match positional.len() {
        0 => (
            keyword_shape.ok_or_else(unsupported_view_argument_error)?,
            false,
        ),
        1 => (positional.get_item(0)?, true),
        _ if keyword_error.is_some() => {
            return Err(unsupported_view_call_error(positional, keywords)?);
        }
        _ => unreachable!("two or more positional view arguments are handled above"),
    };
    let shape = if let Ok(shape) = value.cast::<PyTuple>() {
        ViewShapeArgument::Tuple(shape.clone())
    } else if let Ok(shape) = value.cast::<PyList>() {
        ViewShapeArgument::List(shape.clone())
    } else if let Ok(dtype) = value.cast::<PyDType>() {
        if !positional_dimension {
            if keyword_error.is_none() && keywords.is_some_and(|keywords| keywords.len() == 1) {
                return Err(unsupported_view_size_dtype_error(
                    positional, keywords, dtype,
                )?);
            }
            return Err(unsupported_view_call_error(positional, keywords)?);
        }
        if keyword_error.is_some() {
            return Err(unsupported_view_call_error(positional, keywords)?);
        }
        return Ok(ViewArgument::DType(dtype.clone()));
    } else if is_view_shape_dimension(&value) {
        if !positional_dimension {
            return Err(unsupported_view_integer_error());
        }
        // PyTorch's overloaded argument parser checks the single-integer
        // shape form once for each public overload before mode dispatch, then
        // unpacks it below. Preserve those observable __index__ calls.
        if !is_view_shape_dimension(&value) {
            return Err(unsupported_view_argument_error());
        }
        ViewShapeArgument::Dimension(value)
    } else {
        if positional_dimension && keyword_error.is_some() {
            return Err(unsupported_view_call_error(positional, keywords)?);
        }
        return Err(unsupported_view_argument_error());
    };
    validate_view_shape_first(&shape)?;
    if let Some(error) = keyword_error {
        if positional_dimension {
            return Err(unsupported_view_call_error(positional, keywords)?);
        }
        return Err(error);
    }
    Ok(ViewArgument::Shape(shape))
}

fn validate_view_shape_first(shape: &ViewShapeArgument<'_>) -> PyResult<()> {
    let first = match shape {
        ViewShapeArgument::Dimension(_) => None,
        ViewShapeArgument::Tuple(dimensions) => dimensions.get_item(0).ok(),
        ViewShapeArgument::List(dimensions) => dimensions.get_item(0).ok(),
    };
    let Some(first) = first else {
        return Ok(());
    };
    if is_view_shape_dimension(&first) {
        return Ok(());
    }
    let actual = python_type_name(&first)?;
    Err(PyTypeError::new_err(format!(
        "view(): argument 'size' must be tuple of ints, but found element of type {actual} at pos 0"
    )))
}

fn is_view_shape_dimension(dimension: &Bound<'_, PyAny>) -> bool {
    if dimension.is_instance_of::<PyBool>() {
        return false;
    }
    python_number_index(dimension).is_ok()
}

#[allow(
    unsafe_code,
    reason = "PyNumber_Index invokes the native Python number-index protocol and returns a new reference"
)]
fn python_number_index<'py>(dimension: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>> {
    // SAFETY: `dimension` is live for the call. PyNumber_Index returns a new
    // Python int reference or sets an exception and returns null.
    unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(
            dimension.py(),
            ffi::PyNumber_Index(dimension.as_ptr()),
        )?
        .cast_into::<PyInt>()
        .map_err(Into::into)
    }
}

fn parse_view_shape_argument(shape: ViewShapeArgument<'_>) -> PyResult<Vec<i64>> {
    match shape {
        ViewShapeArgument::Dimension(dimension) => {
            parse_view_shape_dimensions(1, std::iter::once(dimension))
        }
        ViewShapeArgument::Tuple(dimensions) => {
            parse_view_shape_dimensions(dimensions.len(), dimensions.iter())
        }
        ViewShapeArgument::List(dimensions) => {
            parse_view_shape_dimensions(dimensions.len(), dimensions.iter())
        }
    }
}

fn parse_view_shape_dimensions<'py>(
    length: usize,
    dimensions: impl Iterator<Item = Bound<'py, PyAny>>,
) -> PyResult<Vec<i64>> {
    let mut parsed = try_size_vector(length)?;
    for (index, dimension) in dimensions.enumerate() {
        let position = index + 1;
        let indexed = python_number_index(&dimension);
        let Ok(indexed) = indexed else {
            return Err(view_shape_dimension_unpack_error(position, &dimension)?);
        };
        let dimension = indexed.extract::<i64>().map_err(|_| {
            PyTypeError::new_err(format!(
                "view(): argument 'size' failed to unpack the object at pos {position} with error \"Overflow when unpacking long long\""
            ))
        })?;
        try_push_size(&mut parsed, dimension)?;
    }
    Ok(parsed)
}

fn view_shape_dimension_unpack_error(
    position: usize,
    dimension: &Bound<'_, PyAny>,
) -> PyResult<PyErr> {
    let actual = python_type_name(dimension)?;
    Ok(PyTypeError::new_err(format!(
        "view(): argument 'size' failed to unpack the object at pos {position} with error \"type must be tuple of ints,but got {actual}\""
    )))
}

fn unsupported_view_argument_error() -> PyErr {
    PyTypeError::new_err(
        "view() supports exactly one positional integer, tuple, list, or torch.Size shape argument",
    )
}

fn unsupported_view_integer_error() -> PyErr {
    PyTypeError::new_err(
        "view(): variadic integer shapes are not supported; pass a tuple, list, or torch.Size",
    )
}

fn unsupported_view_call_error(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyErr> {
    let summary = call_type_summary(positional, keywords, CallKeywordOrder::PyTorchUnorderedMap)?;
    Ok(PyTypeError::new_err(format!(
        "view() received an invalid combination of arguments - got ({summary}), but expected one of:\n * (torch.dtype dtype)\n * (tuple of ints size)\n"
    )))
}

fn unsupported_view_size_dtype_error(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
    dtype: &Bound<'_, PyDType>,
) -> PyResult<PyErr> {
    let summary = call_type_summary(positional, keywords, CallKeywordOrder::PyTorchUnorderedMap)?;
    let actual = python_type_name(dtype.as_any())?;
    Ok(PyTypeError::new_err(format!(
        "view() received an invalid combination of arguments - got ({summary}), but expected one of:\n * (torch.dtype dtype)\n      didn't match because some of the keywords were incorrect: size\n * (tuple of ints size)\n      didn't match because some of the arguments have invalid types: (!size={actual}!, )\n"
    )))
}

fn unsupported_view_dtype_error() -> PyErr {
    PyTypeError::new_err("view(): dtype reinterpretation is not supported")
}

enum PermuteDimensionArguments<'py> {
    Tuple(Bound<'py, PyTuple>),
    List(Bound<'py, PyList>),
    Variadic(Bound<'py, PyTuple>),
}

fn bind_permute_dimensions(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<i64>> {
    let mut keyword_dimensions = None;
    let mut keyword_error = None;
    if let Some(keywords) = keywords {
        for (key, value) in keywords {
            let key = key.extract::<String>()?;
            if key == "dims" {
                if positional.is_empty() && keyword_dimensions.is_none() {
                    keyword_dimensions = Some(value);
                } else {
                    keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err("permute() got multiple values for argument 'dims'")
                    });
                }
            } else {
                keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "permute() got an unexpected keyword argument '{key}'"
                    ))
                });
            }
        }
    }

    let arguments = if positional.is_empty() {
        let Some(dimensions) = keyword_dimensions else {
            return Err(PyTypeError::new_err(
                "permute() missing 1 required positional arguments: \"dims\"",
            ));
        };
        let Some(arguments) = permute_sequence_arguments(&dimensions) else {
            return Err(permute_argument_type_error(&dimensions, None)?);
        };
        validate_permute_sequence_first(&arguments, &dimensions, None)?;
        if let Some(error) = keyword_error {
            return Err(error);
        }
        arguments
    } else if positional.len() == 1 {
        let first = positional.get_item(0)?;
        let arguments = if let Some(arguments) = permute_sequence_arguments(&first) {
            validate_permute_sequence_first(&arguments, &first, Some(1))?;
            arguments
        } else if is_permute_variadic_dimension(&first)? {
            if !is_permute_variadic_dimension(&first)? {
                return Err(permute_argument_type_error(&first, Some(1))?);
            }
            PermuteDimensionArguments::Variadic(positional.clone())
        } else {
            return Err(permute_argument_type_error(&first, Some(1))?);
        };
        if let Some(error) = keyword_error {
            return Err(error);
        }
        arguments
    } else {
        let first = positional.get_item(0)?;
        if !is_permute_variadic_dimension(&first)? {
            return Err(PyTypeError::new_err(format!(
                "permute() takes 1 positional argument but {} were given",
                positional.len()
            )));
        }
        if !is_permute_variadic_dimension(&first)? {
            return Err(permute_argument_type_error(&first, Some(1))?);
        }
        if let Some(error) = keyword_error {
            return Err(error);
        }
        PermuteDimensionArguments::Variadic(positional.clone())
    };

    parse_permute_dimension_arguments(arguments)
}

fn parse_permute_dimension_arguments(
    arguments: PermuteDimensionArguments<'_>,
) -> PyResult<Vec<i64>> {
    match arguments {
        PermuteDimensionArguments::List(dimensions) => {
            parse_permute_dimensions(dimensions.len(), dimensions.iter())
        }
        PermuteDimensionArguments::Tuple(dimensions)
        | PermuteDimensionArguments::Variadic(dimensions) => {
            parse_permute_dimensions(dimensions.len(), dimensions.iter())
        }
    }
}

fn permute_sequence_arguments<'py>(
    dimensions: &Bound<'py, PyAny>,
) -> Option<PermuteDimensionArguments<'py>> {
    if let Ok(dimensions) = dimensions.cast::<PyTuple>() {
        return Some(PermuteDimensionArguments::Tuple(dimensions.clone()));
    }
    if let Ok(dimensions) = dimensions.cast::<PyList>() {
        return Some(PermuteDimensionArguments::List(dimensions.clone()));
    }
    None
}

fn validate_permute_sequence_first(
    arguments: &PermuteDimensionArguments<'_>,
    outer: &Bound<'_, PyAny>,
    position: Option<usize>,
) -> PyResult<()> {
    let first = match arguments {
        PermuteDimensionArguments::Tuple(dimensions) => dimensions.get_item(0).ok(),
        PermuteDimensionArguments::List(dimensions) => dimensions.get_item(0).ok(),
        PermuteDimensionArguments::Variadic(_) => None,
    };
    let Some(first) = first else {
        return Ok(());
    };
    if !first.is_instance_of::<PyBool>()
        && PyModule::import(first.py(), "operator")?
            .getattr("index")?
            .call1((&first,))
            .is_ok()
    {
        return Ok(());
    }
    let Some(position) = position else {
        return Err(permute_argument_type_error(outer, None)?);
    };
    let actual = python_type_name(&first)?;
    Err(PyTypeError::new_err(format!(
        "permute(): argument 'dims' (position {position}) must be tuple of ints, but found element of type {actual} at pos 0"
    )))
}

fn is_permute_variadic_dimension(dimension: &Bound<'_, PyAny>) -> PyResult<bool> {
    if dimension.is_instance_of::<PyBool>() {
        return Ok(false);
    }
    if dimension.is_instance_of::<PyInt>() {
        return Ok(true);
    }
    Ok(PyModule::import(dimension.py(), "operator")?
        .getattr("index")?
        .call1((dimension,))
        .is_ok())
}

fn permute_argument_type_error(
    dimensions: &Bound<'_, PyAny>,
    position: Option<usize>,
) -> PyResult<PyErr> {
    let position = position.map_or_else(String::new, |position| format!(" (position {position})"));
    let actual = python_type_name(dimensions)?;
    Ok(PyTypeError::new_err(format!(
        "permute(): argument 'dims'{position} must be tuple of ints, not {actual}"
    )))
}

fn parse_permute_dimensions<'py>(
    length: usize,
    dimensions: impl Iterator<Item = Bound<'py, PyAny>>,
) -> PyResult<Vec<i64>> {
    let mut parsed = try_size_vector(length)?;
    for (index, dimension) in dimensions.enumerate() {
        let position = index + 1;
        let indexed = PyModule::import(dimension.py(), "operator")?
            .getattr("index")?
            .call1((&dimension,));
        let Ok(indexed) = indexed else {
            return Err(permute_dimension_unpack_error(position, &dimension)?);
        };
        let dimension = indexed.extract::<i64>().map_err(|_| {
            PyTypeError::new_err(format!(
                "permute(): argument 'dims' failed to unpack the object at pos {position} with error \"Overflow when unpacking long long\""
            ))
        })?;
        try_push_size(&mut parsed, dimension)?;
    }
    Ok(parsed)
}

fn permute_dimension_unpack_error(
    position: usize,
    dimension: &Bound<'_, PyAny>,
) -> PyResult<PyErr> {
    let actual = python_type_name(dimension)?;
    Ok(PyTypeError::new_err(format!(
        "permute(): argument 'dims' failed to unpack the object at pos {position} with error \"type must be tuple of ints,but got {actual}\""
    )))
}

fn permute_tensor(input: &CoreTensor, dimensions: Vec<i64>) -> PyResult<CoreTensor> {
    let rank = input.shape().len();
    if dimensions.len() != rank {
        return Err(permute_error(&TensorError::PermutationRankMismatch {
            dimensions: dimensions.len(),
            rank,
        }));
    }

    let signed_rank = i64::try_from(rank)
        .map_err(|_| PyOverflowError::new_err("tensor rank exceeds the platform limit"))?;
    let mut seen = try_size_vector(rank)?;
    seen.resize(rank, false);
    let mut normalized = try_size_vector(rank)?;
    for dimension in dimensions {
        if dimension < -signed_rank || dimension >= signed_rank {
            return Err(PyIndexError::new_err(format!(
                "Dimension out of range (expected to be in range of [{}, {}], but got {dimension})",
                -signed_rank,
                signed_rank - 1
            )));
        }
        let dimension = if dimension < 0 {
            dimension + signed_rank
        } else {
            dimension
        };
        let dimension = usize::try_from(dimension)
            .map_err(|_| PyOverflowError::new_err("tensor dimension exceeds usize"))?;
        if seen[dimension] {
            return Err(permute_error(&TensorError::DuplicatePermutationDimension {
                dimension,
            }));
        }
        seen[dimension] = true;
        try_push_size(&mut normalized, dimension)?;
    }

    input
        .permute_axes(normalized)
        .map_err(|error| permute_error(&error))
}

fn movedim_tensor(input: &CoreTensor, source: i64, destination: i64) -> PyResult<CoreTensor> {
    let rank = input.shape().len();
    let source = normalize_movedim_dimension(source, rank)?;
    let destination = normalize_movedim_dimension(destination, rank)?;
    let mut dimensions = try_size_vector(rank)?;
    for axis in 0..rank {
        if axis != source {
            let axis = i64::try_from(axis)
                .map_err(|_| PyOverflowError::new_err("tensor rank exceeds the platform limit"))?;
            try_push_size(&mut dimensions, axis)?;
        }
    }
    if rank != 0 {
        let source = i64::try_from(source)
            .map_err(|_| PyOverflowError::new_err("tensor rank exceeds the platform limit"))?;
        dimensions.insert(destination, source);
    }
    permute_tensor(input, dimensions)
}

fn normalize_movedim_dimension(dimension: i64, rank: usize) -> PyResult<usize> {
    let effective_rank = rank.max(1);
    let signed_rank = i64::try_from(effective_rank)
        .map_err(|_| PyOverflowError::new_err("tensor rank exceeds the platform limit"))?;
    if dimension < -signed_rank || dimension >= signed_rank {
        return Err(PyIndexError::new_err(format!(
            "Dimension out of range (expected to be in range of [{}, {}], but got {dimension})",
            -signed_rank,
            signed_rank - 1
        )));
    }
    if rank == 0 {
        return Ok(0);
    }
    usize::try_from(if dimension < 0 {
        dimension + signed_rank
    } else {
        dimension
    })
    .map_err(|_| PyOverflowError::new_err("tensor dimension exceeds usize"))
}

fn validate_dimension_swap_argument_prefix<const N: usize>(
    operation: &str,
    names: &[&str; N],
    arguments: &[Option<ParsedCallArgument<'_>>; N],
    length: usize,
) -> PyResult<()> {
    for (name, argument) in names.iter().zip(arguments.iter()).take(length) {
        let argument = argument
            .as_ref()
            .expect("arguments preceding the first dimension-swap gap are present");
        if *name == "input" {
            if argument.value.cast::<PyTensor>().is_err() {
                let actual = python_type_name(&argument.value)?;
                return Err(dimension_swap_argument_type_error(
                    operation,
                    name,
                    argument.position,
                    "Tensor",
                    &actual,
                ));
            }
        } else {
            validate_dimension_swap_dimension(operation, name, argument.position, &argument.value)?;
        }
    }
    Ok(())
}

#[repr(C)]
struct PyTypeObjectNamePrefix {
    _ob_base: ffi::PyVarObject,
    tp_name: *const c_char,
}

#[allow(
    unsafe_code,
    reason = "PyType_GetFlags reads immutable flags from a live type through the stable ABI"
)]
fn is_native_immutable_python_type(value_type: &Bound<'_, PyType>) -> bool {
    // SAFETY: value_type is a live Python type object for the duration of the call.
    let flags = unsafe { ffi::PyType_GetFlags(value_type.as_type_ptr()) };
    flags & ffi::Py_TPFLAGS_IMMUTABLETYPE != 0 && flags & ffi::Py_TPFLAGS_HEAPTYPE == 0
}

#[allow(
    unsafe_code,
    reason = "CPython exposes the non-overridable tp_name in every live type-object prefix"
)]
fn cpython_type_object_name<'a>(value_type: &'a Bound<'_, PyType>) -> PyResult<&'a CStr> {
    let prefix = value_type.as_type_ptr().cast::<PyTypeObjectNamePrefix>();
    // SAFETY: every classic CPython type object starts with PyVarObject and
    // tp_name, which remains live while value_type is borrowed.
    let name = unsafe { (*prefix).tp_name };
    if name.is_null() {
        return Err(PyRuntimeError::new_err("Python type has no tp_name"));
    }
    // SAFETY: CPython requires tp_name to remain NUL-terminated for the
    // lifetime of the live type object.
    Ok(unsafe { CStr::from_ptr(name) })
}

fn has_numpy_integer_ancestry(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = value.py();
    // Calling type's descriptor directly bypasses metaclass overrides, while
    // __mro__ itself is immutable for the duration of this check.
    let mro = py
        .get_type::<PyType>()
        .getattr("__getattribute__")?
        .call1((value.get_type(), "__mro__"))?
        .cast_into::<PyTuple>()?;
    for base in mro.iter() {
        let base = base.cast_into::<PyType>()?;
        if is_native_immutable_python_type(&base)
            && cpython_type_object_name(&base)? == c"numpy.integer"
        {
            return Ok(true);
        }
    }
    Ok(false)
}

#[allow(
    unsafe_code,
    reason = "CPython exposes tp_name only as a type-object field before Python 3.13"
)]
fn cpython_type_name(value: &Bound<'_, PyAny>) -> PyResult<String> {
    let value_type = value.get_type();
    let prefix = value_type.as_type_ptr().cast::<PyTypeObjectNamePrefix>();
    // SAFETY: every classic CPython type object starts with PyVarObject and
    // tp_name. The value keeps its type alive, and the attached interpreter
    // prevents a concurrent Python-level type-name mutation while it is copied.
    let name = unsafe { (*prefix).tp_name };
    if name.is_null() {
        return Err(PyRuntimeError::new_err("Python type has no tp_name"));
    }
    // SAFETY: CPython requires tp_name to point to a NUL-terminated UTF-8 name
    // while the type name remains unchanged; no Python callback can run before
    // this function copies it into owned Rust storage.
    let name = unsafe { CStr::from_ptr(name) }
        .to_str()
        .map_err(|_| PyRuntimeError::new_err("Python tp_name is not valid UTF-8"))?;
    let mut output = String::new();
    try_push_string(&mut output, name)?;
    Ok(output)
}

#[allow(
    unsafe_code,
    reason = "CPython exposes tp_name only as a type-object field before Python 3.13"
)]
fn cpython_type_name_with(
    value: &Bound<'_, PyAny>,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<String> {
    let value_type = value.get_type();
    let prefix = value_type.as_type_ptr().cast::<PyTypeObjectNamePrefix>();
    // SAFETY: see cpython_type_name; this variant performs the same immediate
    // copy while using the formatter's preallocated allocation error.
    let name = unsafe { (*prefix).tp_name };
    if name.is_null() {
        return Err(PyRuntimeError::new_err("Python type has no tp_name"));
    }
    // SAFETY: no Python callback can mutate the type before this copy ends.
    let name = unsafe { CStr::from_ptr(name) }
        .to_str()
        .map_err(|_| PyRuntimeError::new_err("Python tp_name is not valid UTF-8"))?;
    try_string_from_str_with(name, allocation)
}

pub(crate) fn native_pytorch_type_name(value: &Bound<'_, PyAny>) -> Option<&'static str> {
    if value.is_exact_instance_of::<PyTensor>() {
        Some("Tensor")
    } else if value.is_exact_instance_of::<PyDType>() {
        Some("torch.dtype")
    } else if value.is_exact_instance_of::<PyDevice>() {
        Some("torch.device")
    } else if value.is_exact_instance_of::<PyMemoryFormat>() {
        Some("torch.memory_format")
    } else {
        None
    }
}

fn dynamically_created_pytorch_type_name(name: &str) -> Option<&'static str> {
    match name {
        "torch_rs.Size" => Some("torch.Size"),
        "torch_rs.layout" => Some("torch.layout"),
        "torch_rs.finfo" => Some("torch.finfo"),
        _ => None,
    }
}

pub(crate) fn python_type_name(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Some(name) = native_pytorch_type_name(value) {
        let mut output = String::new();
        try_push_string(&mut output, name)?;
        Ok(output)
    } else {
        let name = cpython_type_name(value)?;
        let Some(pytorch_name) = dynamically_created_pytorch_type_name(&name) else {
            return Ok(name);
        };
        let mut output = String::new();
        try_push_string(&mut output, pytorch_name)?;
        Ok(output)
    }
}

fn python_type_name_with(
    value: &Bound<'_, PyAny>,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<String> {
    if let Some(name) = native_pytorch_type_name(value) {
        try_string_from_str_with(name, allocation)
    } else {
        let name = cpython_type_name_with(value, allocation)?;
        let Some(pytorch_name) = dynamically_created_pytorch_type_name(&name) else {
            return Ok(name);
        };
        try_string_from_str_with(pytorch_name, allocation)
    }
}

fn dimension_swap_argument_type_error(
    operation: &str,
    argument: &str,
    position: Option<usize>,
    expected: &str,
    actual: &str,
) -> PyErr {
    let position = position.map_or_else(String::new, |position| format!(" (position {position})"));
    PyTypeError::new_err(format!(
        "{operation}(): argument '{argument}'{position} must be {expected}, not {actual}"
    ))
}

fn parse_integer_indices<'py>(
    tensor: &CoreTensor,
    length: usize,
    indices: impl Iterator<Item = Bound<'py, PyAny>>,
) -> PyResult<Vec<i64>> {
    let mut parsed = try_size_vector(length)?;
    let mut offset = tensor.storage_offset();
    for (dimension, index) in indices.enumerate() {
        let index = parse_integer_index(&index)?;
        offset = tensor
            .checked_index_offset(offset, dimension, index)
            .map_err(|error| tensor_error(&error))?;
        try_push_size(&mut parsed, index)?;
    }
    Ok(parsed)
}

fn is_fast_integer_index(index: &Bound<'_, PyAny>) -> PyResult<bool> {
    if index.is_instance_of::<PyBool>() {
        return Ok(false);
    }
    if index.is_instance_of::<PyInt>() {
        return Ok(true);
    }
    let Ok(numpy) = PyModule::import(index.py(), "numpy") else {
        return Ok(false);
    };
    index.is_instance(&numpy.getattr("integer")?)
}

fn is_exact_full_slice(index: &Bound<'_, PyAny>) -> PyResult<bool> {
    let Ok(slice) = index.cast::<PySlice>() else {
        return Ok(false);
    };
    Ok(slice.getattr("start")?.is_none()
        && slice.getattr("stop")?.is_none()
        && slice.getattr("step")?.is_none())
}

// The caller checks tuple arity against the tensor rank first so lower-rank
// integer-prefix/full-slice forms retain PyTorch's "too many indices" error
// without converting their integer-like objects.
fn parse_leading_integer_full_slice(
    tensor: &CoreTensor,
    indices: &Bound<'_, PyTuple>,
) -> PyResult<Option<Vec<i64>>> {
    let Some(integer_dimensions) = indices.len().checked_sub(1) else {
        return Ok(None);
    };
    if integer_dimensions == 0 {
        return Ok(None);
    }
    if !is_exact_full_slice(&indices.get_item(integer_dimensions)?)? {
        return Ok(None);
    }
    parse_integer_indices(
        tensor,
        integer_dimensions,
        indices.iter().take(integer_dimensions),
    )
    .map(Some)
}

// Return how many tensor dimensions an alias-only tuple consumes. A single
// Ellipsis consumes the unmentioned dimensions, so it does not increment the
// rank requirement.
fn metadata_alias_tuple_dimensions(indices: &Bound<'_, PyTuple>) -> PyResult<Option<usize>> {
    let mut indexed_dimensions = 0;
    let mut contains_ellipsis = false;
    for index in indices.iter() {
        if index.is_instance_of::<PyEllipsis>() {
            if contains_ellipsis {
                return Ok(None);
            }
            contains_ellipsis = true;
        } else if is_exact_full_slice(&index)? {
            indexed_dimensions += 1;
        } else {
            return Ok(None);
        }
    }
    Ok(Some(indexed_dimensions))
}

fn parse_integer_index(index: &Bound<'_, PyAny>) -> PyResult<i64> {
    if index.is_instance_of::<PyBool>() {
        return Err(invalid_index(index));
    }
    if index.is_instance_of::<PyInt>() {
        return index
            .extract::<i64>()
            .map_err(|_| PyValueError::new_err("Overflow when unpacking long long"));
    }

    let indexed = PyModule::import(index.py(), "operator")
        .and_then(|operator| operator.getattr("index"))
        .and_then(|operator_index| operator_index.call1((index,)));
    match indexed {
        Ok(indexed) => indexed
            .extract::<i64>()
            .map_err(|_| PyValueError::new_err("Overflow when unpacking long long")),
        Err(_) => Err(invalid_index(index)),
    }
}

fn invalid_index(index: &Bound<'_, PyAny>) -> PyErr {
    let type_name = index
        .get_type()
        .name()
        .ok()
        .and_then(|name| name.to_str().ok().map(str::to_owned))
        .unwrap_or_else(|| "unknown".to_owned());
    PyIndexError::new_err(format!(
        "only integers, slices (`:`), ellipsis (`...`), None and long or byte Variables are valid indices (got {type_name})"
    ))
}

fn too_many_indices(dimensions: usize) -> PyErr {
    PyIndexError::new_err(TensorError::TooManyIndices { dimensions }.to_string())
}

pub(crate) fn normalize_dimension(dimension: i64, rank: usize) -> PyResult<usize> {
    let rank = i64::try_from(rank)
        .map_err(|_| PyOverflowError::new_err("tensor rank exceeds the platform limit"))?;
    if rank == 0 {
        return Err(PyIndexError::new_err(format!(
            "Dimension specified as {dimension} but tensor has no dimensions"
        )));
    }
    if dimension < -rank || dimension >= rank {
        return Err(PyIndexError::new_err(format!(
            "Dimension out of range (expected to be in range of [{}, {}], but got {dimension})",
            -rank,
            rank - 1
        )));
    }
    usize::try_from(if dimension < 0 {
        dimension + rank
    } else {
        dimension
    })
    .map_err(|_| PyOverflowError::new_err("tensor dimension exceeds the platform limit"))
}

fn normalize_unbind_dimension(dimension: i64, rank: usize) -> PyResult<usize> {
    if rank != 0 {
        return normalize_dimension(dimension, rank);
    }
    if !(-1..=0).contains(&dimension) {
        return Err(PyIndexError::new_err(format!(
            "Dimension out of range (expected to be in range of [-1, 0], but got {dimension})"
        )));
    }
    Err(PyIndexError::new_err(
        "Dimension specified as 0 but tensor has no dimensions",
    ))
}

fn parse_reshape_shape(
    shape_dimensions: &Bound<'_, PyTuple>,
    keyword_shape: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<i64>> {
    if let Some(shape) = keyword_shape {
        if !shape_dimensions.is_empty() {
            return Err(PyTypeError::new_err(
                "reshape() received both positional and keyword shape arguments",
            ));
        }
        if let Ok(dimensions) = shape.cast::<PyList>() {
            return parse_reshape_dimensions(dimensions.len(), dimensions.iter());
        }
        if let Ok(dimensions) = shape.cast::<PyTuple>() {
            return parse_reshape_dimensions(dimensions.len(), dimensions.iter());
        }
        return Err(PyTypeError::new_err(
            "reshape(): argument 'shape' must be a tuple or list of integers",
        ));
    }

    if shape_dimensions.is_empty() {
        return Err(PyTypeError::new_err(
            "reshape() missing required shape arguments",
        ));
    }
    if shape_dimensions.len() == 1 {
        let shape = shape_dimensions.get_item(0)?;
        if let Ok(dimensions) = shape.cast::<PyList>() {
            return parse_reshape_dimensions(dimensions.len(), dimensions.iter());
        }
        if let Ok(dimensions) = shape.cast::<PyTuple>() {
            return parse_reshape_dimensions(dimensions.len(), dimensions.iter());
        }
    }
    parse_reshape_dimensions(shape_dimensions.len(), shape_dimensions.iter())
}

fn bind_top_level_reshape_shape<'py>(
    argument: &ParsedCallArgument<'py>,
) -> PyResult<BoundTopLevelReshapeShape<'py>> {
    if let Ok(dimensions) = argument.value.cast::<PyList>() {
        bind_top_level_reshape_dimensions(argument, dimensions.len(), dimensions.iter())
    } else if let Ok(dimensions) = argument.value.cast::<PyTuple>() {
        bind_top_level_reshape_dimensions(argument, dimensions.len(), dimensions.iter())
    } else if let Some(probed) = probe_torch_function_override(&argument.value) {
        let mut overrides = Vec::new();
        overrides
            .try_reserve_exact(1)
            .map_err(|_| PyMemoryError::new_err("unable to allocate reshape dispatch operands"))?;
        overrides.push(probed);
        Ok(BoundTopLevelReshapeShape::Override(overrides))
    } else {
        Err(top_level_reshape_shape_type_error(argument)?)
    }
}

fn parse_reshape_dimensions<'py>(
    length: usize,
    dimensions: impl Iterator<Item = Bound<'py, PyAny>>,
) -> PyResult<Vec<i64>> {
    let mut parsed = try_size_vector(length)?;
    for (index, dimension) in dimensions.enumerate() {
        if dimension.is_instance_of::<PyBool>() {
            return Err(invalid_reshape_dimension(
                index,
                "bool is not a valid shape dimension",
            ));
        }
        try_push_size(
            &mut parsed,
            dimension
                .extract::<i64>()
                .map_err(|error| invalid_reshape_dimension(index, &error.to_string()))?,
        )?;
    }
    Ok(parsed)
}

fn bind_top_level_reshape_dimensions<'py>(
    argument: &ParsedCallArgument<'_>,
    length: usize,
    dimensions: impl Iterator<Item = Bound<'py, PyAny>>,
) -> PyResult<BoundTopLevelReshapeShape<'py>> {
    let mut native_dimensions = try_size_vector(length)?;
    let mut overrides = None;
    for (index, dimension) in dimensions.enumerate() {
        if let Some(probed) = probe_torch_function_override(&dimension) {
            if overrides.is_none() {
                let mut collected = Vec::new();
                collected
                    .try_reserve_exact(length.saturating_sub(index))
                    .map_err(|_| {
                        PyMemoryError::new_err("unable to allocate reshape dispatch operands")
                    })?;
                overrides = Some(collected);
            }
            if let Some(overrides) = &mut overrides {
                overrides.push(probed);
            }
        } else if index == 0 {
            validate_top_level_reshape_dimension(argument, index, &dimension)?;
        }

        if overrides.is_none() {
            try_push_size(&mut native_dimensions, dimension)?;
        }
    }
    if let Some(overrides) = overrides {
        Ok(BoundTopLevelReshapeShape::Override(overrides))
    } else {
        Ok(BoundTopLevelReshapeShape::Native(
            BoundTopLevelNativeReshapeShape {
                dimensions: native_dimensions,
            },
        ))
    }
}

fn validate_top_level_reshape_dimension(
    argument: &ParsedCallArgument<'_>,
    index: usize,
    dimension: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if dimension.is_instance_of::<PyBool>() {
        return Err(top_level_reshape_shape_dimension_type_error(
            argument, index, dimension,
        )?);
    }
    let Ok(indexed) = python_number_index(dimension) else {
        return Err(top_level_reshape_shape_dimension_type_error(
            argument, index, dimension,
        )?);
    };
    drop(indexed);
    Ok(())
}

fn parse_top_level_reshape_native_dimensions(
    shape: &BoundTopLevelNativeReshapeShape<'_>,
) -> PyResult<Vec<i64>> {
    let mut parsed = try_size_vector(shape.dimensions.len())?;
    for (index, dimension) in shape.dimensions.iter().enumerate() {
        let position = index + 1;
        let Ok(indexed) = python_number_index(dimension) else {
            return Err(top_level_reshape_shape_dimension_unpack_type_error(
                position, dimension,
            )?);
        };
        let dimension = indexed
            .extract::<i64>()
            .map_err(|_| top_level_reshape_shape_dimension_unpack_error(position))?;
        try_push_size(&mut parsed, dimension)?;
    }
    Ok(parsed)
}

fn top_level_reshape_shape_type_error(argument: &ParsedCallArgument<'_>) -> PyResult<PyErr> {
    let actual = python_type_name(&argument.value)?;
    Ok(PyTypeError::new_err(format!(
        "reshape(): argument 'shape'{} must be tuple of ints, not {actual}",
        position_suffix(argument.position)
    )))
}

fn top_level_reshape_shape_dimension_type_error(
    argument: &ParsedCallArgument<'_>,
    index: usize,
    value: &Bound<'_, PyAny>,
) -> PyResult<PyErr> {
    if argument.position.is_none() {
        top_level_reshape_shape_type_error(argument)
    } else {
        top_level_reshape_shape_element_type_error(argument, index, value)
    }
}

fn top_level_reshape_shape_dimension_unpack_error(position: usize) -> PyErr {
    PyTypeError::new_err(format!(
        "reshape(): argument 'shape' failed to unpack the object at pos {position} with error \"Overflow when unpacking long long\""
    ))
}

fn top_level_reshape_shape_dimension_unpack_type_error(
    position: usize,
    value: &Bound<'_, PyAny>,
) -> PyResult<PyErr> {
    let actual = python_type_name(value)?;
    Ok(PyTypeError::new_err(format!(
        "reshape(): argument 'shape' failed to unpack the object at pos {position} with error \"type must be tuple of ints,but got {actual}\""
    )))
}

fn top_level_reshape_shape_element_type_error(
    argument: &ParsedCallArgument<'_>,
    index: usize,
    value: &Bound<'_, PyAny>,
) -> PyResult<PyErr> {
    let actual = python_type_name(value)?;
    Ok(PyTypeError::new_err(format!(
        "reshape(): argument 'shape'{} must be tuple of ints, but found element of type {actual} at pos {index}",
        position_suffix(argument.position)
    )))
}

fn position_suffix(position: Option<usize>) -> String {
    position.map_or_else(String::new, |position| format!(" (position {position})"))
}

fn invalid_reshape_dimension(index: usize, reason: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "reshape(): shape element at index {index} is invalid: {reason}"
    ))
}

fn parse_size_dimensions<'py>(
    length: usize,
    dimensions: impl Iterator<Item = Bound<'py, PyAny>>,
) -> PyResult<Vec<i64>> {
    let mut parsed = try_size_vector(length)?;

    for (index, dimension) in dimensions.enumerate() {
        if dimension.is_instance_of::<PyBool>() {
            return Err(invalid_size_dimension(
                index,
                "bool is not a valid size dimension",
            ));
        }
        try_push_size(
            &mut parsed,
            dimension
                .extract::<i64>()
                .map_err(|error| invalid_size_dimension(index, &error.to_string()))?,
        )?;
    }

    Ok(parsed)
}

fn validate_size(size: Vec<i64>) -> PyResult<Vec<usize>> {
    if let Some(dimension) = size.iter().find(|dimension| **dimension < 0) {
        return Err(PyRuntimeError::new_err(format!(
            "Trying to create tensor with negative dimension {dimension}: {size:?}"
        )));
    }

    let mut shape = try_size_vector(size.len())?;
    for dimension in size {
        try_push_size(
            &mut shape,
            usize::try_from(dimension).map_err(|_| {
                PyRuntimeError::new_err(format!(
                    "tensor dimension {dimension} exceeds the platform size limit"
                ))
            })?,
        )?;
    }
    Ok(shape)
}

fn try_size_vector<T>(length: usize) -> PyResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| python_allocation_error())?;
    Ok(values)
}

fn try_push_size<T>(values: &mut Vec<T>, value: T) -> PyResult<()> {
    values
        .try_reserve(1)
        .map_err(|_| python_allocation_error())?;
    values.push(value);
    Ok(())
}

fn try_push_string(output: &mut String, value: &str) -> PyResult<()> {
    output
        .try_reserve(value.len())
        .map_err(|_| python_allocation_error())?;
    output.push_str(value);
    Ok(())
}

struct PythonAllocationFallback<'py> {
    py: Python<'py>,
    error: PyErr,
}

impl<'py> PythonAllocationFallback<'py> {
    fn new(py: Python<'py>) -> Self {
        let error = python_allocation_error();
        error.value(py);
        Self { py, error }
    }

    fn error(&self) -> PyErr {
        self.error.clone_ref(self.py)
    }
}

fn try_size_vector_with<T>(
    length: usize,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| allocation.error())?;
    Ok(values)
}

fn try_push_size_with<T>(
    values: &mut Vec<T>,
    value: T,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<()> {
    values.try_reserve(1).map_err(|_| allocation.error())?;
    values.push(value);
    Ok(())
}

fn try_push_string_with(
    output: &mut String,
    value: &str,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<()> {
    output
        .try_reserve(value.len())
        .map_err(|_| allocation.error())?;
    output.push_str(value);
    Ok(())
}

fn try_string_from_str_with(
    value: &str,
    allocation: &PythonAllocationFallback<'_>,
) -> PyResult<String> {
    let mut output = String::new();
    try_push_string_with(&mut output, value, allocation)?;
    Ok(output)
}

fn python_allocation_error() -> PyErr {
    PyRuntimeError::new_err("std::bad_alloc")
}

fn invalid_size_dimension(index: usize, reason: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "full(): size element at index {index} is invalid: {reason}"
    ))
}

fn parse_fill_value(function: &str, fill_value: &Bound<'_, PyAny>) -> PyResult<ParsedFillValue> {
    if let Ok(tensor) = fill_value.cast::<PyTensor>() {
        let tensor = tensor.try_borrow()?;
        if tensor.inner.requires_grad() || !tensor.inner.shape().is_empty() {
            return Err(tensor_fill_value_type_error(function));
        }
        return tensor
            .inner
            .item()
            .map(ParsedFillValue::TensorScalar)
            .map_err(|error| tensor_error(&error));
    }

    if fill_value.is_instance_of::<PyInt>() {
        return parse_integer_fill_value(fill_value);
    }

    if fill_value.is_instance_of::<PyFloat>() {
        return fill_value.extract::<f64>().map(ParsedFillValue::Float);
    }

    parse_numpy_fill_value(function, fill_value)
}

fn parse_arithmetic_scalar(value: &Bound<'_, PyAny>) -> PyResult<Option<ParsedArithmeticScalar>> {
    if value.is_exact_instance_of::<PyBool>() {
        return value
            .is_truthy()
            .map(ParsedArithmeticScalar::PythonBool)
            .map(Some);
    }

    if value.is_instance_of::<PyInt>() {
        return parse_integer_fill_value(value)
            .map(ParsedArithmeticScalar::Number)
            .map(Some);
    }

    if value.is_instance_of::<PyFloat>() {
        return value
            .extract::<f64>()
            .map(ParsedFillValue::Float)
            .map(ParsedArithmeticScalar::Number)
            .map(Some);
    }

    parse_numpy_arithmetic_scalar(value)
}

fn parse_numpy_fill_value(
    function: &str,
    fill_value: &Bound<'_, PyAny>,
) -> PyResult<ParsedFillValue> {
    parse_numpy_value(
        fill_value,
        || invalid_fill_value(function),
        "NumPy integer fill_value is outside the signed 64-bit range",
    )
}

fn parse_numpy_arithmetic_scalar(
    value: &Bound<'_, PyAny>,
) -> PyResult<Option<ParsedArithmeticScalar>> {
    let Ok(numpy) = PyModule::import(value.py(), "numpy") else {
        return Ok(None);
    };
    let generic = numpy.getattr("generic")?;
    if !value.is_instance(&generic)? {
        return Ok(None);
    }

    let numpy_bool = numpy.getattr("bool_")?;
    if value.is_instance(&numpy_bool)? {
        return value
            .is_truthy()
            .map(|value| ParsedFillValue::SignedInteger(i64::from(value)))
            .map(ParsedArithmeticScalar::Number)
            .map(Some);
    }

    let numpy_integer = numpy.getattr("integer")?;
    if value.is_instance(&numpy_integer)? {
        if let Ok(value) = value.extract::<i64>() {
            return Ok(Some(ParsedArithmeticScalar::Number(
                ParsedFillValue::SignedInteger(value),
            )));
        }
        value.extract::<u64>().map_err(|_| {
            PyTypeError::new_err("NumPy integer operand is outside the supported 64-bit range")
        })?;
        return Ok(Some(ParsedArithmeticScalar::WideNumpyUnsigned));
    }

    let numpy_floating = numpy.getattr("floating")?;
    if value.is_instance(&numpy_floating)? {
        return value
            .extract::<f64>()
            .map(ParsedFillValue::Float)
            .map(ParsedArithmeticScalar::Number)
            .map(Some);
    }

    Ok(None)
}

fn parse_numpy_value(
    value: &Bound<'_, PyAny>,
    invalid_value: impl Fn() -> PyErr,
    integer_range_error: &'static str,
) -> PyResult<ParsedFillValue> {
    let numpy = PyModule::import(value.py(), "numpy").map_err(|_| invalid_value())?;
    let generic = numpy.getattr("generic").map_err(|_| invalid_value())?;
    if !value.is_instance(&generic)? {
        return Err(invalid_value());
    }

    let numpy_bool = numpy.getattr("bool_").map_err(|_| invalid_value())?;
    if value.is_instance(&numpy_bool)? {
        return value
            .is_truthy()
            .map(|value| ParsedFillValue::SignedInteger(i64::from(value)));
    }

    let numpy_integer = numpy.getattr("integer").map_err(|_| invalid_value())?;
    if value.is_instance(&numpy_integer)? {
        return value
            .extract::<i64>()
            .map(ParsedFillValue::SignedInteger)
            .map_err(|_| PyTypeError::new_err(integer_range_error));
    }

    let numpy_floating = numpy.getattr("floating").map_err(|_| invalid_value())?;
    if value.is_instance(&numpy_floating)? {
        return value
            .extract::<f64>()
            .map(ParsedFillValue::Float)
            .map_err(|_| invalid_value());
    }

    Err(invalid_value())
}

fn parse_integer_fill_value(fill_value: &Bound<'_, PyAny>) -> PyResult<ParsedFillValue> {
    if let Ok(value) = fill_value.extract::<i64>() {
        return Ok(ParsedFillValue::SignedInteger(value));
    }

    if let Ok(value) = fill_value.extract::<u64>() {
        return Ok(ParsedFillValue::UnsignedInteger(value));
    }

    Err(PyOverflowError::new_err(
        "Python integer is outside the supported scalar range",
    ))
}

fn invalid_fill_value(function: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "{function}(): fill_value must be a number or zero-dimensional tensor"
    ))
}

fn tensor_fill_value_type_error(function: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "{function}(): argument 'fill_value' (position 2) must be Number, not Tensor"
    ))
}

fn bool_subtraction_error() -> PyErr {
    PyRuntimeError::new_err(
        "Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.",
    )
}

fn creation_shape_error(error: &TensorError, shape: &[usize]) -> PyErr {
    if matches!(
        error,
        TensorError::ElementCountOverflow | TensorError::StorageCapacityOverflow { .. }
    ) {
        PyRuntimeError::new_err(format!(
            "Storage size calculation overflowed with sizes={shape:?}"
        ))
    } else {
        tensor_error(error)
    }
}

fn eye_shape_error(error: &TensorError, shape: &[usize]) -> PyErr {
    match error {
        TensorError::ElementCountOverflow => {
            PyRuntimeError::new_err("numel: integer multiplication overflow")
        }
        TensorError::StorageCapacityOverflow { .. } if eye_numel_exceeds_signed_limit(shape) => {
            PyRuntimeError::new_err("numel: integer multiplication overflow")
        }
        TensorError::StorageCapacityOverflow { .. } => PyRuntimeError::new_err(format!(
            "Storage size calculation overflowed with sizes={shape:?}"
        )),
        _ => tensor_error(error),
    }
}

fn eye_numel_exceeds_signed_limit(shape: &[usize]) -> bool {
    shape
        .iter()
        .try_fold(1_usize, |elements, dimension| {
            elements.checked_mul(*dimension)
        })
        .is_none_or(|elements| elements > isize::MAX.unsigned_abs())
}

fn scalar_creation_error(error: &TensorError, scalar_dimension: Option<usize>) -> PyErr {
    if let Some(dimension) = scalar_dimension
        && matches!(
            error,
            TensorError::ElementCountOverflow | TensorError::StorageCapacityOverflow { .. }
        )
    {
        PyRuntimeError::new_err(format!(
            "Storage size calculation overflowed with sizes=[{dimension}]"
        ))
    } else {
        tensor_error(error)
    }
}

fn creation_factory_error(
    error: &TensorError,
    shape: &[usize],
    scalar_dimension: Option<usize>,
) -> PyErr {
    if scalar_dimension.is_some() {
        scalar_creation_error(error, scalar_dimension)
    } else {
        creation_shape_error(error, shape)
    }
}

impl ParsedFillValue {
    fn is_arithmetic_one(&self) -> bool {
        match self {
            Self::Float(value) => value.to_bits() == 1.0_f64.to_bits(),
            Self::SignedInteger(value) => *value == 1,
            Self::UnsignedInteger(value) => *value == 1,
            Self::TensorScalar(value) => value.to_bits() == 1.0_f32.to_bits(),
        }
    }

    fn into_f32(self) -> PyResult<f32> {
        match self {
            Self::Float(value) => {
                if value.is_finite() && value.abs() > f64::from(f32::MAX) {
                    return Err(fill_value_overflow());
                }
                #[allow(clippy::cast_possible_truncation)]
                let converted = value as f32;
                Ok(converted)
            }
            Self::SignedInteger(value) => {
                #[allow(clippy::cast_precision_loss)]
                let converted = value as f32;
                Ok(converted)
            }
            Self::UnsignedInteger(value) => {
                #[allow(clippy::cast_precision_loss)]
                let converted = value as f32;
                Ok(converted)
            }
            Self::TensorScalar(value) => Ok(value),
        }
    }

    fn into_arithmetic_f32(self) -> f32 {
        match self {
            Self::Float(value) => {
                #[allow(clippy::cast_possible_truncation)]
                let converted = value as f32;
                converted
            }
            Self::SignedInteger(value) => {
                #[allow(clippy::cast_precision_loss)]
                let converted = value as f32;
                converted
            }
            Self::UnsignedInteger(value) => {
                #[allow(clippy::cast_precision_loss)]
                let converted = value as f32;
                converted
            }
            Self::TensorScalar(value) => value,
        }
    }

    fn into_scalar_tensor_f32(self) -> PyResult<f32> {
        match self {
            Self::Float(value) => {
                if value.is_finite() && value.abs() > f64::from(f32::MAX) {
                    return Err(scalar_tensor_overflow());
                }
                #[allow(clippy::cast_possible_truncation)]
                let converted = value as f32;
                Ok(converted)
            }
            Self::SignedInteger(value) => {
                #[allow(clippy::cast_precision_loss)]
                let converted = value as f32;
                Ok(converted)
            }
            Self::UnsignedInteger(value) => {
                #[allow(clippy::cast_precision_loss)]
                let converted = value as f32;
                Ok(converted)
            }
            Self::TensorScalar(value) => Ok(value),
        }
    }
}

impl ParsedArithmeticScalar {
    fn is_python_bool(&self) -> bool {
        matches!(self, Self::PythonBool(_))
    }

    fn is_one(&self) -> bool {
        match self {
            Self::PythonBool(value) => *value,
            Self::Number(value) => value.is_arithmetic_one(),
            Self::WideNumpyUnsigned => false,
        }
    }

    fn into_f32(self) -> f32 {
        match self {
            Self::PythonBool(value) => f32::from(u8::from(value)),
            Self::Number(value) => value.into_arithmetic_f32(),
            Self::WideNumpyUnsigned => {
                unreachable!("wide NumPy unsigned operands are dispatched before conversion")
            }
        }
    }
}

fn fill_value_overflow() -> PyErr {
    PyRuntimeError::new_err("value cannot be converted to type float without overflow")
}

fn scalar_tensor_overflow() -> PyErr {
    PyRuntimeError::new_err("value cannot be converted to type float without overflow")
}

fn flatten_buffer(
    value: &Bound<'_, PyAny>,
    dtype_was_explicit: bool,
) -> PyResult<Option<(Vec<f32>, Vec<usize>)>> {
    let view = PyMemoryView::from(value)?;

    let dimensions = view.getattr("ndim")?.extract::<usize>()?;
    if dimensions == 0 {
        if value.py().version_info() < (3, 12) {
            return Err(buffer_shape_error(value)?);
        }
        return Err(PyTypeError::new_err("0-dim memory has no length"));
    }
    let elements = view.len()?;
    if elements == 0 {
        return Ok(Some((Vec::new(), vec![0])));
    }
    if dimensions != 1 {
        return Err(buffer_shape_error(value)?);
    }

    let format_description = view.getattr("format")?.extract::<String>()?;
    let format = match format_description.as_bytes() {
        [format] | [b'@', format] => *format,
        _ => return Err(buffer_shape_error(value)?),
    };
    if format == b'c' && dtype_was_explicit {
        return Ok(None);
    }
    if format == b'c' {
        return Err(PyTypeError::new_err("new(): invalid data type 'bytes'"));
    }

    let item_size = view.getattr("itemsize")?.extract::<usize>()?;
    if !buffer_format_has_item_size(format, item_size) {
        return Err(buffer_shape_error(value)?);
    }
    if format == b'e' && value.py().version_info() < (3, 12) {
        return Err(buffer_shape_error(value)?);
    }
    if format == b'e' || (format == b'?' && value.py().version_info() >= (3, 14)) {
        let mut output = Vec::new();
        output.try_reserve_exact(elements).map_err(|_| {
            PyMemoryError::new_err("unable to allocate native tensor storage for buffer")
        })?;
        for index in 0..elements {
            output.push(view.get_item(index)?.extract::<f32>()?);
        }
        return Ok(Some((output, vec![elements])));
    }

    let contiguous = view.call_method0("tobytes")?;
    let contiguous = contiguous.cast::<PyBytes>()?;
    let bytes = contiguous.as_bytes();
    let expected_bytes = elements
        .checked_mul(item_size)
        .ok_or_else(|| PyOverflowError::new_err("buffer size overflowed usize"))?;
    if bytes.len() != expected_bytes {
        return Err(PyValueError::new_err(
            "buffer length is inconsistent with its shape and item size",
        ));
    }

    let mut output = Vec::new();
    output.try_reserve_exact(elements).map_err(|_| {
        PyMemoryError::new_err("unable to allocate native tensor storage for buffer")
    })?;
    for item in bytes.chunks_exact(item_size) {
        let Some(converted) = buffer_item_as_f32(format, item) else {
            return Err(buffer_shape_error(value)?);
        };
        output.push(converted);
    }
    Ok(Some((output, vec![elements])))
}

fn buffer_shape_error(value: &Bound<'_, PyAny>) -> PyResult<PyErr> {
    let type_name = value.get_type().name()?;
    Ok(PyValueError::new_err(format!(
        "could not determine the shape of object type '{type_name}'"
    )))
}

fn unsupported_tensor_data_error(
    value: &Bound<'_, PyAny>,
    dtype_was_explicit: bool,
) -> PyResult<PyErr> {
    let type_name = python_type_name(value)?;
    if dtype_was_explicit {
        Ok(PyTypeError::new_err(format!(
            "must be real number, not {type_name}"
        )))
    } else {
        Ok(PyRuntimeError::new_err(format!(
            "Could not infer dtype of {type_name}"
        )))
    }
}

fn buffer_format_has_item_size(format: u8, item_size: usize) -> bool {
    match format {
        b'b' | b'B' | b'?' => item_size == 1,
        b'h' | b'H' | b'e' => item_size == 2,
        b'i' | b'I' | b'f' => item_size == 4,
        b'q' | b'Q' | b'd' => item_size == 8,
        b'l' | b'L' => item_size == size_of::<c_long>(),
        b'n' | b'N' | b'P' => item_size == size_of::<usize>(),
        _ => false,
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn buffer_item_as_f32(format: u8, bytes: &[u8]) -> Option<f32> {
    Some(match (format, bytes.len()) {
        (b'b', 1) => f32::from(i8::from_ne_bytes(bytes.try_into().ok()?)),
        (b'B', 1) => f32::from(u8::from_ne_bytes(bytes.try_into().ok()?)),
        (b'?', 1) => f32::from(bytes[0] & 1),
        (b'h', 2) => f32::from(i16::from_ne_bytes(bytes.try_into().ok()?)),
        (b'H', 2) => f32::from(u16::from_ne_bytes(bytes.try_into().ok()?)),
        (b'i' | b'l' | b'n', 4) => i32::from_ne_bytes(bytes.try_into().ok()?) as f32,
        (b'I' | b'L' | b'N' | b'P', 4) => u32::from_ne_bytes(bytes.try_into().ok()?) as f32,
        (b'l' | b'q' | b'n', 8) => i64::from_ne_bytes(bytes.try_into().ok()?) as f32,
        (b'L' | b'Q' | b'N' | b'P', 8) => u64::from_ne_bytes(bytes.try_into().ok()?) as f32,
        (b'e', 2) => half_to_f32(u16::from_ne_bytes(bytes.try_into().ok()?)),
        (b'f', 4) => f32::from_ne_bytes(bytes.try_into().ok()?),
        (b'd', 8) => f64::from_ne_bytes(bytes.try_into().ok()?) as f32,
        _ => return None,
    })
}

#[allow(clippy::cast_precision_loss)]
fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x03ff);
    if exponent == 0 {
        if fraction == 0 {
            return f32::from_bits(sign);
        }
        let value = fraction as f32 * 2.0_f32.powi(-24);
        return if sign == 0 { value } else { -value };
    }
    if exponent == 0x1f {
        return if fraction == 0 {
            f32::from_bits(sign | 0x7f80_0000)
        } else {
            f32::from_bits(sign | 0x7fc0_0000)
        };
    }

    let exponent = exponent + (127 - 15);
    f32::from_bits(sign | (exponent << 23) | (fraction << 13))
}

fn is_sequence_input(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if value.cast::<PySequence>().is_ok() {
        return Ok(true);
    }
    if value.cast::<PyMapping>().is_ok() {
        return Ok(false);
    }
    Ok(value.hasattr("__len__")? && value.hasattr("__getitem__")?)
}

fn flatten_rectangular(value: &Bound<'_, PyAny>, output: &mut Vec<f32>) -> PyResult<Vec<usize>> {
    if let Ok(scalar) = value.extract::<f32>() {
        output.push(scalar);
        return Ok(Vec::new());
    }

    if !is_sequence_input(value)? {
        return Err(PyTypeError::new_err(
            "tensor data must contain real numbers in a rectangular sequence",
        ));
    }
    let length = value.len()?;
    if length == 0 {
        return Ok(vec![0]);
    }

    let first_shape = flatten_rectangular(&value.get_item(0)?, output)?;
    for index in 1..length {
        let shape = flatten_rectangular(&value.get_item(index)?, output)?;
        if shape != first_shape {
            return Err(PyValueError::new_err(
                "expected a rectangular sequence, but nested shapes differ",
            ));
        }
    }

    let mut shape = Vec::with_capacity(first_shape.len() + 1);
    shape.push(length);
    shape.extend(first_shape);
    Ok(shape)
}

fn nested_list(py: Python<'_>, data: &[f32], shape: &[usize]) -> PyResult<Py<PyAny>> {
    if shape.is_empty() {
        return data[0].into_py_any(py);
    }

    let mut items = Vec::new();
    items.try_reserve_exact(shape[0]).map_err(|_| {
        PyMemoryError::new_err("unable to allocate Python list for tensor conversion")
    })?;
    if shape[0] == 0 {
        return Ok(PyList::new(py, items)?.into_any().unbind());
    }
    let chunk_size = if shape[1..].contains(&0) {
        0
    } else {
        shape[1..]
            .iter()
            .try_fold(1_usize, |elements, dimension| {
                elements.checked_mul(*dimension)
            })
            .ok_or_else(|| PyOverflowError::new_err("tensor shape product overflowed usize"))?
    };
    for index in 0..shape[0] {
        let start = index * chunk_size;
        items.push(nested_list(
            py,
            &data[start..start + chunk_size],
            &shape[1..],
        )?);
    }
    Ok(PyList::new(py, items)?.into_any().unbind())
}

#[pymodule]
fn torch_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    cpython_compat::initialize_torch_function_descriptor_caller(py)?;
    for (name, enabled) in NATIVE_BUILD_CAPABILITIES {
        module.add(name, enabled)?;
    }
    module.add_function(wrap_pyfunction!(get_cpu_capability_native, module)?)?;
    module.add_function(wrap_pyfunction!(is_ck_sdpa_available_native, module)?)?;
    module.add_function(wrap_pyfunction!(
        is_flash_attention_available_native,
        module
    )?)?;
    // PyTorch keeps these private build capabilities on torch._C. Removing them from
    // the extension's generated export list also prevents the package wildcard
    // import from copying them onto the public torch_rs module.
    let exports = module.getattr("__all__")?;
    for name in PRIVATE_NATIVE_EXPORTS {
        exports.call_method1("remove", (name,))?;
    }
    module.add("Size", size_type_object(py)?.clone_ref(py))?;
    module.add_class::<PyTensor>()?;
    let tensor_type = py.get_type::<PyTensor>();
    let tensor_base = py.get_type::<PyTensorBase>();
    // PyTorch installs Tensor.__abs__ as the TensorBase.abs descriptor itself.
    // Assigning the descriptor preserves its metadata and dispatch identity while
    // activating Python's unary absolute-value numeric slot.
    let abs_descriptor = tensor_base.getattr("abs")?;
    tensor_type.setattr("__abs__", abs_descriptor)?;
    // PyTorch installs Tensor.__pos__ as the TensorBase.positive descriptor
    // itself. Besides preserving its public metadata and call diagnostics,
    // assigning the descriptor activates the unary-positive numeric slot.
    let positive_descriptor = tensor_base.getattr("positive")?;
    tensor_type.setattr("__pos__", positive_descriptor)?;
    register_scalar_conversions(&tensor_base)?;
    module.add_class::<PyDType>()?;
    module.add("finfo", finfo_type_object(py)?.clone_ref(py))?;
    add_default_dtype_validator(module)?;
    add_warn_always_builtins(module)?;
    add_cudnn_builtins(module)?;
    add_cublas_matmul_builtins(module)?;
    add_flash_sdp_builtins(module)?;
    add_mem_efficient_sdp_builtins(module)?;
    add_math_sdp_builtins(module)?;
    add_cudnn_sdp_builtins(module)?;
    add_math_sdp_reduction_builtins(module)?;
    add_nnpack_builtins(module)?;
    add_compiler_state_builtins(module)?;
    module.add_class::<PyDevice>()?;
    module.add_class::<PyMemoryFormat>()?;
    add_grad_mode_contexts(module)?;
    // Define this public Python helper outside the partially initialized package.
    // A package import binds it to the live public module; direct native module
    // initialization (including Rust tests) falls back to the module being built.
    let is_tensor_helpers = PyModule::from_code(
        py,
        IS_TENSOR_SOURCE,
        c"torch_rs/_is_tensor.py",
        c"torch_rs._is_tensor",
    )?;
    if is_tensor_helpers.getattr("torch")?.is_none() {
        is_tensor_helpers.setattr("torch", module)?;
    }
    module.add("is_tensor", is_tensor_helpers.getattr("is_tensor")?)?;
    add_no_argument_builtins(module)?;
    module.add_function(wrap_pyfunction!(tensor, module)?)?;
    module.add("_MAX_BACKWARD_LEAF_ROOTS", MAX_BACKWARD_LEAF_ROOTS)?;
    module.add_function(wrap_pyfunction!(_backward_leaf_roots, module)?)?;
    let exports = module.getattr("__all__")?;
    for name in ["_MAX_BACKWARD_LEAF_ROOTS", "_backward_leaf_roots"] {
        exports.call_method1("remove", (name,))?;
    }
    torch_function_mode_stack::add_torch_function_mode_stack(module)?;
    add_torch_function_probe(module)?;
    add_variable_functions(module)?;
    module.add_function(wrap_pyfunction!(clone, module)?)?;
    module.add_function(wrap_pyfunction!(relu, module)?)?;
    add_nn_functional_bridges(module)?;
    module.add_function(wrap_pyfunction!(is_same_size, module)?)?;
    module.add_function(wrap_pyfunction!(equal, module)?)?;
    module.add_function(wrap_pyfunction!(t, module)?)?;
    module.add_function(wrap_pyfunction!(transpose, module)?)?;
    module.add_function(wrap_pyfunction!(swapdims, module)?)?;
    module.add_function(wrap_pyfunction!(swapaxes, module)?)?;
    module.add_function(wrap_pyfunction!(squeeze, module)?)?;
    module.add_function(wrap_pyfunction!(flatten, module)?)?;
    add_tensor_queries(module)?;
    module.add_function(wrap_pyfunction!(zeros, module)?)?;
    module.add_function(wrap_pyfunction!(ones, module)?)?;
    module.add_function(wrap_pyfunction!(eye, module)?)?;
    module.add_function(wrap_pyfunction!(full, module)?)?;
    let float32 = dtype_object(py, DType::Float32)?;
    module.add("float32", float32.clone_ref(py))?;
    module.add("float", float32.clone_ref(py))?;
    module.add("layout", layout_objects(py)?.layout.clone_ref(py))?;
    module.add("strided", strided_object(py)?.clone_ref(py))?;
    module
        .getattr("__all__")?
        .call_method1("remove", ("strided",))?;
    for (name, memory_format) in [
        ("preserve_format", MemoryFormat::Preserve),
        ("contiguous_format", MemoryFormat::Contiguous),
        ("channels_last", MemoryFormat::ChannelsLast),
        ("channels_last_3d", MemoryFormat::ChannelsLast3d),
    ] {
        module.add(name, memory_format_object(py, memory_format)?.clone_ref(py))?;
    }
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use pyo3::exceptions::PyTypeError;
    use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyMemoryView, PyModule, PySlice};

    use super::{
        PyTensor, PythonAllocationFallback, flatten_buffer, half_to_f32, libcxx_string_hash,
        nested_list, pytorch_libcxx_keyword_order, pytorch_msvc_keyword_order, torch_rs,
        try_size_vector,
    };

    #[test]
    fn half_precision_buffer_values_convert_to_float32() {
        assert_eq!(half_to_f32(0x0000).to_bits(), 0.0_f32.to_bits());
        assert_eq!(half_to_f32(0x8000).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(half_to_f32(0x0001).to_bits(), 2.0_f32.powi(-24).to_bits());
        assert_eq!(half_to_f32(0x0400).to_bits(), 2.0_f32.powi(-14).to_bits());
        assert_eq!(half_to_f32(0x3c00).to_bits(), 1.0_f32.to_bits());
        assert_eq!(half_to_f32(0xc000).to_bits(), (-2.0_f32).to_bits());
        assert_eq!(half_to_f32(0x7c00).to_bits(), f32::INFINITY.to_bits());
        assert_eq!(half_to_f32(0xfc00).to_bits(), f32::NEG_INFINITY.to_bits());
        assert_eq!(half_to_f32(0x7c01).to_bits(), 0x7fc0_0000);
        assert_eq!(half_to_f32(0xffff).to_bits(), 0xffc0_0000);
    }

    #[test]
    fn one_dimensional_buffer_is_copied_in_logical_stride_order() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let array = PyModule::import(py, "array")
                .unwrap()
                .getattr("array")
                .unwrap()
                .call1(("i", [1_i32, 2, 3, 4]))
                .unwrap();
            let view = PyMemoryView::from(&array).unwrap();
            let reversed = view.get_item(PySlice::new(py, 3, -5, -1)).unwrap();

            let (values, shape) = flatten_buffer(&reversed, true).unwrap().unwrap();
            assert_eq!(shape, [4]);
            assert_eq!(values, [4.0, 3.0, 2.0, 1.0]);

            array.set_item(3, 99).unwrap();
            assert_eq!(values, [4.0, 3.0, 2.0, 1.0]);
        });
    }

    #[test]
    fn size_vector_capacity_overflow_returns_python_error() {
        pyo3::Python::initialize();
        let error = try_size_vector::<i64>(usize::MAX)
            .expect_err("an impossible vector capacity must return an error");
        assert_eq!(error.to_string(), "RuntimeError: std::bad_alloc");
    }

    #[test]
    fn libcxx_keyword_order_matches_pytorch_on_macos() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let allocation = PythonAllocationFallback::new(py);
            let ordered = pytorch_libcxx_keyword_order(
                ["a", "b", "d"]
                    .into_iter()
                    .map(|key| (key.to_owned(), "Tensor".to_owned()))
                    .collect(),
                &allocation,
            )
            .unwrap();
            let keys = ordered.into_iter().map(|(key, _)| key).collect::<Vec<_>>();
            assert_eq!(keys, ["d", "b", "a"]);

            let ordered = pytorch_libcxx_keyword_order(
                (0..14)
                    .map(|index| (format!("key{index}"), "Tensor".to_owned()))
                    .collect(),
                &allocation,
            )
            .unwrap();
            let keys = ordered.into_iter().map(|(key, _)| key).collect::<Vec<_>>();
            assert_eq!(
                keys,
                [
                    "key13", "key11", "key12", "key8", "key1", "key2", "key6", "key5", "key3",
                    "key4", "key0", "key10", "key7", "key9"
                ]
            );
        });
    }

    #[test]
    fn msvc_keyword_order_matches_pytorch_on_windows() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let allocation = PythonAllocationFallback::new(py);
            let ordered = pytorch_msvc_keyword_order(
                ["a", "b", "d"]
                    .into_iter()
                    .map(|key| (key.to_owned(), "Tensor".to_owned()))
                    .collect(),
                &allocation,
            )
            .unwrap();
            let keys = ordered.into_iter().map(|(key, _)| key).collect::<Vec<_>>();
            assert_eq!(keys, ["a", "b", "d"]);

            let collisions = pytorch_msvc_keyword_order(
                ["key0", "key1", "key8"]
                    .into_iter()
                    .map(|key| (key.to_owned(), "Tensor".to_owned()))
                    .collect(),
                &allocation,
            )
            .unwrap();
            let keys = collisions
                .into_iter()
                .map(|(key, _)| key)
                .collect::<Vec<_>>();
            assert_eq!(keys, ["key8", "key0", "key1"]);
        });
    }

    #[test]
    fn libcxx_string_hash_matches_reference_boundaries() {
        for (value, expected) in [
            ("", 11_160_318_154_034_397_263),
            ("a", 2_603_192_927_274_642_682),
            ("key13", 15_487_510_319_299_464_526),
            ("abcdefghijklmnopq", 237_482_408_704_357_350),
            (
                "abcdefghijklmnopqrstuvwxyz0123456",
                11_578_587_182_705_320_317,
            ),
            (
                "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-abcdefghijklmnopqrstuvwxyz",
                14_968_935_594_714_716_411,
            ),
        ] {
            assert_eq!(libcxx_string_hash(value), expected);
        }
    }

    #[test]
    fn nested_list_short_circuits_a_leading_zero_before_shape_multiplication() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let maximum = usize::try_from(i64::MAX).unwrap();
            let list = nested_list(py, &[], &[0, maximum, maximum]).unwrap();
            assert_eq!(list.bind(py).len().unwrap(), 0);
        });
    }

    #[test]
    fn reshape_binding_requires_shape_and_accepts_shape_keyword() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let module = PyModule::new(py, "torch_rs").unwrap();
            torch_rs(&module).unwrap();
            let tensor = module
                .getattr("tensor")
                .unwrap()
                .call1((vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],))
                .unwrap();

            let keywords = PyDict::new(py);
            keywords.set_item("shape", (2, 3)).unwrap();
            let reshaped = tensor.call_method("reshape", (), Some(&keywords)).unwrap();
            assert_eq!(
                reshaped
                    .getattr("shape")
                    .unwrap()
                    .extract::<Vec<usize>>()
                    .unwrap(),
                [2, 3]
            );

            let invalid_keywords = PyDict::new(py);
            invalid_keywords.set_item("shape", -1).unwrap();
            let error = tensor
                .call_method("reshape", (), Some(&invalid_keywords))
                .expect_err("a scalar keyword shape must fail");
            assert!(error.is_instance_of::<PyTypeError>(py));

            let error = tensor
                .call_method0("reshape")
                .expect_err("reshape without a shape must fail");
            assert!(error.is_instance_of::<PyTypeError>(py));
        });
    }

    #[test]
    fn swapdims_binding_returns_a_shared_storage_transpose_view() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let module = PyModule::new(py, "torch_rs").unwrap();
            torch_rs(&module).unwrap();
            let source = module
                .getattr("zeros")
                .unwrap()
                .call1(((2, 3, 4),))
                .unwrap();
            let swapped = source.call_method1("swapdims", (0, -1)).unwrap();
            let source = source.cast::<PyTensor>().unwrap().try_borrow().unwrap();
            let swapped = swapped.cast::<PyTensor>().unwrap().try_borrow().unwrap();

            assert_eq!(swapped.inner.shape(), [4, 3, 2]);
            assert_eq!(swapped.inner.stride(), [1, 4, 12]);
            assert_eq!(swapped.inner.storage_offset(), 0);
            assert!(swapped.inner.shares_storage_with(&source.inner));
        });
    }

    #[test]
    fn swapaxes_binding_preserves_strided_storage_and_offset() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let module = PyModule::new(py, "torch_rs").unwrap();
            torch_rs(&module).unwrap();
            let base = module
                .getattr("zeros")
                .unwrap()
                .call1(((2, 3, 4),))
                .unwrap();
            let transposed = base.call_method1("transpose", (0, 2)).unwrap();
            let source = transposed.get_item(1).unwrap();
            let swapped = source.call_method1("swapaxes", (0, -1)).unwrap();
            let source = source.cast::<PyTensor>().unwrap().try_borrow().unwrap();
            let swapped = swapped.cast::<PyTensor>().unwrap().try_borrow().unwrap();

            assert_eq!(swapped.inner.shape(), [2, 3]);
            assert_eq!(swapped.inner.stride(), [12, 4]);
            assert_eq!(swapped.inner.storage_offset(), 1);
            assert!(swapped.inner.shares_storage_with(&source.inner));
        });
    }
}

use std::cell::Cell;
use std::ffi::CStr;
use std::os::raw::c_long;
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{
    PyIndexError, PyMemoryError, PyOverflowError, PyRuntimeError, PyTypeError, PyUserWarning,
    PyValueError,
};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyAny, PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyMapping, PyMemoryView, PyModule,
    PySequence, PyString, PyTuple,
};

use crate::{
    DType, Device, MemoryFormat, Tensor as CoreTensor, TensorError, enter_no_grad, exit_no_grad,
    is_grad_enabled as core_is_grad_enabled,
    python_layout::{LayoutObjects as PyLayoutObjects, create_layout_objects},
};

static FLOAT32: PyOnceLock<Py<PyDType>> = PyOnceLock::new();
static LAYOUT_OBJECTS: PyOnceLock<PyLayoutObjects> = PyOnceLock::new();
static PRESERVE_FORMAT: PyOnceLock<Py<PyMemoryFormat>> = PyOnceLock::new();
static CONTIGUOUS_FORMAT: PyOnceLock<Py<PyMemoryFormat>> = PyOnceLock::new();
static CHANNELS_LAST: PyOnceLock<Py<PyMemoryFormat>> = PyOnceLock::new();
static CHANNELS_LAST_3D: PyOnceLock<Py<PyMemoryFormat>> = PyOnceLock::new();
static FLOAT_REQUIRES_GRAD_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);
static T_NON_MATRIX_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);
static T_SCALAR_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);
static MT_SCALAR_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);

const NO_GRAD_WRAPPER_SOURCE: &CStr = cr#"
import functools
import inspect
import sys


def _decorate_no_grad(context_factory, function):
    if inspect.isgeneratorfunction(function):
        @functools.wraps(function)
        def generator_context(*args, **kwargs):
            generator = function(*args, **kwargs)
            try:
                with context_factory():
                    response = generator.send(None)

                while True:
                    try:
                        request = yield response
                    except GeneratorExit:
                        with context_factory():
                            generator.close()
                        raise
                    except BaseException:
                        with context_factory():
                            response = generator.throw(*sys.exc_info())
                    else:
                        with context_factory():
                            response = generator.send(request)
            except StopIteration as error:
                return error.value

        return generator_context

    @functools.wraps(function)
    def decorate_context(*args, **kwargs):
        with context_factory():
            return function(*args, **kwargs)

    return decorate_context


def _make_no_grad(context_base):
    class no_grad(context_base):
        def __new__(cls, original_function=None):
            if original_function is not None:
                return cls()(original_function)
            return super().__new__(cls)

        def __call__(self, function):
            return _decorate_no_grad(type(self), function)

    no_grad.__module__ = "torch_rs"
    no_grad.__qualname__ = "no_grad"
    return no_grad
"#;

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

thread_local! {
    static NO_GRAD_CONTEXT_DEPTH: Cell<usize> = const { Cell::new(0) };
}

#[cfg(target_os = "macos")]
const FLOAT_REQUIRES_GRAD_WARNING: &CStr = c"Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.\nConsider using tensor.detach() first. (Triggered internally at /Users/runner/work/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:823.)";
#[cfg(target_os = "macos")]
const T_NON_MATRIX_WARNING: &CStr = c"The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4317.)";
#[cfg(target_os = "macos")]
const T_SCALAR_WARNING: &CStr = c"Tensor.T is deprecated on 0-D tensors. This function is the identity in these cases. (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4322.)";
#[cfg(target_os = "macos")]
const MT_SCALAR_WARNING: &CStr = c"Tensor.mT is deprecated on 0-D tensors. This function is the identity in these cases. (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4374.)";

#[cfg(target_os = "linux")]
const FLOAT_REQUIRES_GRAD_WARNING: &CStr = c"Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.\nConsider using tensor.detach() first. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:822.)";
#[cfg(target_os = "linux")]
const T_NON_MATRIX_WARNING: &CStr = c"The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at /__w/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4314.)";
#[cfg(target_os = "linux")]
const T_SCALAR_WARNING: &CStr = c"Tensor.T is deprecated on 0-D tensors. This function is the identity in these cases. (Triggered internally at /__w/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4321.)";
#[cfg(target_os = "linux")]
const MT_SCALAR_WARNING: &CStr = c"Tensor.mT is deprecated on 0-D tensors. This function is the identity in these cases. (Triggered internally at /__w/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4373.)";

#[cfg(target_os = "windows")]
const FLOAT_REQUIRES_GRAD_WARNING: &CStr = c"Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.\nConsider using tensor.detach() first. (Triggered internally at C:\\actions-runner\\_work\\pytorch\\pytorch\\torch\\csrc\\autograd\\generated\\python_variable_methods.cpp:823.)";
#[cfg(target_os = "windows")]
const T_NON_MATRIX_WARNING: &CStr = c"The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at C:\\actions-runner\\_work\\pytorch\\pytorch\\aten\\src\\ATen\\native\\TensorShape.cpp:4317.)";
#[cfg(target_os = "windows")]
const T_SCALAR_WARNING: &CStr = c"Tensor.T is deprecated on 0-D tensors. This function is the identity in these cases. (Triggered internally at C:\\actions-runner\\_work\\pytorch\\pytorch\\aten\\src\\ATen\\native\\TensorShape.cpp:4322.)";
#[cfg(target_os = "windows")]
const MT_SCALAR_WARNING: &CStr = c"Tensor.mT is deprecated on 0-D tensors. This function is the identity in these cases. (Triggered internally at C:\\actions-runner\\_work\\pytorch\\pytorch\\aten\\src\\ATen\\native\\TensorShape.cpp:4374.)";

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const FLOAT_REQUIRES_GRAD_WARNING: &CStr = c"Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.\nConsider using tensor.detach() first.";
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const T_NON_MATRIX_WARNING: &CStr = c"The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor.";
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const T_SCALAR_WARNING: &CStr =
    c"Tensor.T is deprecated on 0-D tensors. This function is the identity in these cases.";
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const MT_SCALAR_WARNING: &CStr =
    c"Tensor.mT is deprecated on 0-D tensors. This function is the identity in these cases.";

/// Python scalar-type descriptor backed by a native [`DType`].
#[pyclass(name = "dtype", module = "torch_rs", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyDType {
    inner: DType,
}

#[pymethods]
impl PyDType {
    #[getter]
    fn itemsize(&self) -> usize {
        self.inner.element_size()
    }

    fn __repr__(&self) -> &'static str {
        match self.inner {
            DType::Float32 => "torch.float32",
        }
    }

    fn __str__(&self) -> &'static str {
        self.__repr__()
    }
}

/// Python memory-format descriptor backed by a native [`MemoryFormat`].
#[pyclass(
    name = "memory_format",
    module = "torch_rs",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Clone, PartialEq, Eq, Hash)]
struct PyMemoryFormat {
    inner: MemoryFormat,
}

#[pymethods]
impl PyMemoryFormat {
    fn __repr__(&self) -> String {
        format!("torch.{}", self.inner)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Python device descriptor backed by a native [`Device`].
#[pyclass(
    name = "device",
    module = "torch_rs",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Clone, PartialEq, Eq, Hash)]
struct PyDevice {
    inner: Device,
}

#[pymethods]
impl PyDevice {
    #[new]
    fn new(r#type: &Bound<'_, PyAny>) -> PyResult<Self> {
        parse_device_value("device", r#type).map(|inner| Self { inner })
    }

    #[getter]
    fn r#type(&self) -> &'static str {
        match self.inner {
            Device::Cpu => "cpu",
        }
    }

    #[getter]
    fn index(&self) -> Option<usize> {
        self.inner.index()
    }

    fn __repr__(&self) -> &'static str {
        match self.inner {
            Device::Cpu => "device(type='cpu')",
        }
    }

    fn __str__(&self) -> &'static str {
        self.r#type()
    }
}

// Internal descriptor owner matching PyTorch's native tensor base class.
#[pyclass(
    name = "TensorBase",
    module = "torch._C",
    subclass,
    skip_from_py_object
)]
struct PyTensorBase;

#[pymethods]
impl PyTensorBase {
    #[getter]
    fn layout(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        Ok(strided_object(slf.py())?.clone_ref(slf.py()))
    }

    #[pyo3(text_signature = None)]
    fn int_scalar<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyInt>> {
        let value = slf
            .as_any()
            .cast::<PyTensor>()?
            .try_borrow()?
            .inner
            .item()
            .map_err(|error| scalar_conversion_error(&error))?;

        // CPython's float-to-int conversion truncates toward zero, produces a
        // PyLong of any required size, and supplies the canonical infinity and
        // NaN errors. Float32 values are represented exactly as Python floats.
        slf.py()
            .get_type::<PyInt>()
            .call1((f64::from(value),))?
            .cast_into::<PyInt>()
            .map_err(Into::into)
    }

    #[pyo3(text_signature = None)]
    fn float_scalar(slf: &Bound<'_, Self>) -> PyResult<f64> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        if core_is_grad_enabled() && tensor.inner.requires_grad() {
            warn_once(
                slf.py(),
                &FLOAT_REQUIRES_GRAD_WARNING_EMITTED,
                FLOAT_REQUIRES_GRAD_WARNING,
            )?;
        }
        tensor
            .inner
            .item()
            .map(f64::from)
            .map_err(|error| scalar_conversion_error(&error))
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nIs ``True`` if the Tensor is stored on the CPU, ``False`` otherwise.\n"]
    #[getter]
    fn is_cpu(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        Ok(tensor.inner.is_cpu())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nIs ``True`` if the Tensor is stored on the GPU, ``False`` otherwise.\n"]
    #[getter]
    fn is_cuda(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        Ok(tensor.inner.is_cuda())
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
    #[doc = "\nIs ``True`` if the Tensor uses sparse COO storage layout, ``False`` otherwise.\n"]
    #[getter]
    fn is_sparse(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        Ok(tensor.inner.is_sparse())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nget_device() -> Device ordinal (Integer)\n\nFor CUDA tensors, this function returns the device ordinal of the GPU on which the tensor resides.\nFor CPU tensors, this function returns `-1`.\n\nExample::\n\n    >>> x = torch.randn(3, 4, 5, device='cuda:0')\n    >>> x.get_device()\n    0\n    >>> x.cpu().get_device()\n    -1\n"]
    #[pyo3(text_signature = None)]
    fn get_device(slf: &Bound<'_, Self>) -> PyResult<i64> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        tensor
            .inner
            .device()
            .index()
            .map_or(Ok(-1), |index| i64::try_from(index).map_err(Into::into))
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nAlias for :meth:`~Tensor.element_size()`\n"]
    #[getter]
    fn itemsize(slf: &Bound<'_, Self>) -> PyResult<usize> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        Ok(tensor.inner.dtype().element_size())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nelement_size() -> int\n\nReturns the size in bytes of an individual element.\n\nExample::\n\n    >>> torch.tensor([]).element_size()\n    4\n    >>> torch.tensor([], dtype=torch.uint8).element_size()\n    1\n\n"]
    #[pyo3(text_signature = None)]
    fn element_size(slf: &Bound<'_, Self>) -> PyResult<usize> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        Ok(tensor.inner.element_size())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\ndata_ptr() -> int\n\nReturns the address of the first element of :attr:`self` tensor.\n\n.. note::\n\n    If the tensor is a copy-on-write tensor (e.g. created via\n    :meth:`_lazy_clone`), calling this method will materialize the\n    copy. Use :meth:`const_data_ptr` if you only need read-only access\n    to the data pointer.\n"]
    #[pyo3(text_signature = None)]
    fn data_ptr(slf: &Bound<'_, Self>) -> PyResult<usize> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        Ok(tensor.inner.data_ptr())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nis_complex() -> bool\n\nReturns True if the data type of :attr:`self` is a complex data type.\n"]
    #[pyo3(text_signature = None)]
    fn is_complex(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        Ok(tensor.inner.is_complex())
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nis_signed() -> bool\n\nReturns True if the data type of :attr:`self` is a signed data type.\n"]
    #[pyo3(text_signature = None)]
    fn is_signed(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        Ok(tensor.inner.is_signed())
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
        tensor.multiplication_method(MultiplicationMethod::Multiply, args, kwargs)
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

// PyTorch publishes __int__ and __float__ as METH_NOARGS methods on TensorBase
// instead of the slot wrappers CPython normally exposes for extension types.
// Register the same method shapes, then let module initialization connect them
// to the numeric slots.
pyo3::inventory::submit! {
    type Inventory = <PyTensorBase as pyo3::impl_::pyclass::PyClassImpl>::Inventory;
    Inventory::new(pyo3::impl_::pyclass::PyClassItems {
        methods: &[
            pyo3::impl_::pymethods::PyMethodDefType::Method(
                pyo3::impl_::pymethods::PyMethodDef::noargs(
                    c"__int__",
                    pyo3::impl_::trampoline::get_trampoline_function!(
                        noargs,
                        PyTensorBase::__pymethod_int_scalar__
                    ),
                    c"",
                ),
            ),
            pyo3::impl_::pymethods::PyMethodDefType::Method(
                pyo3::impl_::pymethods::PyMethodDef::noargs(
                    c"__float__",
                    pyo3::impl_::trampoline::get_trampoline_function!(
                        noargs,
                        PyTensorBase::__pymethod_float_scalar__
                    ),
                    c"",
                ),
            ),
        ],
        slots: &[],
    })
}

/// Python-facing tensor backed by the native Rust tensor core.
#[pyclass(
    name = "Tensor",
    module = "torch_rs",
    extends = PyTensorBase,
    skip_from_py_object
)]
struct PyTensor {
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
    fn new(inner: CoreTensor) -> Self {
        Self {
            inner,
            grad_cache: PyOnceLock::new(),
        }
    }
}

/// Thread-local autograd recording guard underlying the Python `torch.no_grad` class.
#[pyclass(
    name = "_NoGradContext",
    module = "torch_rs",
    subclass,
    skip_from_py_object
)]
struct PyNoGrad;

#[pymethods]
impl PyNoGrad {
    #[new]
    fn new() -> Self {
        Self
    }

    #[allow(clippy::unused_self)] // Python's context-manager protocol requires an instance method.
    fn __enter__(&self) {
        enter_no_grad();
        NO_GRAD_CONTEXT_DEPTH.set(
            NO_GRAD_CONTEXT_DEPTH
                .get()
                .checked_add(1)
                .expect("Python no-grad nesting depth overflowed usize"),
        );
    }

    #[allow(clippy::unused_self)] // Python's context-manager protocol requires an instance method.
    fn __exit__(
        &self,
        _exception_type: &Bound<'_, PyAny>,
        _exception_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) {
        if let Some(depth) = NO_GRAD_CONTEXT_DEPTH.get().checked_sub(1) {
            NO_GRAD_CONTEXT_DEPTH.set(depth);
            exit_no_grad();
        }
    }
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

struct ParsedCallArgument<'py> {
    value: Bound<'py, PyAny>,
    position: Option<usize>,
}

struct CreationCallArguments<'py> {
    size: Option<Bound<'py, PyAny>>,
    shape: Option<Bound<'py, PyAny>>,
    dtype: Option<Bound<'py, PyAny>>,
    device: Option<Bound<'py, PyAny>>,
    requires_grad: Option<Bound<'py, PyAny>>,
    keyword_error: Option<PyErr>,
}

struct FullCallArguments<'py> {
    size: Option<Bound<'py, PyAny>>,
    fill_value: Option<Bound<'py, PyAny>>,
    dtype: Option<Bound<'py, PyAny>>,
    device: Option<Bound<'py, PyAny>>,
    requires_grad: Option<Bound<'py, PyAny>>,
    keyword_error: Option<PyErr>,
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

#[derive(Clone, Copy)]
enum MultiplicationMethod {
    Mul,
    Multiply,
}

impl MultiplicationMethod {
    const fn name(self) -> &'static str {
        match self {
            Self::Mul => "mul",
            Self::Multiply => "multiply",
        }
    }
}

#[pymethods]
impl PyTensor {
    #[classattr]
    fn __array_priority__() -> f64 {
        1000.0
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.shape().iter().copied())
    }

    /// Alias for [`Tensor.dim()`](https://pytorch.org/docs/stable/generated/torch.Tensor.dim.html).
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.shape().len()
    }

    #[doc = "\nReturns the number of bytes consumed by the \"view\" of elements of the Tensor\nif the Tensor does not use sparse storage layout.\nDefined to be :meth:`~Tensor.numel()` * :meth:`~Tensor.element_size()`\n"]
    #[getter]
    fn nbytes(&self) -> usize {
        self.inner.numel() * self.inner.element_size()
    }

    #[getter]
    fn dtype(&self, py: Python<'_>) -> PyResult<Py<PyDType>> {
        match self.inner.dtype() {
            DType::Float32 => Ok(float32_object(py)?.clone_ref(py)),
        }
    }

    #[getter]
    fn device(&self) -> PyDevice {
        PyDevice {
            inner: self.inner.device(),
        }
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    #[getter]
    fn is_leaf(&self) -> bool {
        self.inner.is_leaf()
    }

    #[getter]
    fn grad(&self, py: Python<'_>) -> PyResult<Option<Py<Self>>> {
        if let Some(gradient) = self.grad_cache.get(py) {
            return Ok(Some(gradient.clone_ref(py)));
        }
        let Some(inner) = self
            .inner
            .live_grad()
            .map_err(|error| tensor_error(&error))?
        else {
            return Ok(None);
        };
        let gradient = self
            .grad_cache
            .get_or_try_init(py, || Py::new(py, Self::new(inner)))?;
        Ok(Some(gradient.clone_ref(py)))
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
            .reverse_dimensions()
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

    fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = if let Ok(indices) = index.cast::<PyTuple>() {
            if indices.len() > self.inner.shape().len() {
                return Err(too_many_indices(self.inner.shape().len()));
            }
            let indices = parse_integer_indices(&self.inner, indices.len(), indices.iter())?;
            self.inner.index(indices)
        } else if is_fast_integer_index(index)? {
            let index = parse_integer_index(index)?;
            self.inner.index_integer(index)
        } else {
            if self.inner.shape().is_empty() {
                return Err(too_many_indices(0));
            }
            let index = parse_integer_index(index)?;
            self.inner.index([index])
        };
        inner.map(Self::new).map_err(|error| tensor_error(&error))
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

    /// Returns the number of dimensions of the tensor.
    #[pyo3(text_signature = None)]
    fn dim(&self) -> usize {
        self.inner.shape().len()
    }

    /// Alias for [`Tensor.dim()`](https://pytorch.org/docs/stable/generated/torch.Tensor.dim.html).
    #[pyo3(text_signature = None)]
    fn ndimension(&self) -> usize {
        self.inner.shape().len()
    }

    /// Alias for [`Tensor.numel()`](https://pytorch.org/docs/stable/generated/torch.Tensor.numel.html).
    #[pyo3(text_signature = None)]
    fn nelement(&self) -> usize {
        self.inner.numel()
    }

    /// Returns the total number of elements in the tensor.
    #[pyo3(text_signature = None)]
    fn numel(&self) -> usize {
        self.inner.numel()
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nis_floating_point() -> bool\n\nReturns True if the data type of :attr:`self` is a floating point data type.\n"]
    #[pyo3(text_signature = None)]
    fn is_floating_point(&self) -> bool {
        self.inner.is_floating_point()
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

    fn sin(&self) -> PyResult<Self> {
        self.inner
            .sin()
            .map(Self::new)
            .map_err(|error| tensor_error(&error))
    }

    fn exp(&self) -> PyResult<Self> {
        self.inner
            .exp()
            .map(Self::new)
            .map_err(|error| tensor_error(&error))
    }

    // Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
    #[allow(clippy::doc_markdown)]
    #[doc = "\nneg() -> Tensor\n\nSee :func:`torch.neg`\n"]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn neg(&self, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        // PyTorch exposes Tensor.neg from its internal TensorBase descriptor,
        // which remains observable in the no-argument binding errors.
        if kwargs.is_some_and(|values| !values.is_empty()) {
            return Err(PyTypeError::new_err(
                "TensorBase.neg() takes no keyword arguments",
            ));
        }
        if !args.is_empty() {
            return Err(PyTypeError::new_err(format!(
                "TensorBase.neg() takes no arguments ({} given)",
                args.len()
            )));
        }
        self.negated()
    }

    fn sum(&self) -> Self {
        Self::new(self.inner.sum())
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
        self.multiplication_method(MultiplicationMethod::Mul, args, kwargs)
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
        self.inner
            .matmul(&other.inner)
            .map(Self::new)
            .map_err(|error| tensor_error(&error))
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
    fn negated(&self) -> PyResult<Self> {
        self.inner
            .negate()
            .map(Self::new)
            .map_err(|error| tensor_error(&error))
    }

    fn multiplication_method(
        &self,
        operation: MultiplicationMethod,
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
                MultiplicationMethod::Mul => {
                    let actual = transpose_type_name(&other.value)?;
                    Err(mul_argument_type_error(other.position, &actual))
                }
                MultiplicationMethod::Multiply => Err(multiply_binding_error(args, kwargs)?),
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
                    let message = if other.value.lt(0_i64)? {
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

    fn truth_value(&self) -> PyResult<bool> {
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

// Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
#[allow(clippy::doc_markdown)]
#[doc = "\nis_grad_enabled() -> (bool)\n\nReturns True if grad mode is currently enabled.\n"]
#[pyfunction(signature = (*args, **kwargs), text_signature = None)]
fn is_grad_enabled(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<bool> {
    if kwargs.is_some_and(|values| !values.is_empty()) {
        return Err(PyTypeError::new_err(
            "torch.is_grad_enabled() takes no keyword arguments",
        ));
    }
    if !args.is_empty() {
        return Err(PyTypeError::new_err(format!(
            "torch.is_grad_enabled() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(core_is_grad_enabled())
}

// Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
#[allow(clippy::doc_markdown)]
#[cfg_attr(
    not(doc),
    doc = "\nget_default_dtype() -> torch.dtype\n\nGet the current default floating point :class:`torch.dtype`.\n\nExample::\n\n    >>> torch.get_default_dtype()  # initial default for floating point is torch.float32\n    torch.float32\n    >>> torch.set_default_dtype(torch.float64)\n    >>> torch.get_default_dtype()  # default is now changed to torch.float64\n    torch.float64\n\n"
)]
#[cfg_attr(doc, doc = "Get the current default floating-point dtype.")]
#[pyfunction(signature = (*args, **kwargs), text_signature = None)]
fn get_default_dtype(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyDType>> {
    if kwargs.is_some_and(|values| !values.is_empty()) {
        return Err(PyTypeError::new_err(
            "torch.get_default_dtype() takes no keyword arguments",
        ));
    }
    if !args.is_empty() {
        return Err(PyTypeError::new_err(format!(
            "torch.get_default_dtype() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(float32_object(py)?.clone_ref(py))
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

fn parse_requires_grad(function: &str, requires_grad: &Bound<'_, PyAny>) -> PyResult<bool> {
    if requires_grad.is_exact_instance_of::<PyBool>() {
        return requires_grad.is_truthy();
    }
    let type_name = transpose_type_name(requires_grad)?;
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
fn detach(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
    let input = bind_detach_argument(args, kwargs)?;
    let input = input
        .value
        .cast::<PyTensor>()
        .expect("the detach input type was checked while binding");
    input
        .try_borrow()?
        .inner
        .detach()
        .map(PyTensor::new)
        .map_err(|error| tensor_error(&error))
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

#[pyfunction(signature = (*args, **kwargs))]
fn transpose(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
    let ([input, dim0, dim1], keyword_error) =
        bind_dimension_swap_arguments("transpose", args, kwargs, ["input", "dim0", "dim1"])?;
    if let Some(keyword_error) = keyword_error {
        return Err(keyword_error);
    }
    let input_type = transpose_type_name(&input.value)?;
    let input_tensor = input.value.cast::<PyTensor>().map_err(|_| {
        dimension_swap_argument_type_error(
            "transpose",
            "input",
            input.position,
            "Tensor",
            &input_type,
        )
    })?;
    let input_tensor = input_tensor.try_borrow()?;
    let [dim0, dim1] =
        parse_dimension_swap_dimensions("transpose", ["dim0", "dim1"], &dim0, &dim1)?;
    input_tensor
        .inner
        .transpose(dim0, dim1)
        .map(PyTensor::new)
        .map_err(|error| transpose_error(&error))
}

#[pyfunction(signature = (*args, **kwargs), text_signature = "(input, dim=None)")]
fn squeeze(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
    let (input, input_position, dimension) = bind_top_level_squeeze_arguments(args, kwargs)?;
    let input_type = transpose_type_name(&input)?;
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

/// Returns the total number of elements in the input tensor.
#[pyfunction(signature = (*args, **kwargs), text_signature = None)]
fn numel(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<usize> {
    if args.len() > 1 {
        return Err(PyTypeError::new_err(format!(
            "numel() takes 1 positional argument but {} were given",
            args.len()
        )));
    }

    let keyword_input = match kwargs {
        Some(values) => values.get_item("input")?,
        None => None,
    };
    if args.is_empty() && keyword_input.is_none() {
        return Err(PyTypeError::new_err(
            "numel() missing 1 required positional arguments: \"input\"",
        ));
    }

    let (input, position) = if args.is_empty() {
        (
            keyword_input
                .as_ref()
                .expect("the required keyword input was checked above"),
            None,
        )
    } else {
        (&args.get_item(0)?, Some(1))
    };
    let Ok(tensor) = input.cast::<PyTensor>() else {
        let position =
            position.map_or_else(String::new, |position| format!(" (position {position})"));
        let input_type = transpose_type_name(input)?;
        return Err(PyTypeError::new_err(format!(
            "numel(): argument 'input'{position} must be Tensor, not {input_type}"
        )));
    };

    if !args.is_empty() && keyword_input.is_some() {
        return Err(PyTypeError::new_err(
            "numel() got multiple values for argument 'input'",
        ));
    }
    if let Some(kwargs) = kwargs {
        for key in kwargs.keys() {
            let key = key.extract::<String>()?;
            if key != "input" {
                return Err(PyTypeError::new_err(format!(
                    "numel() got an unexpected keyword argument '{key}'"
                )));
            }
        }
    }

    Ok(tensor.try_borrow()?.inner.numel())
}

// Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
#[allow(clippy::doc_markdown)]
#[cfg_attr(
    not(doc),
    doc = "\nis_nonzero(input) -> (bool)\n\nReturns True if the :attr:`input` is a single element tensor which is not equal to zero\nafter type conversions.\ni.e. not equal to ``torch.tensor([0.])`` or ``torch.tensor([0])`` or\n``torch.tensor([False])``.\nThrows a ``RuntimeError`` if ``torch.numel() != 1`` (even in case\nof sparse tensors).\n\nArgs:\n    input (Tensor): the input tensor.\n\nExamples::\n\n    >>> torch.is_nonzero(torch.tensor([0.]))\n    False\n    >>> torch.is_nonzero(torch.tensor([1.5]))\n    True\n    >>> torch.is_nonzero(torch.tensor([False]))\n    False\n    >>> torch.is_nonzero(torch.tensor([3]))\n    True\n    >>> torch.is_nonzero(torch.tensor([1, 3, 5]))\n    Traceback (most recent call last):\n    ...\n    RuntimeError: Boolean value of Tensor with more than one value is ambiguous\n    >>> torch.is_nonzero(torch.tensor([]))\n    Traceback (most recent call last):\n    ...\n    RuntimeError: Boolean value of Tensor with no values is ambiguous\n"
)]
#[cfg_attr(doc, doc = "See the runtime Python documentation for examples.")]
#[pyfunction(signature = (*args, **kwargs), text_signature = None)]
fn is_nonzero(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<bool> {
    let input = bind_legacy_single_tensor_argument("is_nonzero", args, kwargs)?;
    let tensor = input
        .value
        .cast::<PyTensor>()
        .expect("the is_nonzero input type was checked while binding");
    tensor.try_borrow()?.truth_value()
}

// Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
#[allow(clippy::doc_markdown)]
#[cfg_attr(
    not(doc),
    doc = "\nis_complex(input: Tensor) -> bool\n\nReturns True if the data type of :attr:`input` is a complex data type i.e.,\none of ``torch.complex64``, and ``torch.complex128``.\n\nArgs:\n    input (Tensor): the input tensor.\n\nExample::\n\n    >>> torch.is_complex(torch.tensor([1, 2, 3], dtype=torch.complex64))\n    True\n    >>> torch.is_complex(torch.tensor([1, 2, 3], dtype=torch.complex128))\n    True\n    >>> torch.is_complex(torch.tensor([1, 2, 3], dtype=torch.int32))\n    False\n    >>> torch.is_complex(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16))\n    False\n"
)]
#[cfg_attr(doc, doc = "See the runtime Python documentation for examples.")]
#[pyfunction(signature = (*args, **kwargs), text_signature = None)]
fn is_complex(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<bool> {
    let input = bind_legacy_single_tensor_argument("is_complex", args, kwargs)?;
    let tensor = input
        .value
        .cast::<PyTensor>()
        .expect("the is_complex input type was checked while binding");
    Ok(tensor.try_borrow()?.inner.is_complex())
}

// Preserve PyTorch's public docstring exactly rather than adding Rust Markdown markup.
#[allow(clippy::doc_markdown)]
#[cfg_attr(
    not(doc),
    doc = "\nis_floating_point(input: Tensor) -> bool\n\nReturns True if the data type of :attr:`input` is a floating point data type i.e.,\none of ``torch.float64``, ``torch.float32``, ``torch.float16``, and ``torch.bfloat16``.\n\nArgs:\n    input (Tensor): the input tensor.\n\nExample::\n\n    >>> torch.is_floating_point(torch.tensor([1.0, 2.0, 3.0]))\n    True\n    >>> torch.is_floating_point(torch.tensor([1, 2, 3], dtype=torch.int32))\n    False\n    >>> torch.is_floating_point(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16))\n    True\n    >>> torch.is_floating_point(torch.tensor([1, 2, 3], dtype=torch.complex64))\n    False\n"
)]
#[cfg_attr(doc, doc = "See the runtime Python documentation for examples.")]
#[pyfunction(signature = (*args, **kwargs), text_signature = None)]
fn is_floating_point(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<bool> {
    let input = bind_legacy_single_tensor_argument("is_floating_point", args, kwargs)?;
    let tensor = input
        .value
        .cast::<PyTensor>()
        .expect("the is_floating_point input type was checked while binding");
    Ok(tensor.try_borrow()?.inner.is_floating_point())
}

#[pyfunction(
    signature = (*args, **kwargs),
    text_signature = "(size=None, *, shape=None, dtype=None, device=None, requires_grad=False)"
)]
fn zeros(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
    let arguments = bind_creation_arguments("zeros", args, kwargs)?;
    let (size, dtype, device, requires_grad) = parse_creation_arguments("zeros", arguments)?;
    CoreTensor::zeros_with_metadata(size, dtype, device)
        .map(|inner| PyTensor::new(inner.with_requires_grad(requires_grad)))
        .map_err(|error| tensor_error(&error))
}

#[pyfunction(
    signature = (*args, **kwargs),
    text_signature = "(size=None, *, shape=None, dtype=None, device=None, requires_grad=False)"
)]
fn ones(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
    let arguments = bind_creation_arguments("ones", args, kwargs)?;
    let (size, dtype, device, requires_grad) = parse_creation_arguments("ones", arguments)?;
    CoreTensor::ones_with_metadata(size, dtype, device)
        .map(|inner| PyTensor::new(inner.with_requires_grad(requires_grad)))
        .map_err(|error| tensor_error(&error))
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
        .map_err(|error| creation_shape_error(&error, &shape))
}

#[pyfunction(
    signature = (*args, **kwargs),
    text_signature = "(size, fill_value, *, dtype=None, device=None, requires_grad=False)"
)]
fn full(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
    let arguments = bind_full_arguments(args, kwargs)?;
    let (size, fill_value, dtype, device, requires_grad) = parse_full_arguments(arguments)?;
    let shape = validate_size(size)?;
    CoreTensor::validate_full_shape(&shape)
        .map_err(|error| creation_shape_error(&error, &shape))?;
    let fill_value = fill_value.into_f32()?;
    CoreTensor::full_with_metadata(shape, fill_value, dtype, device)
        .map(|inner| PyTensor::new(inner.with_requires_grad(requires_grad)))
        .map_err(|error| tensor_error(&error))
}

fn float32_object(py: Python<'_>) -> PyResult<&'static Py<PyDType>> {
    FLOAT32.get_or_try_init(py, || {
        Py::new(
            py,
            PyDType {
                inner: DType::Float32,
            },
        )
    })
}

fn layout_objects(py: Python<'_>) -> PyResult<&'static PyLayoutObjects> {
    LAYOUT_OBJECTS.get_or_try_init(py, || create_layout_objects(py))
}

fn strided_object(py: Python<'_>) -> PyResult<&'static Py<PyAny>> {
    Ok(&layout_objects(py)?.strided)
}

fn memory_format_object(
    py: Python<'_>,
    memory_format: MemoryFormat,
) -> PyResult<&'static Py<PyMemoryFormat>> {
    let object = match memory_format {
        MemoryFormat::Preserve => &PRESERVE_FORMAT,
        MemoryFormat::Contiguous => &CONTIGUOUS_FORMAT,
        MemoryFormat::ChannelsLast => &CHANNELS_LAST,
        MemoryFormat::ChannelsLast3d => &CHANNELS_LAST_3D,
    };
    object.get_or_try_init(py, || {
        Py::new(
            py,
            PyMemoryFormat {
                inner: memory_format,
            },
        )
    })
}

fn warn_once(py: Python<'_>, emitted: &AtomicBool, message: &CStr) -> PyResult<()> {
    if emitted.swap(true, Ordering::Relaxed) {
        return Ok(());
    }
    PyErr::warn(py, &py.get_type::<PyUserWarning>(), message, 1)
}

fn parse_clone_memory_format(memory_format: Option<&Bound<'_, PyAny>>) -> PyResult<MemoryFormat> {
    let Some(memory_format) = memory_format else {
        return Ok(MemoryFormat::Preserve);
    };
    if memory_format.is_none() {
        return Ok(MemoryFormat::Preserve);
    }
    if let Ok(memory_format) = memory_format.cast::<PyMemoryFormat>() {
        return Ok(memory_format.try_borrow()?.inner);
    }

    let type_name = memory_format.get_type().name()?;
    Err(PyTypeError::new_err(format!(
        "clone(): argument 'memory_format' must be torch.memory_format, not {type_name}"
    )))
}

fn parse_is_contiguous_memory_format(memory_format: &Bound<'_, PyAny>) -> PyResult<MemoryFormat> {
    if let Ok(memory_format) = memory_format.cast::<PyMemoryFormat>() {
        return Ok(memory_format.try_borrow()?.inner);
    }

    let type_name = memory_format.get_type().name()?;
    Err(PyTypeError::new_err(format!(
        "is_contiguous(): argument 'memory_format' must be torch.memory_format, not {type_name}"
    )))
}

fn parse_contiguous_memory_format(memory_format: &Bound<'_, PyAny>) -> PyResult<MemoryFormat> {
    if let Ok(memory_format) = memory_format.cast::<PyMemoryFormat>() {
        return Ok(memory_format.try_borrow()?.inner);
    }

    let type_name = memory_format.get_type().name()?;
    Err(PyTypeError::new_err(format!(
        "contiguous(): argument 'memory_format' must be torch.memory_format, not {type_name}"
    )))
}

fn bind_creation_arguments<'py>(
    function: &str,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<CreationCallArguments<'py>> {
    // PyTorch rejects excess positional arguments before inspecting keywords.
    if positional.len() > 1 {
        return Err(PyTypeError::new_err(format!(
            "{function}() takes 1 positional argument but {} were given",
            positional.len()
        )));
    }

    let mut size_was_provided = !positional.is_empty();
    let mut arguments = CreationCallArguments {
        size: if positional.is_empty() {
            None
        } else {
            optional_call_argument(positional.get_item(0)?)
        },
        shape: None,
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
            "size" => {
                if size_was_provided {
                    arguments.keyword_error.get_or_insert_with(|| {
                        PyTypeError::new_err(format!(
                            "{function}() got multiple values for argument 'size'"
                        ))
                    });
                } else {
                    size_was_provided = true;
                    arguments.size = optional_call_argument(value);
                }
            }
            "shape" => arguments.shape = optional_call_argument(value),
            "dtype" => arguments.dtype = optional_call_argument(value),
            "device" => arguments.device = optional_call_argument(value),
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

fn optional_call_argument(value: Bound<'_, PyAny>) -> Option<Bound<'_, PyAny>> {
    if value.is_none() { None } else { Some(value) }
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
) -> PyResult<(Vec<usize>, DType, Device, bool)> {
    let CreationCallArguments {
        size,
        shape,
        dtype,
        device,
        requires_grad,
        keyword_error,
    } = arguments;

    // PyTorch validates declared argument types in signature order, then
    // reports duplicate or unknown keywords, and only then resolves a valid
    // device specification.
    let size = parse_creation_size(function, size.as_ref(), shape.as_ref())?;
    let dtype = parse_dtype(function, dtype.as_ref())?;
    validate_device_argument_type(function, device.as_ref())?;
    let requires_grad = parse_factory_requires_grad(function, requires_grad.as_ref())?;
    if let Some(error) = keyword_error {
        return Err(error);
    }
    let device = parse_device(function, device.as_ref())?;
    Ok((size, dtype, device, requires_grad))
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
            "dtype" => arguments.dtype = optional_call_argument(value),
            "device" => arguments.device = optional_call_argument(value),
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

fn parse_full_arguments(
    arguments: FullCallArguments<'_>,
) -> PyResult<(Vec<i64>, ParsedFillValue, DType, Device, bool)> {
    let FullCallArguments {
        size,
        fill_value,
        dtype,
        device,
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
    let fill_value = parse_fill_value(&fill_value)?;
    let dtype = parse_dtype("full", dtype.as_ref())?;
    validate_device_argument_type("full", device.as_ref())?;
    let requires_grad = parse_factory_requires_grad("full", requires_grad.as_ref())?;
    if let Some(error) = keyword_error {
        return Err(error);
    }
    let device = parse_device("full", device.as_ref())?;
    Ok((size, fill_value, dtype, device, requires_grad))
}

fn parse_creation_size(
    function: &str,
    size: Option<&Bound<'_, PyAny>>,
    shape: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<usize>> {
    let value = match (size, shape) {
        (Some(_), Some(_)) => {
            return Err(PyTypeError::new_err(format!(
                "{function}() received both 'size' and its compatibility alias 'shape'"
            )));
        }
        (Some(value), None) | (None, Some(value)) => value,
        (None, None) => {
            return Err(PyTypeError::new_err(format!(
                "{function}() missing required argument 'size'"
            )));
        }
    };
    value.extract::<Vec<usize>>()
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
        return Ok(dtype.try_borrow()?.inner);
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

fn parse_device_value(function: &str, device: &Bound<'_, PyAny>) -> PyResult<Device> {
    if let Ok(device) = device.cast::<PyDevice>() {
        return Ok(device.try_borrow()?.inner);
    }
    if let Ok(device) = device.cast::<PyString>() {
        let specification = device.to_str()?;
        if specification == "cpu" {
            return Ok(Device::Cpu);
        }
        return Err(PyRuntimeError::new_err(format!(
            "{function}(): device '{specification}' is not supported; only 'cpu' is implemented"
        )));
    }

    let error = device_argument_type_error(function, device)?;
    Err(error)
}

fn device_argument_type_error(function: &str, device: &Bound<'_, PyAny>) -> PyResult<PyErr> {
    let argument = if function == "device" {
        "type"
    } else {
        "device"
    };
    let type_name = device.get_type().name()?;
    Ok(PyTypeError::new_err(format!(
        "{function}(): argument '{argument}' must be torch.device or str, not {type_name}"
    )))
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

fn is_dimension_swap_integer(dimension: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !dimension.is_instance_of::<PyBool>() && dimension.is_instance_of::<PyInt>() {
        return Ok(true);
    }

    if let Ok(numpy) = PyModule::import(dimension.py(), "numpy") {
        let numpy_integer = numpy.getattr("integer")?;
        if dimension.is_instance(&numpy_integer)? {
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

    let type_name = transpose_type_name(dimension)?;
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

fn extract_dimension_swap_dimension(dimension: &Bound<'_, PyAny>) -> PyResult<i64> {
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

    let actual = transpose_type_name(dimension)?;
    let position = position.map_or_else(String::new, |position| format!(" (position {position})"));
    Err(PyTypeError::new_err(format!(
        "flatten(): argument '{argument}'{position} must be int, not {actual}"
    )))
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
    let actual = transpose_type_name(input)?;
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
                let actual = transpose_type_name(&dimension)?;
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

    let actual = transpose_type_name(argument)?;
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
        let actual = transpose_type_name(&dimension)?;
        let parsed_dimension = parse_squeeze_integer(&dimension, true).map_err(|_| {
            PyTypeError::new_err(format!(
                "squeeze(): argument 'dim' failed to unpack the object at pos {} with error \"Overflow when unpacking long long\"",
                index + 1
            ))
        })?;
        let Some(dimension) = parsed_dimension else {
            if index == 0 {
                let sequence_type = transpose_type_name(sequence)?;
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
    if !value.is_instance_of::<PyTuple>() && !value.is_instance_of::<PyList>() {
        return transpose_type_name(value);
    }

    let kind = transpose_type_name(value)?;
    let (opening, closing, trailing, names) = if let Ok(sequence) = value.cast::<PyTuple>() {
        let mut names = try_size_vector(sequence.len())?;
        for index in 0..sequence.len() {
            names.push(transpose_type_name(&sequence.get_item(index)?)?);
        }
        ("(", ")", sequence.len() == 1, names)
    } else {
        let sequence = value.cast::<PyList>()?;
        let mut names = try_size_vector(sequence.len())?;
        for index in 0..sequence.len() {
            names.push(transpose_type_name(&sequence.get_item(index)?)?);
        }
        ("[", "]", false, names)
    };
    let names = names.join(", ");
    let trailing = if trailing { "," } else { "" };
    Ok(format!("{kind} of {opening}{names}{trailing}{closing}"))
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
    let mut positional_names = try_size_vector(positional.len())?;
    for value in positional.iter() {
        positional_names.push(transpose_type_name(&value)?);
    }

    let keyword_length = keywords.map_or(0, PyDictMethods::len);
    let mut keyword_names = try_size_vector(keyword_length)?;
    if let Some(keywords) = keywords {
        for (key, value) in keywords {
            keyword_names.push((key.extract::<String>()?, transpose_type_name(&value)?));
        }
        match keyword_order {
            CallKeywordOrder::Sorted => {
                keyword_names.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            }
            CallKeywordOrder::PyTorchUnorderedMap => {
                keyword_names = pytorch_unordered_keyword_order(keyword_names)?;
            }
        }
    }
    let keyword_names = keyword_names
        .into_iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(", ");

    let positional_names = positional_names.join(", ");
    match (positional_names.is_empty(), keyword_names.is_empty()) {
        (true, true) => Ok(String::new()),
        (false, true) => Ok(positional_names),
        (true, false) => Ok(format!("{keyword_names}, ")),
        (false, false) => Ok(format!("{positional_names}, {keyword_names}")),
    }
}

fn pytorch_unordered_keyword_order(
    keywords: Vec<(String, String)>,
) -> PyResult<Vec<(String, String)>> {
    // PyTorch 2.13's overload formatter copies keyword arguments into
    // libstdc++'s `std::unordered_map`. Reproduce its MurmurHash64A buckets
    // and prime rehash policy so collision groups retain the same order.
    let capacity = keywords.len();
    let mut bucket_count = 13_u64;
    let mut ordered = try_size_vector(capacity)?;

    for (key, value) in keywords {
        if usize::try_from(bucket_count).is_ok_and(|count| ordered.len() == count) {
            bucket_count = next_prime(bucket_count.saturating_mul(2));
            let previous = ordered;
            ordered = try_size_vector(capacity)?;
            for entry in previous {
                insert_unordered_keyword(&mut ordered, entry, bucket_count);
            }
        }
        let hash = pytorch_string_hash(&key);
        insert_unordered_keyword(&mut ordered, (hash, key, value), bucket_count);
    }

    Ok(ordered
        .into_iter()
        .map(|(_, key, value)| (key, value))
        .collect())
}

fn insert_unordered_keyword(
    ordered: &mut Vec<(u64, String, String)>,
    entry: (u64, String, String),
    bucket_count: u64,
) {
    let bucket = entry.0 % bucket_count;
    let position = ordered
        .iter()
        .position(|existing| existing.0 % bucket_count == bucket)
        .unwrap_or(0);
    ordered.insert(position, entry);
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

fn next_prime(mut candidate: u64) -> u64 {
    candidate |= 1;
    while !is_prime(candidate) {
        candidate = candidate.saturating_add(2);
    }
    candidate
}

fn is_prime(candidate: u64) -> bool {
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
    let input_type = transpose_type_name(&input)?;

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
    let dimension_type = transpose_type_name(&dimension_value)?;
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

fn bind_detach_argument<'py>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<ParsedCallArgument<'py>> {
    if positional.len() > 1 {
        return Err(PyTypeError::new_err(format!(
            "detach() takes 1 positional argument but {} were given",
            positional.len()
        )));
    }

    let mut input = if positional.is_empty() {
        None
    } else {
        Some(ParsedCallArgument {
            value: positional.get_item(0)?,
            position: Some(1),
        })
    };
    let mut keyword_error = None;
    if let Some(keywords) = keywords {
        for (key, value) in keywords {
            let key = key.extract::<String>()?;
            if key != "input" {
                keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "detach() got an unexpected keyword argument '{key}'"
                    ))
                });
            } else if input.is_some() {
                keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err("detach() got multiple values for argument 'input'")
                });
            } else {
                input = Some(ParsedCallArgument {
                    value,
                    position: None,
                });
            }
        }
    }

    let input = input.ok_or_else(|| {
        PyTypeError::new_err("detach() missing 1 required positional arguments: \"input\"")
    })?;
    if input.value.cast::<PyTensor>().is_err() {
        let position = input
            .position
            .map_or_else(String::new, |position| format!(" (position {position})"));
        let actual = transpose_type_name(&input.value)?;
        return Err(PyTypeError::new_err(format!(
            "detach(): argument 'input'{position} must be Tensor, not {actual}"
        )));
    }
    if let Some(keyword_error) = keyword_error {
        return Err(keyword_error);
    }
    Ok(input)
}

fn bind_legacy_single_tensor_argument<'py>(
    function: &str,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<ParsedCallArgument<'py>> {
    if positional.len() > 1 {
        return Err(PyTypeError::new_err(format!(
            "{function}() takes 1 positional argument but {} were given",
            positional.len()
        )));
    }

    // PyTorch's legacy parser resolves `input`, then `x`, then `a` for type checking.
    let (keyword_input, keyword_alias) = match keywords {
        Some(values) => {
            if let Some(input) = values.get_item("input")? {
                (Some(input), None)
            } else if let Some(input) = values.get_item("x")? {
                (Some(input), Some("x"))
            } else if let Some(input) = values.get_item("a")? {
                (Some(input), Some("a"))
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
    if input.value.cast::<PyTensor>().is_err() {
        let position = input
            .position
            .map_or_else(String::new, |position| format!(" (position {position})"));
        let input_type = transpose_type_name(&input.value)?;
        return Err(PyTypeError::new_err(format!(
            "{function}(): argument 'input'{position} must be Tensor, not {input_type}"
        )));
    }

    if let Some(keywords) = keywords {
        // The legacy aliases are accepted only as the sole keyword. Mixed calls
        // validate their original keyword order and report an alias as unexpected.
        let sole_alias = if positional.is_empty() && keywords.len() == 1 {
            keyword_alias
        } else {
            None
        };
        for key in keywords.keys() {
            let key = key.extract::<String>()?;
            if sole_alias == Some(key.as_str()) {
                continue;
            }
            if key == "input" {
                if !positional.is_empty() {
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

    Ok(input)
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

fn parse_tensor_argument<'a, 'py>(
    function: &str,
    argument: &str,
    value: &'a ParsedCallArgument<'py>,
) -> PyResult<&'a Bound<'py, PyTensor>> {
    let Ok(tensor) = value.value.cast::<PyTensor>() else {
        let position = value
            .position
            .map_or_else(String::new, |position| format!(" (position {position})"));
        let actual = transpose_type_name(&value.value)?;
        return Err(PyTypeError::new_err(format!(
            "{function}(): argument '{argument}'{position} must be Tensor, not {actual}"
        )));
    };
    Ok(tensor)
}

fn bind_multiplication_argument<'py>(
    operation: MultiplicationMethod,
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
) -> PyResult<(ParsedCallArgument<'py>, Option<PyErr>)> {
    if matches!(operation, MultiplicationMethod::Multiply) {
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

    let function = operation.name();
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
    let mut keyword_error = None;
    if let Some(keywords) = keywords {
        for (key, value) in keywords {
            let key = key.extract::<String>()?;
            if key != "other" {
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

    let other = other.ok_or_else(|| {
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

fn multiply_binding_error(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyErr> {
    let summary = call_type_summary(positional, keywords, CallKeywordOrder::PyTorchUnorderedMap)?;
    let keyword_length = keywords.map_or(0, PyDictMethods::len);
    let mismatch = if positional.len() + keyword_length == 1 {
        if positional.len() == 1 {
            let value = positional.get_item(0)?;
            let detail = call_argument_type_description(&value)?;
            format!(
                "\n      didn't match because some of the arguments have invalid types: (!{detail}!)"
            )
        } else {
            let keywords = keywords.expect("a single keyword argument is present");
            let (key, value) = keywords
                .iter()
                .next()
                .expect("a single keyword argument remains present");
            let key = key.extract::<String>()?;
            if key == "other" {
                let detail = call_argument_type_description(&value)?;
                format!(
                    "\n      didn't match because some of the arguments have invalid types: (!other={detail}!, )"
                )
            } else {
                format!("\n      didn't match because some of the keywords were incorrect: {key}")
            }
        }
    } else {
        String::new()
    };

    Ok(PyTypeError::new_err(format!(
        "multiply() received an invalid combination of arguments - got ({summary}), but expected one of:\n * (Tensor other){mismatch}\n * (Number other){mismatch}\n"
    )))
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
                let actual = transpose_type_name(&argument.value)?;
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

fn transpose_type_name(value: &Bound<'_, PyAny>) -> PyResult<String> {
    // PyTorch reports CPython's `tp_name`: heap types use their unqualified
    // class name, while static extension types retain their module prefix.
    const PY_TPFLAGS_HEAPTYPE: u64 = 1 << 9;

    let value_type = value.get_type();
    let name = value_type.name()?.to_str()?.to_owned();
    let module = value_type.getattr("__module__")?.extract::<String>()?;
    let flags = value_type.getattr("__flags__")?.extract::<u64>()?;
    if module == "torch_rs" && matches!(name.as_str(), "dtype" | "device" | "memory_format") {
        Ok(format!("torch.{name}"))
    } else if flags & PY_TPFLAGS_HEAPTYPE == 0 && module != "builtins" {
        Ok(format!("{module}.{name}"))
    } else {
        Ok(name)
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

fn normalize_dimension(dimension: i64, rank: usize) -> PyResult<usize> {
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
        .map_err(|_| PyRuntimeError::new_err("std::bad_alloc"))?;
    Ok(values)
}

fn try_push_size<T>(values: &mut Vec<T>, value: T) -> PyResult<()> {
    values
        .try_reserve(1)
        .map_err(|_| PyRuntimeError::new_err("std::bad_alloc"))?;
    values.push(value);
    Ok(())
}

fn invalid_size_dimension(index: usize, reason: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "full(): size element at index {index} is invalid: {reason}"
    ))
}

fn parse_fill_value(fill_value: &Bound<'_, PyAny>) -> PyResult<ParsedFillValue> {
    if let Ok(tensor) = fill_value.cast::<PyTensor>() {
        let tensor = tensor.try_borrow()?;
        if !tensor.inner.shape().is_empty() {
            return Err(PyTypeError::new_err(
                "full(): fill_value tensor must be zero-dimensional",
            ));
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

    parse_numpy_fill_value(fill_value)
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

fn parse_numpy_fill_value(fill_value: &Bound<'_, PyAny>) -> PyResult<ParsedFillValue> {
    parse_numpy_value(
        fill_value,
        invalid_fill_value,
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
    invalid_value: fn() -> PyErr,
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

fn invalid_fill_value() -> PyErr {
    PyTypeError::new_err("full(): fill_value must be a number or zero-dimensional tensor")
}

fn bool_subtraction_error() -> PyErr {
    PyRuntimeError::new_err(
        "Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.",
    )
}

fn creation_shape_error(error: &TensorError, shape: &[usize]) -> PyErr {
    if matches!(error, TensorError::ElementCountOverflow) {
        PyRuntimeError::new_err(format!(
            "Storage size calculation overflowed with size {shape:?}"
        ))
    } else {
        tensor_error(error)
    }
}

impl ParsedFillValue {
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
}

impl ParsedArithmeticScalar {
    fn is_python_bool(&self) -> bool {
        matches!(self, Self::PythonBool(_))
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
    PyRuntimeError::new_err("value cannot be converted to float32 without overflow")
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
    let type_name = transpose_type_name(value)?;
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

fn tensor_error(error: &TensorError) -> PyErr {
    match error {
        TensorError::ShapeDataMismatch { .. }
        | TensorError::ShapeMismatch { .. }
        | TensorError::MatmulRequiresMatrices { .. }
        | TensorError::MatmulInnerDimensionMismatch { .. }
        | TensorError::ItemRequiresOneElement { .. }
        | TensorError::InvalidStorageOffset { .. }
        | TensorError::IndexCalculationOverflow
        | TensorError::ReshapeMultipleInferredDimensions
        | TensorError::ReshapeInvalidDimension { .. }
        | TensorError::ReshapeAmbiguousZeroElements { .. }
        | TensorError::ReshapeElementCountMismatch { .. }
        | TensorError::StrideCalculationOverflow
        | TensorError::StorageCapacityOverflow { .. }
        | TensorError::AllocationFailed { .. }
        | TensorError::UnsupportedMemoryFormat { .. }
        | TensorError::ContiguousPreserveFormatUnsupported
        | TensorError::ContiguousMemoryFormatRankMismatch { .. }
        | TensorError::PermutationRankMismatch { .. }
        | TensorError::PermutationDimensionOutOfRange { .. }
        | TensorError::DuplicatePermutationDimension { .. }
        | TensorError::MatrixTransposeRequiresMatrix { .. }
        | TensorError::DuplicateDimension { .. }
        | TensorError::SqueezeDimensionsRankLimit
        | TensorError::FlattenStartAfterEnd
        | TensorError::FlattenNonConcreteInteger
        | TensorError::ElementCountOverflow
        | TensorError::BackwardRequiresScalar { .. }
        | TensorError::DoesNotRequireGrad
        | TensorError::BackwardGraphFreed => PyRuntimeError::new_err(error.to_string()),
        TensorError::InvalidScalarIndex
        | TensorError::TooManyIndices { .. }
        | TensorError::IndexOutOfBounds { .. }
        | TensorError::DimensionOutOfRange { .. } => PyIndexError::new_err(error.to_string()),
    }
}

fn item_error(error: &TensorError) -> PyErr {
    if let TensorError::ItemRequiresOneElement { elements } = error {
        PyRuntimeError::new_err(format!(
            "a Tensor with {elements} elements cannot be converted to Scalar"
        ))
    } else {
        tensor_error(error)
    }
}

fn scalar_conversion_error(error: &TensorError) -> PyErr {
    if matches!(error, TensorError::ItemRequiresOneElement { .. }) {
        PyValueError::new_err("only one element tensors can be converted to Python scalars")
    } else {
        tensor_error(error)
    }
}

fn transpose_error(error: &TensorError) -> PyErr {
    if matches!(error, TensorError::ElementCountOverflow) {
        PyRuntimeError::new_err("numel: integer multiplication overflow")
    } else {
        tensor_error(error)
    }
}

#[pymodule]
fn torch_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<PyTensor>()?;
    let tensor_base = py.get_type::<PyTensorBase>();
    let int_descriptor = tensor_base.getattr("__int__")?;
    tensor_base.setattr("__int__", int_descriptor)?;
    let float_descriptor = tensor_base.getattr("__float__")?;
    tensor_base.setattr("__float__", float_descriptor)?;
    if tensor_base.hasattr("int_scalar")? {
        tensor_base.delattr("int_scalar")?;
    }
    if tensor_base.hasattr("float_scalar")? {
        tensor_base.delattr("float_scalar")?;
    }
    module.add_class::<PyDType>()?;
    module.add_class::<PyDevice>()?;
    module.add_class::<PyMemoryFormat>()?;
    module.add_class::<PyNoGrad>()?;
    let no_grad_helpers = PyModule::from_code(
        py,
        NO_GRAD_WRAPPER_SOURCE,
        c"torch_rs/_no_grad.py",
        c"torch_rs._no_grad",
    )?;
    let no_grad_class = no_grad_helpers
        .getattr("_make_no_grad")?
        .call1((module.getattr("_NoGradContext")?,))?;
    module
        .getattr("__all__")?
        .call_method1("remove", ("_NoGradContext",))?;
    module.delattr("_NoGradContext")?;
    module.add("no_grad", no_grad_class)?;
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
    module.add_function(wrap_pyfunction!(is_grad_enabled, module)?)?;
    module.add_function(wrap_pyfunction!(get_default_dtype, module)?)?;
    module.add_function(wrap_pyfunction!(tensor, module)?)?;
    module.add_function(wrap_pyfunction!(clone, module)?)?;
    module.add_function(wrap_pyfunction!(detach, module)?)?;
    module.add_function(wrap_pyfunction!(equal, module)?)?;
    module.add_function(wrap_pyfunction!(transpose, module)?)?;
    module.add_function(wrap_pyfunction!(squeeze, module)?)?;
    module.add_function(wrap_pyfunction!(flatten, module)?)?;
    module.add_function(wrap_pyfunction!(numel, module)?)?;
    module.add_function(wrap_pyfunction!(is_nonzero, module)?)?;
    module.add_function(wrap_pyfunction!(is_complex, module)?)?;
    module.add_function(wrap_pyfunction!(is_floating_point, module)?)?;
    module.add_function(wrap_pyfunction!(zeros, module)?)?;
    module.add_function(wrap_pyfunction!(ones, module)?)?;
    module.add_function(wrap_pyfunction!(eye, module)?)?;
    module.add_function(wrap_pyfunction!(full, module)?)?;
    let float32 = float32_object(py)?;
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

    use super::{PyTensor, flatten_buffer, half_to_f32, nested_list, torch_rs, try_size_vector};

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

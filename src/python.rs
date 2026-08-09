use std::ffi::CStr;
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{
    PyIndexError, PyMemoryError, PyOverflowError, PyRuntimeError, PyTypeError, PyUserWarning,
    PyValueError,
};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyAny, PyBool, PyDict, PyFloat, PyInt, PyList, PyModule, PySequence, PyString, PyTuple,
};

use crate::{DType, Device, MemoryFormat, Tensor as CoreTensor, TensorError};

static FLOAT32: PyOnceLock<Py<PyDType>> = PyOnceLock::new();
static PRESERVE_FORMAT: PyOnceLock<Py<PyMemoryFormat>> = PyOnceLock::new();
static CONTIGUOUS_FORMAT: PyOnceLock<Py<PyMemoryFormat>> = PyOnceLock::new();
static CHANNELS_LAST: PyOnceLock<Py<PyMemoryFormat>> = PyOnceLock::new();
static CHANNELS_LAST_3D: PyOnceLock<Py<PyMemoryFormat>> = PyOnceLock::new();
static T_NON_MATRIX_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);
static T_SCALAR_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);
static MT_SCALAR_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);

#[cfg(target_os = "macos")]
const T_NON_MATRIX_WARNING: &CStr = c"The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4317.)";
#[cfg(target_os = "macos")]
const T_SCALAR_WARNING: &CStr = c"Tensor.T is deprecated on 0-D tensors. This function is the identity in these cases. (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4322.)";
#[cfg(target_os = "macos")]
const MT_SCALAR_WARNING: &CStr = c"Tensor.mT is deprecated on 0-D tensors. This function is the identity in these cases. (Triggered internally at /Users/runner/work/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4374.)";

#[cfg(target_os = "linux")]
const T_NON_MATRIX_WARNING: &CStr = c"The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:4317.)";
#[cfg(target_os = "linux")]
const T_SCALAR_WARNING: &CStr = c"Tensor.T is deprecated on 0-D tensors. This function is the identity in these cases. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:4322.)";
#[cfg(target_os = "linux")]
const MT_SCALAR_WARNING: &CStr = c"Tensor.mT is deprecated on 0-D tensors. This function is the identity in these cases. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:4374.)";

#[cfg(target_os = "windows")]
const T_NON_MATRIX_WARNING: &CStr = c"The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at D:\\a\\pytorch\\pytorch\\aten\\src\\ATen\\native\\TensorShape.cpp:4317.)";
#[cfg(target_os = "windows")]
const T_SCALAR_WARNING: &CStr = c"Tensor.T is deprecated on 0-D tensors. This function is the identity in these cases. (Triggered internally at D:\\a\\pytorch\\pytorch\\aten\\src\\ATen\\native\\TensorShape.cpp:4322.)";
#[cfg(target_os = "windows")]
const MT_SCALAR_WARNING: &CStr = c"Tensor.mT is deprecated on 0-D tensors. This function is the identity in these cases. (Triggered internally at D:\\a\\pytorch\\pytorch\\aten\\src\\ATen\\native\\TensorShape.cpp:4374.)";

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
        match self.inner {
            Device::Cpu => None,
        }
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

/// Python-facing tensor backed by the native Rust tensor core.
#[pyclass(name = "Tensor", module = "torch_rs", skip_from_py_object)]
#[derive(Clone)]
struct PyTensor {
    inner: CoreTensor,
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

enum EyeDimensionArgument {
    Omitted,
    Provided(Py<PyAny>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for EyeDimensionArgument {
    type Error = PyErr;

    fn extract(object: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self::Provided(object.into()))
    }
}

struct ParsedCallArgument<'py> {
    value: Bound<'py, PyAny>,
    position: Option<usize>,
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
enum ConstantFactory {
    Zeros,
    Ones,
}

enum ParsedSizeDimension {
    Value(i64),
    Overflow,
    Invalid(String),
}

enum PreparedSizeDimension<'py> {
    Parsed(ParsedSizeDimension),
    Deferred(Bound<'py, PyAny>),
}

struct PreparedFactorySize<'py> {
    dimensions: Vec<PreparedSizeDimension<'py>>,
}

enum FirstSizeErrorStyle {
    PositionalValue,
    PositionalCollection,
    KeywordCollection(String),
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
            .map(|inner| Self { inner })
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
        Py::new(slf.py(), Self { inner })
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
        Py::new(slf.py(), Self { inner })
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn transpose(
        &self,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let [dim0, dim1] = bind_transpose_arguments(args, kwargs, ["dim0", "dim1"])?;
        let dim0 = parse_transpose_dimension("dim0", dim0.position, &dim0.value)?;
        let dim1 = parse_transpose_dimension("dim1", dim1.position, &dim1.value)?;
        self.inner
            .transpose(dim0, dim1)
            .map(|inner| Self { inner })
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
            .map(|inner| Self { inner })
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
        Py::new(slf.py(), Self { inner })
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
        inner
            .map(|inner| Self { inner })
            .map_err(|error| tensor_error(&error))
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
            .map(|inner| Self { inner })
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

    fn numel(&self) -> usize {
        self.inner.numel()
    }

    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let values = self
            .inner
            .try_to_vec()
            .map_err(|error| tensor_error(&error))?;
        nested_list(py, &values, self.inner.shape())
    }

    fn item(&self) -> PyResult<f32> {
        self.inner.item().map_err(|error| tensor_error(&error))
    }

    #[pyo3(signature = (*, memory_format=None))]
    fn clone(&self, memory_format: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let memory_format = parse_clone_memory_format(memory_format)?;
        self.inner
            .try_clone_with_memory_format(memory_format)
            .map(|inner| Self { inner })
            .map_err(|error| tensor_error(&error))
    }

    fn relu(&self) -> PyResult<Self> {
        self.inner
            .relu()
            .map(|inner| Self { inner })
            .map_err(|error| tensor_error(&error))
    }

    fn sin(&self) -> PyResult<Self> {
        self.inner
            .sin()
            .map(|inner| Self { inner })
            .map_err(|error| tensor_error(&error))
    }

    fn exp(&self) -> PyResult<Self> {
        self.inner
            .exp()
            .map(|inner| Self { inner })
            .map_err(|error| tensor_error(&error))
    }

    fn sum(&self) -> Self {
        Self {
            inner: self.inner.sum(),
        }
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

    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary_operation(py, other, BinaryOperation::Multiply, true)
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
            .map(|inner| Self { inner })
            .map_err(|error| tensor_error(&error))
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

        Self {
            inner: result.map_err(|error| tensor_error(&error))?,
        }
        .into_py_any(py)
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

#[pyfunction(signature = (data, *, dtype=None, device=None))]
fn tensor(
    data: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyTensor> {
    let (dtype, device) = parse_metadata("tensor", dtype, device)?;
    let mut flattened = Vec::new();
    let shape = flatten_rectangular(data, &mut flattened)?;
    CoreTensor::from_vec_with_metadata(flattened, shape, dtype, device)
        .map(|inner| PyTensor { inner })
        .map_err(|error| tensor_error(&error))
}

#[pyfunction(signature = (input, *, memory_format=None))]
fn clone(input: &PyTensor, memory_format: Option<&Bound<'_, PyAny>>) -> PyResult<PyTensor> {
    let memory_format = parse_clone_memory_format(memory_format)?;
    input
        .inner
        .try_clone_with_memory_format(memory_format)
        .map(|inner| PyTensor { inner })
        .map_err(|error| tensor_error(&error))
}

#[pyfunction(signature = (*args, **kwargs))]
fn transpose(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
    let [input, dim0, dim1] = bind_transpose_arguments(args, kwargs, ["input", "dim0", "dim1"])?;
    let input_type = transpose_type_name(&input.value)?;
    let input_tensor = input.value.cast::<PyTensor>().map_err(|_| {
        transpose_argument_type_error("input", input.position, "Tensor", &input_type)
    })?;
    let input_tensor = input_tensor.try_borrow()?;
    let dim0 = parse_transpose_dimension("dim0", dim0.position, &dim0.value)?;
    let dim1 = parse_transpose_dimension("dim1", dim1.position, &dim1.value)?;
    input_tensor
        .inner
        .transpose(dim0, dim1)
        .map(|inner| PyTensor { inner })
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
        .map(|inner| PyTensor { inner })
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
    Py::new(args.py(), PyTensor { inner })
}

#[pyfunction(
    signature = (*args, **kwargs),
    text_signature = "(*size, shape=None, dtype=None, device=None)"
)]
fn zeros(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
    create_constant_tensor(ConstantFactory::Zeros, args, kwargs)
}

#[pyfunction(
    signature = (*args, **kwargs),
    text_signature = "(*size, shape=None, dtype=None, device=None)"
)]
fn ones(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyTensor> {
    create_constant_tensor(ConstantFactory::Ones, args, kwargs)
}

#[pyfunction(
    signature = (n, m=EyeDimensionArgument::Omitted, *, dtype=None, device=None),
    text_signature = "(n, m=None, *, dtype=None, device=None)"
)]
fn eye(
    n: &Bound<'_, PyAny>,
    m: EyeDimensionArgument,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyTensor> {
    let py = n.py();
    let (dtype, device) = parse_metadata("eye", dtype, device)?;
    let n = parse_eye_dimension("n", n)?;
    let m = match m {
        EyeDimensionArgument::Omitted => n,
        EyeDimensionArgument::Provided(m) => parse_eye_dimension("m", m.bind(py))?,
    };
    let n = validate_eye_dimension("n", n)?;
    let m = validate_eye_dimension("m", m)?;
    let shape = [n, m];

    CoreTensor::eye_with_metadata(n, m, dtype, device)
        .map(|inner| PyTensor { inner })
        .map_err(|error| creation_shape_error(&error, &shape))
}

#[pyfunction(signature = (size, fill_value, *, dtype=None, device=None))]
fn full(
    size: &Bound<'_, PyAny>,
    fill_value: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyTensor> {
    let (dtype, device) = parse_metadata("full", dtype, device)?;
    let size = parse_size(size)?;
    let fill_value = parse_fill_value(fill_value)?;
    let shape = validate_size(size)?;
    CoreTensor::validate_full_shape(&shape)
        .map_err(|error| creation_shape_error(&error, &shape))?;
    let fill_value = fill_value.into_f32()?;
    CoreTensor::full_with_metadata(shape, fill_value, dtype, device)
        .map(|inner| PyTensor { inner })
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

impl ConstantFactory {
    const fn name(self) -> &'static str {
        match self {
            Self::Zeros => "zeros",
            Self::Ones => "ones",
        }
    }
}

fn create_constant_tensor(
    factory: ConstantFactory,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyTensor> {
    let function = factory.name();
    let size_keyword = match kwargs {
        Some(kwargs) => kwargs.get_item("size")?,
        None => None,
    };
    let shape_keyword = match kwargs {
        Some(kwargs) => kwargs.get_item("shape")?,
        None => None,
    };

    if args.is_empty() && size_keyword.is_none() && shape_keyword.is_none() {
        return Err(PyTypeError::new_err(format!(
            "{function}() missing 1 required positional arguments: \"size\""
        )));
    }

    // PyTorch chooses the overload from the first size element before it
    // binds metadata or reports duplicate keywords. Preparing the input here
    // preserves that order while deferring later element conversion errors.
    let parsed_size = if args.is_empty() {
        let value = size_keyword
            .as_ref()
            .or(shape_keyword.as_ref())
            .expect("a size source was checked above");
        prepare_keyword_factory_size(function, value)?
    } else {
        prepare_positional_factory_size(function, args)?
    };

    if !args.is_empty() && size_keyword.is_some() {
        return Err(PyTypeError::new_err(format!(
            "{function}() got multiple values for argument 'size'"
        )));
    }
    if !args.is_empty() && shape_keyword.is_some()
        || size_keyword.is_some() && shape_keyword.is_some()
    {
        return Err(PyTypeError::new_err(format!(
            "{function}() got an unexpected keyword argument 'shape'"
        )));
    }

    if let Some(kwargs) = kwargs {
        for (key, _) in kwargs {
            let key = key.extract::<String>()?;
            if !matches!(key.as_str(), "size" | "shape" | "dtype" | "device") {
                return Err(PyTypeError::new_err(format!(
                    "{function}() got an unexpected keyword argument '{key}'"
                )));
            }
        }
    }

    let dtype = match kwargs {
        Some(kwargs) => kwargs.get_item("dtype")?,
        None => None,
    };
    let device = match kwargs {
        Some(kwargs) => kwargs.get_item("device")?,
        None => None,
    };
    let (dtype, device) = parse_metadata(function, dtype.as_ref(), device.as_ref())?;
    let signed_size = parsed_size.finish(function)?;
    let shape = validate_factory_size(factory, signed_size)?;

    let result = match factory {
        ConstantFactory::Zeros => CoreTensor::zeros_with_metadata(shape.clone(), dtype, device),
        ConstantFactory::Ones => CoreTensor::ones_with_metadata(shape.clone(), dtype, device),
    };
    result
        .map(|inner| PyTensor { inner })
        .map_err(|error| constant_factory_shape_error(&error, &shape))
}

fn prepare_positional_factory_size<'py>(
    function: &str,
    args: &Bound<'py, PyTuple>,
) -> PyResult<PreparedFactorySize<'py>> {
    if args.len() > 1 {
        let first = parse_size_dimension(&args.get_item(0)?, true)?;
        if matches!(first, ParsedSizeDimension::Invalid(_)) {
            return Err(PyTypeError::new_err(format!(
                "{function}() takes 1 positional argument but {} were given",
                args.len()
            )));
        }

        let mut dimensions = try_size_vector(args.len())?;
        try_push_size(&mut dimensions, PreparedSizeDimension::Parsed(first))?;
        for dimension in args.iter().skip(1) {
            try_push_size(&mut dimensions, PreparedSizeDimension::Deferred(dimension))?;
        }
        return Ok(PreparedFactorySize { dimensions });
    }

    let value = args.get_item(0)?;
    if let Ok(dimensions) = value.cast::<PyList>() {
        return prepare_factory_size_dimensions(
            function,
            dimensions.len(),
            dimensions.iter(),
            &FirstSizeErrorStyle::PositionalCollection,
        );
    }
    if let Ok(dimensions) = value.cast::<PyTuple>() {
        return prepare_factory_size_dimensions(
            function,
            dimensions.len(),
            dimensions.iter(),
            &FirstSizeErrorStyle::PositionalCollection,
        );
    }
    prepare_factory_size_dimensions(
        function,
        1,
        std::iter::once(value),
        &FirstSizeErrorStyle::PositionalValue,
    )
}

fn prepare_keyword_factory_size<'py>(
    function: &str,
    value: &Bound<'py, PyAny>,
) -> PyResult<PreparedFactorySize<'py>> {
    let container_type = transpose_type_name(value)?;
    if let Ok(dimensions) = value.cast::<PyList>() {
        return prepare_factory_size_dimensions(
            function,
            dimensions.len(),
            dimensions.iter(),
            &FirstSizeErrorStyle::KeywordCollection(container_type),
        );
    }
    if let Ok(dimensions) = value.cast::<PyTuple>() {
        return prepare_factory_size_dimensions(
            function,
            dimensions.len(),
            dimensions.iter(),
            &FirstSizeErrorStyle::KeywordCollection(container_type),
        );
    }
    Err(PyTypeError::new_err(format!(
        "{function}(): argument 'size' must be tuple of ints, not {container_type}"
    )))
}

fn prepare_factory_size_dimensions<'py>(
    function: &str,
    length: usize,
    dimensions: impl Iterator<Item = Bound<'py, PyAny>>,
    first_error_style: &FirstSizeErrorStyle,
) -> PyResult<PreparedFactorySize<'py>> {
    let mut parsed = try_size_vector(length)?;
    for (index, dimension) in dimensions.enumerate() {
        let value = if index == 0 {
            let value = parse_size_dimension(&dimension, true)?;
            if let ParsedSizeDimension::Invalid(type_name) = &value {
                return Err(first_factory_size_error(
                    function,
                    type_name,
                    first_error_style,
                ));
            }
            PreparedSizeDimension::Parsed(value)
        } else {
            PreparedSizeDimension::Deferred(dimension)
        };
        try_push_size(&mut parsed, value)?;
    }
    Ok(PreparedFactorySize { dimensions: parsed })
}

fn parse_size_dimension(
    dimension: &Bound<'_, PyAny>,
    reject_bool: bool,
) -> PyResult<ParsedSizeDimension> {
    let type_name = transpose_type_name(dimension)?;
    if reject_bool && dimension.is_instance_of::<PyBool>() {
        return Ok(ParsedSizeDimension::Invalid(type_name));
    }
    match dimension.extract::<i64>() {
        Ok(value) => Ok(ParsedSizeDimension::Value(value)),
        Err(error) if error.is_instance_of::<PyOverflowError>(dimension.py()) => {
            Ok(ParsedSizeDimension::Overflow)
        }
        Err(_) => Ok(ParsedSizeDimension::Invalid(type_name)),
    }
}

fn first_factory_size_error(function: &str, type_name: &str, style: &FirstSizeErrorStyle) -> PyErr {
    let message = match style {
        FirstSizeErrorStyle::PositionalValue => format!(
            "{function}(): argument 'size' (position 1) must be tuple of ints, not {type_name}"
        ),
        FirstSizeErrorStyle::PositionalCollection => format!(
            "{function}(): argument 'size' (position 1) must be tuple of ints, but found element of type {type_name} at pos 0"
        ),
        FirstSizeErrorStyle::KeywordCollection(container_type) => {
            format!("{function}(): argument 'size' must be tuple of ints, not {container_type}")
        }
    };
    PyTypeError::new_err(message)
}

impl PreparedFactorySize<'_> {
    fn finish(self, function: &str) -> PyResult<Vec<i64>> {
        let mut size = try_size_vector(self.dimensions.len())?;
        for (index, dimension) in self.dimensions.into_iter().enumerate() {
            let parsed = match dimension {
                PreparedSizeDimension::Parsed(value) => value,
                PreparedSizeDimension::Deferred(value) => parse_size_dimension(&value, false)?,
            };
            let value = match parsed {
                ParsedSizeDimension::Value(value) => value,
                ParsedSizeDimension::Overflow => {
                    return Err(factory_size_unpack_error(
                        function,
                        index + 1,
                        "Overflow when unpacking long long",
                    ));
                }
                ParsedSizeDimension::Invalid(type_name) => {
                    return Err(factory_size_unpack_error(
                        function,
                        index + 1,
                        &format!("type must be tuple of ints,but got {type_name}"),
                    ));
                }
            };
            try_push_size(&mut size, value)?;
        }
        Ok(size)
    }
}

fn factory_size_unpack_error(function: &str, position: usize, reason: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "{function}(): argument 'size' failed to unpack the object at pos {position} with error \"{reason}\""
    ))
}

fn validate_factory_size(factory: ConstantFactory, size: Vec<i64>) -> PyResult<Vec<usize>> {
    if let Some(dimension) = size.iter().find(|dimension| **dimension < 0) {
        return match factory {
            ConstantFactory::Zeros => Err(PyRuntimeError::new_err(
                "zeros: Dimension size must be non-negative.",
            )),
            ConstantFactory::Ones => Err(PyRuntimeError::new_err(format!(
                "Trying to create tensor with negative dimension {dimension}: {size:?}"
            ))),
        };
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

fn constant_factory_shape_error(error: &TensorError, shape: &[usize]) -> PyErr {
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
    if dtype.is_none() {
        return Ok(DType::Float32);
    }
    if let Ok(dtype) = dtype.cast::<PyDType>() {
        return Ok(dtype.try_borrow()?.inner);
    }

    let type_name = dtype.get_type().name()?;
    Err(PyTypeError::new_err(format!(
        "{function}(): argument 'dtype' must be torch.dtype, not {type_name}"
    )))
}

fn parse_device(function: &str, device: Option<&Bound<'_, PyAny>>) -> PyResult<Device> {
    match device {
        None => Ok(Device::Cpu),
        Some(device) if device.is_none() => Ok(Device::Cpu),
        Some(device) => parse_device_value(function, device),
    }
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

    let argument = if function == "device" {
        "type"
    } else {
        "device"
    };
    let expected = if function == "device" {
        "torch.device or str"
    } else {
        "torch.device"
    };
    let type_name = device.get_type().name()?;
    Err(PyTypeError::new_err(format!(
        "{function}(): argument '{argument}' must be {expected}, not {type_name}"
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

fn parse_transpose_dimension(
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

    let type_name = transpose_type_name(dimension)?;
    Err(transpose_argument_type_error(
        argument, position, "int", &type_name,
    ))
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
                let detail = squeeze_sequence_type_description(sequence)?;
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

fn squeeze_sequence_type_description(sequence: &Bound<'_, PyAny>) -> PyResult<String> {
    let (kind, opening, closing, trailing) = if let Ok(sequence) = sequence.cast::<PyTuple>() {
        ("tuple", "(", ")", sequence.len() == 1)
    } else {
        sequence.cast::<PyList>()?;
        ("list", "[", "]", false)
    };
    let sequence = sequence.cast::<PySequence>()?;
    let mut names = try_size_vector(sequence.len()?)?;
    for item in sequence.try_iter()? {
        names.push(transpose_type_name(&item?)?);
    }
    let names = names.join(", ");
    let trailing = if trailing { "," } else { "" };
    Ok(format!("{kind} of {opening}{names}{trailing}{closing}"))
}

fn squeeze_call_summary(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
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
        keyword_names.sort_unstable_by(|left, right| left.0.cmp(&right.0));
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

fn squeeze_method_binding_error(
    positional: &Bound<'_, PyTuple>,
    keywords: Option<&Bound<'_, PyDict>>,
    unknown_keyword: Option<&str>,
) -> PyResult<PyErr> {
    let summary = squeeze_call_summary(positional, keywords)?;
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
    let summary = squeeze_call_summary(positional, keywords)?;
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
    let summary = squeeze_call_summary(positional, keywords)?;
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
        squeeze_sequence_type_description(&dimension_value)?
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

fn bind_transpose_arguments<'py, const N: usize>(
    positional: &Bound<'py, PyTuple>,
    keywords: Option<&Bound<'py, PyDict>>,
    names: [&str; N],
) -> PyResult<[ParsedCallArgument<'py>; N]> {
    if positional.len() > N {
        return Err(PyTypeError::new_err(format!(
            "transpose() takes {N} positional arguments but {} were given",
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
                        "transpose() got an unexpected keyword argument '{key}'"
                    ))
                });
                continue;
            };
            if arguments[index].is_some() {
                keyword_error.get_or_insert_with(|| {
                    PyTypeError::new_err(format!(
                        "transpose() got multiple values for argument '{}'",
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
            "transpose() missing {} required positional {argument}: {quoted_names}",
            missing.len()
        )));
    }

    if let Some(keyword_error) = keyword_error {
        return Err(keyword_error);
    }

    Ok(arguments
        .map(|argument| argument.expect("all required transpose arguments were checked above")))
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

fn transpose_argument_type_error(
    argument: &str,
    position: Option<usize>,
    expected: &str,
    actual: &str,
) -> PyErr {
    let position = position.map_or_else(String::new, |position| format!(" (position {position})"));
    PyTypeError::new_err(format!(
        "transpose(): argument '{argument}'{position} must be {expected}, not {actual}"
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

fn flatten_rectangular(value: &Bound<'_, PyAny>, output: &mut Vec<f32>) -> PyResult<Vec<usize>> {
    if let Ok(scalar) = value.extract::<f32>() {
        output.push(scalar);
        return Ok(Vec::new());
    }

    let sequence = value.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err("tensor data must contain real numbers in a rectangular sequence")
    })?;
    let length = sequence.len()?;
    if length == 0 {
        return Ok(vec![0]);
    }

    let first_shape = flatten_rectangular(&sequence.get_item(0)?, output)?;
    for index in 1..length {
        let shape = flatten_rectangular(&sequence.get_item(index)?, output)?;
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
        | TensorError::ElementCountOverflow => PyRuntimeError::new_err(error.to_string()),
        TensorError::InvalidScalarIndex
        | TensorError::TooManyIndices { .. }
        | TensorError::IndexOutOfBounds { .. }
        | TensorError::DimensionOutOfRange { .. } => PyIndexError::new_err(error.to_string()),
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
    module.add_class::<PyDType>()?;
    module.add_class::<PyDevice>()?;
    module.add_class::<PyMemoryFormat>()?;
    module.add_function(wrap_pyfunction!(tensor, module)?)?;
    module.add_function(wrap_pyfunction!(clone, module)?)?;
    module.add_function(wrap_pyfunction!(transpose, module)?)?;
    module.add_function(wrap_pyfunction!(squeeze, module)?)?;
    module.add_function(wrap_pyfunction!(flatten, module)?)?;
    module.add_function(wrap_pyfunction!(zeros, module)?)?;
    module.add_function(wrap_pyfunction!(ones, module)?)?;
    module.add_function(wrap_pyfunction!(eye, module)?)?;
    module.add_function(wrap_pyfunction!(full, module)?)?;
    let float32 = float32_object(py)?;
    module.add("float32", float32.clone_ref(py))?;
    module.add("float", float32.clone_ref(py))?;
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
    use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyModule};

    use super::{nested_list, torch_rs, try_size_vector};

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
}

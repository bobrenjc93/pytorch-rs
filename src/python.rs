use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{
    PyIndexError, PyMemoryError, PyOverflowError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyAny, PyBool, PyByteArray, PyBytes, PyDict, PyFloat, PyInt, PyList, PyModule, PySequence,
    PyString, PyTuple,
};

use crate::{DType, Device, Scalar, Tensor as CoreTensor, TensorData, TensorDataRef, TensorError};

static FLOAT32: PyOnceLock<Py<PyDType>> = PyOnceLock::new();
static INT64: PyOnceLock<Py<PyDType>> = PyOnceLock::new();

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
            DType::Int64 => "torch.int64",
        }
    }

    fn __str__(&self) -> &'static str {
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

#[derive(Clone, Copy)]
enum ParsedFillValue {
    Float(f64),
    NumpyFloat(f64),
    SignedInteger(i64),
    UnsignedInteger(u64),
    NumpyUint64(u64),
    WideInteger(f64),
    Boolean(bool),
    NumpyBoolean(bool),
    TensorScalar(Scalar),
}

enum ParsedArithmeticScalar {
    PythonBool(bool),
    NumpyBool(bool),
    Number(ParsedFillValue),
    WideNumpyUnsigned,
}

#[derive(Clone, Copy)]
enum BinaryOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
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
            DType::Int64 => Ok(int64_object(py)?.clone_ref(py)),
        }
    }

    #[getter]
    fn device(&self) -> PyDevice {
        PyDevice {
            inner: self.inner.device(),
        }
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
        nested_list(py, self.inner.data(), self.inner.shape())
    }

    fn item(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scalar_into_py(
            py,
            self.inner
                .item_scalar()
                .map_err(|error| tensor_error(&error))?,
        )
    }

    fn relu(&self) -> PyResult<Self> {
        self.inner
            .relu()
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

    fn __repr__(&self) -> String {
        match self.inner.data() {
            TensorDataRef::Float32(values) => {
                format!("tensor({values:?}, shape={:?})", self.inner.shape())
            }
            TensorDataRef::Int64(values) => format!(
                "tensor({values:?}, shape={:?}, dtype=torch.int64)",
                self.inner.shape()
            ),
        }
    }
}

impl PyTensor {
    fn numpy_array_copy(
        &self,
        py: Python<'_>,
        dtype: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let numpy = PyModule::import(py, "numpy")?;
        let values = match self.inner.data() {
            TensorDataRef::Float32(values) => PyList::new(py, values.iter().copied())?,
            TensorDataRef::Int64(values) => PyList::new(py, values.iter().copied())?,
        };
        let arguments = PyDict::new(py);
        if let Some(dtype) = dtype {
            arguments.set_item("dtype", dtype)?;
        } else {
            let numpy_dtype = match self.inner.dtype() {
                DType::Float32 => "float32",
                DType::Int64 => "int64",
            };
            arguments.set_item("dtype", numpy.getattr(numpy_dtype)?)?;
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
            operation.apply_scalar(
                &self.inner,
                scalar.into_scalar(self.inner.dtype(), operation)?,
                reverse,
            )
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
        scalar: Scalar,
        reverse: bool,
    ) -> Result<CoreTensor, TensorError> {
        match (self, reverse) {
            (Self::Add, _) => tensor.add_typed_scalar(scalar),
            (Self::Subtract, false) => tensor.sub_typed_scalar(scalar, false),
            (Self::Subtract, true) => tensor.sub_typed_scalar(scalar, true),
            (Self::Multiply, _) => tensor.mul_typed_scalar(scalar),
            (Self::Divide, false) => tensor.div_typed_scalar(scalar, false),
            (Self::Divide, true) => tensor.div_typed_scalar(scalar, true),
        }
    }
}

#[pyfunction(signature = (data, *, dtype=None, device=None))]
fn tensor(
    data: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyTensor> {
    let requested_dtype = parse_optional_dtype("tensor", dtype)?;
    let explicit_dtype = requested_dtype.is_some();
    let device = parse_device("tensor", device)?;
    let mut flattened = Vec::new();
    let shape = flatten_rectangular(data, &mut flattened, requested_dtype)?;
    let dtype = match requested_dtype {
        Some(dtype) => dtype,
        None => infer_tensor_dtype(&flattened)?,
    };
    let data = convert_tensor_data(flattened, dtype, explicit_dtype)?;
    CoreTensor::from_data_with_metadata(data, shape, device)
        .map(|inner| PyTensor { inner })
        .map_err(|error| tensor_error(&error))
}

#[pyfunction(signature = (size=None, *, shape=None, dtype=None, device=None))]
fn zeros(
    size: Option<&Bound<'_, PyAny>>,
    shape: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyTensor> {
    let size = parse_creation_size("zeros", size, shape)?;
    let (dtype, device) = parse_metadata("zeros", dtype, device)?;
    CoreTensor::zeros_with_metadata(size, dtype, device)
        .map(|inner| PyTensor { inner })
        .map_err(|error| tensor_error(&error))
}

#[pyfunction(signature = (size=None, *, shape=None, dtype=None, device=None))]
fn ones(
    size: Option<&Bound<'_, PyAny>>,
    shape: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyTensor> {
    let size = parse_creation_size("ones", size, shape)?;
    let (dtype, device) = parse_metadata("ones", dtype, device)?;
    CoreTensor::ones_with_metadata(size, dtype, device)
        .map(|inner| PyTensor { inner })
        .map_err(|error| tensor_error(&error))
}

#[pyfunction(signature = (size, fill_value, *, dtype=None, device=None))]
fn full(
    size: &Bound<'_, PyAny>,
    fill_value: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyTensor> {
    let dtype = parse_optional_dtype("full", dtype)?;
    let device = parse_device("full", device)?;
    let size = parse_size(size)?;
    let fill_value = parse_fill_value(fill_value)?;
    let dtype = match dtype {
        Some(dtype) => dtype,
        None => infer_full_dtype(fill_value)?,
    };
    let shape = validate_size(size)?;
    CoreTensor::validate_full_shape(&shape, dtype)
        .map_err(|error| full_shape_error(&error, &shape))?;
    let fill_value = fill_value.into_fill_scalar(dtype)?;
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

fn int64_object(py: Python<'_>) -> PyResult<&'static Py<PyDType>> {
    INT64.get_or_try_init(py, || {
        Py::new(
            py,
            PyDType {
                inner: DType::Int64,
            },
        )
    })
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
    Ok(parse_optional_dtype(function, dtype)?.unwrap_or(DType::Float32))
}

fn parse_optional_dtype(
    function: &str,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<DType>> {
    let Some(dtype) = dtype else {
        return Ok(None);
    };
    if let Ok(dtype) = dtype.cast::<PyDType>() {
        return Ok(Some(dtype.try_borrow()?.inner));
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
    let type_name = device.get_type().name()?;
    Err(PyTypeError::new_err(format!(
        "{function}(): argument '{argument}' must be torch.device or str, not {type_name}"
    )))
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
            .item_scalar()
            .map(ParsedFillValue::TensorScalar)
            .map_err(|error| tensor_error(&error));
    }

    if fill_value.is_instance_of::<PyInt>() {
        return parse_bounded_integer_value(fill_value);
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
        return parse_bounded_integer_value(value)
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
            .map(ParsedArithmeticScalar::NumpyBool)
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
        return value.is_truthy().map(ParsedFillValue::NumpyBoolean);
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
            .map(ParsedFillValue::NumpyFloat)
            .map_err(|_| invalid_value());
    }

    Err(invalid_value())
}

fn parse_integer_value(value: &Bound<'_, PyAny>) -> PyResult<ParsedFillValue> {
    match parse_bounded_integer_value(value) {
        Ok(value) => Ok(value),
        Err(_) => value
            .extract::<f64>()
            .map(ParsedFillValue::WideInteger)
            .map_err(|_| {
                PyOverflowError::new_err(
                    "Python integer is too large to convert to a supported tensor dtype",
                )
            }),
    }
}

fn parse_bounded_integer_value(value: &Bound<'_, PyAny>) -> PyResult<ParsedFillValue> {
    if value.is_exact_instance_of::<PyBool>() {
        return value.is_truthy().map(ParsedFillValue::Boolean);
    }
    if let Ok(value) = value.extract::<i64>() {
        return Ok(ParsedFillValue::SignedInteger(value));
    }

    if let Ok(value) = value.extract::<u64>() {
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

fn full_shape_error(error: &TensorError, shape: &[usize]) -> PyErr {
    if matches!(error, TensorError::ElementCountOverflow) {
        PyRuntimeError::new_err(format!(
            "Storage size calculation overflowed with size {shape:?}"
        ))
    } else {
        tensor_error(error)
    }
}

impl ParsedFillValue {
    fn into_fill_scalar(self, dtype: DType) -> PyResult<Scalar> {
        match dtype {
            DType::Float32 => self.into_fill_f32().map(Scalar::Float32),
            DType::Int64 => self.into_fill_i64().map(Scalar::Int64),
        }
    }

    fn into_fill_f32(self) -> PyResult<f32> {
        match self {
            Self::Float(value) | Self::NumpyFloat(value) | Self::WideInteger(value) => {
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
            Self::UnsignedInteger(value) | Self::NumpyUint64(value) => {
                #[allow(clippy::cast_precision_loss)]
                let converted = value as f32;
                Ok(converted)
            }
            Self::Boolean(value) | Self::NumpyBoolean(value) => Ok(f32::from(u8::from(value))),
            Self::TensorScalar(value) => Ok(value.as_f32()),
        }
    }

    fn into_fill_i64(self) -> PyResult<i64> {
        match self {
            Self::Float(value) | Self::NumpyFloat(value) => checked_float_to_i64(value),
            Self::SignedInteger(value) | Self::TensorScalar(Scalar::Int64(value)) => Ok(value),
            Self::UnsignedInteger(value) => {
                i64::try_from(value).map_err(|_| integer_conversion_overflow())
            }
            Self::NumpyUint64(_) | Self::WideInteger(_) => Err(integer_conversion_overflow()),
            Self::Boolean(value) | Self::NumpyBoolean(value) => Ok(i64::from(value)),
            Self::TensorScalar(Scalar::Float32(value)) => checked_float_to_i64(f64::from(value)),
        }
    }

    fn into_tensor_f32(self) -> f32 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let value = match self {
            Self::Float(value) | Self::NumpyFloat(value) | Self::WideInteger(value) => value as f32,
            Self::SignedInteger(value) => value as f32,
            Self::UnsignedInteger(value) | Self::NumpyUint64(value) => value as f32,
            Self::Boolean(value) | Self::NumpyBoolean(value) => f32::from(u8::from(value)),
            Self::TensorScalar(value) => value.as_f32(),
        };
        value
    }

    fn into_tensor_i64(self, explicit_dtype: bool) -> PyResult<i64> {
        match self {
            Self::Float(value) => checked_float_to_i64(value),
            Self::NumpyFloat(_) | Self::NumpyBoolean(_) => {
                Err(invalid_numpy_tensor_int64_conversion())
            }
            Self::SignedInteger(value) | Self::TensorScalar(Scalar::Int64(value)) => Ok(value),
            Self::UnsignedInteger(value) => {
                i64::try_from(value).map_err(|_| tensor_integer_conversion_overflow())
            }
            Self::NumpyUint64(value) => {
                if !explicit_dtype {
                    return Err(invalid_inferred_numpy_uint64());
                }
                i64::try_from(value).map_err(|_| tensor_integer_conversion_overflow())
            }
            Self::WideInteger(_) => Err(tensor_integer_conversion_overflow()),
            Self::Boolean(value) => Ok(i64::from(value)),
            Self::TensorScalar(Scalar::Float32(value)) => checked_float_to_i64(f64::from(value)),
        }
    }

    fn into_arithmetic_scalar(self, tensor_dtype: DType) -> PyResult<Scalar> {
        match self {
            Self::Float(value) | Self::NumpyFloat(value) => {
                #[allow(clippy::cast_possible_truncation)]
                let converted = value as f32;
                Ok(Scalar::Float32(converted))
            }
            Self::SignedInteger(value) => Ok(Scalar::Int64(value)),
            Self::UnsignedInteger(value) | Self::NumpyUint64(value) => match i64::try_from(value) {
                Ok(value) => Ok(Scalar::Int64(value)),
                Err(_) if matches!(tensor_dtype, DType::Float32) => {
                    #[allow(clippy::cast_precision_loss)]
                    let converted = value as f32;
                    Ok(Scalar::Float32(converted))
                }
                Err(_) => Err(integer_conversion_overflow()),
            },
            Self::WideInteger(_) => Err(integer_conversion_overflow()),
            Self::Boolean(value) => Ok(Scalar::Int64(i64::from(value))),
            Self::NumpyBoolean(value) => Ok(Scalar::Float32(f32::from(u8::from(value)))),
            Self::TensorScalar(value) => Ok(value),
        }
    }
}

impl ParsedArithmeticScalar {
    fn is_python_bool(&self) -> bool {
        matches!(self, Self::PythonBool(_))
    }

    fn into_scalar(self, tensor_dtype: DType, operation: BinaryOperation) -> PyResult<Scalar> {
        match self {
            Self::PythonBool(value) => Ok(Scalar::Int64(i64::from(value))),
            Self::NumpyBool(value) => Ok(Scalar::Float32(f32::from(u8::from(value)))),
            Self::Number(ParsedFillValue::UnsignedInteger(value))
                if value > i64::MAX.cast_unsigned() && matches!(tensor_dtype, DType::Int64) =>
            {
                if matches!(operation, BinaryOperation::Divide) {
                    #[allow(clippy::cast_precision_loss)]
                    let value = value as f32;
                    Ok(Scalar::Float32(value))
                } else {
                    Ok(Scalar::Int64(value.cast_signed()))
                }
            }
            Self::Number(value) => value.into_arithmetic_scalar(tensor_dtype),
            Self::WideNumpyUnsigned => {
                unreachable!("wide NumPy unsigned operands are dispatched before conversion")
            }
        }
    }
}

fn integer_conversion_overflow() -> PyErr {
    PyRuntimeError::new_err("value cannot be converted to int64 without overflow")
}

fn tensor_integer_conversion_overflow() -> PyErr {
    PyValueError::new_err("integer value is outside the int64 range")
}

fn invalid_numpy_tensor_int64_conversion() -> PyErr {
    PyTypeError::new_err("NumPy floating and boolean scalars cannot be converted to int64")
}

fn invalid_inferred_numpy_uint64() -> PyErr {
    PyTypeError::new_err("NumPy uint64 tensor values require an explicit supported dtype")
}

fn checked_float_to_i64(value: f64) -> PyResult<i64> {
    const INCLUSIVE_UPPER_BOUND: f64 = 9_223_372_036_854_775_808.0;
    if !value.is_finite() || !(-INCLUSIVE_UPPER_BOUND..=INCLUSIVE_UPPER_BOUND).contains(&value) {
        return Err(integer_conversion_overflow());
    }
    if value == INCLUSIVE_UPPER_BOUND {
        return Ok(i64::MAX);
    }
    #[allow(clippy::cast_possible_truncation)]
    Ok(value as i64)
}

fn fill_value_overflow() -> PyErr {
    PyRuntimeError::new_err("value cannot be converted to float32 without overflow")
}

fn flatten_rectangular(
    value: &Bound<'_, PyAny>,
    output: &mut Vec<ParsedFillValue>,
    requested_dtype: Option<DType>,
) -> PyResult<Vec<usize>> {
    if let Some(scalar) = parse_tensor_scalar(value, requested_dtype)? {
        try_push_size(output, scalar)?;
        return Ok(Vec::new());
    }

    if value.is_instance_of::<PyString>()
        || value.is_instance_of::<PyBytes>()
        || value.is_instance_of::<PyByteArray>()
    {
        return Err(invalid_tensor_data());
    }

    let sequence = value.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err("tensor data must contain real numbers in a rectangular sequence")
    })?;
    let length = sequence.len()?;
    if length == 0 {
        return Ok(vec![0]);
    }

    let first_shape = flatten_rectangular(&sequence.get_item(0)?, output, requested_dtype)?;
    for index in 1..length {
        let shape = flatten_rectangular(&sequence.get_item(index)?, output, requested_dtype)?;
        if shape != first_shape {
            return Err(PyValueError::new_err(
                "expected a rectangular sequence, but nested shapes differ",
            ));
        }
    }

    let shape_capacity = first_shape
        .len()
        .checked_add(1)
        .ok_or_else(|| PyMemoryError::new_err("unable to allocate tensor shape"))?;
    let mut shape = try_size_vector(shape_capacity)?;
    shape.push(length);
    shape.extend(first_shape);
    Ok(shape)
}

fn parse_tensor_scalar(
    value: &Bound<'_, PyAny>,
    requested_dtype: Option<DType>,
) -> PyResult<Option<ParsedFillValue>> {
    if value.is_instance_of::<PyBool>() {
        return value.is_truthy().map(ParsedFillValue::Boolean).map(Some);
    }
    if value.is_exact_instance_of::<PyInt>() {
        return parse_integer_value(value).map(Some);
    }
    if value.is_instance_of::<PyFloat>() {
        return value.extract::<f64>().map(ParsedFillValue::Float).map(Some);
    }

    let Ok(numpy) = PyModule::import(value.py(), "numpy") else {
        return parse_tensor_integer_subclass(value, requested_dtype);
    };
    let generic = numpy.getattr("generic")?;
    if value.is_instance(&generic)? {
        return parse_numpy_tensor_value(value, &numpy).map(Some);
    }
    parse_tensor_integer_subclass(value, requested_dtype)
}

fn parse_tensor_integer_subclass(
    value: &Bound<'_, PyAny>,
    requested_dtype: Option<DType>,
) -> PyResult<Option<ParsedFillValue>> {
    if !value.is_instance_of::<PyInt>() {
        return Ok(None);
    }
    if matches!(requested_dtype, Some(DType::Float32)) {
        return value.extract::<f64>().map(ParsedFillValue::Float).map(Some);
    }
    parse_integer_value(value).map(Some)
}

fn parse_numpy_tensor_value(
    value: &Bound<'_, PyAny>,
    numpy: &Bound<'_, PyModule>,
) -> PyResult<ParsedFillValue> {
    if value.is_instance(&numpy.getattr("bool_")?)? {
        return value.is_truthy().map(ParsedFillValue::NumpyBoolean);
    }
    if value.is_instance(&numpy.getattr("uint64")?)? {
        return value.extract::<u64>().map(ParsedFillValue::NumpyUint64);
    }
    if value.is_instance(&numpy.getattr("integer")?)? {
        return value
            .extract::<i64>()
            .map(ParsedFillValue::SignedInteger)
            .map_err(|_| {
                PyTypeError::new_err(
                    "NumPy integer tensor value is outside the supported 64-bit range",
                )
            });
    }
    if value.is_instance(&numpy.getattr("floating")?)? {
        return value.extract::<f64>().map(ParsedFillValue::NumpyFloat);
    }
    Err(invalid_tensor_data())
}

fn invalid_tensor_data() -> PyErr {
    PyTypeError::new_err("tensor data must contain real numbers in a rectangular sequence")
}

fn infer_tensor_dtype(values: &[ParsedFillValue]) -> PyResult<DType> {
    if values.is_empty()
        || values.iter().any(|value| {
            matches!(
                value,
                ParsedFillValue::Float(_) | ParsedFillValue::NumpyFloat(_)
            )
        })
    {
        return Ok(DType::Float32);
    }
    if values.iter().any(|value| {
        !matches!(
            value,
            ParsedFillValue::Boolean(_) | ParsedFillValue::NumpyBoolean(_)
        )
    }) {
        return Ok(DType::Int64);
    }
    Err(unsupported_bool_storage())
}

fn infer_full_dtype(value: ParsedFillValue) -> PyResult<DType> {
    match value {
        ParsedFillValue::Float(_)
        | ParsedFillValue::NumpyFloat(_)
        | ParsedFillValue::NumpyBoolean(_)
        | ParsedFillValue::TensorScalar(Scalar::Float32(_)) => Ok(DType::Float32),
        ParsedFillValue::SignedInteger(_)
        | ParsedFillValue::UnsignedInteger(_)
        | ParsedFillValue::NumpyUint64(_)
        | ParsedFillValue::WideInteger(_)
        | ParsedFillValue::TensorScalar(Scalar::Int64(_)) => Ok(DType::Int64),
        ParsedFillValue::Boolean(_) => Err(unsupported_bool_storage()),
    }
}

fn unsupported_bool_storage() -> PyErr {
    PyRuntimeError::new_err(
        "bool tensor storage is not supported; specify dtype=torch.float32 or torch.int64",
    )
}

fn convert_tensor_data(
    values: Vec<ParsedFillValue>,
    dtype: DType,
    explicit_dtype: bool,
) -> PyResult<TensorData> {
    match dtype {
        DType::Float32 => {
            let mut converted = try_size_vector(values.len())?;
            for value in values {
                converted.push(value.into_tensor_f32());
            }
            Ok(TensorData::Float32(converted))
        }
        DType::Int64 => {
            let mut converted = try_size_vector(values.len())?;
            for value in values {
                converted.push(value.into_tensor_i64(explicit_dtype)?);
            }
            Ok(TensorData::Int64(converted))
        }
    }
}

fn scalar_into_py(py: Python<'_>, scalar: Scalar) -> PyResult<Py<PyAny>> {
    match scalar {
        Scalar::Float32(value) => value.into_py_any(py),
        Scalar::Int64(value) => value.into_py_any(py),
    }
}

fn nested_list(py: Python<'_>, data: TensorDataRef<'_>, shape: &[usize]) -> PyResult<Py<PyAny>> {
    if shape.is_empty() {
        return match data {
            TensorDataRef::Float32(values) => values[0].into_py_any(py),
            TensorDataRef::Int64(values) => values[0].into_py_any(py),
        };
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
            match data {
                TensorDataRef::Float32(values) => {
                    TensorDataRef::Float32(&values[start..start + chunk_size])
                }
                TensorDataRef::Int64(values) => {
                    TensorDataRef::Int64(&values[start..start + chunk_size])
                }
            },
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
        | TensorError::MatmulDTypeMismatch { .. }
        | TensorError::ItemRequiresOneElement { .. }
        | TensorError::ReshapeMultipleInferredDimensions
        | TensorError::ReshapeInvalidDimension { .. }
        | TensorError::ReshapeAmbiguousZeroElements { .. }
        | TensorError::ReshapeElementCountMismatch { .. }
        | TensorError::StrideCalculationOverflow
        | TensorError::StorageCapacityOverflow { .. }
        | TensorError::AllocationFailed { .. }
        | TensorError::ElementCountOverflow => PyRuntimeError::new_err(error.to_string()),
    }
}

#[pymodule]
fn torch_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<PyTensor>()?;
    module.add_class::<PyDType>()?;
    module.add_class::<PyDevice>()?;
    module.add_function(wrap_pyfunction!(tensor, module)?)?;
    module.add_function(wrap_pyfunction!(zeros, module)?)?;
    module.add_function(wrap_pyfunction!(ones, module)?)?;
    module.add_function(wrap_pyfunction!(full, module)?)?;
    let float32 = float32_object(py)?;
    module.add("float32", float32.clone_ref(py))?;
    module.add("float", float32.clone_ref(py))?;
    let int64 = int64_object(py)?;
    module.add("int64", int64.clone_ref(py))?;
    module.add("long", int64.clone_ref(py))?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use pyo3::exceptions::PyTypeError;
    use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyModule};

    use crate::TensorDataRef;

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
            let list =
                nested_list(py, TensorDataRef::Float32(&[]), &[0, maximum, maximum]).unwrap();
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

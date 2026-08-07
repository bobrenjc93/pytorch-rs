use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyAny, PyBool, PyFloat, PyInt, PyList, PyMemoryView, PyModule, PySequence, PyTuple,
};

use crate::{Tensor as CoreTensor, TensorError};

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

#[pymethods]
impl PyTensor {
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.shape().iter().copied())
    }

    fn numel(&self) -> usize {
        self.inner.numel()
    }

    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        nested_list(py, self.inner.as_slice(), self.inner.shape())
    }

    fn item(&self) -> PyResult<f32> {
        self.inner.item().map_err(|error| tensor_error(&error))
    }

    fn relu(&self) -> Self {
        Self {
            inner: self.inner.relu(),
        }
    }

    fn sum(&self) -> Self {
        Self {
            inner: self.inner.sum(),
        }
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        self.inner
            .add(&other.inner)
            .map(|inner| Self { inner })
            .map_err(|error| tensor_error(&error))
    }

    fn __mul__(&self, other: &Self) -> PyResult<Self> {
        self.inner
            .mul(&other.inner)
            .map(|inner| Self { inner })
            .map_err(|error| tensor_error(&error))
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
        format!(
            "tensor({:?}, shape={:?})",
            self.inner.as_slice(),
            self.inner.shape()
        )
    }
}

#[pyfunction]
fn tensor(data: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    let mut flattened = Vec::new();
    let shape = flatten_rectangular(data, &mut flattened)?;
    CoreTensor::from_vec(flattened, shape)
        .map(|inner| PyTensor { inner })
        .map_err(|error| tensor_error(&error))
}

#[pyfunction]
fn zeros(shape: Vec<usize>) -> PyResult<PyTensor> {
    CoreTensor::zeros(shape)
        .map(|inner| PyTensor { inner })
        .map_err(|error| tensor_error(&error))
}

#[pyfunction]
fn ones(shape: Vec<usize>) -> PyResult<PyTensor> {
    CoreTensor::ones(shape)
        .map(|inner| PyTensor { inner })
        .map_err(|error| tensor_error(&error))
}

#[pyfunction(signature = (size, fill_value))]
fn full(size: &Bound<'_, PyAny>, fill_value: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    let size = parse_size(size)?;
    let fill_value = parse_fill_value(fill_value)?;
    let shape = validate_size(size)?;
    CoreTensor::validate_full_shape(&shape).map_err(|error| full_shape_error(&error, &shape))?;
    let fill_value = fill_value.into_f32()?;
    CoreTensor::full(shape, fill_value)
        .map(|inner| PyTensor { inner })
        .map_err(|error| tensor_error(&error))
}

fn parse_size(size: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let dimensions = if let Ok(size) = size.cast::<PyList>() {
        size.iter().collect::<Vec<_>>()
    } else if let Ok(size) = size.cast::<PyTuple>() {
        size.iter().collect::<Vec<_>>()
    } else {
        return Err(PyTypeError::new_err(
            "full(): argument 'size' must be a tuple or list of integers",
        ));
    };

    dimensions
        .into_iter()
        .enumerate()
        .map(|(index, dimension)| {
            if dimension.is_instance_of::<PyBool>() {
                return Err(invalid_size_dimension(
                    index,
                    "bool is not a valid size dimension",
                ));
            }
            dimension
                .extract::<i64>()
                .map_err(|error| invalid_size_dimension(index, &error.to_string()))
        })
        .collect()
}

fn validate_size(size: Vec<i64>) -> PyResult<Vec<usize>> {
    if let Some(dimension) = size.iter().find(|dimension| **dimension < 0) {
        return Err(PyRuntimeError::new_err(format!(
            "Trying to create tensor with negative dimension {dimension}: {size:?}"
        )));
    }

    size.into_iter()
        .map(|dimension| {
            usize::try_from(dimension).map_err(|_| {
                PyRuntimeError::new_err(format!(
                    "tensor dimension {dimension} exceeds the platform size limit"
                ))
            })
        })
        .collect()
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

    parse_buffer_fill_value(fill_value)
}

fn parse_buffer_fill_value(fill_value: &Bound<'_, PyAny>) -> PyResult<ParsedFillValue> {
    let view = PyMemoryView::from(fill_value).map_err(|_| invalid_fill_value())?;
    let dimensions = view
        .getattr("ndim")
        .and_then(|value| value.extract::<usize>())
        .map_err(|_| invalid_fill_value())?;
    if dimensions != 0 {
        return Err(invalid_fill_value());
    }

    let format = view
        .getattr("format")
        .and_then(|value| value.extract::<String>())
        .map_err(|_| invalid_fill_value())?;
    if !is_real_numeric_buffer_format(&format) {
        return Err(invalid_fill_value());
    }

    if let Ok(scalar) = view.call_method0("tolist") {
        if scalar.is_instance_of::<PyInt>() {
            return parse_integer_fill_value(&scalar);
        }
        if scalar.is_instance_of::<PyFloat>() {
            return scalar.extract::<f64>().map(ParsedFillValue::Float);
        }
    }

    fill_value
        .extract::<f64>()
        .map(ParsedFillValue::Float)
        .map_err(|_| invalid_fill_value())
}

fn is_real_numeric_buffer_format(format: &str) -> bool {
    let type_code = match format.as_bytes() {
        [type_code] | [b'@' | b'=' | b'<' | b'>' | b'!', type_code] => *type_code,
        _ => return false,
    };
    matches!(
        type_code,
        b'?' | b'b'
            | b'B'
            | b'h'
            | b'H'
            | b'i'
            | b'I'
            | b'l'
            | b'L'
            | b'q'
            | b'Q'
            | b'n'
            | b'N'
            | b'e'
            | b'f'
            | b'd'
            | b'g'
    )
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

    let chunk_size = shape[1..].iter().product::<usize>();
    let mut items = Vec::with_capacity(shape[0]);
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
        | TensorError::StrideCalculationOverflow
        | TensorError::StorageCapacityOverflow { .. }
        | TensorError::AllocationFailed { .. }
        | TensorError::ElementCountOverflow => PyRuntimeError::new_err(error.to_string()),
    }
}

#[pymodule]
fn torch_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTensor>()?;
    module.add_function(wrap_pyfunction!(tensor, module)?)?;
    module.add_function(wrap_pyfunction!(zeros, module)?)?;
    module.add_function(wrap_pyfunction!(ones, module)?)?;
    module.add_function(wrap_pyfunction!(full, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

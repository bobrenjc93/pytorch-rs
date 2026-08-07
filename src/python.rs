use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyMemoryError, PyOverflowError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyFloat, PyInt, PyList, PyModule, PySequence, PyTuple};

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

fn parse_numpy_fill_value(fill_value: &Bound<'_, PyAny>) -> PyResult<ParsedFillValue> {
    let numpy = PyModule::import(fill_value.py(), "numpy").map_err(|_| invalid_fill_value())?;
    let generic = numpy.getattr("generic").map_err(|_| invalid_fill_value())?;
    if !fill_value.is_instance(&generic)? {
        return Err(invalid_fill_value());
    }

    let numpy_bool = numpy.getattr("bool_").map_err(|_| invalid_fill_value())?;
    if fill_value.is_instance(&numpy_bool)? {
        return fill_value
            .is_truthy()
            .map(|value| ParsedFillValue::SignedInteger(i64::from(value)));
    }

    let numpy_integer = numpy.getattr("integer").map_err(|_| invalid_fill_value())?;
    if fill_value.is_instance(&numpy_integer)? {
        return fill_value
            .extract::<i64>()
            .map(ParsedFillValue::SignedInteger)
            .map_err(|_| {
                PyTypeError::new_err("NumPy integer fill_value is outside the signed 64-bit range")
            });
    }

    let numpy_floating = numpy
        .getattr("floating")
        .map_err(|_| invalid_fill_value())?;
    if fill_value.is_instance(&numpy_floating)? {
        return fill_value
            .extract::<f64>()
            .map(ParsedFillValue::Float)
            .map_err(|_| invalid_fill_value());
    }

    Err(invalid_fill_value())
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
    let mut items = Vec::new();
    items.try_reserve_exact(shape[0]).map_err(|_| {
        PyMemoryError::new_err("unable to allocate Python list for tensor conversion")
    })?;
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

#[cfg(test)]
mod tests {
    use super::try_size_vector;

    #[test]
    fn size_vector_capacity_overflow_returns_python_error() {
        pyo3::Python::initialize();
        let error = try_size_vector::<i64>(usize::MAX)
            .expect_err("an impossible vector capacity must return an error");
        assert_eq!(error.to_string(), "RuntimeError: std::bad_alloc");
    }
}

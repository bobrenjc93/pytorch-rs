use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PyModule, PySequence, PyTuple};

use crate::{Tensor as CoreTensor, TensorError};

/// Python-facing tensor backed by the native Rust tensor core.
#[pyclass(name = "Tensor", module = "torch_rs", skip_from_py_object)]
#[derive(Clone)]
struct PyTensor {
    inner: CoreTensor,
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
        | TensorError::ItemRequiresOneElement { .. } => PyRuntimeError::new_err(error.to_string()),
        TensorError::ElementCountOverflow => PyValueError::new_err(error.to_string()),
    }
}

#[pymodule]
fn torch_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTensor>()?;
    module.add_function(wrap_pyfunction!(tensor, module)?)?;
    module.add_function(wrap_pyfunction!(zeros, module)?)?;
    module.add_function(wrap_pyfunction!(ones, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

//! Reusable Python argument-schema parsing for native operator bridges.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyFloat, PyInt, PyModule};

use crate::{
    python::{PyTensor, python_type_name},
    python_tensor_errors::item_error,
};

#[derive(Clone, Copy)]
pub(crate) struct ArgumentSchema {
    operation: &'static str,
    name: &'static str,
    position: Option<usize>,
    expected: &'static str,
}

impl ArgumentSchema {
    pub(crate) const fn new(
        operation: &'static str,
        name: &'static str,
        position: usize,
        expected: &'static str,
    ) -> Self {
        Self {
            operation,
            name,
            position: Some(position),
            expected,
        }
    }

    pub(crate) const fn keyword(
        operation: &'static str,
        name: &'static str,
        expected: &'static str,
    ) -> Self {
        Self {
            operation,
            name,
            position: None,
            expected,
        }
    }

    pub(crate) fn type_error(self, value: &Bound<'_, PyAny>) -> PyResult<PyErr> {
        let actual = python_type_name(value)?;
        let position = self
            .position
            .map_or_else(String::new, |position| format!(" (position {position})"));
        Ok(PyTypeError::new_err(format!(
            "{}(): argument '{}'{} must be {}, not {actual}",
            self.operation, self.name, position, self.expected
        )))
    }

    pub(crate) fn parse_exact_bool(self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        if !value.is_exact_instance_of::<PyBool>() {
            return Err(self.type_error(value)?);
        }
        value.is_truthy()
    }
}

pub(crate) fn parse_float_like_argument(
    schema: ArgumentSchema,
    value: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    if let Ok(tensor) = value.cast::<PyTensor>() {
        let tensor = tensor.try_borrow()?;
        if tensor.inner().shape().is_empty() && !tensor.inner().requires_grad() {
            return tensor
                .inner()
                .item()
                .map(f64::from)
                .map_err(|error| item_error(&error));
        }
        return Err(schema.type_error(value)?);
    }

    if value.is_instance_of::<PyInt>() || value.is_instance_of::<PyFloat>() {
        return value.extract::<f64>();
    }

    let Ok(numpy) = PyModule::import(value.py(), "numpy") else {
        return Err(schema.type_error(value)?);
    };
    let generic = numpy.getattr("generic")?;
    if !value.is_instance(&generic)? {
        return Err(schema.type_error(value)?);
    }

    if value.is_instance(&numpy.getattr("bool_")?)? {
        return value.is_truthy().map(|value| if value { 1.0 } else { 0.0 });
    }
    for scalar_type in ["integer", "floating", "complexfloating"] {
        if value.is_instance(&numpy.getattr(scalar_type)?)? {
            return value.extract::<f64>();
        }
    }

    Err(schema.type_error(value)?)
}

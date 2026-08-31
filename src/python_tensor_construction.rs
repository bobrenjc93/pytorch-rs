//! Python object flattening for native tensor construction.

use std::mem::size_of;
use std::os::raw::c_long;

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{
    PyMemoryError, PyNotImplementedError, PyOverflowError, PyRuntimeError, PyTypeError,
    PyValueError,
};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{
    PyAny, PyBool, PyBytes, PyFloat, PyInt, PyList, PyMapping, PyMemoryView, PySequence, PyTuple,
};

use crate::python::python_type_name;

const AS_TENSOR_MAX_SEQUENCE_DIMENSIONS: usize = 128;

pub(crate) struct FlattenedTensorData {
    values: Vec<f32>,
    shape: Vec<usize>,
}

impl FlattenedTensorData {
    fn new(values: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { values, shape }
    }

    pub(crate) fn into_parts(self) -> (Vec<f32>, Vec<usize>) {
        (self.values, self.shape)
    }
}

pub(crate) fn flatten_as_tensor_data(value: &Bound<'_, PyAny>) -> PyResult<FlattenedTensorData> {
    if let Some(scalar) = extract_as_tensor_python_real_scalar(value)? {
        return Ok(FlattenedTensorData::new(vec![scalar], Vec::new()));
    }
    if !is_as_tensor_list_or_tuple(value) {
        return Err(as_tensor_unsupported_conversion_error());
    }

    let mut output = Vec::new();
    let mut active_sequences = Vec::new();
    let shape = flatten_as_tensor_first_shape(value, 0, &mut output, &mut active_sequences)?;
    Ok(FlattenedTensorData::new(output, shape))
}

pub(crate) fn flatten_tensor_data(
    value: &Bound<'_, PyAny>,
    dtype_was_explicit: bool,
) -> PyResult<FlattenedTensorData> {
    if let Ok(scalar) = value.extract::<f32>() {
        Ok(FlattenedTensorData::new(vec![scalar], Vec::new()))
    } else if value.cast::<PyBytes>().is_ok() {
        Err(PyTypeError::new_err("new(): invalid data type 'bytes'"))
    } else if value.cast::<PyMemoryView>().is_ok() {
        if let Some(buffer) = flatten_buffer(value, dtype_was_explicit)? {
            Ok(buffer)
        } else {
            let mut flattened = Vec::new();
            let shape = flatten_rectangular(value, &mut flattened)?;
            Ok(FlattenedTensorData::new(flattened, shape))
        }
    } else if is_sequence_input(value)? {
        let mut flattened = Vec::new();
        let shape = flatten_rectangular(value, &mut flattened)?;
        Ok(FlattenedTensorData::new(flattened, shape))
    } else {
        Err(unsupported_tensor_data_error(value, dtype_was_explicit)?)
    }
}

fn flatten_as_tensor_first_shape(
    value: &Bound<'_, PyAny>,
    dim: usize,
    output: &mut Vec<f32>,
    active_sequences: &mut Vec<*mut ffi::PyObject>,
) -> PyResult<Vec<usize>> {
    if let Some(scalar) = extract_as_tensor_python_real_scalar(value)? {
        output
            .try_reserve(1)
            .map_err(|_| python_allocation_error())?;
        output.push(scalar);
        return Ok(Vec::new());
    }
    if !is_as_tensor_list_or_tuple(value) {
        return Err(as_tensor_nested_infer_dtype_error(value)?);
    }

    enter_as_tensor_sequence(value, dim, active_sequences)?;
    let result = (|| {
        let sequence = value.cast::<PySequence>()?;
        let length = sequence.len()?;
        if length == 0 {
            return Ok(vec![0]);
        }

        let first_shape = flatten_as_tensor_first_shape(
            &sequence.get_item(0)?,
            dim + 1,
            output,
            active_sequences,
        )?;
        if first_shape.contains(&0) {
            for index in 1..length {
                validate_as_tensor_empty_branch(
                    &sequence.get_item(index)?,
                    dim + 1,
                    active_sequences,
                )?;
            }
        } else {
            for index in 1..length {
                flatten_as_tensor_with_shape(
                    &sequence.get_item(index)?,
                    &first_shape,
                    dim + 1,
                    output,
                    active_sequences,
                )?;
            }
        }

        let mut shape = Vec::with_capacity(first_shape.len() + 1);
        shape.push(length);
        shape.extend(first_shape);
        Ok(shape)
    })();
    active_sequences.pop();
    result
}

fn flatten_as_tensor_with_shape(
    value: &Bound<'_, PyAny>,
    shape: &[usize],
    dim: usize,
    output: &mut Vec<f32>,
    active_sequences: &mut Vec<*mut ffi::PyObject>,
) -> PyResult<()> {
    if shape.is_empty() {
        let Some(scalar) = extract_as_tensor_python_real_scalar(value)? else {
            return Err(as_tensor_nested_scalar_error(value)?);
        };
        output
            .try_reserve(1)
            .map_err(|_| python_allocation_error())?;
        output.push(scalar);
        return Ok(());
    }
    if shape.contains(&0) {
        return validate_as_tensor_empty_branch(value, dim, active_sequences);
    }
    if !is_as_tensor_list_or_tuple(value) {
        if extract_as_tensor_python_real_scalar(value)?.is_some() {
            return Err(PyTypeError::new_err("not a sequence"));
        }
        return Err(as_tensor_nested_infer_dtype_error(value)?);
    }

    enter_as_tensor_sequence(value, dim, active_sequences)?;
    let result = (|| {
        let sequence = value.cast::<PySequence>()?;
        let length = sequence.len()?;
        if length != shape[0] {
            return Err(PyValueError::new_err(format!(
                "expected sequence of length {} at dim {dim} (got {length})",
                shape[0]
            )));
        }
        for index in 0..length {
            flatten_as_tensor_with_shape(
                &sequence.get_item(index)?,
                &shape[1..],
                dim + 1,
                output,
                active_sequences,
            )?;
        }
        Ok(())
    })();
    active_sequences.pop();
    result
}

fn validate_as_tensor_empty_branch(
    value: &Bound<'_, PyAny>,
    dim: usize,
    active_sequences: &mut Vec<*mut ffi::PyObject>,
) -> PyResult<()> {
    if extract_as_tensor_python_real_scalar(value)?.is_some() {
        return Ok(());
    }
    if !is_as_tensor_list_or_tuple(value) {
        return Err(as_tensor_nested_infer_dtype_error(value)?);
    }

    enter_as_tensor_sequence(value, dim, active_sequences)?;
    let result = (|| {
        let sequence = value.cast::<PySequence>()?;
        for index in 0..sequence.len()? {
            validate_as_tensor_empty_branch(&sequence.get_item(index)?, dim + 1, active_sequences)?;
        }
        Ok(())
    })();
    active_sequences.pop();
    result
}

fn enter_as_tensor_sequence(
    value: &Bound<'_, PyAny>,
    dim: usize,
    active_sequences: &mut Vec<*mut ffi::PyObject>,
) -> PyResult<()> {
    if dim >= AS_TENSOR_MAX_SEQUENCE_DIMENSIONS || active_sequences.contains(&value.as_ptr()) {
        return Err(as_tensor_too_many_dimensions_error(value));
    }
    active_sequences
        .try_reserve(1)
        .map_err(|_| python_allocation_error())?;
    active_sequences.push(value.as_ptr());
    Ok(())
}

fn is_as_tensor_list_or_tuple(value: &Bound<'_, PyAny>) -> bool {
    value.is_exact_instance_of::<PyList>() || value.is_exact_instance_of::<PyTuple>()
}

#[allow(clippy::cast_possible_truncation)]
fn extract_as_tensor_python_real_scalar(value: &Bound<'_, PyAny>) -> PyResult<Option<f32>> {
    if value.is_exact_instance_of::<PyBool>() {
        return Ok(None);
    }
    if value.is_exact_instance_of::<PyInt>() || value.is_exact_instance_of::<PyFloat>() {
        let converted = value.extract::<f64>()? as f32;
        return Ok(Some(converted));
    }
    Ok(None)
}

fn as_tensor_unsupported_conversion_error() -> PyErr {
    PyNotImplementedError::new_err(
        "as_tensor(): only exact native CPU float32 Tensor inputs, Python real scalars, and rectangular list/tuple inputs are supported; NumPy arrays, buffers, dtype conversions, CUDA/meta/indexed CPU devices, copy, pinned memory, tensor subclasses, and __torch_function__ argument dispatch are not implemented",
    )
}

fn as_tensor_nested_scalar_error(value: &Bound<'_, PyAny>) -> PyResult<PyErr> {
    let actual = python_type_name(value)?;
    if is_as_tensor_list_or_tuple(value) {
        Ok(PyTypeError::new_err(format!(
            "must be real number, not {actual}"
        )))
    } else {
        Ok(PyRuntimeError::new_err(format!(
            "Could not infer dtype of {actual}"
        )))
    }
}

fn as_tensor_nested_infer_dtype_error(value: &Bound<'_, PyAny>) -> PyResult<PyErr> {
    let actual = python_type_name(value)?;
    Ok(PyRuntimeError::new_err(format!(
        "Could not infer dtype of {actual}"
    )))
}

fn as_tensor_too_many_dimensions_error(value: &Bound<'_, PyAny>) -> PyErr {
    let sequence_type = if value.is_exact_instance_of::<PyTuple>() {
        "tuple"
    } else {
        "list"
    };
    PyValueError::new_err(format!("too many dimensions '{sequence_type}'"))
}

fn flatten_buffer(
    value: &Bound<'_, PyAny>,
    dtype_was_explicit: bool,
) -> PyResult<Option<FlattenedTensorData>> {
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
        return Ok(Some(FlattenedTensorData::new(Vec::new(), vec![0])));
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
        return Ok(Some(FlattenedTensorData::new(output, vec![elements])));
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
    Ok(Some(FlattenedTensorData::new(output, vec![elements])))
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
        (b'?', 1) => f32::from(u8::from(bytes[0] != 0)),
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

pub(crate) fn nested_list(py: Python<'_>, data: &[f32], shape: &[usize]) -> PyResult<Py<PyAny>> {
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

fn python_allocation_error() -> PyErr {
    PyRuntimeError::new_err("std::bad_alloc")
}

#[cfg(test)]
mod tests {
    use pyo3::types::{PyAnyMethods, PyMemoryView, PyModule, PySlice};

    use super::{flatten_buffer, half_to_f32, nested_list};

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

            let data = flatten_buffer(&reversed, true).unwrap().unwrap();
            let (values, shape) = data.into_parts();
            assert_eq!(shape, [4]);
            assert_eq!(values, [4.0, 3.0, 2.0, 1.0]);

            array.set_item(3, 99).unwrap();
            assert_eq!(values, [4.0, 3.0, 2.0, 1.0]);
        });
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
}

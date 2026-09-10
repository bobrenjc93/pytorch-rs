//! One bridge for composed CUDA captures; eager Tensor methods stay unchanged.
use super::{CoreTensor, DType, PyTensor, compile_trace_mul_scalar_value, tensor_error};
use crate::tensor::cuda_graph::{Layout, Operation};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

// Shape/stride declarations have already passed the frontend's whole-graph
// checks. Recompute them natively too, before any storage allocation or launch.
type Node<'py> = (
    String,
    Vec<usize>,
    Bound<'py, PyAny>,
    Vec<usize>,
    Vec<usize>,
);

#[pyfunction(name = "_compile_trace_cuda_graph", signature = (inputs, nodes, /))]
pub(super) fn execute(
    inputs: &Bound<'_, PyTuple>,
    nodes: Vec<Node<'_>>,
) -> PyResult<Vec<PyTensor>> {
    let mut owners = Vec::with_capacity(inputs.len());
    for input in inputs.iter() {
        if !input.is_exact_instance_of::<PyTensor>() {
            return Err(PyTypeError::new_err(
                "CUDA graph requires exact native Tensor inputs",
            ));
        }
        owners.push(input.cast_into::<PyTensor>()?);
    }
    let guards = owners
        .iter()
        .map(|input| input.try_borrow().map_err(PyErr::from))
        .collect::<PyResult<Vec<_>>>()?;
    let tensors: Vec<&CoreTensor> = guards.iter().map(|input| &input.inner).collect();
    let Some(first) = tensors.first() else {
        return Err(PyValueError::new_err("CUDA graph requires inputs"));
    };
    for input in &tensors {
        if !input.is_cuda()
            || input.device() != first.device()
            || input.dtype() != DType::Float32
            || input.requires_grad()
            || !input.is_contiguous()
        {
            return Err(PyNotImplementedError::new_err(
                "CUDA graph requires contiguous same-device CUDA float32 inputs without gradients",
            ));
        }
    }
    let mut layouts: Vec<_> = tensors
        .iter()
        .map(|tensor| Layout::from_tensor(tensor))
        .collect();
    let mut operations = Vec::with_capacity(nodes.len());
    for (target, indices, payload, shape, strides) in nodes {
        let operation = match (target.as_str(), indices.as_slice()) {
            ("mul_scalar", &[input]) => {
                Operation::MulScalar(input, compile_trace_mul_scalar_value(&payload)?)
            }
            ("sum", &[input]) => {
                let options = payload.cast::<PyTuple>()?;
                if options.len() != 2 {
                    return Err(PyNotImplementedError::new_err("invalid reduction options"));
                }
                Operation::SumRows(
                    input,
                    super::compile_trace_sum_options(&options.get_item(0)?, &options.get_item(1)?)?,
                )
            }
            ("neg", &[input]) if payload.is_none() => Operation::Neg(input),
            ("add", &[left, right]) if payload.is_none() => Operation::Add(left, right),
            ("matmul", &[left, right]) if payload.is_none() => Operation::Matmul(left, right),
            _ => {
                return Err(PyNotImplementedError::new_err(
                    "unsupported CUDA graph node",
                ));
            }
        };
        let layout = operation
            .layout(&layouts)
            .map_err(|error| tensor_error(&error))?;
        if layout.shape != shape || layout.strides != strides {
            return Err(PyValueError::new_err("CUDA graph output metadata mismatch"));
        }
        layouts.push(layout);
        operations.push(operation);
    }
    // Retain all input borrows and every intermediate, including unused nodes.
    // Existing core operations synchronize even on launch errors and restore
    // the current device. No tensor clone/copy, Python call, or PyO3 crossing is
    // performed between nodes. Every call still allocates fresh result storage.
    let mut outputs = Vec::with_capacity(operations.len());
    for (index, operation) in operations.into_iter().enumerate() {
        let output = operation
            .execute(&tensors, &outputs)
            .map_err(|error| tensor_error(&error))?;
        if !layouts[tensors.len() + index].matches(&output)
            || output.device() != first.device()
            || output.dtype() != DType::Float32
            || output.requires_grad()
            || output.storage_offset() != 0
        {
            return Err(PyRuntimeError::new_err(
                "CUDA graph native result metadata mismatch",
            ));
        }
        outputs.push(output);
    }
    Ok(outputs.into_iter().map(PyTensor::new).collect())
}

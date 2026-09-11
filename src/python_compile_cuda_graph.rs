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
) -> PyResult<Vec<Py<PyTensor>>> {
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
        {
            return Err(PyNotImplementedError::new_err(
                "CUDA graph requires same-device CUDA float32 inputs without gradients",
            ));
        }
    }
    let mut layouts: Vec<_> = tensors
        .iter()
        .map(|tensor| Layout::from_tensor(tensor))
        .collect();
    let mut operations = Vec::with_capacity(nodes.len());
    let mut aliases = Vec::with_capacity(nodes.len());
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
            ("contiguous", &[input]) if payload.is_none() => Operation::Contiguous(input),
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
        aliases.push(match operation {
            Operation::Contiguous(input)
                if layouts[input]
                    .is_contiguous()
                    .map_err(|error| tensor_error(&error))? =>
            {
                Some(input)
            }
            _ => None,
        });
        layouts.push(layout);
        operations.push(operation);
    }
    // Retain all input borrows and every intermediate, including unused nodes.
    // Existing core operations synchronize even on launch errors and restore
    // the current device. No Python call or PyO3 crossing occurs between nodes.
    // Packing stays native; contiguous aliases preserve their original owner.
    let mut outputs = Vec::with_capacity(operations.len());
    for (index, operation) in operations.into_iter().enumerate() {
        let output = operation
            .execute(&tensors, &outputs)
            .map_err(|error| tensor_error(&error))?;
        if !layouts[tensors.len() + index].matches(&output)
            || output.device() != first.device()
            || output.dtype() != DType::Float32
            || output.requires_grad()
        {
            return Err(PyRuntimeError::new_err(
                "CUDA graph native result metadata mismatch",
            ));
        }
        outputs.push(output);
    }
    drop(tensors);
    drop(guards);
    wrap_outputs(inputs.py(), owners, outputs, aliases)
}

fn wrap_outputs<'py>(
    py: Python<'py>,
    owners: Vec<Bound<'py, PyTensor>>,
    outputs: Vec<CoreTensor>,
    aliases: Vec<Option<usize>>,
) -> PyResult<Vec<Py<PyTensor>>> {
    let input_count = owners.len();
    let mut objects: Vec<Py<PyTensor>> = owners.into_iter().map(Bound::unbind).collect();
    for (output, alias) in outputs.into_iter().zip(aliases) {
        let object = match alias {
            Some(input) => objects[input].clone_ref(py),
            None => Py::new(py, PyTensor::new(output))?,
        };
        objects.push(object);
    }
    Ok(objects.drain(input_count..).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, cuda, tensor::cuda_graph::EXECUTIONS};

    #[test]
    fn invalid_late_nodes_execute_zero_native_operations() {
        if cuda::device_count() == 0 {
            eprintln!("skipping native CUDA graph launch accounting: no CUDA device");
            return;
        }
        Python::initialize();
        Python::attach(|py| {
            let base = CoreTensor::from_vec(vec![1.; 21], [3, 7])
                .unwrap()
                .try_copy_cpu_to_cuda(Device::Cuda(0))
                .unwrap();
            let input = Py::new(py, PyTensor::new(base.transpose(0, 1).unwrap())).unwrap();
            let inputs = PyTuple::new(py, [input]).unwrap();
            let node =
                |target: &str, indices: Vec<usize>, shape: Vec<usize>, strides: Vec<usize>| {
                    (
                        target.to_owned(),
                        indices,
                        py.None().into_bound(py),
                        shape,
                        strides,
                    )
                };
            let bad = [
                node("neg", vec![0], vec![7, 3], vec![3, 1]),
                node("mul_scalar", vec![0], vec![7, 3], vec![3, 1]),
                node("sum", vec![0], vec![7], vec![1]),
                node("add", vec![1, 0], vec![7, 3], vec![3, 1]),
                node("matmul", vec![1, 0], vec![7, 3], vec![3, 1]),
                node("contiguous", vec![99], vec![7, 3], vec![3, 1]),
                node("contiguous", vec![1], vec![7, 3], vec![1, 7]),
                node("contiguous", vec![0, 1], vec![7, 3], vec![3, 1]),
                node("relu", vec![1], vec![7, 3], vec![3, 1]),
            ];
            for last in bad {
                EXECUTIONS.with(|count| count.set(0));
                assert!(
                    execute(
                        &inputs,
                        vec![node("contiguous", vec![0], vec![7, 3], vec![3, 1]), last]
                    )
                    .is_err()
                );
                EXECUTIONS.with(|count| assert_eq!(count.get(), 0));
            }
            EXECUTIONS.with(|count| count.set(0));
            let outputs = execute(
                &inputs,
                vec![
                    node("contiguous", vec![0], vec![7, 3], vec![3, 1]),
                    node("contiguous", vec![1], vec![7, 3], vec![3, 1]),
                ],
            )
            .unwrap();
            EXECUTIONS.with(|count| assert_eq!(count.get(), 2));
            assert!(outputs[0].is(&outputs[1]));
        });
    }
}

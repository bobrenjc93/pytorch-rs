//! One bridge for composed CUDA captures; eager Tensor methods stay unchanged.
use super::{CoreTensor, DType, PyTensor, compile_trace_mul_scalar_value, tensor_error};
use crate::tensor::cuda_graph::{Layout, Operation, Shape};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyInt, PyList, PyTuple};

// Shape/stride declarations have already passed the frontend's whole-graph
// checks. Recompute them natively too, before any storage allocation or launch.
type Node<'py> = (
    String,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
);

fn exact_indices(value: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if !value.is_exact_instance_of::<PyTuple>() && !value.is_exact_instance_of::<PyList>() {
        return Err(PyTypeError::new_err(
            "CUDA graph metadata requires tuple/list of exact integers",
        ));
    }
    value
        .try_iter()?
        .map(|item| {
            let item = item?;
            if !item.is_exact_instance_of::<PyInt>() {
                return Err(PyTypeError::new_err(
                    "CUDA graph metadata requires exact integers",
                ));
            }
            item.extract()
        })
        .collect()
}

fn exact_shape_dimensions(payload: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    if !payload.is_exact_instance_of::<PyTuple>() {
        return Err(PyTypeError::new_err(
            "shape payload requires an exact tuple",
        ));
    }
    let dimensions = payload.cast::<PyTuple>()?;
    dimensions
        .iter()
        .map(|dim| {
            if !dim.is_exact_instance_of::<PyInt>() {
                return Err(PyTypeError::new_err(
                    "shape dimensions require exact integers",
                ));
            }
            dim.extract::<i64>()
                .map_err(|_| PyTypeError::new_err("shape dimension overflow"))
        })
        .collect()
}

fn exact_shape(payload: &Bound<'_, PyAny>) -> PyResult<Shape> {
    let requested = exact_shape_dimensions(payload)?;
    Shape::new(&requested).map_err(|error| tensor_error(&error))
}

// Metadata only: use exactly the planner used again by the whole-graph bridge.
#[pyfunction(name = "_compile_trace_cuda_reshape_metadata")]
pub(super) fn reshape_metadata(
    shape: &Bound<'_, PyAny>,
    strides: &Bound<'_, PyAny>,
    offset: &Bound<'_, PyAny>,
    requested: &Bound<'_, PyAny>,
) -> PyResult<(Vec<usize>, Vec<usize>, usize)> {
    shape_metadata(shape, strides, offset, requested, false)
}

#[pyfunction(name = "_compile_trace_cuda_view_metadata")]
pub(super) fn view_metadata(
    shape: &Bound<'_, PyAny>,
    strides: &Bound<'_, PyAny>,
    offset: &Bound<'_, PyAny>,
    requested: &Bound<'_, PyAny>,
) -> PyResult<(Vec<usize>, Vec<usize>, usize)> {
    shape_metadata(shape, strides, offset, requested, true)
}

fn shape_metadata(
    shape: &Bound<'_, PyAny>,
    strides: &Bound<'_, PyAny>,
    offset: &Bound<'_, PyAny>,
    requested: &Bound<'_, PyAny>,
    alias_only: bool,
) -> PyResult<(Vec<usize>, Vec<usize>, usize)> {
    if !offset.is_exact_instance_of::<PyInt>() {
        return Err(PyTypeError::new_err(
            "shape offset requires an exact integer",
        ));
    }
    let input = Layout {
        shape: exact_indices(shape)?,
        strides: exact_indices(strides)?,
        offset: offset.extract()?,
    };
    let dimensions = exact_shape_dimensions(requested)?;
    if dimensions.len() > 2 && input.shape.len() <= 2 {
        // Error validation does not admit higher-rank operations. Use eager's
        // checked resolver before applying the unchanged graph rank bound.
        input
            .validate_requested_shape(&dimensions)
            .map_err(|error| tensor_error(&error))?;
    }
    let requested = Shape::new(&dimensions).map_err(|error| tensor_error(&error))?;
    let operation = if alias_only {
        Operation::View(0, requested)
    } else {
        Operation::Reshape(0, requested)
    };
    let output = operation
        .layout(&[input])
        .map_err(|error| tensor_error(&error))?;
    Ok((output.shape, output.strides, output.offset))
}

fn parse_operation(
    target: &str,
    indices: &[usize],
    payload: &Bound<'_, PyAny>,
) -> PyResult<Operation> {
    let operation = match (target, indices) {
        ("mul_scalar", &[input]) => {
            Operation::MulScalar(input, compile_trace_mul_scalar_value(payload)?)
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
        ("reshape", &[input]) => Operation::Reshape(input, exact_shape(payload)?),
        ("view", &[input]) => Operation::View(input, exact_shape(payload)?),
        ("transpose", &[input]) => {
            if !payload.is_exact_instance_of::<PyTuple>() {
                return Err(PyTypeError::new_err(
                    "transpose axes require an exact tuple",
                ));
            }
            let axes = payload.cast::<PyTuple>()?;
            if axes.len() != 2
                || axes
                    .iter()
                    .any(|axis| !axis.is_exact_instance_of::<PyInt>())
            {
                return Err(PyTypeError::new_err(
                    "transpose requires two exact integer axes",
                ));
            }
            Operation::Transpose(
                input,
                axes.get_item(0)?.extract()?,
                axes.get_item(1)?.extract()?,
            )
        }
        ("t", &[input]) if payload.is_none() => Operation::T(input),
        ("contiguous", &[input]) if payload.is_none() => Operation::Contiguous(input),
        ("neg", &[input]) if payload.is_none() => Operation::Neg(input),
        ("relu", &[input]) if payload.is_none() => Operation::Relu(input),
        ("add", &[left, right]) if payload.is_none() => Operation::Add(left, right),
        ("matmul", &[left, right]) if payload.is_none() => Operation::Matmul(left, right),
        _ => {
            return Err(PyNotImplementedError::new_err(
                "unsupported CUDA graph node",
            ));
        }
    };
    Ok(operation)
}

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
        let indices = exact_indices(&indices)?;
        let shape = exact_indices(&shape)?;
        let strides = exact_indices(&strides)?;
        let operation = parse_operation(&target, &indices, &payload)?;
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
    // T, Transpose, Reshape and View always wrap fresh objects, even for unchanged metadata.
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
    fn shape_payloads_reject_before_early_or_late_execution() {
        if cuda::device_count() == 0 {
            eprintln!("skipping CUDA reshape bridge checks: no CUDA device");
            return;
        }
        Python::initialize();
        Python::attach(|py| {
            for target in ["reshape", "view"] {
                let tensor = CoreTensor::from_vec(vec![1.; 6], [2, 3])
                    .unwrap()
                    .try_copy_cpu_to_cuda(Device::Cuda(0))
                    .unwrap()
                    .t()
                    .unwrap();
                let tensor = if target == "view" {
                    tensor.t().unwrap()
                } else {
                    tensor
                };
                let inputs =
                    PyTuple::new(py, [Py::new(py, PyTensor::new(tensor)).unwrap()]).unwrap();
                let valid = (
                    target.to_owned(),
                    PyTuple::new(py, [0]).unwrap().into_any(),
                    PyTuple::new(py, [-1]).unwrap().into_any(),
                    PyTuple::new(py, [6]).unwrap().into_any(),
                    PyTuple::new(py, [1]).unwrap().into_any(),
                );
                let mut bad = Vec::new();
                for expr in [
                    c"None",
                    c"[6]",
                    c"(True, 6)",
                    c"(1, True)",
                    c"(6.0,)",
                    c"(-1, -1)",
                    c"(-2,)",
                    c"(7,)",
                    c"(1, 2, 3)",
                    c"()",
                    c"(9223372036854775808,)",
                    c"(9223372036854775807, 2)",
                ] {
                    let mut node = valid.clone();
                    node.2 = py.eval(expr, None, None).unwrap();
                    bad.push(node);
                }
                for expr in [c"(True,)", c"(6.0,)", c"(7,)"] {
                    let mut node = valid.clone();
                    node.3 = py.eval(expr, None, None).unwrap();
                    bad.push(node);
                }
                for expr in [c"(True,)", c"(1.0,)", c"(2,)"] {
                    let mut node = valid.clone();
                    node.4 = py.eval(expr, None, None).unwrap();
                    bad.push(node);
                }
                for expr in [c"(False,)", c"(0.0,)", c"(99,)", c"()", c"(0, 0)"] {
                    let mut node = valid.clone();
                    node.1 = py.eval(expr, None, None).unwrap();
                    bad.push(node);
                }
                for invalid in bad {
                    for nodes in [
                        vec![invalid.clone(), valid.clone()],
                        vec![valid.clone(), invalid],
                    ] {
                        EXECUTIONS.with(|count| count.set(0));
                        assert!(execute(&inputs, nodes).is_err());
                        EXECUTIONS.with(|count| assert_eq!(count.get(), 0));
                    }
                }
                let outputs = execute(&inputs, vec![valid.clone(), valid]).unwrap();
                assert!(!outputs[0].is(&outputs[1]));
                assert!(!outputs[0].bind(py).is(inputs.get_item(0).unwrap()));
            }
        });
    }

    #[test]
    fn incompatible_view_rejects_before_even_an_earlier_pack() {
        if cuda::device_count() == 0 {
            eprintln!("skipping CUDA view prevalidation: no CUDA device");
            return;
        }
        Python::initialize();
        Python::attach(|py| {
            let tensor = CoreTensor::from_vec(vec![1.; 6], [2, 3])
                .unwrap()
                .try_copy_cpu_to_cuda(Device::Cuda(0))
                .unwrap()
                .t()
                .unwrap();
            let inputs = PyTuple::new(py, [Py::new(py, PyTensor::new(tensor)).unwrap()]).unwrap();
            let pack = (
                "contiguous".to_owned(),
                PyTuple::new(py, [0]).unwrap().into_any(),
                py.None().into_bound(py),
                PyTuple::new(py, [3, 2]).unwrap().into_any(),
                PyTuple::new(py, [2, 1]).unwrap().into_any(),
            );
            let view = (
                "view".to_owned(),
                PyTuple::new(py, [0]).unwrap().into_any(),
                PyTuple::new(py, [-1]).unwrap().into_any(),
                PyTuple::new(py, [6]).unwrap().into_any(),
                PyTuple::new(py, [1]).unwrap().into_any(),
            );
            for nodes in [vec![view.clone(), pack.clone()], vec![pack, view]] {
                EXECUTIONS.with(|count| count.set(0));
                let error = execute(&inputs, nodes).unwrap_err();
                assert!(error.is_instance_of::<PyRuntimeError>(py));
                assert!(error.to_string().contains("view size is not compatible"));
                EXECUTIONS.with(|count| assert_eq!(count.get(), 0));
            }
        });
    }

    #[test]
    fn t_metadata_types_reject_before_any_native_operation() {
        if cuda::device_count() == 0 {
            eprintln!("skipping CUDA graph type checks: no CUDA device");
            return;
        }
        Python::initialize();
        Python::attach(|py| {
            let base = CoreTensor::from_vec(vec![1.], [1, 1])
                .unwrap()
                .try_copy_cpu_to_cuda(Device::Cuda(0))
                .unwrap();
            let input = Py::new(py, PyTensor::new(base)).unwrap();
            let inputs = PyTuple::new(py, [input]).unwrap();
            let valid = (
                "t".to_owned(),
                PyTuple::new(py, [0]).unwrap().into_any(),
                py.None().into_bound(py),
                PyTuple::new(py, [1, 1]).unwrap().into_any(),
                PyTuple::new(py, [1, 1]).unwrap().into_any(),
            );
            let mut bad = Vec::new();
            for expr in [c"(True, 1)", c"(1.0, 1)", c"(1, False)"] {
                let value = py.eval(expr, None, None).unwrap();
                let mut shape = valid.clone();
                shape.3 = value.clone();
                bad.push(shape);
                let mut stride = valid.clone();
                stride.4 = value;
                bad.push(stride);
            }
            for expr in [c"(False,)", c"(0.0,)", c"(99,)", c"()", c"(0, 0)"] {
                let mut node = valid.clone();
                node.1 = py.eval(expr, None, None).unwrap();
                bad.push(node);
            }
            let mut payload = valid.clone();
            payload.2 = py.eval(c"True", None, None).unwrap();
            bad.push(payload);
            for invalid in bad {
                for nodes in [
                    vec![invalid.clone(), valid.clone()],
                    vec![valid.clone(), invalid],
                ] {
                    EXECUTIONS.with(|count| count.set(0));
                    assert!(execute(&inputs, nodes).is_err());
                    EXECUTIONS.with(|count| assert_eq!(count.get(), 0));
                }
            }
            let outputs = execute(&inputs, vec![valid.clone(), valid]).unwrap();
            assert!(!outputs[0].is(&outputs[1]));
            assert!(!outputs[0].bind(py).is(inputs.get_item(0).unwrap()));
            // The bridge must also reject a higher rank after an earlier valid node.
            let rank3 = CoreTensor::from_vec(vec![1.], [1, 1, 1])
                .unwrap()
                .try_copy_cpu_to_cuda(Device::Cuda(0))
                .unwrap();
            let inputs = PyTuple::new(py, [Py::new(py, PyTensor::new(rank3)).unwrap()]).unwrap();
            let node = |target: &str| {
                (
                    target.to_owned(),
                    PyTuple::new(py, [0]).unwrap().into_any(),
                    py.None().into_bound(py),
                    PyTuple::new(py, [1, 1, 1]).unwrap().into_any(),
                    PyTuple::new(py, [1, 1, 1]).unwrap().into_any(),
                )
            };
            EXECUTIONS.with(|count| count.set(0));
            assert!(execute(&inputs, vec![node("neg"), node("t")]).is_err());
            EXECUTIONS.with(|count| assert_eq!(count.get(), 0));
        });
    }

    #[test]
    fn transpose_axes_reject_before_early_or_late_execution() {
        if cuda::device_count() == 0 {
            eprintln!("skipping transpose graph validation: no CUDA device");
            return;
        }
        Python::initialize();
        Python::attach(|py| {
            let tensor = CoreTensor::from_vec(vec![1.], [1, 1])
                .unwrap()
                .try_copy_cpu_to_cuda(Device::Cuda(0))
                .unwrap();
            let inputs = PyTuple::new(py, [Py::new(py, PyTensor::new(tensor)).unwrap()]).unwrap();
            let valid = (
                "transpose".to_owned(),
                PyTuple::new(py, [0]).unwrap().into_any(),
                PyTuple::new(py, [-1, -2]).unwrap().into_any(),
                PyTuple::new(py, [1, 1]).unwrap().into_any(),
                PyTuple::new(py, [1, 1]).unwrap().into_any(),
            );
            let mut bad = Vec::new();
            for expr in [
                c"None",
                c"(True, 1)",
                c"(0, False)",
                c"(0.0, 1)",
                c"(0,)",
                c"[0, 1]",
                c"(0, 2)",
                c"(-3, 0)",
                c"(0, 9223372036854775808)",
            ] {
                let mut node = valid.clone();
                node.2 = py.eval(expr, None, None).unwrap();
                bad.push(node);
            }
            for expr in [c"(True, 1)", c"(1.0, 1)", c"(1, False)"] {
                for field in [3, 4] {
                    let mut node = valid.clone();
                    let value = py.eval(expr, None, None).unwrap();
                    if field == 3 {
                        node.3 = value;
                    } else {
                        node.4 = value;
                    }
                    bad.push(node);
                }
            }
            for expr in [c"(False,)", c"(0.0,)", c"(99,)", c"()", c"(0, 0)"] {
                let mut node = valid.clone();
                node.1 = py.eval(expr, None, None).unwrap();
                bad.push(node);
            }
            for invalid in bad {
                for nodes in [
                    vec![invalid.clone(), valid.clone()],
                    vec![valid.clone(), invalid],
                ] {
                    EXECUTIONS.with(|count| count.set(0));
                    assert!(execute(&inputs, nodes).is_err());
                    EXECUTIONS.with(|count| assert_eq!(count.get(), 0));
                }
            }
            let outputs = execute(&inputs, vec![valid.clone(), valid]).unwrap();
            assert!(!outputs[0].is(&outputs[1]));
            assert!(!outputs[0].bind(py).is(inputs.get_item(0).unwrap()));
        });
    }

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
                        PyTuple::new(py, indices).unwrap().into_any(),
                        py.None().into_bound(py),
                        PyTuple::new(py, shape).unwrap().into_any(),
                        PyTuple::new(py, strides).unwrap().into_any(),
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
                node("abs", vec![1], vec![7, 3], vec![3, 1]),
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
    #[test]
    fn relu_graph_prevalidates_early_and_late_nodes() {
        if cuda::device_count() == 0 {
            eprintln!("skipping CUDA ReLU graph accounting: no CUDA device");
            return;
        }
        Python::initialize();
        Python::attach(|py| {
            let base = CoreTensor::from_vec(vec![-1., 2., -3., 4., -5., 6.], [2, 3])
                .unwrap()
                .try_copy_cpu_to_cuda(Device::Cuda(0))
                .unwrap();
            let input = Py::new(py, PyTensor::new(base)).unwrap();
            let inputs = PyTuple::new(py, [input]).unwrap();
            let node = |indices: Vec<usize>, shape: Vec<usize>, strides: Vec<usize>| {
                (
                    "relu".to_owned(),
                    PyTuple::new(py, indices).unwrap().into_any(),
                    py.None().into_bound(py),
                    PyTuple::new(py, shape).unwrap().into_any(),
                    PyTuple::new(py, strides).unwrap().into_any(),
                )
            };
            let valid = node(vec![0], vec![2, 3], vec![3, 1]);
            let mut bad = vec![
                node(vec![], vec![2, 3], vec![3, 1]),
                node(vec![0, 0], vec![2, 3], vec![3, 1]),
                node(vec![99], vec![2, 3], vec![3, 1]),
                node(vec![0], vec![3, 2], vec![2, 1]),
                node(vec![0], vec![2, 3], vec![1, 2]),
            ];
            for payload in [
                false.into_pyobject(py).unwrap().to_owned().into_any(),
                1_i32.into_pyobject(py).unwrap().into_any(),
                PyTuple::empty(py).into_any(),
            ] {
                let mut wrong = valid.clone();
                wrong.2 = payload;
                bad.push(wrong);
            }
            for wrong in bad {
                for nodes in [
                    vec![wrong.clone(), valid.clone()],
                    vec![valid.clone(), wrong],
                ] {
                    EXECUTIONS.with(|count| count.set(0));
                    assert!(execute(&inputs, nodes).is_err());
                    EXECUTIONS.with(|count| assert_eq!(count.get(), 0));
                }
            }
            EXECUTIONS.with(|count| count.set(0));
            let out = execute(&inputs, vec![valid, node(vec![1], vec![2, 3], vec![3, 1])]).unwrap();
            EXECUTIONS.with(|count| assert_eq!(count.get(), 2));
            assert!(!out[0].is(&out[1]));
        });
    }
}

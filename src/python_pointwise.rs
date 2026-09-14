//! Private typed JIT bridge; no arbitrary source strings or raw pointers accepted.
use super::{CoreTensor, PyTensor, tensor_error};
use crate::{
    cuda::jit::Kernel,
    pointwise_ir::{Graph, Node},
};
use pyo3::{
    exceptions::{PyNotImplementedError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyFloat, PyInt, PyString, PyTuple},
};

type Payload = (String, usize, usize, u64);

fn graph(
    nodes: &Bound<'_, PyTuple>,
    outputs: &Bound<'_, PyTuple>,
    arity: usize,
) -> PyResult<Graph> {
    if !outputs.is_exact_instance_of::<PyTuple>()
        || outputs
            .iter()
            .any(|root| !root.is_exact_instance_of::<PyInt>())
    {
        return Err(PyTypeError::new_err(
            "expected an exact tuple of integer output roots",
        ));
    }
    if !(1..=64).contains(&outputs.len()) {
        return Err(PyValueError::new_err(
            "expected 1 to 64 computed output roots",
        ));
    }
    let outputs = outputs.extract::<Vec<usize>>()?;
    if nodes.len() > 4096 {
        return Err(PyValueError::new_err("pointwise graph exceeds node limit"));
    }
    let mut parsed = Vec::with_capacity(nodes.len());
    for item in nodes.iter() {
        if !item.is_exact_instance_of::<PyTuple>() {
            return Err(PyTypeError::new_err("expected exact node tuples"));
        }
        let tuple = item.cast::<PyTuple>()?;
        if tuple.len() != 4
            || !tuple.get_item(0)?.is_exact_instance_of::<PyString>()
            || tuple
                .iter()
                .skip(1)
                .any(|x| !x.is_exact_instance_of::<PyInt>())
        {
            return Err(PyTypeError::new_err(
                "expected (operation, index, index, bits)",
            ));
        }
        let (op, a, b, bits): Payload = tuple.extract()?;
        let node = match (op.as_str(), a, b, bits) {
            ("input", a, 0, 0) => Node::Input(a),
            ("scalar", a, negative @ 0..=1, 0) => Node::RuntimeScalar(a, negative != 0),
            ("constant", 0, 0, bits) => Node::Constant(bits),
            ("boolean", 0, 0, bits @ 0..=1) => Node::Boolean(bits != 0),
            ("integer", 0, 0, bits) => Node::Integer(bits),
            ("add", a, b, 0) => Node::Add(a, b),
            ("sub", a, b, 0) => Node::Sub(a, b),
            ("mul", a, b, 0) => Node::Mul(a, b),
            ("neg", a, 0, 0) => Node::Neg(a),
            ("relu", a, 0, 0) => Node::Relu(a),
            ("sin", a, 0, 0) => Node::Sin(a),
            ("cos", a, 0, 0) => Node::Cos(a),
            _ => {
                return Err(PyNotImplementedError::new_err(
                    "unsupported typed pointwise node",
                ));
            }
        };
        parsed.push(node);
    }
    let graph = Graph {
        nodes: parsed,
        inputs: arity,
        outputs,
    };
    graph
        .validate()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(graph)
}

fn with_inputs<T>(
    inputs: &Bound<'_, PyTuple>,
    f: impl FnOnce(&[&CoreTensor]) -> PyResult<T>,
) -> PyResult<T> {
    let owners = inputs
        .iter()
        .map(|input| {
            if !input.is_exact_instance_of::<PyTensor>() {
                return Err(PyTypeError::new_err(
                    "pointwise JIT requires exact native tensors",
                ));
            }
            Ok(input.cast_into::<PyTensor>()?)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let borrows = owners
        .iter()
        .map(|x| x.try_borrow().map_err(PyErr::from))
        .collect::<PyResult<Vec<_>>>()?;
    let tensors: Vec<_> = borrows.iter().map(|x| &x.inner).collect();
    // Hold Python borrows and storage owners through synchronous completion.
    f(&tensors)
}

#[pyfunction(name = "_pointwise_validate_inputs")]
pub(super) fn validate_inputs(inputs: &Bound<'_, PyTuple>) -> PyResult<usize> {
    with_inputs(inputs, |tensors| {
        CoreTensor::validate_pointwise_inputs(tensors)
            .map_err(|e| PyNotImplementedError::new_err(e.to_string()))
    })
}

#[pyfunction(name = "_pointwise_source")]
#[pyo3(signature = (nodes, outputs, arity, shapes=None))]
pub(super) fn source(
    nodes: &Bound<'_, PyTuple>,
    outputs: &Bound<'_, PyTuple>,
    arity: usize,
    shapes: Option<Vec<Vec<usize>>>,
) -> PyResult<String> {
    let graph = graph(nodes, outputs, arity)?;
    if let Some(shapes) = shapes {
        let indexing = graph
            .indexing(&shapes.iter().map(Vec::as_slice).collect::<Vec<_>>())
            .map_err(|e| tensor_error(&e))?;
        graph
            .indexed_source(&indexing.addresses)
            .map_err(|e| tensor_error(&e))
    } else {
        graph.source().map_err(|e| tensor_error(&e))
    }
}

#[pyfunction(name = "_pointwise_compile")]
pub(super) fn compile(
    inputs: &Bound<'_, PyTuple>,
    nodes: &Bound<'_, PyTuple>,
    outputs: &Bound<'_, PyTuple>,
) -> PyResult<Compiled> {
    let graph = graph(nodes, outputs, inputs.len())?;
    let device = validate_inputs(inputs)?;
    let indexing = with_inputs(inputs, |tensors| {
        graph
            .indexing(&tensors.iter().map(|x| x.shape()).collect::<Vec<_>>())
            .map_err(|e| PyNotImplementedError::new_err(e.to_string()))
    })?;
    let kernel = Kernel::compile_indexed(&graph, device, indexing.addresses)
        .map_err(|e| tensor_error(&e))?;
    Ok(Compiled {
        kernel,
        arity: graph.inputs,
    })
}

pub(crate) fn convert_outputs(
    py: Python<'_>,
    outputs: Vec<CoreTensor>,
    convert: impl FnMut(CoreTensor) -> PyResult<Py<PyTensor>>,
) -> PyResult<Bound<'_, PyTuple>> {
    let objects = outputs
        .into_iter()
        .map(convert)
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, objects)
}

#[pyclass(frozen, module = "torch_rs.torch_rs", name = "_PointwiseKernel")]
pub(super) struct Compiled {
    kernel: Kernel,
    arity: usize,
}
#[pymethods]
impl Compiled {
    #[pyo3(signature = (inputs, scalars=None))]
    fn run<'py>(
        &self,
        inputs: &Bound<'py, PyTuple>,
        scalars: Option<&Bound<'_, PyTuple>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        // Exact types are checked before conversion; no user float hooks run.
        let values = scalars.map_or_else(
            || Ok(Vec::new()),
            |scalars| {
                if !scalars.is_exact_instance_of::<PyTuple>() {
                    return Err(PyTypeError::new_err("expected an exact scalar tuple"));
                }
                scalars
                    .iter()
                    .map(|value| {
                        if !value.is_exact_instance_of::<PyFloat>() {
                            return Err(PyTypeError::new_err(
                                "runtime scalars must be exact floats",
                            ));
                        }
                        // The graph's runtime scalar boundary is float32 materialization.
                        #[allow(clippy::cast_possible_truncation)]
                        let value = value.extract::<f64>()? as f32;
                        Ok(value)
                    })
                    .collect::<PyResult<Vec<_>>>()
            },
        )?;
        self.kernel
            .validate_scalars(&values)
            .map_err(|e| tensor_error(&e))?;
        if inputs.len() != self.arity {
            return Err(PyValueError::new_err(
                "pointwise input arity guard mismatch",
            ));
        }
        with_inputs(inputs, |tensors| {
            let outputs = CoreTensor::pointwise_jit(tensors, &self.kernel, &values)
                .map_err(|e| tensor_error(&e))?;
            // Keep every storage owner and input borrow through fallible Python
            // conversion. A partial conversion only drops local, unpublished owners.
            convert_outputs(inputs.py(), outputs, |output| {
                Py::new(inputs.py(), PyTensor::new(output))
            })
        })
    }
    #[getter]
    fn source(&self) -> &str {
        &self.kernel.source
    }
    #[getter]
    fn ptx(&self) -> &str {
        &self.kernel.ptx
    }
    #[getter]
    fn nvrtc_version(&self) -> (i32, i32) {
        self.kernel.version
    }
    #[getter]
    fn options(&self) -> Vec<String> {
        self.kernel.options.clone()
    }
    #[getter]
    fn device(&self) -> usize {
        self.kernel.device
    }
}

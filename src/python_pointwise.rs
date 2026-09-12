//! Private typed JIT bridge; no arbitrary source strings or raw pointers accepted.
use super::{CoreTensor, PyTensor, tensor_error};
use crate::{
    cuda::jit::Kernel,
    pointwise_ir::{Graph, Node},
};
use pyo3::{
    exceptions::{PyNotImplementedError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyInt, PyString, PyTuple},
};

type Payload = (String, usize, usize, u32);

fn graph(nodes: &Bound<'_, PyTuple>, output: usize, arity: usize) -> PyResult<Graph> {
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
            ("constant", 0, 0, bits) => Node::Constant(bits),
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
        output,
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
pub(super) fn source(nodes: &Bound<'_, PyTuple>, output: usize, arity: usize) -> PyResult<String> {
    graph(nodes, output, arity)?
        .source()
        .map_err(|e| tensor_error(&e))
}

#[pyfunction(name = "_pointwise_compile")]
pub(super) fn compile(
    inputs: &Bound<'_, PyTuple>,
    nodes: &Bound<'_, PyTuple>,
    output: usize,
) -> PyResult<Compiled> {
    let graph = graph(nodes, output, inputs.len())?;
    let device = validate_inputs(inputs)?;
    let kernel = Kernel::compile(&graph, device).map_err(|e| tensor_error(&e))?;
    Ok(Compiled {
        kernel,
        arity: graph.inputs,
    })
}

#[pyclass(frozen, module = "torch_rs.torch_rs", name = "_PointwiseKernel")]
pub(super) struct Compiled {
    kernel: Kernel,
    arity: usize,
}
#[pymethods]
impl Compiled {
    fn run(&self, inputs: &Bound<'_, PyTuple>) -> PyResult<PyTensor> {
        if inputs.len() != self.arity {
            return Err(PyValueError::new_err(
                "pointwise input arity guard mismatch",
            ));
        }
        with_inputs(inputs, |tensors| {
            CoreTensor::pointwise_jit(tensors, &self.kernel)
                .map(PyTensor::new)
                .map_err(|e| tensor_error(&e))
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

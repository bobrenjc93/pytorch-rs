//! Private typed JIT bridge; no arbitrary source strings or raw pointers accepted.
use super::{CoreTensor, PyTensor, tensor_error};
use crate::{
    cuda::jit::Kernel,
    pointwise_ir::{Graph, Node},
    tensor::pointwise_jit::{HostAdmission, HostPointwise, PreparedPointwise},
};
use pyo3::{
    exceptions::{PyNotImplementedError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyString, PyTuple},
};
use std::sync::Arc;

type Payload = (String, usize, usize, u64);

#[pyfunction(name = "_pointwise_namespace_keys_exact")]
#[allow(unsafe_code)] // Borrowed dict keys stay inside the callback-free critical section.
pub(super) fn namespace_keys_exact(value: &Bound<'_, PyAny>) -> bool {
    let Ok(namespace) = value.cast_exact::<PyDict>() else {
        return false;
    };
    // Protect iterator construction as well as traversal. Exact type checks
    // invoke no callbacks; Python retains namespace selection and error policy.
    pyo3::sync::critical_section::with_critical_section(value, || {
        let mut position = 0;
        let mut key = std::ptr::null_mut();
        // SAFETY: namespace is an exact, live dict. The critical section (or
        // GIL) prevents mutation throughout traversal; neither C API releases
        // it or calls Python. PyDict_Next lends a non-null key on success. We
        // inspect only its exact type and never retain it or read the value.
        // Avoid creating owned key/value references for every warm guard scan.
        unsafe {
            while pyo3::ffi::PyDict_Next(
                namespace.as_ptr(),
                &raw mut position,
                &raw mut key,
                std::ptr::null_mut(),
            ) != 0
            {
                if pyo3::ffi::PyUnicode_CheckExact(key) == 0 {
                    return false;
                }
            }
        }
        true
    })
}

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

#[pyfunction(name = "_pointwise_admit_inputs")]
pub(super) fn admit_inputs(py: Python<'_>, inputs: &Bound<'_, PyTuple>) -> PyResult<Py<PyTuple>> {
    with_inputs(inputs, |tensors| {
        let metadata = tensors
            .iter()
            .map(|tensor| super::compile_tensor_metadata(py, tensor))
            .collect::<PyResult<Vec<_>>>()?;
        // Preserve the frontend's any-CPU diagnostic before whole-input admission,
        // including when another slot has unsupported dtype, gradients or layout.
        if tensors
            .iter()
            .any(|tensor| tensor.device() == crate::Device::Cpu)
        {
            return Err(PyNotImplementedError::new_err(
                "torch.compile(): native CUDA pointwise: default backend does not compile CPU tensors; use backend='eager' for the documented CPU capture subset; see docs/compile-pointwise-jit.md",
            ));
        }
        CoreTensor::validate_pointwise_inputs(tensors)
            .map_err(|e| PyNotImplementedError::new_err(e.to_string()))?;
        Ok(PyTuple::new(py, metadata)?.unbind())
    })
}

fn parse_numerical_hint(hint: Option<&Bound<'_, PyAny>>) -> PyResult<Option<u64>> {
    hint.map(|hint| {
        if !hint.is_exact_instance_of::<PyInt>() {
            return Err(PyTypeError::new_err(
                "numerical hint must be an exact integer",
            ));
        }
        hint.extract::<u64>()
    })
    .transpose()
}

fn parse_output_order(order: Option<&Bound<'_, PyAny>>, count: usize) -> PyResult<Vec<usize>> {
    let Some(order) = order else {
        return Ok((0..count).collect());
    };
    if !order.is_exact_instance_of::<PyTuple>() {
        return Err(PyTypeError::new_err("output order must be an exact tuple"));
    }
    let order = order.cast::<PyTuple>()?;
    if order.len() != count
        || order
            .iter()
            .any(|slot| !slot.is_exact_instance_of::<PyInt>())
    {
        return Err(PyValueError::new_err("expected an output-slot permutation"));
    }
    let slots = order.extract::<Vec<usize>>()?;
    let mut seen = vec![false; slots.len()];
    for &slot in &slots {
        if slot >= seen.len() || std::mem::replace(&mut seen[slot], true) {
            return Err(PyValueError::new_err("expected an output-slot permutation"));
        }
    }
    Ok(slots)
}

fn parse_scalars(scalars: Option<&Bound<'_, PyTuple>>) -> PyResult<Vec<f32>> {
    scalars.map_or_else(
        || Ok(Vec::new()),
        |scalars| {
            if !scalars.is_exact_instance_of::<PyTuple>() {
                return Err(PyTypeError::new_err("expected an exact scalar tuple"));
            }
            scalars
                .iter()
                .map(|value| {
                    if !value.is_exact_instance_of::<PyFloat>() {
                        return Err(PyTypeError::new_err("runtime scalars must be exact floats"));
                    }
                    // Exact types precede conversion; no user float hooks run.
                    // The runtime scalar boundary is float32 materialization.
                    #[allow(clippy::cast_possible_truncation)]
                    let value = value.extract::<f64>()? as f32;
                    Ok(value)
                })
                .collect::<PyResult<Vec<_>>>()
        },
    )
}

/// Diagnostic rendering of the same immutable program executed by the VM.
/// This never compiles or retains a topology-specific executable.
#[pyfunction(name = "_pointwise_plan")]
#[pyo3(signature = (nodes, outputs, arity, shapes=None, numerical_hint=None, output_order=None))]
pub(super) fn plan(
    nodes: &Bound<'_, PyTuple>,
    outputs: &Bound<'_, PyTuple>,
    arity: usize,
    shapes: Option<Vec<Vec<usize>>>,
    numerical_hint: Option<&Bound<'_, PyAny>>,
    output_order: Option<&Bound<'_, PyAny>>,
) -> PyResult<String> {
    let graph = graph(nodes, outputs, arity)?;
    let numerical_hint = parse_numerical_hint(numerical_hint)?;
    let order = parse_output_order(output_order, graph.outputs.len())?;
    let (addresses, default_hint, scalar_output) = if let Some(shapes) = shapes {
        let indexing = graph
            .indexing(&shapes.iter().map(Vec::as_slice).collect::<Vec<_>>())
            .map_err(|error| tensor_error(&error))?;
        (
            indexing.addresses,
            indexing.output.elements as u64,
            indexing.output.shape.is_empty(),
        )
    } else {
        (
            vec![crate::pointwise_ir::indexing::Address::Linear; arity],
            u64::MAX,
            false,
        )
    };
    crate::pointwise_ir::program::Program::describe(
        &graph,
        &addresses,
        numerical_hint.unwrap_or(default_hint),
        &order,
        scalar_output,
    )
    .map_err(|error| tensor_error(&error))
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

/// Typed preparation precedes any compiler discovery or device upload.
#[pyfunction(name = "_pointwise_host_plan")]
#[pyo3(signature = (inputs, nodes, outputs, numerical_hint=None, output_order=None))]
pub(super) fn host_plan(
    inputs: &Bound<'_, PyTuple>,
    nodes: &Bound<'_, PyTuple>,
    outputs: &Bound<'_, PyTuple>,
    numerical_hint: Option<&Bound<'_, PyAny>>,
    output_order: Option<&Bound<'_, PyAny>>,
) -> PyResult<HostPlan> {
    let graph = graph(nodes, outputs, inputs.len())?;
    let output_count = graph.outputs.len();
    with_inputs(inputs, |tensors| {
        let admission = HostAdmission::new(tensors, graph)
            .map_err(|error| PyNotImplementedError::new_err(error.to_string()))?;
        let hint = parse_numerical_hint(numerical_hint)?;
        let order = parse_output_order(output_order, output_count)?;
        admission
            .prepare(hint, &order)
            .map(|host| HostPlan { host })
            .map_err(|error| tensor_error(&error))
    })
}

#[pyclass(frozen, module = "torch_rs.torch_rs", name = "_PointwiseHostPlan")]
pub(super) struct HostPlan {
    host: HostPointwise,
}
#[pymethods]
impl HostPlan {
    #[getter]
    fn executable_identity<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.host.identity.bytes)
    }
    fn compile(&self) -> PyResult<Executable> {
        self.host
            .compile()
            .map(|kernel| Executable { kernel })
            .map_err(|error| tensor_error(&error))
    }
}

/// Ordinary-default handles cannot call the legacy VM re-planning APIs.
#[pyclass(frozen, module = "torch_rs.torch_rs", name = "_PointwiseExecutable")]
pub(super) struct Executable {
    kernel: Arc<Kernel>,
}
#[pymethods]
impl Executable {
    fn bind(&self, host: &HostPlan) -> PyResult<Prepared> {
        host.host
            .bind(Arc::clone(&self.kernel))
            .map(|invocation| Prepared { invocation })
            .map_err(|error| tensor_error(&error))
    }
    #[getter]
    fn executable_identity<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(
            py,
            &self
                .kernel
                .identity
                .as_ref()
                .expect("selected executable")
                .bytes,
        )
    }
    #[getter]
    fn kind(&self) -> &'static str {
        if self
            .kernel
            .identity
            .as_ref()
            .expect("selected executable")
            .direct
        {
            "direct"
        } else {
            "vm"
        }
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
    #[getter]
    fn context(&self) -> usize {
        self.kernel
            .identity
            .as_ref()
            .expect("selected executable")
            .context
    }
    /// Regenerated diagnostic only, never selected-invocation evidence.
    #[pyo3(signature = (numerical_hint, output_order=None, scalar_output=None))]
    fn plan(
        &self,
        numerical_hint: &Bound<'_, PyAny>,
        output_order: Option<&Bound<'_, PyAny>>,
        scalar_output: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        Compiled {
            kernel: Arc::clone(&self.kernel),
            arity: self.kernel.graph.inputs,
        }
        .plan(numerical_hint, output_order, scalar_output)
    }
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
    kernel: Arc<Kernel>,
    arity: usize,
}
#[pymethods]
impl Compiled {
    #[pyo3(signature = (inputs, numerical_hint=None, output_order=None))]
    fn prepare(
        &self,
        inputs: &Bound<'_, PyTuple>,
        numerical_hint: Option<&Bound<'_, PyAny>>,
        output_order: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Prepared> {
        let numerical_hint = parse_numerical_hint(numerical_hint)?;
        let output_order = parse_output_order(output_order, self.kernel.graph.outputs.len())?;
        if inputs.len() != self.arity {
            return Err(PyValueError::new_err(
                "pointwise input arity guard mismatch",
            ));
        }
        with_inputs(inputs, |tensors| {
            CoreTensor::prepare_pointwise(
                tensors,
                Arc::clone(&self.kernel),
                numerical_hint,
                Some(&output_order),
            )
            .map(|invocation| Prepared { invocation })
            .map_err(|error| tensor_error(&error))
        })
    }

    #[pyo3(signature = (inputs, scalars=None, numerical_hint=None, output_order=None))]
    fn run<'py>(
        &self,
        inputs: &Bound<'py, PyTuple>,
        scalars: Option<&Bound<'_, PyTuple>>,
        numerical_hint: Option<&Bound<'_, PyAny>>,
        output_order: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        // Validate before input borrowing or any conversion can call Python.
        // A specialization hint controls numerical planning, never storage bounds.
        let numerical_hint = parse_numerical_hint(numerical_hint)?;
        let output_order = parse_output_order(output_order, self.kernel.graph.outputs.len())?;
        let values = parse_scalars(scalars)?;
        self.kernel
            .validate_scalars(&values)
            .map_err(|e| tensor_error(&e))?;
        if inputs.len() != self.arity {
            return Err(PyValueError::new_err(
                "pointwise input arity guard mismatch",
            ));
        }
        with_inputs(inputs, |tensors| {
            let outputs = CoreTensor::pointwise_jit(
                tensors,
                &self.kernel,
                &values,
                numerical_hint,
                Some(&output_order),
            )
            .map_err(|e| tensor_error(&e))?;
            // Keep every storage owner and input borrow through fallible Python
            // conversion. A partial conversion only drops local, unpublished owners.
            convert_outputs(inputs.py(), outputs, |output| {
                Py::new(inputs.py(), PyTensor::new(output))
            })
        })
    }
    #[pyo3(signature = (numerical_hint, output_order=None, scalar_output=None))]
    fn plan(
        &self,
        numerical_hint: &Bound<'_, PyAny>,
        output_order: Option<&Bound<'_, PyAny>>,
        scalar_output: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        let scalar_output = scalar_output
            .map(|value| {
                if !value.is_exact_instance_of::<PyBool>() {
                    return Err(PyTypeError::new_err("scalar output must be an exact bool"));
                }
                value.extract::<bool>()
            })
            .transpose()?
            .unwrap_or(false);
        let hint = parse_numerical_hint(Some(numerical_hint))?.expect("required hint");
        let order = parse_output_order(output_order, self.kernel.graph.outputs.len())?;
        crate::pointwise_ir::program::Program::describe(
            &self.kernel.graph,
            &self.kernel.addresses,
            hint,
            &order,
            scalar_output,
        )
        .map_err(|error| tensor_error(&error))
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

#[pyclass(frozen, module = "torch_rs.torch_rs", name = "_PointwisePrepared")]
pub(super) struct Prepared {
    invocation: PreparedPointwise,
}

#[pymethods]
impl Prepared {
    fn belongs_to(&self, executable: &Executable) -> bool {
        Arc::ptr_eq(self.invocation.kernel(), &executable.kernel)
    }
    #[getter]
    fn executable_identity<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.invocation
            .kernel()
            .identity
            .as_ref()
            .map(|id| PyBytes::new(py, &id.bytes))
    }
    #[getter]
    fn kind(&self) -> &'static str {
        if self
            .invocation
            .kernel()
            .identity
            .as_ref()
            .is_some_and(|id| id.direct)
        {
            "direct"
        } else {
            "vm"
        }
    }
    #[getter]
    fn source(&self) -> &str {
        &self.invocation.kernel().source
    }
    #[getter]
    fn ptx(&self) -> &str {
        &self.invocation.kernel().ptx
    }
    #[getter]
    fn nvrtc_version(&self) -> (i32, i32) {
        self.invocation.kernel().version
    }
    #[getter]
    fn options(&self) -> Vec<String> {
        self.invocation.kernel().options.clone()
    }
    #[getter]
    fn device(&self) -> usize {
        self.invocation.kernel().device
    }
    #[getter]
    fn context(&self) -> Option<usize> {
        self.invocation
            .kernel()
            .identity
            .as_ref()
            .map(|id| id.context)
    }
    #[getter]
    fn instruction_count(&self) -> usize {
        self.invocation.instruction_count()
    }
    #[getter]
    fn register_count(&self) -> usize {
        self.invocation.register_count()
    }
    #[pyo3(signature = (inputs, scalars=None))]
    fn run<'py>(
        &self,
        inputs: &Bound<'py, PyTuple>,
        scalars: Option<&Bound<'_, PyTuple>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let values = parse_scalars(scalars)?;
        with_inputs(inputs, |tensors| {
            let outputs = self
                .invocation
                .run(tensors, &values)
                .map_err(|error| tensor_error(&error))?;
            // The borrowed preparation/kernel, inputs and every output remain
            // owned through completion and this fallible conversion boundary.
            convert_outputs(inputs.py(), outputs, |output| {
                Py::new(inputs.py(), PyTensor::new(output))
            })
        })
    }

    #[getter]
    fn input_shapes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let shapes = self
            .invocation
            .input_shapes()
            .iter()
            .map(|shape| PyTuple::new(py, shape.iter().copied()))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, shapes)
    }

    #[getter]
    fn retained_bytes(&self) -> usize {
        self.invocation.retained_bytes()
    }
}

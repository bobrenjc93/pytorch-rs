//! Private typed JIT bridge; no arbitrary source strings or raw pointers accepted.
use super::{CoreTensor, PyTensor, tensor_error};
use crate::{
    cuda::{jit::Kernel, jit_module::Module, leading_sum::Kernel as LeadingKernel},
    pointwise_ir::{Graph, Node},
    tensor::{
        leading_sum::{Divisor, HostLeadingSum, LeadingSum, PreparedLeadingSum, ScalarKind},
        pointwise_jit::{HostAdmission, HostPointwise, PreparedPointwise},
    },
};
use pyo3::{
    exceptions::{PyNotImplementedError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyString, PyTuple},
};
use std::sync::Arc;

type Payload = (String, usize, usize, u64);

#[cfg(test)]
#[path = "python_pointwise_admission_tests.rs"]
mod admission_tests;

#[pyfunction(name = "_pointwise_namespace_keys_exact")]
pub(super) fn namespace_keys_exact(value: &Bound<'_, PyAny>) -> bool {
    let Ok(namespace) = value.cast_exact::<PyDict>() else {
        return false;
    };
    // Protect iterator construction as well as traversal. Exact type checks
    // invoke no callbacks; Python retains namespace selection and error policy.
    pyo3::sync::critical_section::with_critical_section(value, || {
        namespace
            .iter()
            .all(|(key, _)| key.is_exact_instance_of::<PyString>())
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
            ("erf", a, 0, 0) => Node::Erf(a),
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
pub(super) fn admit_inputs<'py>(inputs: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyTuple>> {
    with_inputs(inputs, |tensors| {
        let metadata = tensors
            .iter()
            .map(|tensor| super::compile_tensor_metadata(inputs.py(), tensor))
            .collect::<PyResult<Vec<_>>>()?;
        // Preserve whole-input CPU precedence after collecting every record.
        if tensors.iter().any(|tensor| tensor.device().is_cpu()) {
            return Err(PyNotImplementedError::new_err(
                "torch.compile(): native CUDA pointwise: default backend does not compile CPU tensors; use backend='eager' for the documented CPU capture subset; see docs/compile-pointwise-jit.md",
            ));
        }
        CoreTensor::validate_pointwise_inputs(tensors)
            .map_err(|e| PyNotImplementedError::new_err(e.to_string()))?;
        PyTuple::new(inputs.py(), metadata)
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
            .map(|host| HostPlan {
                host: HostVariant::Pointwise(host),
            })
            .map_err(|error| tensor_error(&error))
    })
}

/// The descriptor is immutable scalar/provenance data, never Python owners.
/// Exact-kind checks precede conversions, indexing and any native discovery.
fn leading_sum_descriptor(value: &Bound<'_, PyAny>) -> PyResult<LeadingSum> {
    let descriptor = value
        .cast_exact::<PyTuple>()
        .map_err(|_| PyTypeError::new_err("expected an exact leading-sum descriptor tuple"))?;
    if descriptor.len() != 5 {
        return Err(PyValueError::new_err(
            "expected five leading-sum descriptor fields",
        ));
    }
    let tag = descriptor.get_item(0)?;
    let axis = descriptor.get_item(1)?;
    let keepdim = descriptor.get_item(2)?;
    if !tag.is_exact_instance_of::<PyString>()
        || !axis.is_exact_instance_of::<PyInt>()
        || !keepdim.is_exact_instance_of::<PyBool>()
    {
        return Err(PyTypeError::new_err(
            "expected leading_sum, exact integer axis and bool keepdim",
        ));
    }
    if tag.extract::<String>()? != "leading_sum" {
        return Err(PyValueError::new_err("unknown leading-sum descriptor kind"));
    }
    let axis = axis.extract::<i64>()?;
    if axis != 0 && axis != -2 {
        return Err(PyValueError::new_err("leading sum requires axis 0 or -2"));
    }
    let row_certificate = parse_dimension_certificate(&descriptor.get_item(3)?)?;
    let divisor = parse_leading_sum_divisor(&descriptor.get_item(4)?)?;
    let descriptor = LeadingSum {
        axis,
        keepdim: keepdim.extract::<bool>()?,
        row_certificate,
        divisor,
    };
    descriptor
        .validate()
        .map_err(|error| tensor_error(&error))?;
    Ok(descriptor)
}

fn parse_leading_sum_divisor(value: &Bound<'_, PyAny>) -> PyResult<Divisor> {
    let divisor = value
        .cast_exact::<PyTuple>()
        .map_err(|_| PyTypeError::new_err("expected an exact divisor descriptor tuple"))?;
    if divisor.is_empty() || !divisor.get_item(0)?.is_exact_instance_of::<PyString>() {
        return Err(PyTypeError::new_err("expected a divisor kind"));
    }
    let kind = divisor.get_item(0)?.extract::<String>()?;
    Ok(match (kind.as_str(), divisor.len()) {
        ("none", 1) => Divisor::None,
        ("constant", 3) => {
            let kind = divisor.get_item(1)?;
            let bits = divisor.get_item(2)?;
            if !kind.is_exact_instance_of::<PyString>() || !bits.is_exact_instance_of::<PyInt>() {
                return Err(PyTypeError::new_err(
                    "expected exact constant kind and integer bits",
                ));
            }
            let kind = match kind.extract::<String>()?.as_str() {
                "bool" => ScalarKind::Boolean,
                "int" => ScalarKind::Integer,
                "float" => ScalarKind::Float,
                _ => return Err(PyValueError::new_err("unknown leading-sum scalar kind")),
            };
            Divisor::Constant {
                kind,
                bits: bits.extract::<u64>()?,
            }
        }
        ("runtime", 3) => {
            let slot = divisor.get_item(1)?;
            let negative = divisor.get_item(2)?;
            if !slot.is_exact_instance_of::<PyInt>() || !negative.is_exact_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err(
                    "expected exact runtime slot and bool sign",
                ));
            }
            Divisor::Runtime {
                slot: slot.extract::<usize>()?,
                negative: negative.extract::<bool>()?,
            }
        }
        ("dimension", 3) => {
            let axis = divisor.get_item(1)?;
            if !axis.is_exact_instance_of::<PyInt>() {
                return Err(PyTypeError::new_err(
                    "expected an exact divisor dimension axis",
                ));
            }
            let axis = axis.extract::<usize>()?;
            if axis > 1 {
                return Err(PyValueError::new_err(
                    "divisor dimension axis must be 0 or 1",
                ));
            }
            Divisor::Dimension {
                axis,
                certificate: parse_dimension_certificate(&divisor.get_item(2)?)?,
            }
        }
        _ => {
            return Err(PyValueError::new_err(
                "unsupported leading-sum divisor descriptor",
            ));
        }
    })
}

fn parse_dimension_certificate(value: &Bound<'_, PyAny>) -> PyResult<Option<u64>> {
    if value.is_none() {
        return Ok(None);
    }
    if !value.is_exact_instance_of::<PyInt>() {
        return Err(PyTypeError::new_err(
            "dimension certificate must be None or an exact integer",
        ));
    }
    value.extract::<u64>().map(Some)
}

#[pyfunction(name = "_leading_sum_host_plan")]
pub(super) fn leading_sum_host_plan(
    inputs: &Bound<'_, PyTuple>,
    descriptor: &Bound<'_, PyAny>,
) -> PyResult<HostPlan> {
    let descriptor = leading_sum_descriptor(descriptor)?;
    if !inputs.is_exact_instance_of::<PyTuple>() {
        return Err(PyTypeError::new_err(
            "expected an exact leading-sum input tuple",
        ));
    }
    with_inputs(inputs, |tensors| {
        HostLeadingSum::new(tensors, descriptor)
            .map(|host| HostPlan {
                host: HostVariant::LeadingSum(host),
            })
            .map_err(|error| tensor_error(&error))
    })
}

enum HostVariant {
    Pointwise(HostPointwise),
    LeadingSum(HostLeadingSum),
}

enum ExecutableKernel {
    Pointwise(Arc<Kernel>),
    LeadingSum(Arc<LeadingKernel>),
}

impl ExecutableKernel {
    fn module(&self) -> &Module {
        match self {
            Self::Pointwise(kernel) => &kernel.module,
            Self::LeadingSum(kernel) => &kernel.module,
        }
    }
    fn identity(&self) -> &[u8] {
        match self {
            Self::Pointwise(kernel) => {
                &kernel.identity.as_ref().expect("selected executable").bytes
            }
            Self::LeadingSum(kernel) => &kernel.identity,
        }
    }
    fn kind(&self) -> &'static str {
        match self {
            Self::Pointwise(kernel)
                if kernel
                    .identity
                    .as_ref()
                    .expect("selected executable")
                    .direct =>
            {
                "direct"
            }
            Self::Pointwise(_) => "vm",
            Self::LeadingSum(_) => "leading_sum",
        }
    }
}

#[pyclass(frozen, module = "torch_rs.torch_rs", name = "_PointwiseHostPlan")]
pub(super) struct HostPlan {
    host: HostVariant,
}
#[pymethods]
impl HostPlan {
    #[getter]
    fn executable_identity<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let identity = match &self.host {
            HostVariant::Pointwise(host) => &host.identity.bytes,
            HostVariant::LeadingSum(host) => &host.identity,
        };
        PyBytes::new(py, identity)
    }
    fn compile(&self) -> PyResult<Executable> {
        let kernel = match &self.host {
            HostVariant::Pointwise(host) => {
                ExecutableKernel::Pointwise(host.compile().map_err(|error| tensor_error(&error))?)
            }
            HostVariant::LeadingSum(host) => {
                ExecutableKernel::LeadingSum(host.compile().map_err(|error| tensor_error(&error))?)
            }
        };
        Ok(Executable { kernel })
    }
}

/// Ordinary-default handles cannot call the legacy VM re-planning APIs.
#[pyclass(frozen, module = "torch_rs.torch_rs", name = "_PointwiseExecutable")]
pub(super) struct Executable {
    kernel: ExecutableKernel,
}
#[pymethods]
impl Executable {
    fn bind(&self, host: &HostPlan) -> PyResult<Prepared> {
        let invocation = match (&self.kernel, &host.host) {
            (ExecutableKernel::Pointwise(kernel), HostVariant::Pointwise(host)) => {
                PreparedVariant::Pointwise(
                    host.bind(Arc::clone(kernel))
                        .map_err(|error| tensor_error(&error))?,
                )
            }
            (ExecutableKernel::LeadingSum(kernel), HostVariant::LeadingSum(host)) => {
                PreparedVariant::LeadingSum(
                    host.bind(Arc::clone(kernel))
                        .map_err(|error| tensor_error(&error))?,
                )
            }
            _ => return Err(PyValueError::new_err("host plan executable kind mismatch")),
        };
        Ok(Prepared { invocation })
    }
    #[getter]
    fn executable_identity<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.kernel.identity())
    }
    #[getter]
    fn kind(&self) -> &'static str {
        self.kernel.kind()
    }
    #[getter]
    fn source(&self) -> &str {
        &self.kernel.module().source
    }
    #[getter]
    fn ptx(&self) -> &str {
        &self.kernel.module().ptx
    }
    #[getter]
    fn nvrtc_version(&self) -> (i32, i32) {
        self.kernel.module().version
    }
    #[getter]
    fn options(&self) -> Vec<String> {
        self.kernel.module().options.clone()
    }
    #[getter]
    fn device(&self) -> usize {
        self.kernel.module().device
    }
    #[getter]
    fn context(&self) -> usize {
        self.kernel.module().context
    }
    /// Regenerated diagnostic only, never selected-invocation evidence.
    #[pyo3(signature = (numerical_hint, output_order=None, scalar_output=None))]
    fn plan(
        &self,
        numerical_hint: &Bound<'_, PyAny>,
        output_order: Option<&Bound<'_, PyAny>>,
        scalar_output: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        let ExecutableKernel::Pointwise(kernel) = &self.kernel else {
            return Err(PyNotImplementedError::new_err(
                "leading_sum has no pointwise Program",
            ));
        };
        Compiled {
            kernel: Arc::clone(kernel),
            arity: kernel.graph.inputs,
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
            .map(|invocation| Prepared {
                invocation: PreparedVariant::Pointwise(invocation),
            })
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

enum PreparedVariant {
    Pointwise(PreparedPointwise),
    LeadingSum(PreparedLeadingSum),
}
impl PreparedVariant {
    fn module(&self) -> &Module {
        match self {
            Self::Pointwise(invocation) => &invocation.kernel().module,
            Self::LeadingSum(invocation) => &invocation.kernel().module,
        }
    }
}

#[pyclass(frozen, module = "torch_rs.torch_rs", name = "_PointwisePrepared")]
pub(super) struct Prepared {
    invocation: PreparedVariant,
}

#[pymethods]
impl Prepared {
    fn belongs_to(&self, executable: &Executable) -> bool {
        match (&self.invocation, &executable.kernel) {
            (PreparedVariant::Pointwise(invocation), ExecutableKernel::Pointwise(kernel)) => {
                Arc::ptr_eq(invocation.kernel(), kernel)
            }
            (PreparedVariant::LeadingSum(invocation), ExecutableKernel::LeadingSum(kernel)) => {
                Arc::ptr_eq(invocation.kernel(), kernel)
            }
            _ => false,
        }
    }
    #[getter]
    fn executable_identity<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        match &self.invocation {
            PreparedVariant::Pointwise(invocation) => invocation
                .kernel()
                .identity
                .as_ref()
                .map(|id| PyBytes::new(py, &id.bytes)),
            PreparedVariant::LeadingSum(invocation) => {
                Some(PyBytes::new(py, &invocation.kernel().identity))
            }
        }
    }
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.invocation {
            PreparedVariant::Pointwise(invocation)
                if invocation
                    .kernel()
                    .identity
                    .as_ref()
                    .is_some_and(|id| id.direct) =>
            {
                "direct"
            }
            PreparedVariant::Pointwise(_) => "vm",
            PreparedVariant::LeadingSum(_) => "leading_sum",
        }
    }
    #[getter]
    fn source(&self) -> &str {
        &self.invocation.module().source
    }
    #[getter]
    fn ptx(&self) -> &str {
        &self.invocation.module().ptx
    }
    #[getter]
    fn nvrtc_version(&self) -> (i32, i32) {
        self.invocation.module().version
    }
    #[getter]
    fn options(&self) -> Vec<String> {
        self.invocation.module().options.clone()
    }
    #[getter]
    fn device(&self) -> usize {
        self.invocation.module().device
    }
    #[getter]
    fn context(&self) -> Option<usize> {
        match &self.invocation {
            PreparedVariant::Pointwise(invocation) => {
                invocation.kernel().identity.as_ref().map(|id| id.context)
            }
            PreparedVariant::LeadingSum(invocation) => Some(invocation.kernel().module.context),
        }
    }
    #[getter]
    fn instruction_count(&self) -> PyResult<usize> {
        match &self.invocation {
            PreparedVariant::Pointwise(invocation) => Ok(invocation.instruction_count()),
            PreparedVariant::LeadingSum(_) => Err(PyNotImplementedError::new_err(
                "leading_sum has no pointwise instructions",
            )),
        }
    }
    #[getter]
    fn register_count(&self) -> PyResult<usize> {
        match &self.invocation {
            PreparedVariant::Pointwise(invocation) => Ok(invocation.register_count()),
            PreparedVariant::LeadingSum(_) => Err(PyNotImplementedError::new_err(
                "leading_sum has no pointwise registers",
            )),
        }
    }
    #[pyo3(signature = (inputs, scalars=None))]
    fn run<'py>(
        &self,
        inputs: &Bound<'py, PyTuple>,
        scalars: Option<&Bound<'_, PyTuple>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let values = parse_scalars(scalars)?;
        if matches!(self.invocation, PreparedVariant::LeadingSum(_))
            && !inputs.is_exact_instance_of::<PyTuple>()
        {
            return Err(PyTypeError::new_err(
                "expected an exact leading-sum input tuple",
            ));
        }
        with_inputs(inputs, |tensors| {
            let outputs = match &self.invocation {
                PreparedVariant::Pointwise(invocation) => invocation.run(tensors, &values),
                PreparedVariant::LeadingSum(invocation) => invocation.run(tensors, &values),
            }
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
        let shapes = match &self.invocation {
            PreparedVariant::Pointwise(invocation) => invocation.input_shapes(),
            PreparedVariant::LeadingSum(invocation) => invocation.input_shapes(),
        };
        let shapes = shapes
            .iter()
            .map(|shape| PyTuple::new(py, shape.iter().copied()))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, shapes)
    }

    #[getter]
    fn retained_bytes(&self) -> usize {
        match &self.invocation {
            PreparedVariant::Pointwise(invocation) => invocation.retained_bytes(),
            PreparedVariant::LeadingSum(invocation) => invocation.retained_bytes(),
        }
    }
}

#[cfg(test)]
mod leading_sum_descriptor_tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn leading_sum_exact_descriptor_parsing_and_certificates() {
        Python::initialize();
        Python::attach(|py| {
            for expression in [
                "('leading_sum', 0, False, 3, ('none',))",
                "('leading_sum', -2, True, None, ('dimension', 1, 7))",
                "('leading_sum', 0, False, 0, ('dimension', 0, 0))",
                "('leading_sum', 0, False, 1, ('constant', 'float', 9223372036854775808))",
                "('leading_sum', 0, False, None, ('runtime', 0, True))",
            ] {
                let expression = CString::new(expression).unwrap();
                let value = py.eval(&expression, None, None).unwrap();
                assert!(leading_sum_descriptor(&value).is_ok(), "{expression:?}");
            }
            for expression in [
                "[]",
                "('leading_sum', 0, False, 1)",
                "('other', 0, False, 1, ('none',))",
                "('leading_sum', False, False, 1, ('none',))",
                "('leading_sum', 1, False, 1, ('none',))",
                "('leading_sum', 0, 0, 1, ('none',))",
                "('leading_sum', 0, False, True, ('none',))",
                "('leading_sum', 0, False, -1, ('none',))",
                "('leading_sum', 0, False, 2**64, ('none',))",
                "('leading_sum', 0, False, 1, [])",
                "('leading_sum', 0, False, 1, ('none', 0))",
                "('leading_sum', 0, False, 1, ('constant', 'float', True))",
                "('leading_sum', 0, False, 1, ('constant', 'wrong', 0))",
                "('leading_sum', 0, False, 1, ('constant', 'bool', 1))",
                "('leading_sum', 0, False, 1, ('constant', 'int', 4609434218613702656))",
                "('leading_sum', 0, False, 1, ('runtime', True, False))",
                "('leading_sum', 0, False, 1, ('runtime', -1, False))",
                "('leading_sum', 0, False, 1, ('runtime', 0, 0))",
                "('leading_sum', 0, False, 1, ('dimension', False, 1))",
                "('leading_sum', 0, False, 1, ('dimension', 2, 1))",
                "('leading_sum', 0, False, 1, ('dimension', 0, 2))",
                "('leading_sum', 0, False, 1, ('dimension', 0, None))",
            ] {
                let expression = CString::new(expression).unwrap();
                let value = py.eval(&expression, None, None).unwrap();
                assert!(leading_sum_descriptor(&value).is_err(), "{expression:?}");
            }
        });
    }

    #[test]
    fn leading_sum_descriptor_rejects_subclasses_without_callbacks() {
        Python::initialize();
        Python::attach(|py| {
            let namespace = PyDict::new(py);
            py.run(c"calls = []\nclass BadInt(int):\n def __index__(self): calls.append('index'); raise AssertionError\n def __int__(self): calls.append('int'); raise AssertionError\nclass BadTuple(tuple):\n def __iter__(self): calls.append('iter'); raise AssertionError\nclass BadString(str):\n def __eq__(self, other): calls.append('eq'); raise AssertionError\n", Some(&namespace), None).unwrap();
            for expression in [
                "BadTuple(('leading_sum', 0, False, 1, ('none',)))",
                "('leading_sum', BadInt(0), False, 1, ('none',))",
                "(BadString('leading_sum'), 0, False, 1, ('none',))",
                "('leading_sum', 0, False, BadInt(1), ('none',))",
                "('leading_sum', 0, False, 1, BadTuple(('none',)))",
                "('leading_sum', 0, False, 1, ('constant', 'float', BadInt(0)))",
            ] {
                let expression = CString::new(expression).unwrap();
                let value = py.eval(&expression, Some(&namespace), None).unwrap();
                assert!(leading_sum_descriptor(&value).is_err());
            }
            assert_eq!(
                namespace.get_item("calls").unwrap().unwrap().len().unwrap(),
                0
            );
        });
    }
}

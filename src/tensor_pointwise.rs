//! Whole-input admission and fresh output construction for the pointwise JIT.
use super::{Arc, DType, Device, Tensor, requires_grad_flag};

#[cfg(test)]
#[path = "tensor_pointwise_identity_tests.rs"]
mod identity_tests;
use crate::{
    cuda::{PointwisePlan, jit::Kernel},
    pointwise_ir::{indexing::Layout, invalid},
    tensor_error::TensorError,
};

/// Immutable metadata and completed instructions, never inputs, results or
/// scratch. The exact native shapes certify the original Graph.indexing checks.
pub(crate) struct PreparedPointwise {
    kernel: Arc<Kernel>,
    input_shapes: Vec<Vec<usize>>,
    output: Layout,
    input_elements: [usize; 2],
    plan: PointwisePlan,
}

/// Construction-only admission certificate. It owns no input storage and is
/// released after binding; only the executable LRU retains native code identity.
pub(crate) struct HostPointwise {
    graph: crate::pointwise_ir::Graph,
    addresses: Vec<crate::pointwise_ir::indexing::Address>,
    device: usize,
    pub(crate) identity: crate::cuda::jit::ExecutableIdentity,
    source: String,
    input_shapes: Vec<Vec<usize>>,
    output: Layout,
    input_elements: [usize; 2],
    program: crate::pointwise_ir::program::Program,
}

/// Original input/Graph admission precedes interpretation of planning options.
/// This preserves admission errors even when Python's incompatible-shape hint
/// sentinel cannot be represented as a native unsigned hint.
pub(crate) struct HostAdmission {
    graph: crate::pointwise_ir::Graph,
    device: usize,
    indexing: crate::pointwise_ir::indexing::Indexing,
    input_shapes: Vec<Vec<usize>>,
}

impl HostAdmission {
    pub(crate) fn new(
        inputs: &[&Tensor],
        graph: crate::pointwise_ir::Graph,
    ) -> Result<Self, TensorError> {
        let device = Tensor::validate_pointwise_inputs(inputs)?;
        let indexing = graph.indexing(
            &inputs
                .iter()
                .map(|x| x.shape.as_slice())
                .collect::<Vec<_>>(),
        )?;
        Ok(Self {
            graph,
            device,
            indexing,
            input_shapes: inputs.iter().map(|x| x.shape.clone()).collect(),
        })
    }

    pub(crate) fn prepare(
        self,
        numerical_hint: Option<u64>,
        output_order: &[usize],
    ) -> Result<HostPointwise, TensorError> {
        let Self {
            graph,
            device,
            indexing,
            input_shapes,
        } = self;
        let program = crate::pointwise_ir::program::Program::build(
            &graph,
            &indexing.addresses,
            numerical_hint.unwrap_or(indexing.output.elements as u64),
            output_order,
            indexing.output.shape.is_empty(),
        )?;
        let direct = if indexing.output.elements == 0 {
            None
        } else {
            program.direct_source(&graph, &indexing.addresses)?
        };
        let context = Kernel::checked_context(device)?;
        let identity = crate::cuda::jit::ExecutableIdentity::new(
            &graph,
            &indexing.addresses,
            device,
            context,
            direct.as_ref().map(|_| &program),
        );
        let source = match direct {
            Some(source) => source,
            None => graph.indexed_source(&indexing.addresses)?,
        };
        let input_elements = [
            indexing.input_elements[0],
            *indexing
                .input_elements
                .get(1)
                .unwrap_or(&indexing.input_elements[0]),
        ];
        Ok(HostPointwise {
            graph,
            addresses: indexing.addresses,
            device,
            identity,
            source,
            input_shapes,
            output: indexing.output,
            input_elements,
            program,
        })
    }
}

impl HostPointwise {
    #[cfg(test)]
    pub(crate) fn new(
        inputs: &[&Tensor],
        graph: crate::pointwise_ir::Graph,
        numerical_hint: Option<u64>,
        output_order: &[usize],
    ) -> Result<Self, TensorError> {
        HostAdmission::new(inputs, graph)?.prepare(numerical_hint, output_order)
    }

    pub(crate) fn compile(&self) -> Result<Arc<Kernel>, TensorError> {
        Kernel::compile_selected(
            &self.graph,
            self.device,
            self.addresses.clone(),
            self.source.clone(),
            self.identity.clone(),
        )
    }

    pub(crate) fn bind(&self, kernel: Arc<Kernel>) -> Result<PreparedPointwise, TensorError> {
        if kernel.identity.as_ref() != Some(&self.identity) {
            return Err(invalid("host plan executable identity mismatch"));
        }
        let plan = PointwisePlan::new(&kernel, self.output.elements, &self.program)?;
        Ok(PreparedPointwise {
            kernel,
            input_shapes: self.input_shapes.clone(),
            output: Layout::new(&self.output.shape)?,
            input_elements: self.input_elements,
            plan,
        })
    }
}

impl Tensor {
    pub(crate) fn validate_pointwise_inputs(inputs: &[&Self]) -> Result<usize, TensorError> {
        let Some(first) = inputs.first() else {
            return Err(invalid("missing tensor input"));
        };
        if !(1..=2).contains(&inputs.len()) {
            return Err(invalid("expected one or two inputs"));
        }
        let Device::Cuda(device) = first.device() else {
            return Err(invalid("requires native CUDA inputs"));
        };
        for input in inputs {
            if input.device() != first.device()
                || input.dtype() != DType::Float32
                || input.requires_grad()
                || !input.is_contiguous()
            {
                return Err(invalid(
                    "requires same-device contiguous CUDA float32 tensors without gradients",
                ));
            }
            let layout = crate::pointwise_ir::indexing::Layout::new(&input.shape)?;
            if layout.elements != input.elements {
                return Err(invalid("inconsistent pointwise input element count"));
            }
            // Empty offset views are valid without forming a pointer.
            if input.elements != 0
                && input
                    .offset
                    .checked_add(input.elements)
                    .is_none_or(|end| end > input.storage.len())
            {
                return Err(invalid("input storage bounds exceeded"));
            }
        }
        Ok(device)
    }

    pub(crate) fn pointwise_jit(
        inputs: &[&Self],
        kernel: &Arc<Kernel>,
        scalars: &[f32],
        numerical_hint: Option<u64>,
        output_order: Option<&[usize]>,
    ) -> Result<Vec<Self>, TensorError> {
        kernel.validate_scalars(scalars)?;
        Self::prepare_pointwise(inputs, Arc::clone(kernel), numerical_hint, output_order)?
            .run(inputs, scalars)
    }

    pub(crate) fn prepare_pointwise(
        inputs: &[&Self],
        kernel: Arc<Kernel>,
        numerical_hint: Option<u64>,
        output_order: Option<&[usize]>,
    ) -> Result<PreparedPointwise, TensorError> {
        let device = Self::validate_pointwise_inputs(inputs)?;
        if device != kernel.device {
            return Err(invalid("kernel device guard mismatch"));
        }
        let indexing = kernel.graph.indexing(
            &inputs
                .iter()
                .map(|x| x.shape.as_slice())
                .collect::<Vec<_>>(),
        )?;
        if indexing.addresses != kernel.addresses {
            return Err(invalid("kernel broadcast indexing guard mismatch"));
        }
        let elements = indexing.output.elements;
        let numerical_hint = numerical_hint.unwrap_or(elements as u64);
        let canonical_order: Vec<_> = (0..kernel.graph.outputs.len()).collect();
        let program = crate::pointwise_ir::program::Program::build(
            &kernel.graph,
            &kernel.addresses,
            numerical_hint,
            output_order.unwrap_or(&canonical_order),
            indexing.output.shape.is_empty(),
        )?;
        let counts = [
            indexing.input_elements[0],
            *indexing
                .input_elements
                .get(1)
                .unwrap_or(&indexing.input_elements[0]),
        ];
        let plan = PointwisePlan::new(&kernel, elements, &program)?;
        Ok(PreparedPointwise {
            kernel,
            input_shapes: inputs.iter().map(|input| input.shape.clone()).collect(),
            output: indexing.output,
            input_elements: counts,
            plan,
        })
    }
}

impl PreparedPointwise {
    pub(crate) fn kernel(&self) -> &Arc<Kernel> {
        &self.kernel
    }
    pub(crate) fn instruction_count(&self) -> usize {
        self.plan.instruction_count
    }
    pub(crate) fn register_count(&self) -> usize {
        self.plan.register_count
    }

    pub(crate) fn input_shapes(&self) -> &[Vec<usize>] {
        &self.input_shapes
    }

    /// Incremental native retained bytes: inline owner and actual vector/device
    /// capacities. The shared Kernel and allocator pools are separate owners.
    /// Saturation makes an unrepresentable total uncacheable, not unexecutable.
    pub(crate) fn retained_bytes(&self) -> usize {
        let shapes = self.input_shapes.iter().fold(
            self.input_shapes
                .capacity()
                .saturating_mul(size_of::<Vec<usize>>()),
            |bytes, shape| {
                bytes.saturating_add(shape.capacity().saturating_mul(size_of::<usize>()))
            },
        );
        size_of::<Self>()
            .saturating_add(shapes)
            .saturating_add(
                self.output
                    .shape
                    .capacity()
                    .saturating_mul(size_of::<usize>()),
            )
            .saturating_add(
                self.output
                    .strides
                    .capacity()
                    .saturating_mul(size_of::<usize>()),
            )
            .saturating_add(self.plan.retained_heap_bytes())
    }

    pub(crate) fn run(
        &self,
        inputs: &[&Tensor],
        scalars: &[f32],
    ) -> Result<Vec<Tensor>, TensorError> {
        self.kernel.validate_scalars(scalars)?;
        let device = Tensor::validate_pointwise_inputs(inputs)?;
        if device != self.kernel.device {
            return Err(invalid("kernel device guard mismatch"));
        }
        if inputs.len() != self.input_shapes.len()
            || inputs
                .iter()
                .zip(&self.input_shapes)
                .any(|(input, shape)| input.shape != *shape)
        {
            return Err(invalid("prepared pointwise input shape guard mismatch"));
        }
        // Current whole-input metadata was checked above. Exact shape equality
        // reuses indexing, including admission of unused arguments and dead IR.
        // Offsets, storage pointers and runtime scalar values are never cached.
        let first = inputs[0];
        let second = inputs.get(1).copied().unwrap_or(first);
        let elements = self.output.elements;
        let storages = first.storage.cuda_pointwise_jit(
            &second.storage,
            [first.offset, second.offset],
            elements,
            self.input_elements,
            &self.kernel,
            scalars,
            &self.plan,
        )?;
        Ok(storages
            .into_iter()
            .map(|storage| Tensor {
                storage: Arc::new(storage),
                shape: self.output.shape.clone(),
                strides: self.output.strides.clone(),
                offset: 0,
                elements,
                output_nr: 0,
                leaf_requires_grad: requires_grad_flag(false),
                view_requires_grad: None,
                autograd: None,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pointwise_ir::{Graph, Node};

    #[test]
    fn prepared_reuse_skips_build_and_upload_but_uses_current_inputs_and_scalars() {
        if crate::cuda::device_count() == 0 {
            eprintln!("skipping prepared pointwise reuse: CUDA unavailable");
            return;
        }
        let mut input = Tensor::from_vec(vec![1., -2., 3., 4., 5., -6.], [6])
            .unwrap()
            .try_copy_cpu_to_cuda(Device::Cuda(0))
            .unwrap();
        input.shape = vec![3];
        input.elements = 3;
        let graph = Graph {
            inputs: 1,
            nodes: vec![
                Node::Input(0),
                Node::RuntimeScalar(0, false),
                Node::Mul(0, 1),
            ],
            outputs: vec![2],
        };
        let kernel = Kernel::compile(&graph, 0).unwrap();
        let kernel_owner = Arc::downgrade(&kernel);
        let builds = crate::pointwise_ir::program::build_count();
        let uploads = crate::cuda::pointwise_upload_count();
        let prepared =
            Tensor::prepare_pointwise(&[&input], Arc::clone(&kernel), None, None).unwrap();
        assert_eq!(crate::pointwise_ir::program::build_count(), builds + 1);
        assert_eq!(crate::cuda::pointwise_upload_count(), uploads + 1);
        assert_eq!(prepared.input_shapes(), &[vec![3]]);
        assert_eq!(Arc::strong_count(&input.storage), 1);
        assert_eq!(Arc::strong_count(&kernel), 2);
        let bytes = prepared.retained_bytes();
        assert!(bytes > size_of::<PreparedPointwise>());
        let first = prepared.run(&[&input], &[2.]).unwrap().remove(0);
        input.offset = 3;
        let second = prepared.run(&[&input], &[-2.]).unwrap().remove(0);
        assert_eq!(
            first.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
            [2., -4., 6.]
        );
        assert_eq!(
            second.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
            [-8., -10., 12.]
        );
        assert!(!first.shares_storage_with(&second));
        assert!(!first.shares_storage_with(&input));
        assert!(!second.shares_storage_with(&input));
        assert_eq!(prepared.retained_bytes(), bytes);
        assert_eq!(crate::pointwise_ir::program::build_count(), builds + 1);
        assert_eq!(crate::cuda::pointwise_upload_count(), uploads + 1);
        let input_owner = Arc::downgrade(&input.storage);
        let first_owner = Arc::downgrade(&first.storage);
        drop((input, first, second, kernel));
        assert!(input_owner.upgrade().is_none());
        assert!(first_owner.upgrade().is_none());
        assert_eq!(kernel_owner.strong_count(), 1);
        drop(prepared);
        assert!(kernel_owner.upgrade().is_none());
    }

    #[test]
    fn prepared_run_revalidates_current_admission_and_exact_shape_not_just_count() {
        if crate::cuda::device_count() == 0 {
            eprintln!("skipping prepared pointwise admission: CUDA unavailable");
            return;
        }
        let graph = Graph {
            inputs: 1,
            nodes: vec![Node::Input(0), Node::Neg(0)],
            outputs: vec![1],
        };
        let kernel = Kernel::compile(&graph, 0).unwrap();
        let mut input = Tensor::from_vec(vec![1., 2., 3., 4.], [2, 2])
            .unwrap()
            .try_copy_cpu_to_cuda(Device::Cuda(0))
            .unwrap();
        let prepared = Tensor::prepare_pointwise(&[&input], kernel, None, None).unwrap();
        for phase in ["grad", "count", "offset", "strides", "shape"] {
            match phase {
                "grad" => input.view_requires_grad = Some(requires_grad_flag(true)),
                "count" => input.elements = 3,
                "offset" => input.offset = usize::MAX,
                "strides" => input.strides = vec![1, 2],
                "shape" => {
                    input.shape = vec![4];
                    input.strides = vec![1];
                }
                _ => unreachable!(),
            }
            assert!(prepared.run(&[&input], &[]).is_err(), "{phase}");
            input.view_requires_grad = None;
            input.elements = 4;
            input.offset = 0;
            input.shape = vec![2, 2];
            input.strides = vec![2, 1];
        }
        assert!(prepared.run(&[], &[]).is_err());
        assert!(prepared.run(&[&input, &input], &[]).is_err());
        assert!(prepared.run(&[&input], &[1.]).is_err());
        let cpu = Tensor::from_vec(vec![1., 2., 3., 4.], [2, 2]).unwrap();
        assert!(prepared.run(&[&cpu], &[]).is_err());
        assert_eq!(prepared.run(&[&input], &[]).unwrap()[0].shape(), &[2, 2]);
    }

    #[test]
    fn empty_preparation_checks_unused_nonempty_inputs_without_upload() {
        if crate::cuda::device_count() == 0 {
            eprintln!("skipping empty prepared pointwise: CUDA unavailable");
            return;
        }
        let unused = Tensor::from_vec(vec![1., 2., 3.], [3])
            .unwrap()
            .try_copy_cpu_to_cuda(Device::Cuda(0))
            .unwrap();
        let mut empty = Tensor::from_vec(vec![], [0])
            .unwrap()
            .try_copy_cpu_to_cuda(Device::Cuda(0))
            .unwrap();
        empty.offset = usize::MAX;
        let graph = Graph {
            inputs: 2,
            nodes: vec![Node::Input(0), Node::Input(1), Node::Neg(1)],
            outputs: vec![2],
        };
        let indexing = graph.indexing(&[&[3], &[0]]).unwrap();
        let kernel = Kernel::compile_indexed(&graph, 0, indexing.addresses).unwrap();
        let uploads = crate::cuda::pointwise_upload_count();
        let prepared = Tensor::prepare_pointwise(&[&unused, &empty], kernel, None, None).unwrap();
        assert_eq!(prepared.input_shapes(), &[vec![3], vec![0]]);
        for _ in 0..2 {
            let output = prepared.run(&[&unused, &empty], &[]).unwrap().remove(0);
            assert_eq!(output.shape(), &[0]);
            assert!(!output.shares_storage_with(&empty));
        }
        assert_eq!(crate::cuda::pointwise_upload_count(), uploads);
        let changed_unused = Tensor::from_vec(vec![1., 2.], [2])
            .unwrap()
            .try_copy_cpu_to_cuda(Device::Cuda(0))
            .unwrap();
        assert!(prepared.run(&[&changed_unused, &empty], &[]).is_err());
        assert!(prepared.run(&[&unused, &unused], &[]).is_err());
    }

    #[test]
    fn prepared_calls_can_share_only_immutable_data_across_threads() {
        if crate::cuda::device_count() == 0 {
            eprintln!("skipping concurrent prepared pointwise: CUDA unavailable");
            return;
        }
        let input = Tensor::from_vec(vec![1.; 257], [257])
            .unwrap()
            .try_copy_cpu_to_cuda(Device::Cuda(0))
            .unwrap();
        let graph = Graph {
            inputs: 1,
            nodes: vec![Node::Input(0), Node::Mul(0, 0), Node::Neg(1)],
            outputs: vec![1, 2],
        };
        let kernel = Kernel::compile(&graph, 0).unwrap();
        let prepared = Tensor::prepare_pointwise(&[&input], kernel, None, None).unwrap();
        let results = std::thread::scope(|scope| {
            [2., 3.]
                .map(|value| {
                    let prepared = &prepared;
                    scope.spawn(move || {
                        let input = Tensor::from_vec(vec![value; 257], [257])
                            .unwrap()
                            .try_copy_cpu_to_cuda(Device::Cuda(0))
                            .unwrap();
                        prepared.run(&[&input], &[]).unwrap()
                    })
                })
                .map(|thread| thread.join().unwrap())
        });
        for (outputs, square) in results.iter().zip([4., 9.]) {
            for (output, expected) in outputs.iter().zip([square, -square]) {
                assert_eq!(
                    output.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
                    vec![expected; 257]
                );
            }
        }
        for output in &results[0] {
            assert!(
                results[1]
                    .iter()
                    .all(|other| !output.shares_storage_with(other))
            );
        }
    }

    #[cfg(feature = "python-bindings")]
    #[test]
    fn prepared_conversion_failure_keeps_plan_and_module_owned() {
        use pyo3::{exceptions::PyMemoryError, prelude::*};
        if crate::cuda::device_count() == 0 {
            eprintln!("skipping prepared conversion ownership: CUDA unavailable");
            return;
        }
        let input = Tensor::from_vec(vec![1., 2.], [2])
            .unwrap()
            .try_copy_cpu_to_cuda(Device::Cuda(0))
            .unwrap();
        let graph = Graph {
            inputs: 1,
            nodes: vec![Node::Input(0), Node::Neg(0), Node::Mul(0, 0)],
            outputs: vec![1, 2],
        };
        let kernel = Kernel::compile(&graph, 0).unwrap();
        let prepared =
            Tensor::prepare_pointwise(&[&input], Arc::clone(&kernel), None, None).unwrap();
        Python::initialize();
        Python::attach(|py| {
            let outputs = prepared.run(&[&input], &[]).unwrap();
            let owners: Vec<_> = outputs
                .iter()
                .map(|output| Arc::downgrade(&output.storage))
                .collect();
            let mut conversions = 0;
            let result = crate::python::convert_outputs(py, outputs, |output| {
                conversions += 1;
                assert_eq!(Arc::strong_count(&kernel), 2);
                assert_eq!(Arc::strong_count(&input.storage), 1);
                assert!(owners.iter().all(|owner| owner.strong_count() == 1));
                if conversions == 2 {
                    Err(PyMemoryError::new_err(
                        "injected prepared conversion failure",
                    ))
                } else {
                    Py::new(py, crate::python::PyTensor::new(output))
                }
            });
            assert!(result.unwrap_err().is_instance_of::<PyMemoryError>(py));
            assert!(owners.iter().all(|owner| owner.upgrade().is_none()));
        });
        assert_eq!(Arc::strong_count(&kernel), 2);
        assert_eq!(prepared.run(&[&input], &[]).unwrap().len(), 2);
    }

    #[cfg(feature = "python-bindings")]
    #[test]
    fn python_conversion_failure_drops_converted_and_pending_output_owners() {
        use pyo3::{exceptions::PyMemoryError, prelude::*};
        Python::initialize();
        Python::attach(|py| {
            let outputs: Vec<_> = (0..3)
                .map(|_| Tensor::from_vec(vec![1.], [1]).unwrap())
                .collect();
            let storage: Vec<_> = outputs
                .iter()
                .map(|output| Arc::downgrade(&output.storage))
                .collect();
            let mut conversions = 0;
            let result = crate::python::convert_outputs(py, outputs, |output| {
                conversions += 1;
                // The first converted object and every pending tensor still own
                // their storage when conversion of the second leaf fails.
                assert!(storage.iter().all(|owner| owner.strong_count() == 1));
                if conversions == 2 {
                    Err(PyMemoryError::new_err(
                        "injected second output conversion failure",
                    ))
                } else {
                    Py::new(py, crate::python::PyTensor::new(output))
                }
            });
            assert!(result.unwrap_err().is_instance_of::<PyMemoryError>(py));
            assert_eq!(conversions, 2);
            assert!(storage.iter().all(|owner| owner.upgrade().is_none()));
        });
    }

    #[test]
    fn multiple_outputs_own_distinct_storage_and_survive_subsequent_calls() {
        if crate::cuda::device_count() == 0 {
            eprintln!("skipping multi-output storage: CUDA unavailable");
            return;
        }
        let input = Tensor::from_vec(vec![1., -2., 3.], [3])
            .unwrap()
            .try_copy_cpu_to_cuda(Device::Cuda(0))
            .unwrap();
        let graph = Graph {
            inputs: 1,
            nodes: vec![Node::Input(0), Node::Neg(0), Node::Neg(0), Node::Mul(1, 1)],
            outputs: vec![1, 2, 3],
        };
        let kernel = Kernel::compile(&graph, 0).unwrap();
        let first = Tensor::pointwise_jit(&[&input], &kernel, &[], None, None).unwrap();
        let changed = Tensor::from_vec(vec![4., 5., -6.], [3])
            .unwrap()
            .try_copy_cpu_to_cuda(Device::Cuda(0))
            .unwrap();
        let second = Tensor::pointwise_jit(&[&changed], &kernel, &[], None, None).unwrap();
        for (call, expected) in [(&first, [-1., 2., -3.]), (&second, [-4., -5., 6.])] {
            for (i, output) in call.iter().enumerate() {
                assert!(!input.shares_storage_with(output));
                for sibling in &call[..i] {
                    assert!(!sibling.shares_storage_with(output));
                }
                for old in &first {
                    if std::ptr::eq(call, &raw const second) {
                        assert!(!old.shares_storage_with(output));
                    }
                }
                let expected = if i == 2 {
                    expected.map(|value| value * value)
                } else {
                    expected
                };
                assert_eq!(
                    output.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
                    expected
                );
            }
        }
    }

    #[test]
    fn numerical_hint_selects_plan_without_changing_output_extent() {
        if crate::cuda::device_count() == 0 {
            eprintln!("skipping pointwise numerical hint: CUDA unavailable");
            return;
        }
        let graph = Graph {
            inputs: 1,
            nodes: vec![Node::Input(0), Node::Mul(0, 0), Node::Neg(1), Node::Sin(1)],
            outputs: vec![2, 3],
        };
        let kernel = Kernel::compile(&graph, 0).unwrap();
        for elements in [1, 13] {
            let input = Tensor::from_vec(vec![1e-38; elements], [elements])
                .unwrap()
                .try_copy_cpu_to_cuda(Device::Cuda(0))
                .unwrap();
            for hint in [None, Some(1), Some(13)] {
                let outputs = Tensor::pointwise_jit(&[&input], &kernel, &[], hint, None).unwrap();
                assert_eq!(outputs.len(), 2);
                let negative_zero = hint.unwrap_or(elements as u64) < 3;
                for (slot, output) in outputs.iter().enumerate() {
                    assert_eq!(output.shape(), &[elements]);
                    let values = output.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap();
                    assert_eq!(values.len(), elements);
                    let expected = if slot == 0 && negative_zero {
                        -0.0f32
                    } else {
                        0.0f32
                    };
                    assert!(
                        values
                            .iter()
                            .all(|value| value.to_bits() == expected.to_bits())
                    );
                }
            }
        }
    }

    #[test]
    fn admission_rejects_gradients_ranges_and_layouts_before_jit() {
        if crate::cuda::device_count() == 0 {
            eprintln!("skipping pointwise storage admission: CUDA unavailable");
            return;
        }
        let mut input = Tensor::from_vec(vec![1., 2., 3., 4.], [2, 2])
            .unwrap()
            .try_copy_cpu_to_cuda(Device::Cuda(0))
            .unwrap();
        input.view_requires_grad = Some(requires_grad_flag(true));
        assert!(Tensor::validate_pointwise_inputs(&[&input]).is_err());
        input.view_requires_grad = None;
        input.elements = 3;
        assert!(Tensor::validate_pointwise_inputs(&[&input]).is_err());
        input.elements = 4;
        input.offset = usize::MAX;
        assert!(Tensor::validate_pointwise_inputs(&[&input]).is_err());
        input.offset = 0;
        input.strides = vec![1, 2];
        assert!(Tensor::validate_pointwise_inputs(&[&input]).is_err());
        input.strides = vec![2, 1];
        let graph = Graph {
            inputs: 1,
            nodes: vec![Node::Input(0), Node::Neg(0)],
            outputs: vec![1],
        };
        let kernel = Kernel::compile(&graph, 0).unwrap();
        let output = Tensor::pointwise_jit(&[&input], &kernel, &[], None, None)
            .unwrap()
            .remove(0);
        assert!(!input.shares_storage_with(&output));
        assert_eq!(
            output.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
            vec![-1., -2., -3., -4.]
        );
        input.elements = 0;
        input.shape = vec![0];
        input.strides = vec![1];
        input.offset = usize::MAX;
        let empty = Tensor::pointwise_jit(&[&input], &kernel, &[], None, None)
            .unwrap()
            .remove(0);
        assert!(!input.shares_storage_with(&empty));
        assert_eq!(empty.shape(), &[0]);
    }
}

//! Whole-input admission and fresh output construction for the pointwise JIT.
use super::{Arc, DType, Device, Tensor, requires_grad_flag};
use crate::{cuda::jit::Kernel, pointwise_ir::invalid, tensor_error::TensorError};

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
        kernel: &Kernel,
        scalars: &[f32],
        numerical_hint: Option<u64>,
        output_order: Option<&[usize]>,
    ) -> Result<Vec<Self>, TensorError> {
        kernel.validate_scalars(scalars)?;
        let device = Self::validate_pointwise_inputs(inputs)?;
        if device != kernel.device {
            return Err(invalid("kernel device guard mismatch"));
        }
        let first = inputs[0];
        let second = inputs.get(1).copied().unwrap_or(first);
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
        let shape = indexing.output.shape;
        let strides = indexing.output.strides;
        let counts = [
            indexing.input_elements[0],
            *indexing
                .input_elements
                .get(1)
                .unwrap_or(&indexing.input_elements[0]),
        ];
        let storages = first.storage.cuda_pointwise_jit(
            &second.storage,
            [first.offset, second.offset],
            elements,
            counts,
            kernel,
            scalars,
            &program,
        )?;
        Ok(storages
            .into_iter()
            .map(|storage| Self {
                storage: Arc::new(storage),
                shape: shape.clone(),
                strides: strides.clone(),
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

//! Whole-input admission and fresh output construction for the pointwise JIT.
use super::{Arc, DType, Device, Tensor, contiguous_strides, requires_grad_flag};
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
                || input.shape != first.shape
            {
                return Err(invalid(
                    "requires same-shape, same-device contiguous CUDA float32 tensors without gradients",
                ));
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
        // Validate output layout before compilation, allocation or launch.
        contiguous_strides(&first.shape, first.elements)?;
        Ok(device)
    }

    pub(crate) fn pointwise_jit(inputs: &[&Self], kernel: &Kernel) -> Result<Self, TensorError> {
        let device = Self::validate_pointwise_inputs(inputs)?;
        if device != kernel.device {
            return Err(invalid("kernel device guard mismatch"));
        }
        let first = inputs[0];
        let second = inputs.get(1).copied().unwrap_or(first);
        let shape = first.shape.clone();
        let strides = contiguous_strides(&shape, first.elements)?;
        let storage = first.storage.cuda_pointwise_jit(
            first.offset,
            &second.storage,
            second.offset,
            first.elements,
            kernel,
        )?;
        Ok(Self {
            storage: Arc::new(storage),
            shape,
            strides,
            offset: 0,
            elements: first.elements,
            output_nr: 0,
            leaf_requires_grad: requires_grad_flag(false),
            view_requires_grad: None,
            autograd: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pointwise_ir::{Graph, Node};

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
        input.offset = usize::MAX;
        assert!(Tensor::validate_pointwise_inputs(&[&input]).is_err());
        input.offset = 0;
        input.strides = vec![1, 2];
        assert!(Tensor::validate_pointwise_inputs(&[&input]).is_err());
        input.strides = vec![2, 1];
        let graph = Graph {
            inputs: 1,
            nodes: vec![Node::Input(0), Node::Neg(0)],
            output: 1,
        };
        let kernel = Kernel::compile(&graph, 0).unwrap();
        let output = Tensor::pointwise_jit(&[&input], &kernel).unwrap();
        assert!(!input.shares_storage_with(&output));
        assert_eq!(
            output.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
            vec![-1., -2., -3., -4.]
        );
        input.elements = 0;
        input.shape = vec![0];
        input.strides = vec![1];
        input.offset = usize::MAX;
        let empty = Tensor::pointwise_jit(&[&input], &kernel).unwrap();
        assert!(!input.shares_storage_with(&empty));
        assert_eq!(empty.shape(), &[0]);
    }
}

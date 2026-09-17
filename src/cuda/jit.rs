//! NVRTC compilation and owned driver modules for generated pointwise kernels.
//! Storage owners and the runtime device guard must outlive every launch/wait.
use super::super::{TensorError, c_void};
use super::driver;
use super::jit_module::Module;
#[cfg(test)]
use super::jit_module::Nvrtc;
use crate::pointwise_ir::{Graph, invalid};
use std::sync::Arc;

#[path = "pointwise_identity.rs"]
mod identity;
pub(crate) use identity::ExecutableIdentity;

/// Scratch is register-major and bounded independently of tensor extent.
/// The VM uses grid-stride iteration, so fewer workers never change indexing.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LaunchLayout {
    pub(crate) blocks: u32,
    pub(crate) threads: u32,
    pub(crate) workers: usize,
    pub(crate) scratch_elements: usize,
}
impl LaunchLayout {
    const SCRATCH_BYTES: usize = 64 * 1024 * 1024;

    pub(crate) fn new(elements: usize, registers: usize) -> Result<Self, TensorError> {
        let register_bytes = registers
            .checked_mul(std::mem::size_of::<f32>())
            .filter(|bytes| *bytes != 0)
            .ok_or(TensorError::IndexCalculationOverflow)?;
        if elements == 0 {
            return Ok(Self {
                blocks: 0,
                threads: 0,
                workers: 0,
                scratch_elements: 0,
            });
        }
        let budget_workers = Self::SCRATCH_BYTES / register_bytes;
        let threads = budget_workers.min(256);
        if threads == 0 {
            return Err(invalid("pointwise register storage exceeds scratch budget"));
        }
        let blocks = elements
            .div_ceil(threads)
            .min(65535)
            .min(budget_workers / threads);
        let workers = blocks
            .checked_mul(threads)
            .ok_or(TensorError::IndexCalculationOverflow)?;
        let scratch_elements = workers
            .checked_mul(registers)
            .ok_or(TensorError::IndexCalculationOverflow)?;
        Ok(Self {
            blocks: u32::try_from(blocks).map_err(|_| TensorError::IndexCalculationOverflow)?,
            threads: u32::try_from(threads).map_err(|_| TensorError::IndexCalculationOverflow)?,
            workers,
            scratch_elements,
        })
    }
}

pub(crate) struct Kernel {
    pub(crate) identity: Option<ExecutableIdentity>,
    pub(crate) graph: Graph,
    pub(crate) addresses: Vec<crate::pointwise_ir::indexing::Address>,
    scalar_count: usize,
    pub(crate) module: Module,
}
impl Kernel {
    #[cfg(test)]
    pub(crate) fn compile(graph: &Graph, device: usize) -> Result<Arc<Self>, TensorError> {
        Self::compile_indexed(
            graph,
            device,
            vec![crate::pointwise_ir::indexing::Address::Linear; graph.inputs],
        )
    }
    pub(crate) fn compile_indexed(
        graph: &Graph,
        device: usize,
        addresses: Vec<crate::pointwise_ir::indexing::Address>,
    ) -> Result<Arc<Self>, TensorError> {
        let compilation = graph.compilation(&addresses)?;
        Self::compile_source(graph, device, addresses, compilation, None)
    }

    pub(crate) fn checked_context(device: usize) -> Result<usize, TensorError> {
        Module::checked_context(device)
    }

    pub(crate) fn compile_selected(
        graph: &Graph,
        device: usize,
        addresses: Vec<crate::pointwise_ir::indexing::Address>,
        compilation: crate::pointwise_ir::program::Compilation,
        identity: ExecutableIdentity,
    ) -> Result<Arc<Self>, TensorError> {
        Self::compile_source(graph, device, addresses, compilation, Some(identity))
    }

    fn compile_source(
        graph: &Graph,
        device: usize,
        addresses: Vec<crate::pointwise_ir::indexing::Address>,
        compilation: crate::pointwise_ir::program::Compilation,
        identity: Option<ExecutableIdentity>,
    ) -> Result<Arc<Self>, TensorError> {
        let module = Module::compile(
            compilation.source,
            device,
            identity.as_ref().map(|id| id.context),
            compilation.erf,
            c"torch_rs_pointwise",
        )?;
        Ok(Arc::new(Self {
            identity,
            graph: graph.clone(),
            addresses,
            scalar_count: graph.scalar_count(),
            module,
        }))
    }
    pub(crate) fn validate_scalars(&self, scalars: &[f32]) -> Result<(), TensorError> {
        if scalars.len() != self.scalar_count {
            return Err(invalid("pointwise runtime scalar arity mismatch"));
        }
        Ok(())
    }
    /// Call under the runtime device guard, before uploading or using plan data.
    pub(crate) fn validate_context(&self) -> Result<(), TensorError> {
        self.module.validate_context()
    }
    /// Input pointers must cover the validated broadcast address maps on this
    /// device. Each output covers count disjoint elements. Synchronize before releasing owners.
    #[allow(clippy::too_many_arguments)] // One checked native launch ABI.
    pub(crate) unsafe fn launch(
        &self,
        mut x0: u64,
        mut x1: u64,
        outputs: &[u64],
        mut count: u64,
        mut program: u64,
        mut instruction_count: u64,
        mut scratch: u64,
        layout: LaunchLayout,
        scalars: &[f32],
    ) -> Result<(), TensorError> {
        self.validate_scalars(scalars)?;
        self.validate_context()?;
        if outputs.len() != self.graph.outputs.len() {
            return Err(invalid("pointwise output pointer arity mismatch"));
        }
        let driver = driver()?;
        let mut output_values = outputs.to_vec();
        let mut args = vec![(&raw mut x0).cast(), (&raw mut x1).cast()];
        args.extend(
            output_values
                .iter_mut()
                .map(|value| std::ptr::from_mut(value).cast()),
        );
        args.push((&raw mut count).cast());
        args.push((&raw mut program).cast());
        args.push((&raw mut instruction_count).cast());
        args.push((&raw mut scratch).cast());
        let mut scratch_stride = layout.workers as u64;
        args.push((&raw mut scratch_stride).cast());
        // Driver arguments reference stable host values until cuLaunchKernel
        // has copied them. No scalar device buffer or cached value is retained.
        let mut scalar_values = scalars.to_vec();
        args.extend(
            scalar_values
                .iter_mut()
                .map(|value| std::ptr::from_mut(value).cast()),
        );
        // SAFETY: caller holds all checked storage through legacy-stream completion.
        driver.check(
            unsafe {
                (driver.launch)(
                    self.function as *mut c_void,
                    layout.blocks,
                    1,
                    1,
                    layout.threads,
                    1,
                    1,
                    0,
                    std::ptr::without_provenance_mut(1),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }
}
impl std::ops::Deref for Kernel {
    type Target = Module;
    fn deref(&self) -> &Module {
        &self.module
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pointwise_ir::Node;

    #[test]
    fn preparation_rejects_kernel_context_mismatch_before_upload() {
        if crate::cuda::device_count() == 0 {
            eprintln!("skipping prepared context admission: CUDA unavailable");
            return;
        }
        let input = crate::Tensor::from_vec(vec![1.], [1])
            .unwrap()
            .try_copy_cpu_to_cuda(crate::Device::Cuda(0))
            .unwrap();
        let graph = Graph {
            inputs: 1,
            nodes: vec![Node::Input(0), Node::Neg(0)],
            outputs: vec![1],
        };
        let mut kernel = Kernel::compile(&graph, 0).unwrap();
        let context = kernel.context;
        Arc::get_mut(&mut kernel).unwrap().module.context = usize::MAX;
        let uploads = crate::cuda::pointwise_upload_count();
        let result = crate::Tensor::prepare_pointwise(&[&input], Arc::clone(&kernel), None, None);
        let rejected =
            matches!(&result, Err(error) if error.to_string().contains("context mismatch"));
        drop(result);
        // Restore the real module context before any assertion can unwind.
        Arc::get_mut(&mut kernel).unwrap().module.context = context;
        assert!(rejected);
        assert_eq!(crate::cuda::pointwise_upload_count(), uploads);
        assert!(crate::Tensor::prepare_pointwise(&[&input], kernel, None, None).is_ok());
    }

    #[test]
    fn vm_scratch_is_bounded_for_large_graphs_and_extents() {
        for registers in [1, 4096, 4096 * 64] {
            for elements in [0, 1, 257, usize::MAX] {
                let layout = LaunchLayout::new(elements, registers).unwrap();
                assert_eq!(
                    layout.workers,
                    layout.blocks as usize * layout.threads as usize
                );
                assert_eq!(layout.scratch_elements, layout.workers * registers);
                assert!(layout.scratch_elements * 4 <= LaunchLayout::SCRATCH_BYTES);
                assert_eq!(layout.workers == 0, elements == 0);
            }
        }
        assert!(LaunchLayout::new(1, usize::MAX).is_err());
        assert!(LaunchLayout::new(1, 0).is_err());
    }

    #[test]
    fn failed_nvrtc_compilation_destroys_program_and_allows_retry() {
        if crate::cuda::device_count() == 0 {
            eprintln!("skipping NVRTC failure/retry: CUDA unavailable");
            return;
        }
        let compiler = Nvrtc::load().unwrap();
        let failure = compiler.compile("this is not CUDA C", &[]).unwrap_err();
        assert!(failure.to_string().contains("NVRTC compilation failed"));
        let graph = Graph {
            inputs: 1,
            nodes: vec![Node::Input(0), Node::Sin(0)],
            outputs: vec![1],
        };
        let kernel = Kernel::compile(&graph, 0).unwrap();
        assert!(kernel.ptx.contains("torch_rs_pointwise"));
        assert_eq!(kernel.source, graph.source().unwrap());
        assert!(kernel.version.0 >= 12);
        assert!(
            kernel
                .options
                .iter()
                .any(|x| x.starts_with("--gpu-architecture=compute_"))
        );
        assert!(kernel.options.contains(&"--ftz=false".into()));
        assert!(!kernel.options.contains(&"--use_fast_math".into()));
        assert!(
            !kernel
                .options
                .contains(&"--relocatable-device-code=true".into())
        );
        assert!(!kernel.source.contains("torch_rs_erf"));
    }
}

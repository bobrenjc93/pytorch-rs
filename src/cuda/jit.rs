//! NVRTC compilation and owned driver modules for generated pointwise kernels.
//! Storage owners and the runtime device guard must outlive every launch/wait.
use super::super::{Library, TensorError, c_char, c_int, c_void, runtime};
use super::{current_context, driver};
use crate::pointwise_ir::{Graph, invalid};
use std::ffi::{CStr, CString};
use std::sync::Arc;

#[path = "link.rs"]
mod link;

type Program = *mut c_void;
struct Nvrtc {
    _library: Library,
    version: unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int,
    create: unsafe extern "C" fn(
        *mut Program,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> c_int,
    compile: unsafe extern "C" fn(Program, c_int, *const *const c_char) -> c_int,
    log_size: unsafe extern "C" fn(Program, *mut usize) -> c_int,
    log: unsafe extern "C" fn(Program, *mut c_char) -> c_int,
    ptx_size: unsafe extern "C" fn(Program, *mut usize) -> c_int,
    ptx: unsafe extern "C" fn(Program, *mut c_char) -> c_int,
    destroy: unsafe extern "C" fn(*mut Program) -> c_int,
}
struct Owner<'a>(&'a Nvrtc, Program);
impl Drop for Owner<'_> {
    fn drop(&mut self) {
        // SAFETY: this owner exclusively owns the successfully created program.
        unsafe {
            (self.0.destroy)(&raw mut self.1);
        }
    }
}

impl Nvrtc {
    fn load() -> Result<Self, TensorError> {
        let paths = std::env::var("TORCH_RS_NVRTC").map_or_else(
            |_| {
                vec![
                    "libnvrtc.so.13".into(),
                    "libnvrtc.so.12".into(),
                    "libnvrtc.so".into(),
                ]
            },
            |path| vec![path],
        );
        let mut errors = Vec::new();
        for path in paths {
            // SAFETY: documented NVRTC C ABI; the library owns all symbols.
            let loaded = unsafe {
                (|| -> Result<Self, libloading::Error> {
                    let library = Library::new(&path)?;
                    Ok(Self {
                        version: *library.get(b"nvrtcVersion\0")?,
                        create: *library.get(b"nvrtcCreateProgram\0")?,
                        compile: *library.get(b"nvrtcCompileProgram\0")?,
                        log_size: *library.get(b"nvrtcGetProgramLogSize\0")?,
                        log: *library.get(b"nvrtcGetProgramLog\0")?,
                        ptx_size: *library.get(b"nvrtcGetPTXSize\0")?,
                        ptx: *library.get(b"nvrtcGetPTX\0")?,
                        destroy: *library.get(b"nvrtcDestroyProgram\0")?,
                        _library: library,
                    })
                })()
            };
            match loaded {
                Ok(api) => return Ok(api),
                Err(error) => errors.push(format!("{path}: {error}")),
            }
        }
        Err(invalid(format!(
            "cannot load NVRTC (set TORCH_RS_NVRTC): {}",
            errors.join("; ")
        )))
    }
    fn check(status: c_int) -> Result<(), TensorError> {
        if status == 0 {
            Ok(())
        } else {
            Err(invalid(format!("NVRTC status {status}")))
        }
    }
    fn compile(
        &self,
        source: &str,
        options: &[String],
    ) -> Result<(Vec<u8>, (i32, i32)), TensorError> {
        let source = CString::new(source).map_err(|_| invalid("NUL in generated source"))?;
        let options: Vec<_> = options
            .iter()
            .map(|s| CString::new(s.as_str()).unwrap())
            .collect();
        let pointers: Vec<_> = options.iter().map(|s| s.as_ptr()).collect();
        let mut program = std::ptr::null_mut();
        let mut major = 0;
        let mut minor = 0;
        // SAFETY: valid C strings, documented ABI, and writable program handle.
        unsafe {
            Self::check((self.version)(&raw mut major, &raw mut minor))?;
            Self::check((self.create)(
                &raw mut program,
                source.as_ptr(),
                c"pointwise.cu".as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            ))?;
        }
        let owner = Owner(self, program);
        // SAFETY: live program, option strings and correctly sized writable buffers.
        unsafe {
            let status = (self.compile)(
                owner.1,
                c_int::try_from(pointers.len()).unwrap(),
                pointers.as_ptr(),
            );
            if status != 0 {
                let mut size = 0;
                Self::check((self.log_size)(owner.1, &raw mut size))?;
                let mut log = vec![0_u8; size.max(1)];
                Self::check((self.log)(owner.1, log.as_mut_ptr().cast()))?;
                return Err(invalid(format!(
                    "NVRTC compilation failed ({status}): {}",
                    String::from_utf8_lossy(&log)
                )));
            }
            let mut size = 0;
            Self::check((self.ptx_size)(owner.1, &raw mut size))?;
            let mut ptx = vec![0_u8; size];
            Self::check((self.ptx)(owner.1, ptx.as_mut_ptr().cast()))?;
            Ok((ptx, (major, minor)))
        }
    }
}

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
    pub(crate) graph: Graph,
    pub(crate) addresses: Vec<crate::pointwise_ir::indexing::Address>,
    pub(crate) device: usize,
    scalar_count: usize,
    context: usize,
    module: usize,
    function: usize,
    pub(crate) source: String,
    pub(crate) ptx: String,
    pub(crate) version: (i32, i32),
    pub(crate) options: Vec<String>,
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
        let source = compilation.source;
        let _guard = runtime()?.guard(device)?;
        let context = current_context()?;
        let driver = driver()?;
        let mut ordinal = 0;
        let mut major = 0;
        let mut minor = 0;
        // SAFETY: documented attributes and valid writable integers under device guard.
        unsafe {
            driver.check((driver.device)(&raw mut ordinal), "cuCtxGetDevice")?;
            driver.check(
                (driver.attribute)(&raw mut major, 75, ordinal),
                "cuDeviceGetAttribute",
            )?;
            driver.check(
                (driver.attribute)(&raw mut minor, 76, ordinal),
                "cuDeviceGetAttribute",
            )?;
        }
        let mut options = vec![
            format!("--gpu-architecture=compute_{major}{minor}"),
            "--fmad=true".into(),
            "--ftz=false".into(),
            "--prec-div=true".into(),
            "--prec-sqrt=true".into(),
        ];
        if compilation.erf {
            options.push("--relocatable-device-code=true".into());
        }
        let (ptx, version) = Nvrtc::load()?.compile(&source, &options)?;
        let ptx_text = CStr::from_bytes_until_nul(&ptx)
            .map_err(|_| invalid("invalid NVRTC PTX"))?
            .to_string_lossy()
            .into_owned();
        let (module, function) = if compilation.erf {
            link::load(
                driver,
                &ptx,
                concat!(include_str!("erf_provider.ptx"), "\0").as_bytes(),
                c"torch_rs_pointwise",
            )?
        } else {
            // SAFETY: NUL-terminated NVRTC PTX lives through module loading.
            unsafe { load_module(driver, ptx.as_ptr().cast(), c"torch_rs_pointwise")? }
        };
        Ok(Arc::new(Self {
            graph: graph.clone(),
            addresses,
            device,
            scalar_count: graph.scalar_count(),
            context,
            module,
            function,
            source,
            ptx: ptx_text,
            version,
            options,
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
        if current_context()? != self.context {
            return Err(invalid("generated kernel context mismatch"));
        }
        Ok(())
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
/// The caller supplies valid PTX or cubin storage that survives module loading.
unsafe fn load_module(
    driver: &super::Driver,
    image: *const c_void,
    name: &CStr,
) -> Result<(usize, usize), TensorError> {
    let mut module = std::ptr::null_mut();
    let mut function = std::ptr::null_mut();
    // SAFETY: caller owns the input image; both output handles are writable.
    unsafe {
        driver.check((driver.load)(&raw mut module, image), "cuModuleLoadData")?;
        if let Err(error) = driver.check(
            (driver.function)(&raw mut function, module, name.as_ptr()),
            "cuModuleGetFunction",
        ) {
            (driver.unload)(module);
            return Err(error);
        }
    }
    Ok((module as usize, function as usize))
}

impl Drop for Kernel {
    fn drop(&mut self) {
        if let (Ok(runtime), Ok(driver)) = (runtime(), driver())
            && let Ok(_guard) = runtime.guard(self.device)
            && current_context().ok() == Some(self.context)
        {
            // SAFETY: all launches complete synchronously before releasing the kernel.
            unsafe {
                (driver.unload)(self.module as *mut c_void);
            }
        }
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
        Arc::get_mut(&mut kernel).unwrap().context = usize::MAX;
        let uploads = crate::cuda::pointwise_upload_count();
        let result = crate::Tensor::prepare_pointwise(&[&input], Arc::clone(&kernel), None, None);
        let rejected =
            matches!(&result, Err(error) if error.to_string().contains("context mismatch"));
        drop(result);
        // Restore the real module context before any assertion can unwind.
        Arc::get_mut(&mut kernel).unwrap().context = context;
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

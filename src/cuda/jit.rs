//! NVRTC compilation and owned driver modules for generated pointwise kernels.
//! Storage owners and the runtime device guard must outlive every launch/wait.
use super::super::{Library, TensorError, c_char, c_int, c_void, runtime};
use super::{current_context, driver};
use crate::pointwise_ir::{Graph, invalid};
use std::ffi::{CStr, CString};

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

pub(crate) struct Kernel {
    pub(crate) device: usize,
    context: usize,
    module: usize,
    function: usize,
    pub(crate) source: String,
    pub(crate) ptx: String,
    pub(crate) version: (i32, i32),
    pub(crate) options: Vec<String>,
}
impl Kernel {
    pub(crate) fn compile(graph: &Graph, device: usize) -> Result<Self, TensorError> {
        let source = graph.source()?;
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
        let options = vec![
            format!("--gpu-architecture=compute_{major}{minor}"),
            "--fmad=true".into(),
            "--ftz=false".into(),
            "--prec-div=true".into(),
            "--prec-sqrt=true".into(),
        ];
        let (ptx, version) = Nvrtc::load()?.compile(&source, &options)?;
        let ptx_text = CStr::from_bytes_until_nul(&ptx)
            .map_err(|_| invalid("invalid NVRTC PTX"))?
            .to_string_lossy()
            .into_owned();
        let mut module = std::ptr::null_mut();
        let mut function = std::ptr::null_mut();
        // SAFETY: NUL-terminated PTX produced by NVRTC, valid output handles.
        unsafe {
            driver.check(
                (driver.load)(&raw mut module, ptx.as_ptr().cast()),
                "cuModuleLoadData",
            )?;
            if let Err(error) = driver.check(
                (driver.function)(&raw mut function, module, c"torch_rs_pointwise".as_ptr()),
                "cuModuleGetFunction",
            ) {
                (driver.unload)(module);
                return Err(error);
            }
        }
        Ok(Self {
            device,
            context,
            module: module as usize,
            function: function as usize,
            source,
            ptx: ptx_text,
            version,
            options,
        })
    }
    /// Pointers must reference count live contiguous float32 elements on this
    /// device. Output must be disjoint. Synchronize before releasing owners.
    pub(crate) unsafe fn launch(
        &self,
        mut x0: u64,
        mut x1: u64,
        mut output: u64,
        mut count: u64,
    ) -> Result<(), TensorError> {
        if current_context()? != self.context {
            return Err(invalid("generated kernel context mismatch"));
        }
        let driver = driver()?;
        let mut args = [
            (&raw mut x0).cast(),
            (&raw mut x1).cast(),
            (&raw mut output).cast(),
            (&raw mut count).cast(),
        ];
        let blocks = u32::try_from(count.div_ceil(256).min(65535)).unwrap();
        // SAFETY: caller holds all checked storage through legacy-stream completion.
        driver.check(
            unsafe {
                (driver.launch)(
                    self.function as *mut c_void,
                    blocks,
                    1,
                    1,
                    256,
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
            output: 1,
        };
        let kernel = Kernel::compile(&graph, 0).unwrap();
        assert!(kernel.ptx.contains("torch_rs_pointwise"));
    }
}

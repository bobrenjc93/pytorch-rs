//! Shared owned NVRTC compilation, link/load and context lifetime.
use super::super::{Library, TensorError, c_char, c_int, c_void, runtime};
use super::{current_context, driver};
use crate::pointwise_ir::invalid;
use std::ffi::{CStr, CString};
#[path = "link.rs"]
mod link;

type Program = *mut c_void;
pub(super) struct Nvrtc {
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
    pub(super) fn load() -> Result<Self, TensorError> {
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
    pub(super) fn compile(
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

pub(crate) struct Module {
    pub(crate) device: usize,
    pub(crate) context: usize,
    handle: usize,
    pub(crate) function: usize,
    pub(crate) source: String,
    pub(crate) ptx: String,
    pub(crate) version: (i32, i32),
    pub(crate) options: Vec<String>,
}
impl Module {
    pub(crate) fn checked_context(device: usize) -> Result<usize, TensorError> {
        let _guard = runtime()?.guard(device)?;
        current_context()
    }
    pub(crate) fn compile(
        source: String,
        device: usize,
        expected_context: Option<usize>,
        erf: bool,
        entry: &CStr,
    ) -> Result<Self, TensorError> {
        let _guard = runtime()?.guard(device)?;
        let context = current_context()?;
        if expected_context.is_some_and(|expected| expected != context) {
            return Err(invalid("host plan context mismatch"));
        }
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
        if erf {
            options.push("--relocatable-device-code=true".into());
        }
        let (ptx, version) = Nvrtc::load()?.compile(&source, &options)?;
        let ptx_text = CStr::from_bytes_until_nul(&ptx)
            .map_err(|_| invalid("invalid NVRTC PTX"))?
            .to_string_lossy()
            .into_owned();
        let (module, function) = if erf {
            link::load(
                driver,
                &ptx,
                concat!(include_str!("erf_provider.ptx"), "\0").as_bytes(),
                entry,
            )?
        } else {
            // SAFETY: NUL-terminated NVRTC PTX lives through module loading.
            unsafe { load_module(driver, ptx.as_ptr().cast(), entry)? }
        };
        Ok(Self {
            device,
            context,
            handle: module,
            function,
            source,
            ptx: ptx_text,
            version,
            options,
        })
    }
    pub(crate) fn validate_context(&self) -> Result<(), TensorError> {
        if current_context()? != self.context {
            return Err(invalid("generated kernel context mismatch"));
        }
        Ok(())
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

impl Drop for Module {
    fn drop(&mut self) {
        if let (Ok(runtime), Ok(driver)) = (runtime(), driver())
            && let Ok(_guard) = runtime.guard(self.device)
            && current_context().ok() == Some(self.context)
        {
            // SAFETY: all launches complete synchronously before releasing the kernel.
            unsafe {
                (driver.unload)(self.handle as *mut c_void);
            }
        }
    }
}

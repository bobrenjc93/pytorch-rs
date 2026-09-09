//! Optional driver ABI for native pointwise kernels. PTX is embedded at build
//! time and JIT-compiled by the installed NVIDIA driver, without nvcc or NVRTC.
use super::{CStr, Library, Mutex, OnceLock, Status, TensorError, c_char, c_int, c_void};

struct Driver {
    _library: Library,
    context: unsafe extern "C" fn(*mut *mut c_void) -> Status,
    load: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> Status,
    function: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> Status,
    unload: unsafe extern "C" fn(*mut c_void) -> Status,
    launch: unsafe extern "C" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> Status,
    error_name: unsafe extern "C" fn(c_int, *mut *const c_char) -> Status,
    // Context, module, function handles. Successful modules and the driver
    // library remain live for the process lifetime, like the CUDA runtime.
    modules: Mutex<Vec<(usize, usize, usize)>>,
}
static DRIVER: OnceLock<Result<Driver, String>> = OnceLock::new();

fn driver() -> Result<&'static Driver, TensorError> {
    DRIVER
        .get_or_init(|| {
            // SAFETY: documented driver C ABI; the library outlives all symbols.
            unsafe {
                let library = Library::new(if cfg!(windows) {
                    "nvcuda.dll"
                } else {
                    "libcuda.so.1"
                })
                .map_err(|error| error.to_string())?;
                let loaded = || -> Result<Driver, libloading::Error> {
                    Ok(Driver {
                        context: *library.get(b"cuCtxGetCurrent\0")?,
                        load: *library.get(b"cuModuleLoadData\0")?,
                        function: *library.get(b"cuModuleGetFunction\0")?,
                        unload: *library.get(b"cuModuleUnload\0")?,
                        launch: *library.get(b"cuLaunchKernel\0")?,
                        error_name: *library.get(b"cuGetErrorName\0")?,
                        _library: library,
                        modules: Mutex::new(Vec::new()),
                    })
                };
                loaded().map_err(|error| error.to_string())
            }
        })
        .as_ref()
        .map_err(|message| TensorError::CudaRuntimeError {
            operation: "driver",
            message: message.clone(),
        })
}

impl Driver {
    fn check(&self, status: Status, operation: &'static str) -> Result<(), TensorError> {
        if status == 0 {
            return Ok(());
        }
        let mut name = std::ptr::null();
        // SAFETY: valid output pointer; driver owns the returned static string.
        let message = unsafe {
            (self.error_name)(status, &raw mut name);
            if name.is_null() {
                format!("CUDA driver error {status}")
            } else {
                CStr::from_ptr(name).to_string_lossy().into_owned()
            }
        };
        Err(TensorError::CudaRuntimeError { operation, message })
    }

    fn add_function(&self) -> Result<usize, TensorError> {
        let mut context = std::ptr::null_mut();
        // SAFETY: writable context handle; the caller holds the runtime device guard.
        self.check(
            unsafe { (self.context)(&raw mut context) },
            "cuCtxGetCurrent",
        )?;
        if context.is_null() {
            // A new host thread can hit the allocation cache without calling
            // cudaMalloc; cudaGetDevice alone does not bind a driver context.
            // SAFETY: cudaFree(NULL) initializes the guarded runtime device's
            // primary context without releasing any allocation. Do this only
            // on the cold-thread path, before loading or launching a module.
            let runtime = super::runtime()?;
            runtime.check(unsafe { (runtime.free)(std::ptr::null_mut()) }, "cudaFree")?;
            self.check(
                unsafe { (self.context)(&raw mut context) },
                "cuCtxGetCurrent",
            )?;
        }
        let mut modules = self
            .modules
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(entry) = modules.iter().find(|entry| entry.0 == context as usize) {
            return Ok(entry.2);
        }
        modules
            .try_reserve(1)
            .map_err(|_| TensorError::AllocationFailed { elements: 1 })?;
        let mut module = std::ptr::null_mut();
        let mut function = std::ptr::null_mut();
        // SAFETY: static NUL-terminated PTX and entry name; writable handles.
        unsafe {
            self.check(
                (self.load)(
                    &raw mut module,
                    concat!(include_str!("add.ptx"), "\0").as_ptr().cast(),
                ),
                "cuModuleLoadData",
            )?;
            if let Err(error) = self.check(
                (self.function)(&raw mut function, module, c"add_f32".as_ptr()),
                "cuModuleGetFunction",
            ) {
                (self.unload)(module);
                return Err(error);
            }
        }
        modules.push((context as usize, module as usize, function as usize));
        Ok(function as usize)
    }
}

/// # Safety
/// All pointers refer to `elements` live contiguous floats on the current
/// guarded device. Output does not alias either input. Caller must synchronize
/// the legacy stream before releasing any allocation, also on launch errors.
pub(super) unsafe fn launch_add(
    mut left: u64,
    mut right: u64,
    mut output: u64,
    elements: usize,
) -> Result<(), TensorError> {
    let driver = driver()?;
    let function = driver.add_function()?;
    let mut count = elements as u64;
    let mut arguments = [
        (&raw mut left).cast(),
        (&raw mut right).cast(),
        (&raw mut output).cast(),
        (&raw mut count).cast(),
    ];
    let blocks = u32::try_from(elements.div_ceil(256).min(4096)).expect("bounded grid");
    // SAFETY: parameters live through launch's argument copy. Handle is cached
    // in this context. CU_STREAM_LEGACY (1) explicitly matches runtime copies
    // and zero-fill even when another CUDA user uses a per-thread default.
    driver.check(
        unsafe {
            (driver.launch)(
                function as *mut c_void,
                blocks,
                1,
                1,
                256,
                1,
                1,
                0,
                std::ptr::without_provenance_mut(1),
                arguments.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        },
        "cuLaunchKernel",
    )
}

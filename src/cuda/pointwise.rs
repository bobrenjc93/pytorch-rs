//! Optional driver ABI for native kernels. PTX is embedded at build
//! time and JIT-compiled by the installed NVIDIA driver, without nvcc or NVRTC.
use super::{CStr, Library, Mutex, OnceLock, Status, TensorError, c_char, c_int, c_void};

#[path = "sum_rows.rs"]
mod sum_rows;
pub(super) use sum_rows::{RowSumConfig, RowSumSlice, launch_sum_rows};

struct Driver {
    _library: Library,
    device: unsafe extern "C" fn(*mut c_int) -> Status,
    attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> Status,
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
    modules: Mutex<Vec<Module>>,
}
struct Module {
    context: usize,
    _handle: usize,
    functions: [usize; 8],
}

enum Kernel {
    Add,
    AddVector,
    MultiplyScalar,
    AddTrailingVector,
    SumRows,
    SumRowsFinalize,
    Matmul,
    #[cfg(any(feature = "python-bindings", test))]
    Negate,
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
                        device: *library.get(b"cuCtxGetDevice\0")?,
                        attribute: *library.get(b"cuDeviceGetAttribute\0")?,
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

    fn function(&self, kernel: Kernel) -> Result<usize, TensorError> {
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
        if let Some(entry) = modules
            .iter()
            .find(|entry| entry.context == context as usize)
        {
            return Ok(entry.functions[kernel as usize]);
        }
        modules
            .try_reserve(1)
            .map_err(|_| TensorError::AllocationFailed { elements: 1 })?;
        let mut module = std::ptr::null_mut();
        let mut functions = [0; 8];
        // SAFETY: static NUL-terminated PTX and entry names; writable handles.
        unsafe {
            self.check(
                (self.load)(
                    &raw mut module,
                    concat!(
                        include_str!("add.ptx"),
                        "\n",
                        include_str!("mul_scalar.ptx"),
                        "\n",
                        include_str!("add_trailing_vector.ptx"),
                        "\n",
                        include_str!("sum_rows.ptx"),
                        "\n",
                        include_str!("matmul.ptx"),
                        "\n",
                        include_str!("neg.ptx"),
                        "\0"
                    )
                    .as_ptr()
                    .cast(),
                ),
                "cuModuleLoadData",
            )?;
            for (slot, name) in functions.iter_mut().zip([
                c"add_f32",
                c"add_f32x4",
                c"mul_scalar_f32",
                c"add_trailing_vector_f32",
                c"sum_rows_f32",
                c"sum_rows_finalize_f32",
                c"matmul_f32",
                c"neg_f32",
            ]) {
                let mut function = std::ptr::null_mut();
                if let Err(error) = self.check(
                    (self.function)(&raw mut function, module, name.as_ptr()),
                    "cuModuleGetFunction",
                ) {
                    (self.unload)(module);
                    return Err(error);
                }
                *slot = function as usize;
            }
        }
        modules.push(Module {
            context: context as usize,
            _handle: module as usize,
            functions,
        });
        Ok(functions[kernel as usize])
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
) -> (
    Result<(), TensorError>,
    Option<std::sync::Arc<super::replay::Replay>>,
) {
    let mut keepalive = None;
    let result = (|| {
        let driver = driver()?;
        let vectorized = (left | right | output).is_multiple_of(16);
        let function = driver.function(if vectorized {
            Kernel::AddVector
        } else {
            Kernel::Add
        })?;
        let mut count = elements as u64;
        let mut arguments = [
            (&raw mut left).cast(),
            (&raw mut right).cast(),
            (&raw mut output).cast(),
            (&raw mut count).cast(),
        ];
        let lanes = if vectorized {
            elements.div_ceil(4)
        } else {
            elements
        };
        let blocks = u32::try_from(lanes.div_ceil(256).min(4096)).expect("bounded grid");
        let kernel = super::replay::Kernel {
            function: function as *mut c_void,
            grid: [blocks, 1, 1],
            block: [256, 1, 1],
            shared_bytes: 0,
            arguments: arguments.as_mut_ptr(),
            extra: std::ptr::null_mut(),
        };
        keepalive = unsafe {
            super::replay::get(
                super::replay::Key(function, left, right, output, elements),
                &kernel,
            )
        };
        if let Some(replay) = &keepalive {
            return driver.check(unsafe { replay.launch() }, "cuGraphLaunch");
        }
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
    })();
    (result, keepalive)
}

/// # Safety
/// Matrix/output refer to `elements` live contiguous floats, vector to `columns`
/// floats on the guarded device; columns is nonzero and divides elements.
/// Output is fresh. Caller must complete the legacy stream even on launch error.
pub(super) unsafe fn launch_add_trailing_vector(
    mut matrix: u64,
    mut vector: u64,
    mut output: u64,
    elements: usize,
    columns: usize,
) -> Result<(), TensorError> {
    let driver = driver()?;
    let function = driver.function(Kernel::AddTrailingVector)?;
    let mut count = elements as u64;
    let mut columns = columns as u64;
    let mut arguments = [
        (&raw mut matrix).cast(),
        (&raw mut vector).cast(),
        (&raw mut output).cast(),
        (&raw mut count).cast(),
        (&raw mut columns).cast(),
    ];
    let blocks = u32::try_from(elements.div_ceil(256).min(4096)).expect("bounded grid");
    // SAFETY: arguments survive the launch copy and the function belongs to
    // this context. The caller holds allocations through legacy-stream completion.
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

/// # Safety
/// Input and output refer to `elements` live contiguous floats on the guarded
/// device, without aliasing. Caller must synchronize the legacy stream before
/// releasing either allocation, including on launch errors.
#[cfg(any(feature = "python-bindings", test))]
pub(super) unsafe fn launch_negate(
    mut input: u64,
    mut output: u64,
    elements: usize,
) -> Result<(), TensorError> {
    let driver = driver()?;
    let function = driver.function(Kernel::Negate)?;
    let mut count = elements as u64;
    let mut arguments = [
        (&raw mut input).cast(),
        (&raw mut output).cast(),
        (&raw mut count).cast(),
    ];
    let blocks = u32::try_from(elements.div_ceil(256).min(4096)).expect("bounded grid");
    // SAFETY: parameters survive the launch argument copy; the cached function
    // belongs to this context. CU_STREAM_LEGACY matches runtime copies/zero-fill.
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

/// # Safety
/// Input and output refer to `elements` live contiguous floats on the guarded
/// device, without aliasing. Caller must synchronize the legacy stream before
/// releasing either allocation, including on launch errors.
pub(super) unsafe fn launch_mul_scalar(
    mut input: u64,
    mut output: u64,
    elements: usize,
    mut scalar: f32,
) -> Result<(), TensorError> {
    let driver = driver()?;
    let function = driver.function(Kernel::MultiplyScalar)?;
    let mut count = elements as u64;
    let mut arguments = [
        (&raw mut input).cast(),
        (&raw mut output).cast(),
        (&raw mut count).cast(),
        (&raw mut scalar).cast(),
    ];
    let blocks = u32::try_from(elements.div_ceil(256).min(4096)).expect("bounded grid");
    // SAFETY: parameters survive the launch argument copy; the cached function
    // belongs to this context. CU_STREAM_LEGACY matches runtime copies/zero-fill.
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

/// # Safety
/// Inputs cover the checked rank-2 product on the guarded device; output is
/// disjoint and covers `elements` floats. For inner=0 inputs may be null. The
/// caller must retain all allocations through legacy-stream completion,
/// including on launch errors. Nonzero elements requires nonzero columns.
pub(super) unsafe fn launch_matmul(
    mut left: u64,
    mut right: u64,
    mut output: u64,
    elements: usize,
    inner: usize,
    columns: usize,
) -> Result<(), TensorError> {
    let driver = driver()?;
    let function = driver.function(Kernel::Matmul)?;
    let mut count = elements as u64;
    let mut k = inner as u64;
    let mut n = columns as u64;
    let mut arguments = [
        (&raw mut left).cast(),
        (&raw mut right).cast(),
        (&raw mut output).cast(),
        (&raw mut count).cast(),
        (&raw mut k).cast(),
        (&raw mut n).cast(),
    ];
    let blocks = u32::try_from(elements.div_ceil(256).min(4096)).expect("bounded grid");
    // SAFETY: argument values survive launch, function belongs to this context,
    // and the caller provides the checked allocation/completion contract above.
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

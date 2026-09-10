//! Lazy native cuBLAS SGEMM. No Python/libTorch calls or build-time CUDA linkage.
use super::{CANDIDATES, Library, Mutex, OnceLock, Status, TensorError, c_int, c_void};

type Handle = *mut c_void;
type Gemm = unsafe extern "C" fn(
    Handle,
    c_int,
    c_int,
    i64,
    i64,
    i64,
    *const f32,
    *const f32,
    i64,
    *const f32,
    i64,
    *const f32,
    *mut f32,
    i64,
) -> Status;

struct Blas {
    _library: Library,
    create: unsafe extern "C" fn(*mut Handle) -> Status,
    destroy: unsafe extern "C" fn(Handle) -> Status,
    set_stream: unsafe extern "C" fn(Handle, *mut c_void) -> Status,
    set_math: unsafe extern "C" fn(Handle, c_int) -> Status,
    gemm: Gemm,
    // Handles are context-owned and retained with the library for process life.
    // The lock serializes handle access; inputs survive stream completion in
    // CudaFloat32Storage::unary_output, after the launch lock is released.
    handles: Mutex<Vec<(usize, usize)>>,
}
static BLAS: OnceLock<Result<Blas, String>> = OnceLock::new();

fn blas() -> Result<&'static Blas, TensorError> {
    BLAS.get_or_init(|| {
        let paths = std::env::var("TORCH_RS_CUBLAS").map_or_else(
            |_| {
                let names = if cfg!(windows) {
                    vec!["cublas64_13.dll", "cublas64_12.dll"]
                } else {
                    vec!["libcublas.so.13", "libcublas.so.12", "libcublas.so"]
                };
                let mut paths = Vec::new();
                // Wheel discovery supplies CUDA runtime paths without importing
                // PyTorch. CUDA 13 wheels colocate BLAS; CUDA 12 uses nvidia/cublas.
                for runtime in std::env::var("TORCH_RS_CUDART")
                    .ok()
                    .into_iter()
                    .chain(CANDIDATES.get().into_iter().flatten().cloned())
                {
                    if let Some(parent) = std::path::Path::new(&runtime).parent() {
                        for name in &names {
                            paths.push(parent.join(name).to_string_lossy().into_owned());
                            paths.push(
                                parent
                                    .join("../../cublas/lib")
                                    .join(name)
                                    .to_string_lossy()
                                    .into_owned(),
                            );
                        }
                    }
                }
                paths.extend(names.into_iter().map(str::to_owned));
                paths
            },
            |path| vec![path],
        );
        let mut errors = Vec::new();
        for path in paths {
            // SAFETY: cuBLAS's documented C ABI; the Library owns the symbols.
            match unsafe { Blas::load(&path) } {
                Ok(blas) => return Ok(blas),
                Err(error) => errors.push(format!("{path}: {error}")),
            }
        }
        Err(format!(
            "cannot load native cuBLAS (set TORCH_RS_CUBLAS): {}",
            errors.join("; ")
        ))
    })
    .as_ref()
    .map_err(|message| TensorError::CudaRuntimeError {
        operation: "cuBLAS",
        message: message.clone(),
    })
}

fn check(status: Status, operation: &'static str) -> Result<(), TensorError> {
    if status == 0 {
        Ok(())
    } else {
        Err(TensorError::CudaRuntimeError {
            operation,
            message: format!("cuBLAS status {status}"),
        })
    }
}

impl Blas {
    unsafe fn load(path: &str) -> Result<Self, libloading::Error> {
        // SAFETY: exact signatures from cublas_api.h, including the 64-bit
        // dimension ABI. Library is retained longer than every handle/symbol.
        unsafe {
            let library = Library::new(path)?;
            Ok(Self {
                create: *library.get(b"cublasCreate_v2\0")?,
                destroy: *library.get(b"cublasDestroy_v2\0")?,
                set_stream: *library.get(b"cublasSetStream_v2\0")?,
                set_math: *library.get(b"cublasSetMathMode\0")?,
                gemm: *library.get(b"cublasSgemm_v2_64\0")?,
                _library: library,
                handles: Mutex::new(Vec::new()),
            })
        }
    }
}

/// # Safety
/// Checked contiguous matrices on the guarded current device, fresh disjoint
/// output. All allocations must survive legacy-stream completion, even on error.
/// Dimensions are nonzero and their products fit live float32 allocations.
pub(super) unsafe fn launch(
    left: u64,
    right: u64,
    output: u64,
    rows: usize,
    inner: usize,
    columns: usize,
) -> Result<(), TensorError> {
    let blas = blas()?;
    let context = super::pointwise::current_context()?;
    let mut handles = blas
        .handles
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let handle = if let Some((_, handle)) = handles.iter().find(|(key, _)| *key == context) {
        *handle as Handle
    } else {
        handles
            .try_reserve(1)
            .map_err(|_| TensorError::AllocationFailed { elements: 1 })?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: current context is initialized and guarded; writable handle.
        unsafe {
            check((blas.create)(&raw mut handle), "cublasCreate")?;
        }
        // CUBLAS_DEFAULT_MATH uses full float32 SGEMM, not TF32 or float64.
        // Match the reference's float32 reduction/overflow behavior. Explicit
        // legacy stream also works when other libraries use per-thread defaults.
        let configured = unsafe {
            check((blas.set_math)(handle, 0), "cublasSetMathMode").and_then(|()| {
                check(
                    (blas.set_stream)(handle, std::ptr::without_provenance_mut(1)),
                    "cublasSetStream",
                )
            })
        };
        if let Err(error) = configured {
            unsafe {
                (blas.destroy)(handle);
            }
            return Err(error);
        }
        handles.push((context, handle as usize));
        handle
    };
    let [m, k, n] =
        [rows, inner, columns].map(|v| i64::try_from(v).expect("checked allocation size"));
    // Row-major C = A B is column-major C^T = B^T A^T, with no transposes.
    // Host alpha/beta are copied by cuBLAS before returning. beta=0 ensures
    // fresh/recycled output contents cannot affect results (including NaNs).
    unsafe {
        check(
            (blas.gemm)(
                handle,
                0,
                0,
                n,
                m,
                k,
                &1.0,
                right as *const f32,
                n,
                left as *const f32,
                k,
                &0.0,
                output as *mut f32,
                n,
            ),
            "cublasSgemm",
        )
    }
}

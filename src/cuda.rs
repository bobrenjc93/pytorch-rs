//! Optional CUDA runtime backend. No Python objects, interpreter, or toolkit
//! linkage are needed by native tensors. Unsafe code is confined to this ABI.
#![allow(unsafe_code)]
use crate::tensor_error::TensorError;
use libloading::Library;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::{
    Mutex, OnceLock,
    atomic::{AtomicBool, Ordering},
};

type Status = c_int;
struct Runtime {
    _library: Library,
    count: unsafe extern "C" fn(*mut c_int) -> Status,
    get_device: unsafe extern "C" fn(*mut c_int) -> Status,
    set_device: unsafe extern "C" fn(c_int) -> Status,
    malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> Status,
    free: unsafe extern "C" fn(*mut c_void) -> Status,
    memset: unsafe extern "C" fn(*mut c_void, c_int, usize) -> Status,
    memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> Status,
    error_name: unsafe extern "C" fn(Status) -> *const c_char,
}
static CANDIDATES: OnceLock<Vec<String>> = OnceLock::new();
static RUNTIME: OnceLock<Result<Runtime, String>> = OnceLock::new();
static INITIALIZED: AtomicBool = AtomicBool::new(false);
// At most 32 allocations and 64 MiB retained. Entries are inaccessible to live
// tensors; storage Arc ownership prevents reuse while any view still exists.
static CACHE: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());
const CACHE_BYTES: usize = 64 * 1024 * 1024;

/// Supplies optional runtime library paths before the first backend use.
/// `TORCH_RS_CUDART` takes precedence; system library names remain fallbacks.
pub fn configure_candidates(paths: Vec<String>) {
    let _ = CANDIDATES.set(paths);
}
/// Whether this process has allocated native CUDA storage.
#[must_use]
pub fn is_initialized() -> bool {
    INITIALIZED.load(Ordering::Relaxed)
}
/// Returns the visible device count, or zero if the optional runtime is unavailable.
#[must_use]
pub fn device_count() -> usize {
    runtime().map_or(0, |runtime| {
        let mut count = 0;
        // SAFETY: writable integer, correct runtime ABI.
        if unsafe { (runtime.count)(&raw mut count) } == 0 {
            usize::try_from(count).unwrap_or(0)
        } else {
            0
        }
    })
}
fn runtime() -> Result<&'static Runtime, TensorError> {
    RUNTIME
        .get_or_init(|| {
            let paths = std::env::var("TORCH_RS_CUDART").map_or_else(
                |_| {
                    let mut paths = CANDIDATES.get().cloned().unwrap_or_default();
                    paths.extend(
                        [
                            "libcudart.so.13",
                            "libcudart.so.12",
                            "libcudart.so",
                            "cudart64_13.dll",
                            "cudart64_12.dll",
                        ]
                        .map(str::to_owned),
                    );
                    paths
                },
                |path| vec![path],
            );
            let mut errors = Vec::new();
            for path in paths {
                // SAFETY: the user-selected CUDA runtime implements the documented
                // C ABI. Retaining Library keeps all resolved symbols alive.
                let loaded = unsafe { Runtime::load(&path) };
                match loaded {
                    Ok(runtime) => return Ok(runtime),
                    Err(error) => errors.push(error.to_string()),
                }
            }
            Err(format!(
                "cannot load CUDA runtime (set TORCH_RS_CUDART): {}",
                errors.join("; ")
            ))
        })
        .as_ref()
        .map_err(|message| TensorError::CudaRuntimeError {
            operation: "runtime",
            message: message.clone(),
        })
}
impl Runtime {
    unsafe fn load(path: &str) -> Result<Self, libloading::Error> {
        // SAFETY: every symbol uses CUDA's declared ABI and the library is retained.
        unsafe {
            let library = Library::new(path)?;
            Ok(Self {
                count: *library.get(b"cudaGetDeviceCount\0")?,
                get_device: *library.get(b"cudaGetDevice\0")?,
                set_device: *library.get(b"cudaSetDevice\0")?,
                malloc: *library.get(b"cudaMalloc\0")?,
                free: *library.get(b"cudaFree\0")?,
                memset: *library.get(b"cudaMemset\0")?,
                memcpy: *library.get(b"cudaMemcpy\0")?,
                error_name: *library.get(b"cudaGetErrorName\0")?,
                _library: library,
            })
        }
    }
    fn check(&self, status: Status, operation: &'static str) -> Result<(), TensorError> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: CUDA returns a static NUL-terminated error string or null.
        let name = unsafe {
            let pointer = (self.error_name)(status);
            if pointer.is_null() {
                format!("CUDA error {status}")
            } else {
                CStr::from_ptr(pointer).to_string_lossy().into_owned()
            }
        };
        Err(TensorError::CudaRuntimeError {
            operation,
            message: name,
        })
    }
    fn guard(&'static self, device: usize) -> Result<DeviceGuard, TensorError> {
        let device = c_int::try_from(device).map_err(|_| TensorError::CudaRuntimeError {
            operation: "device",
            message: "device ordinal exceeds int".into(),
        })?;
        let mut previous = 0;
        // SAFETY: valid output pointer and runtime-validated ordinal.
        unsafe {
            self.check((self.get_device)(&raw mut previous), "cudaGetDevice")?;
            if previous != device {
                self.check((self.set_device)(device), "cudaSetDevice")?;
            }
        }
        Ok(DeviceGuard {
            runtime: self,
            previous,
            changed: previous != device,
        })
    }
}
struct DeviceGuard {
    runtime: &'static Runtime,
    previous: c_int,
    changed: bool,
}
impl Drop for DeviceGuard {
    fn drop(&mut self) {
        if self.changed {
            // SAFETY: previous is an ordinal returned by this runtime.
            unsafe {
                (self.runtime.set_device)(self.previous);
            }
        }
    }
}

pub(crate) struct CudaFloat32Storage {
    pub(crate) elements: usize,
    pub(crate) device_index: usize,
    pub(crate) data_ptr: usize,
    runtime: &'static Runtime,
}
impl CudaFloat32Storage {
    pub(crate) fn zeros(elements: usize, device_index: usize) -> Result<Self, TensorError> {
        let bytes = elements
            .checked_mul(4)
            .filter(|bytes| isize::try_from(*bytes).is_ok())
            .ok_or(TensorError::AllocationFailed { elements })?;
        let runtime = runtime()?;
        let _guard = runtime.guard(device_index)?;
        let cached = {
            let mut cache = CACHE
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            cache
                .iter()
                .position(|&(device, size, _)| device == device_index && size == bytes)
                .map(|index| cache.swap_remove(index).2)
        };
        let mut pointer = cached.unwrap_or(0) as *mut c_void;
        if bytes != 0 && cached.is_none() {
            // SAFETY: valid output pointer; size checked above.
            runtime.check(
                unsafe { (runtime.malloc)(&raw mut pointer, bytes) },
                "cudaMalloc",
            )?;
        }
        let result = Self {
            elements,
            device_index,
            data_ptr: pointer as usize,
            runtime,
        };
        if bytes != 0 {
            // SAFETY: this allocation owns at least bytes bytes. Legacy default
            // stream ordering composes memset, reuse, and blocking host copies.
            runtime.check(unsafe { (runtime.memset)(pointer, 0, bytes) }, "cudaMemset")?;
        }
        INITIALIZED.store(true, Ordering::Relaxed);
        Ok(result)
    }
    pub(crate) fn copy_range(
        &self,
        start: usize,
        elements: usize,
    ) -> Result<Vec<f32>, TensorError> {
        // Empty views can carry an offset past storage's end without reading it.
        if elements == 0 {
            return Ok(Vec::new());
        }
        if start
            .checked_add(elements)
            .is_none_or(|end| end > self.elements)
        {
            return Err(TensorError::IndexCalculationOverflow);
        }
        let mut values = Vec::<f32>::new();
        values
            .try_reserve_exact(elements)
            .map_err(|_| TensorError::AllocationFailed { elements })?;
        if elements != 0 {
            let _guard = self.runtime.guard(self.device_index)?;
            // SAFETY: destination has reserved capacity, source range was
            // checked, and synchronous D2H initializes all elements before set_len.
            unsafe {
                self.runtime.check(
                    (self.runtime.memcpy)(
                        values.as_mut_ptr().cast(),
                        (self.data_ptr + start * 4) as *const c_void,
                        elements * 4,
                        2,
                    ),
                    "cudaMemcpy",
                )?;
                values.set_len(elements);
            }
        }
        Ok(values)
    }
}
impl Drop for CudaFloat32Storage {
    fn drop(&mut self) {
        if self.data_ptr == 0 {
            return;
        }
        let bytes = self.elements * 4;
        {
            let mut cache = CACHE
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if cache.len() < 32
                && bytes <= CACHE_BYTES
                && cache.iter().map(|entry| entry.1).sum::<usize>() <= CACHE_BYTES - bytes
            {
                cache.push((self.device_index, bytes, self.data_ptr));
                return;
            }
        }
        if let Ok(_guard) = self.runtime.guard(self.device_index) {
            // SAFETY: this pointer is uniquely owned and no longer exposed.
            unsafe {
                (self.runtime.free)(self.data_ptr as *mut c_void);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Device, Tensor};
    #[test]
    fn standalone_native_cuda_roundtrip() {
        if super::device_count() == 0 {
            eprintln!("skipping native CUDA roundtrip: no CUDA runtime/device");
            return;
        }
        let tensor = Tensor::cuda_zeros_float32(vec![17], Device::Cuda(0)).unwrap();
        assert_eq!(
            tensor.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
            vec![0.0; 17]
        );
        assert!(super::is_initialized());
        let empty = Tensor::cuda_zeros_float32(vec![0], Device::Cuda(0)).unwrap();
        assert!(
            empty
                .try_copy_cuda_to_cpu()
                .unwrap()
                .try_to_vec()
                .unwrap()
                .is_empty()
        );
    }
}

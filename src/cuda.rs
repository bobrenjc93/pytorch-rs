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

mod pointwise;
mod pool;
mod replay;

type Status = c_int;
struct Runtime {
    _library: Library,
    pool: Option<pool::Api>,
    count: unsafe extern "C" fn(*mut c_int) -> Status,
    get_device: unsafe extern "C" fn(*mut c_int) -> Status,
    set_device: unsafe extern "C" fn(c_int) -> Status,
    malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> Status,
    free: unsafe extern "C" fn(*mut c_void) -> Status,
    memset: unsafe extern "C" fn(*mut c_void, c_int, usize) -> Status,
    memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> Status,
    stream_synchronize: unsafe extern "C" fn(*mut c_void) -> Status,
    memcpy_2d: unsafe extern "C" fn(
        *mut c_void,
        usize,
        *const c_void,
        usize,
        usize,
        usize,
        c_int,
    ) -> Status,
    error_name: unsafe extern "C" fn(Status) -> *const c_char,
    get_last_error: unsafe extern "C" fn() -> Status,
}
static CANDIDATES: OnceLock<Vec<String>> = OnceLock::new();
static RUNTIME: OnceLock<Result<Runtime, String>> = OnceLock::new();
static INITIALIZED: AtomicBool = AtomicBool::new(false);
// At most 32 allocations and 64 MiB retained in the front cache. Entries are inaccessible to live
// tensors; storage Arc ownership prevents reuse while any view still exists.
static CACHE: Mutex<Vec<(usize, usize, usize, bool)>> = Mutex::new(Vec::new());
// A failed launch/completion disables reuse. Ordinary allocations are freed;
// uncertain stream-ordered allocations are quarantined.
static CACHE_HEALTHY: AtomicBool = AtomicBool::new(true);
const CACHE_BYTES: usize = 64 * 1024 * 1024;
const COPY_STAGING_ELEMENTS: usize = 256 * 1024 / size_of::<f32>();

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
                pool: pool::Api::load(&library).ok(),
                count: *library.get(b"cudaGetDeviceCount\0")?,
                get_device: *library.get(b"cudaGetDevice\0")?,
                set_device: *library.get(b"cudaSetDevice\0")?,
                malloc: *library.get(b"cudaMalloc\0")?,
                free: *library.get(b"cudaFree\0")?,
                memset: *library.get(b"cudaMemset\0")?,
                memcpy: *library.get(b"cudaMemcpy\0")?,
                stream_synchronize: *library.get(b"cudaStreamSynchronize\0")?,
                memcpy_2d: *library.get(b"cudaMemcpy2D\0")?,
                error_name: *library.get(b"cudaGetErrorName\0")?,
                get_last_error: *library.get(b"cudaGetLastError\0")?,
                _library: library,
            })
        }
    }
    fn check(&self, status: Status, operation: &'static str) -> Result<(), TensorError> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: no-argument CUDA ABI. We report this call's status below;
        // consume its thread-local error so a handled failure (e.g. invalid
        // device) cannot poison a subsequent launch by another runtime user.
        unsafe { (self.get_last_error)() };
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

/// A rectangle of logical runs, measured in float32 elements. Destination
/// rows are packed; source rows may be separated by holes in device storage.
pub(crate) struct CudaCopyRegion {
    pub(crate) start: usize,
    pub(crate) width: usize,
    pub(crate) rows: usize,
    pub(crate) source_pitch: usize,
}

pub(crate) struct CudaFloat32Storage {
    pub(crate) elements: usize,
    pub(crate) device_index: usize,
    pub(crate) data_ptr: usize,
    allocation_bytes: usize,
    pooled: bool,
    runtime: &'static Runtime,
}
impl CudaFloat32Storage {
    pub(crate) fn zeros(elements: usize, device_index: usize) -> Result<Self, TensorError> {
        let (result, _guard) = Self::allocate(elements, device_index)?;
        if elements != 0 {
            // SAFETY: allocation size was checked; legacy default stream
            // ordering composes memset, reuse, and blocking host copies.
            result.runtime.check(
                unsafe { (result.runtime.memset)(result.data_ptr as *mut c_void, 0, elements * 4) },
                "cudaMemset",
            )?;
        }
        Ok(result)
    }

    pub(crate) fn from_host(values: &[f32], device_index: usize) -> Result<Self, TensorError> {
        let (result, _guard) = Self::allocate(values.len(), device_index)?;
        if !values.is_empty() {
            // SAFETY: the source slice is live throughout the transfer and
            // the destination owns exactly values.len() checked floats.
            unsafe {
                result.runtime.check(
                    (result.runtime.memcpy)(
                        result.data_ptr as *mut c_void,
                        values.as_ptr().cast(),
                        values.len() * 4,
                        1, // cudaMemcpyHostToDevice
                    ),
                    "cudaMemcpy",
                )?;
                // Pageable H2D cudaMemcpy can return after staging. Complete
                // the default-stream transfer before publishing the storage.
                result.runtime.check(
                    (result.runtime.stream_synchronize)(std::ptr::null_mut()),
                    "cudaStreamSynchronize",
                )?;
            }
        }
        Ok(result)
    }

    pub(crate) fn add(
        &self,
        left_offset: usize,
        other: &Self,
        right_offset: usize,
        elements: usize,
    ) -> Result<Self, TensorError> {
        if self.device_index != other.device_index {
            return Err(TensorError::UnsupportedCudaAddition {
                reason: "mixed devices",
            });
        }
        // Empty views may point beyond storage; no pointer is formed or read.
        if elements != 0 {
            for (storage, offset) in [(self, left_offset), (other, right_offset)] {
                if offset
                    .checked_add(elements)
                    .is_none_or(|end| end > storage.elements)
                {
                    return Err(TensorError::IndexCalculationOverflow);
                }
            }
        }
        let (result, _guard) = Self::allocate(elements, self.device_index)?;
        if elements != 0 {
            // SAFETY: bounds and device checked above, output is fresh and all
            // three storages remain borrowed/owned until stream completion.
            let (launched, replay) = unsafe {
                pointwise::launch_add(
                    (self.data_ptr + left_offset * 4) as u64,
                    (other.data_ptr + right_offset * 4) as u64,
                    result.data_ptr as u64,
                    elements,
                )
            };
            // Always wait, including after a launch error, before any borrowed
            // input or unpublished output can be dropped or cached. Explicit
            // legacy-stream launch composes with native zero-fill and copies.
            let completed = self.runtime.check(
                unsafe { (self.runtime.stream_synchronize)(std::ptr::without_provenance_mut(1)) },
                "cudaStreamSynchronize",
            );
            if completed.is_err() {
                // Completion is uncertain: quarantine executable metadata too.
                std::mem::forget(replay);
            } else {
                drop(replay);
            }
            if launched.is_err() || completed.is_err() {
                CACHE_HEALTHY.store(false, Ordering::Relaxed);
            }
            launched?;
            completed?;
        }
        Ok(result)
    }

    #[cfg(any(feature = "python-bindings", test))]
    pub(crate) fn negate(&self, offset: usize, elements: usize) -> Result<Self, TensorError> {
        self.unary_pointwise(offset, elements, |input, output, count| {
            // SAFETY: unary_pointwise checks bounds, guards the device and holds
            // both allocations through legacy-stream completion, even on errors.
            unsafe { pointwise::launch_negate(input, output, count) }
        })
    }

    pub(crate) fn mul_scalar(
        &self,
        offset: usize,
        elements: usize,
        scalar: f32,
    ) -> Result<Self, TensorError> {
        self.unary_pointwise(offset, elements, |input, output, count| {
            // SAFETY: the shared pointwise lifetime and device contract above.
            unsafe { pointwise::launch_mul_scalar(input, output, count, scalar) }
        })
    }

    fn unary_pointwise(
        &self,
        offset: usize,
        elements: usize,
        launch: impl FnOnce(u64, u64, usize) -> Result<(), TensorError>,
    ) -> Result<Self, TensorError> {
        // Empty views may have offsets beyond storage; never form their pointer.
        if elements != 0
            && offset
                .checked_add(elements)
                .is_none_or(|end| end > self.elements)
        {
            return Err(TensorError::IndexCalculationOverflow);
        }
        let (result, _guard) = Self::allocate(elements, self.device_index)?;
        if elements != 0 {
            // Bounds are checked and both allocations stay live through completion.
            let launched = launch(
                (self.data_ptr + offset * 4) as u64,
                result.data_ptr as u64,
                elements,
            );
            let completed = self.runtime.check(
                unsafe { (self.runtime.stream_synchronize)(std::ptr::without_provenance_mut(1)) },
                "cudaStreamSynchronize",
            );
            if launched.is_err() || completed.is_err() {
                CACHE_HEALTHY.store(false, Ordering::Relaxed);
            }
            launched?;
            completed?;
        }
        Ok(result)
    }

    // Only the initializing constructors above may publish this allocation.
    // Retain one guard across allocation and initialization, including errors.
    fn allocate(elements: usize, device_index: usize) -> Result<(Self, DeviceGuard), TensorError> {
        let bytes = elements
            .checked_mul(4)
            .filter(|bytes| isize::try_from(*bytes).is_ok())
            .ok_or(TensorError::AllocationFailed { elements })?;
        let runtime = runtime()?;
        let guard = runtime.guard(device_index)?;
        let cached = if CACHE_HEALTHY.load(Ordering::Relaxed) {
            let mut cache = CACHE
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            cache
                .iter()
                .enumerate()
                // Best fit accommodates nearby sizes without keeping a live
                // alias or exposing padding as logical tensor elements. Bound
                // excess capacity to avoid pinning a huge block for tiny work.
                .filter(|(_, entry)| {
                    entry.0 == device_index && entry.1 >= bytes && entry.1 - bytes <= bytes / 4
                })
                .min_by_key(|(_, entry)| entry.1)
                .map(|(index, _)| index)
                .map(|index| cache.remove(index))
        } else {
            None
        };
        let mut pointer = cached.map_or(0, |entry| entry.2) as *mut c_void;
        let mut pooled = cached.is_some_and(|entry| entry.3);
        if bytes != 0
            && cached.is_none()
            && CACHE_HEALTHY.load(Ordering::Relaxed)
            && let Some(api) = &runtime.pool
            && let Some(allocation) = api.allocate(runtime, device_index, bytes)?
        {
            pointer = allocation as *mut c_void;
            pooled = true;
        }
        if bytes != 0 && cached.is_none() && !pooled {
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
            allocation_bytes: cached.map_or(bytes, |entry| entry.1),
            pooled,
            runtime,
        };
        INITIALIZED.store(true, Ordering::Relaxed);
        Ok((result, guard))
    }
    pub(crate) fn copy_range(
        &self,
        start: usize,
        elements: usize,
    ) -> Result<Vec<f32>, TensorError> {
        self.copy_regions(
            elements,
            [Ok(CudaCopyRegion {
                start,
                width: elements,
                rows: 1,
                source_pitch: elements,
            })],
        )
    }

    pub(crate) fn copy_regions(
        &self,
        elements: usize,
        regions: impl IntoIterator<Item = Result<CudaCopyRegion, TensorError>>,
    ) -> Result<Vec<f32>, TensorError> {
        // Empty views can carry an offset past storage's end without reading it.
        if elements == 0 {
            return Ok(Vec::new());
        }
        let mut values = Vec::<f32>::new();
        values
            .try_reserve_exact(elements)
            .map_err(|_| TensorError::AllocationFailed { elements })?;
        let _guard = self.runtime.guard(self.device_index)?;
        for region in regions {
            let CudaCopyRegion {
                start,
                width,
                rows,
                source_pitch,
            } = region?;
            let copied = width
                .checked_mul(rows)
                .filter(|&count| count > 0)
                .ok_or(TensorError::IndexCalculationOverflow)?;
            let end = values
                .len()
                .checked_add(copied)
                .filter(|&end| end <= elements)
                .ok_or(TensorError::IndexCalculationOverflow)?;
            let source_end = (rows - 1)
                .checked_mul(source_pitch)
                .and_then(|span| start.checked_add(span))
                .and_then(|last| last.checked_add(width))
                .ok_or(TensorError::IndexCalculationOverflow)?;
            if source_end > self.elements {
                return Err(TensorError::IndexCalculationOverflow);
            }
            // SAFETY: all source rows are within the allocation, whose byte
            // size is checked at construction. The packed destination lies
            // within reserved capacity. Blocking copies initialize every row
            // before set_len; failed copies never expose uninitialized values.
            unsafe {
                let destination = values.as_mut_ptr().add(values.len());
                let source = (self.data_ptr + start * 4) as *const c_void;
                if rows == 1 {
                    self.runtime.check(
                        (self.runtime.memcpy)(destination.cast(), source, width * 4, 2),
                        "cudaMemcpy",
                    )?;
                } else if rows >= 32
                    && source_pitch <= COPY_STAGING_ELEMENTS / 32
                    && source_pitch <= width * 8
                    && source_pitch >= width
                {
                    // Short, closely spaced rows are slow through the 2-D copy
                    // engine. Stage bounded chunks with at most 8x logical
                    // traffic, never the arbitrary backing span of a view.
                    self.copy_staged_rows(destination, start, width, rows, source_pitch)?;
                } else {
                    // Overlapping rows cannot use cudaMemcpy2D. Excessively
                    // wide pitches may also exceed the device's pitch limit;
                    // copy their logical runs individually without staging.
                    let status = if source_pitch >= width {
                        (self.runtime.memcpy_2d)(
                            destination.cast(),
                            width * 4,
                            source,
                            source_pitch * 4,
                            width * 4,
                            rows,
                            2,
                        )
                    } else {
                        12 // cudaErrorInvalidPitchValue
                    };
                    if status == 12 {
                        for row in 0..rows {
                            self.runtime.check(
                                (self.runtime.memcpy)(
                                    destination.add(row * width).cast(),
                                    (self.data_ptr + (start + row * source_pitch) * 4)
                                        as *const c_void,
                                    width * 4,
                                    2,
                                ),
                                "cudaMemcpy",
                            )?;
                        }
                    } else {
                        self.runtime.check(status, "cudaMemcpy2D")?;
                    }
                }
                values.set_len(end);
            }
        }
        if values.len() != elements {
            return Err(TensorError::IndexCalculationOverflow);
        }
        Ok(values)
    }

    /// # Safety
    /// The caller has validated every source row and reserved width * rows
    /// destination floats. The destination must not overlap CUDA storage.
    unsafe fn copy_staged_rows(
        &self,
        destination: *mut f32,
        start: usize,
        width: usize,
        rows: usize,
        source_pitch: usize,
    ) -> Result<(), TensorError> {
        // Source bounds were checked by copy_regions, so this span and all
        // byte offsets below fit in the existing device allocation.
        let elements = ((rows - 1) * source_pitch + width).min(COPY_STAGING_ELEMENTS);
        let mut staging = Vec::<f32>::new();
        staging
            .try_reserve_exact(elements)
            .map_err(|_| TensorError::AllocationFailed { elements })?;
        let chunk_rows = (elements - width) / source_pitch + 1;
        for first_row in (0..rows).step_by(chunk_rows) {
            let copied_rows = (rows - first_row).min(chunk_rows);
            let span = (copied_rows - 1) * source_pitch + width;
            // SAFETY: the staging buffer has capacity for span floats; all
            // source/destination offsets lie in the caller-validated regions.
            // Blocking D2H initializes staging before the host gather reads it.
            unsafe {
                self.runtime.check(
                    (self.runtime.memcpy)(
                        staging.as_mut_ptr().cast(),
                        (self.data_ptr + (start + first_row * source_pitch) * 4) as *const c_void,
                        span * 4,
                        2,
                    ),
                    "cudaMemcpy",
                )?;
                staging.set_len(span);
                if width == 1 {
                    for row in 0..copied_rows {
                        *destination.add(first_row + row) = staging[row * source_pitch];
                    }
                } else {
                    for row in 0..copied_rows {
                        std::ptr::copy_nonoverlapping(
                            staging.as_ptr().add(row * source_pitch),
                            destination.add((first_row + row) * width),
                            width,
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
impl Drop for CudaFloat32Storage {
    fn drop(&mut self) {
        if self.data_ptr == 0 {
            return;
        }
        let bytes = self.allocation_bytes;
        let mut evicted = Vec::new();
        {
            let mut cache = CACHE
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if CACHE_HEALTHY.load(Ordering::Relaxed) && bytes <= CACHE_BYTES {
                // Replace the least recently returned allocations instead of
                // permanently rejecting new sizes after mixed-size workloads.
                while cache.len() >= 32
                    || cache.iter().map(|entry| entry.1).sum::<usize>() > CACHE_BYTES - bytes
                {
                    evicted.push(cache.remove(0));
                }
                cache.push((self.device_index, bytes, self.data_ptr, self.pooled));
            } else {
                evicted.push((self.device_index, bytes, self.data_ptr, self.pooled));
            }
        }
        for (device, capacity, pointer, pooled) in evicted {
            if let Ok(_guard) = self.runtime.guard(device) {
                if pooled {
                    // Never recycle pool memory after a failed launch/wait.
                    // Quarantine it if completion could not be established.
                    if CACHE_HEALTHY.load(Ordering::Relaxed) {
                        // SAFETY: no owner remains; zero-fill, transfers, add
                        // and release are ordered on this device's legacy
                        // stream, even when their host threads differ. Add
                        // still completes before return; host readback blocks.
                        let status = unsafe {
                            self.runtime
                                .pool
                                .as_ref()
                                .expect("pooled allocation")
                                .free(pointer, capacity, device)
                        };
                        if status != 0 {
                            CACHE_HEALTHY.store(false, Ordering::Relaxed);
                        }
                    }
                } else {
                    // SAFETY: removed cache entries have no live owners.
                    unsafe {
                        (self.runtime.free)(pointer as *mut c_void);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Device, Tensor};
    #[test]
    fn cuda_allocation_checks_byte_overflow_before_loading_runtime() {
        for elements in [usize::MAX, isize::MAX as usize / size_of::<f32>() + 1] {
            assert!(matches!(
                super::CudaFloat32Storage::allocate(elements, 0),
                Err(crate::TensorError::AllocationFailed { .. })
            ));
        }
    }

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

    #[test]
    fn negation_bounds_and_bitwise_values_without_python() {
        if super::device_count() == 0 {
            eprintln!("skipping native CUDA negation: no CUDA runtime/device");
            return;
        }
        let bits = [
            0,
            0x8000_0000,
            1,
            0x8000_0001,
            0x007f_ffff,
            0x0080_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0xffc1_2345,
            0x7f80_0001,
        ];
        let input = Tensor::from_vec(bits.map(f32::from_bits).to_vec(), [bits.len()])
            .unwrap()
            .try_copy_cpu_to_cuda(Device::Cuda(0))
            .unwrap();
        let result = input.negate().unwrap();
        let read_bits = |tensor: &Tensor| {
            tensor
                .try_copy_cuda_to_cpu()
                .unwrap()
                .try_to_vec()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        assert_eq!(read_bits(&input), bits);
        drop(input);
        assert_eq!(read_bits(&result), bits.map(|bit| bit ^ 0x8000_0000));

        let input = super::CudaFloat32Storage::from_host(&[9.0, 1.25, -2.5], 0).unwrap();
        for (offset, elements) in [(3, 1), (usize::MAX, 2), (0, 4)] {
            assert!(matches!(
                input.negate(offset, elements),
                Err(crate::TensorError::IndexCalculationOverflow)
            ));
        }
        assert_eq!(
            input.negate(1, 2).unwrap().copy_range(0, 2).unwrap(),
            [-1.25, 2.5]
        );
        assert_eq!(input.negate(usize::MAX, 0).unwrap().elements, 0);
    }

    #[test]
    fn addition_checks_storage_bounds_and_restores_device() {
        if super::device_count() == 0 {
            eprintln!("skipping CUDA addition bounds: no CUDA runtime/device");
            return;
        }
        let left = super::CudaFloat32Storage::from_host(&[1.25, -2.5, 4.0], 0).unwrap();
        let right = super::CudaFloat32Storage::from_host(&[3.0, 7.5, -8.0], 0).unwrap();
        for (a, b, n) in [(3, 0, 1), (0, 3, 1), (usize::MAX, 0, 2), (0, 0, 4)] {
            assert!(matches!(
                left.add(a, &right, b, n),
                Err(crate::TensorError::IndexCalculationOverflow)
            ));
        }
        assert_eq!(
            left.add(1, &right, 0, 2).unwrap().copy_range(0, 2).unwrap(),
            [0.5, 11.5]
        );
        assert_eq!(
            left.add(usize::MAX, &right, usize::MAX, 0)
                .unwrap()
                .elements,
            0
        );
        if std::env::var("CUDA_VISIBLE_DEVICES").as_deref() != Ok("0,1")
            || super::device_count() < 2
        {
            eprintln!(
                "skipping two-device CUDA addition guard check: requires CUDA_VISIBLE_DEVICES=0,1"
            );
            return;
        }
        let runtime = super::runtime().unwrap();
        let _current = runtime.guard(1).unwrap();
        let output = left.add(0, &right, 0, 3).unwrap();
        let other_device = super::CudaFloat32Storage::zeros(3, 1).unwrap();
        assert!(matches!(
            left.add(0, &other_device, 0, 3),
            Err(crate::TensorError::UnsupportedCudaAddition { .. })
        ));
        assert_eq!(output.copy_range(0, 3).unwrap(), [4.25, 5.0, -4.0]);
        drop(output);
        let mut current = -1;
        // SAFETY: writable output pointer and the loaded runtime's device ABI.
        runtime
            .check(
                unsafe { (runtime.get_device)(&raw mut current) },
                "cudaGetDevice",
            )
            .unwrap();
        assert_eq!(current, 1);
    }

    #[test]
    fn scalar_multiplication_bounds_and_launch_failure_cleanup() {
        use super::{CACHE_HEALTHY, CudaFloat32Storage, Ordering};
        use crate::TensorError;
        if super::device_count() == 0 {
            eprintln!("skipping CUDA multiplication errors: no CUDA runtime/device");
            return;
        }
        let input = CudaFloat32Storage::from_host(&[1.25, -2.5, 4.0], 0).unwrap();
        for (offset, count) in [(3, 1), (usize::MAX, 2), (0, 4)] {
            assert!(matches!(
                input.mul_scalar(offset, count, 2.0),
                Err(TensorError::IndexCalculationOverflow)
            ));
        }
        assert_eq!(
            input.mul_scalar(usize::MAX, 0, f32::NAN).unwrap().elements,
            0
        );
        assert_eq!(
            input
                .mul_scalar(1, 2, -1.5)
                .unwrap()
                .copy_range(0, 2)
                .unwrap(),
            [3.75, -6.0]
        );
        let runtime = super::runtime().unwrap();
        let current = usize::from(
            std::env::var("CUDA_VISIBLE_DEVICES").as_deref() == Ok("0,1")
                && super::device_count() >= 2,
        );
        let _guard = runtime.guard(current).unwrap();
        // Inject a launch failure at the shared launch boundary, after a real
        // allocation on device 0. Never induce a device fault or exhaust the GPU.
        let error = TensorError::CudaRuntimeError {
            operation: "cuLaunchKernel",
            message: "injected launch failure".into(),
        };
        let result = input.unary_pointwise(0, 3, |_, _, _| Err(error.clone()));
        assert!(matches!(result, Err(actual) if actual == error));
        assert!(!CACHE_HEALTHY.load(Ordering::Relaxed));
        let mut restored = -1;
        // SAFETY: valid writable ordinal and legacy stream handle.
        unsafe {
            runtime
                .check((runtime.get_device)(&raw mut restored), "cudaGetDevice")
                .unwrap();
            runtime
                .check(
                    (runtime.stream_synchronize)(std::ptr::without_provenance_mut(1)),
                    "cudaStreamSynchronize",
                )
                .unwrap();
        }
        assert_eq!(usize::try_from(restored).unwrap(), current);
        assert_eq!(input.copy_range(0, 3).unwrap(), [1.25, -2.5, 4.0]);
        // A handled error leaves future correct execution usable, with reuse disabled.
        assert_eq!(
            input
                .mul_scalar(0, 3, 2.0)
                .unwrap()
                .copy_range(0, 3)
                .unwrap(),
            [2.5, -5.0, 8.0]
        );
    }

    #[test]
    fn packed_copy_regions_validate_bounds_and_output_length() {
        use super::{CudaCopyRegion, CudaFloat32Storage};
        if super::device_count() == 0 {
            eprintln!("skipping CUDA region copies: no CUDA runtime/device");
            return;
        }
        let storage = CudaFloat32Storage::zeros(16, 0).unwrap();
        for source_pitch in [1, 5] {
            let region = CudaCopyRegion {
                start: 1,
                width: 2,
                rows: 3,
                source_pitch,
            };
            assert_eq!(storage.copy_regions(6, [Ok(region)]).unwrap(), [0.0; 6]);
        }
        for (elements, start, width, rows, source_pitch) in [
            (1, 16, 1, 1, 1), // source out of bounds
            (4, 0, 2, 2, 16), // final source row out of bounds
            (1, 0, 2, 1, 2),  // output overrun
            (2, 0, 1, 1, 1),  // output underrun after a successful copy
            (1, 0, 1, 0, 1),
            (1, 0, 0, 1, 1),
            (1, 0, usize::MAX, 2, 1),
            (2, 1, 1, 2, usize::MAX),
        ] {
            let region = CudaCopyRegion {
                start,
                width,
                rows,
                source_pitch,
            };
            assert!(matches!(
                storage.copy_regions(elements, [Ok(region)]),
                Err(crate::TensorError::IndexCalculationOverflow)
            ));
        }
    }
}

//! Optional, privately owned CUDA stream-ordered pools. The default pool used
//! by other libraries is never changed. Handles live as long as the runtime.
use super::{Library, Mutex, Runtime, Status, TensorError, c_int, c_void};

#[repr(C)]
struct Properties {
    allocation_type: c_int,
    handle_types: c_int,
    location_type: c_int,
    location_id: c_int,
    security: *mut c_void,
    // CUDA 12/13's maxSize, usage and reserved fields occupy these 64 bytes.
    // Zero selects the default maximum, no special usage and no reserved flags.
    reserved: [u8; 64],
}

pub(super) struct Api {
    attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> Status,
    create: unsafe extern "C" fn(*mut *mut c_void, *const Properties) -> Status,
    destroy: unsafe extern "C" fn(*mut c_void) -> Status,
    set_attribute: unsafe extern "C" fn(*mut c_void, c_int, *mut c_void) -> Status,
    allocate: unsafe extern "C" fn(*mut *mut c_void, usize, *mut c_void, *mut c_void) -> Status,
    free: unsafe extern "C" fn(*mut c_void, *mut c_void) -> Status,
    // None is cached only for devices reporting no pool support.
    devices: Mutex<Vec<(usize, Option<Pool>)>>,
}

struct Pool {
    handle: usize,
    // Includes cached allocations until they are returned with freeAsync.
    active_bytes: usize,
}

impl Api {
    pub(super) unsafe fn load(library: &Library) -> Result<Self, libloading::Error> {
        // SAFETY: CUDA runtime C ABI; the parent Runtime retains the library.
        unsafe {
            Ok(Self {
                attribute: *library.get(b"cudaDeviceGetAttribute\0")?,
                create: *library.get(b"cudaMemPoolCreate\0")?,
                destroy: *library.get(b"cudaMemPoolDestroy\0")?,
                set_attribute: *library.get(b"cudaMemPoolSetAttribute\0")?,
                allocate: *library.get(b"cudaMallocFromPoolAsync\0")?,
                free: *library.get(b"cudaFreeAsync\0")?,
                devices: Mutex::new(Vec::new()),
            })
        }
    }

    fn create_pool(&self, runtime: &Runtime, device: usize) -> Result<Option<Pool>, TensorError> {
        // The runtime guard has already validated that the ordinal fits int.
        let ordinal = c_int::try_from(device).expect("guarded device ordinal");
        let mut supported = 0;
        // SAFETY: writable integer and cudaDevAttrMemoryPoolsSupported (115).
        runtime.check(
            unsafe { (self.attribute)(&raw mut supported, 115, ordinal) },
            "cudaDeviceGetAttribute",
        )?;
        if supported == 0 {
            return Ok(None);
        }
        let properties = Properties {
            allocation_type: 1, // cudaMemAllocationTypePinned
            handle_types: 0,
            location_type: 1, // cudaMemLocationTypeDevice
            location_id: ordinal,
            security: std::ptr::null_mut(),
            reserved: [0; 64],
        };
        let mut pool = std::ptr::null_mut();
        // SAFETY: documented, zero-initialized CUDA 12/13 property layout.
        runtime.check(
            unsafe { (self.create)(&raw mut pool, &raw const properties) },
            "cudaMemPoolCreate",
        )?;
        let mut threshold: u64 = 256 * 1024 * 1024;
        // Initially there is no live storage. Allocation/release adjust the
        // threshold to active bytes + 256 MiB so large live inputs do not
        // force every reusable output block out of the pool at each wait.
        let configured = runtime.check(
            unsafe { (self.set_attribute)(pool, 4, (&raw mut threshold).cast()) },
            "cudaMemPoolSetAttribute",
        );
        if let Err(error) = configured {
            // SAFETY: this pool has not published any allocations.
            unsafe {
                (self.destroy)(pool);
            }
            return Err(error);
        }
        Ok(Some(Pool {
            handle: pool as usize,
            active_bytes: 0,
        }))
    }

    fn threshold(&self, pool: &Pool, active_bytes: usize) -> Status {
        let mut threshold = (active_bytes as u64).saturating_add(256 * 1024 * 1024);
        // SAFETY: valid owned pool and a host u64 copied during this ABI call.
        unsafe { (self.set_attribute)(pool.handle as *mut c_void, 4, (&raw mut threshold).cast()) }
    }

    /// Caller holds the device guard and must initialize on the legacy stream.
    pub(super) fn allocate(
        &self,
        runtime: &Runtime,
        device: usize,
        bytes: usize,
    ) -> Result<Option<usize>, TensorError> {
        let mut devices = self
            .devices
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let index = if let Some(index) = devices.iter().position(|entry| entry.0 == device) {
            index
        } else {
            devices
                .try_reserve(1)
                .map_err(|_| TensorError::AllocationFailed { elements: 1 })?;
            let pool = self.create_pool(runtime, device)?;
            devices.push((device, pool));
            devices.len() - 1
        };
        let Some(pool) = devices[index].1.as_mut() else {
            return Ok(None);
        };
        let active = pool
            .active_bytes
            .checked_add(bytes)
            .ok_or(TensorError::AllocationFailed {
                elements: bytes / 4,
            })?;
        runtime.check(self.threshold(pool, active), "cudaMemPoolSetAttribute")?;
        let mut pointer = std::ptr::null_mut();
        // SAFETY: pool belongs to the guarded device. Every native operation,
        // including driver launches, uses this same explicit legacy stream.
        let allocated = runtime.check(
            unsafe {
                (self.allocate)(
                    &raw mut pointer,
                    bytes,
                    pool.handle as *mut c_void,
                    std::ptr::without_provenance_mut(1),
                )
            },
            "cudaMallocFromPoolAsync",
        );
        if let Err(error) = allocated {
            self.threshold(pool, pool.active_bytes);
            return Err(error);
        }
        pool.active_bytes = active;
        Ok(Some(pointer as usize))
    }

    /// # Safety
    /// Pointer belongs to this API on the guarded device, has no live owners,
    /// and all its native uses were enqueued on the legacy stream. `bytes` is
    /// its original allocation capacity, including after front-cache reuse.
    pub(super) unsafe fn free(&self, pointer: usize, bytes: usize, device: usize) -> Status {
        let mut devices = self
            .devices
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let pool = devices
            .iter_mut()
            .find(|entry| entry.0 == device)
            .and_then(|entry| entry.1.as_mut())
            .expect("owned allocation pool");
        let active = pool
            .active_bytes
            .checked_sub(bytes)
            .expect("owned allocation capacity");
        let configured = self.threshold(pool, active);
        if configured != 0 {
            return configured;
        }
        // SAFETY: the caller owns the pointer and holds its device guard.
        let status =
            unsafe { (self.free)(pointer as *mut c_void, std::ptr::without_provenance_mut(1)) };
        if status == 0 {
            pool.active_bytes = active;
        } else {
            self.threshold(pool, pool.active_bytes);
        }
        status
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // The ownership-only field is intentionally named _library in production.
    #[allow(clippy::used_underscore_binding)]
    fn pool_budgets_unused_memory_independently_of_large_live_allocations() {
        if super::super::device_count() == 0 {
            eprintln!("skipping CUDA pool accounting: no CUDA device");
            return;
        }
        let runtime = super::super::runtime().unwrap();
        let _guard = runtime.guard(0).unwrap();
        // SAFETY: runtime owns the library for the duration of this test.
        let Ok(api) = (unsafe { Api::load(&runtime._library) }) else {
            eprintln!("skipping CUDA pool accounting: no pool symbols");
            return;
        };
        let first_bytes = 300 * 1024 * 1024;
        let second_bytes = 128 * 1024 * 1024;
        let Some(first) = api.allocate(runtime, 0, first_bytes).unwrap() else {
            eprintln!("skipping CUDA pool accounting: device lacks pool support");
            return;
        };
        let second = api.allocate(runtime, 0, second_bytes).unwrap().unwrap();
        let handle = api.devices.lock().unwrap()[0].1.as_ref().unwrap().handle;
        // SAFETY: CUDA's documented attribute ABI and a live private pool.
        let attribute = unsafe {
            *runtime
                ._library
                .get::<unsafe extern "C" fn(*mut c_void, c_int, *mut c_void) -> Status>(
                    b"cudaMemPoolGetAttribute\0",
                )
                .unwrap()
        };
        let query = |kind| {
            let mut value = 0_u64;
            assert_eq!(
                unsafe { attribute(handle as *mut c_void, kind, (&raw mut value).cast()) },
                0
            );
            value
        };
        assert_eq!(
            query(4),
            (first_bytes + second_bytes + 256 * 1024 * 1024) as u64
        );
        // SAFETY: both pointers are exclusively owned, with allocation and
        // release ordered on the same legacy stream. No data is read.
        assert_eq!(unsafe { api.free(first, first_bytes, 0) }, 0);
        assert_eq!(query(4), (second_bytes + 256 * 1024 * 1024) as u64);
        assert_eq!(unsafe { api.free(second, second_bytes, 0) }, 0);
        assert_eq!(query(4), 256 * 1024 * 1024);
        assert_eq!(
            unsafe { (runtime.stream_synchronize)(std::ptr::without_provenance_mut(1)) },
            0
        );
        assert_eq!(query(7), 0); // cudaMemPoolAttrUsedMemCurrent
        assert!(query(5) <= 256 * 1024 * 1024); // reserved backing after draining
        // SAFETY: the private test pool has no live or pending allocations.
        assert_eq!(unsafe { (api.destroy)(handle as *mut c_void) }, 0);
    }
}

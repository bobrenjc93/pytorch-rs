//! Bounded exact-argument graph replay. Entries own executable metadata, never
//! tensor storage. A hit is usable only while the caller borrows live buffers
//! with exactly the recorded addresses, element count, function and context.
use super::{Library, Mutex, OnceLock, c_void};
use std::sync::Arc;

type Handle = *mut c_void;
type Status = i32;

#[repr(C)]
pub(super) struct Kernel {
    pub function: Handle,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_bytes: u32,
    pub arguments: *mut Handle,
    pub extra: *mut Handle,
}

struct Api {
    _library: Library,
    create: unsafe extern "C" fn(*mut Handle, u32) -> Status,
    add: unsafe extern "C" fn(*mut Handle, Handle, *const Handle, usize, *const Kernel) -> Status,
    instantiate: unsafe extern "C" fn(*mut Handle, Handle, u64) -> Status,
    launch: unsafe extern "C" fn(Handle, Handle) -> Status,
    destroy: unsafe extern "C" fn(Handle) -> Status,
    destroy_exec: unsafe extern "C" fn(Handle) -> Status,
    context: unsafe extern "C" fn(*mut Handle) -> Status,
    push: unsafe extern "C" fn(Handle) -> Status,
    pop: unsafe extern "C" fn(*mut Handle) -> Status,
}
static API: OnceLock<Option<Api>> = OnceLock::new();

fn api() -> Option<&'static Api> {
    API.get_or_init(|| {
        // SAFETY: optional documented driver ABI; the library owns all symbols.
        unsafe {
            let library = Library::new(if cfg!(windows) {
                "nvcuda.dll"
            } else {
                "libcuda.so.1"
            })
            .ok()?;
            Some(Api {
                create: *library.get(b"cuGraphCreate\0").ok()?,
                // Explicit v1 symbol and layout, not the CUDA header's v2 alias.
                add: *library.get(b"cuGraphAddKernelNode\0").ok()?,
                instantiate: *library.get(b"cuGraphInstantiateWithFlags\0").ok()?,
                launch: *library.get(b"cuGraphLaunch\0").ok()?,
                destroy: *library.get(b"cuGraphDestroy\0").ok()?,
                destroy_exec: *library.get(b"cuGraphExecDestroy\0").ok()?,
                context: *library.get(b"cuCtxGetCurrent\0").ok()?,
                push: *library.get(b"cuCtxPushCurrent_v2\0").ok()?,
                pop: *library.get(b"cuCtxPopCurrent_v2\0").ok()?,
                _library: library,
            })
        }
    })
    .as_ref()
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Key(pub usize, pub u64, pub u64, pub u64, pub usize);

pub(super) struct Replay {
    exec: usize,
    context: usize,
    api: &'static Api,
    // CUDA graph objects are not thread safe. Serialize host launch calls for
    // one executable; GPU submissions still follow explicit legacy ordering.
    launch_lock: Mutex<()>,
}

impl Replay {
    /// Caller must keep this Arc AND all matching buffers alive through the
    /// subsequent stream wait, including when launch returns an error.
    pub(super) unsafe fn launch(&self) -> Status {
        let _lock = self
            .launch_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        unsafe { (self.api.launch)(self.exec as Handle, std::ptr::without_provenance_mut(1)) }
    }
}

impl Drop for Replay {
    fn drop(&mut self) {
        // Last ownership is released only after all callers have waited. Cache
        // eviction may occur on a different device/thread: restore its context
        // stack after destroying metadata in the executable's owning context.
        unsafe {
            if (self.api.push)(self.context as Handle) == 0 {
                (self.api.destroy_exec)(self.exec as Handle);
                let mut previous = std::ptr::null_mut();
                (self.api.pop)(&raw mut previous);
            }
        }
    }
}

#[derive(Default)]
struct Cache {
    seen: Vec<(Key, u8)>,
    entries: Vec<(Key, Arc<Replay>)>,
}
static CACHE: Mutex<Cache> = Mutex::new(Cache {
    seen: Vec::new(),
    entries: Vec::new(),
});
const LIMIT: usize = 64;

/// The function handle is context-specific and its module is never unloaded.
/// All launch dimensions derive solely from this key. Pointer reuse is safe:
/// this checks current live arguments, not identities of former allocations.
pub(super) unsafe fn get(key: Key, kernel: &Kernel) -> Option<Arc<Replay>> {
    let api = api()?;
    let mut cache = CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((_, replay)) = cache.entries.iter().find(|entry| entry.0 == key) {
        return Some(Arc::clone(replay));
    }
    if let Some((_, hits)) = cache.seen.iter_mut().find(|entry| entry.0 == key) {
        *hits = hits.saturating_add(1);
        if *hits < 3 {
            return None;
        }
    } else {
        if cache.seen.len() == LIMIT {
            cache.seen.remove(0);
        }
        cache.seen.push((key, 1));
        return None;
    }
    let mut context = std::ptr::null_mut();
    let mut graph = std::ptr::null_mut();
    let mut node = std::ptr::null_mut();
    let mut exec = std::ptr::null_mut();
    // Build only repeated keys. Graph creation copies kernel parameters and
    // performs no GPU work. Optional graph failure falls back to direct launch.
    unsafe {
        if (api.context)(&raw mut context) != 0
            || context.is_null()
            || (api.create)(&raw mut graph, 0) != 0
        {
            return None;
        }
        let added = (api.add)(&raw mut node, graph, std::ptr::null(), 0, kernel);
        let built = added == 0 && (api.instantiate)(&raw mut exec, graph, 0) == 0;
        (api.destroy)(graph);
        if !built {
            return None;
        }
    }
    let replay = Arc::new(Replay {
        exec: exec as usize,
        context: context as usize,
        api,
        launch_lock: Mutex::new(()),
    });
    let retired = if cache.entries.len() == LIMIT {
        Some(cache.entries.remove(0))
    } else {
        None
    };
    cache.entries.push((key, Arc::clone(&replay)));
    drop(cache);
    drop(retired);
    Some(replay)
}

#[cfg(test)]
mod tests {
    #[test]
    fn repeated_live_arguments_use_bounded_graph_cache() {
        if crate::cuda::device_count() == 0 || super::api().is_none() {
            eprintln!("skipping graph replay: no CUDA device/graph driver API");
            return;
        }
        let x = crate::Tensor::from_vec(vec![1.25; 3079], [3079])
            .unwrap()
            .try_copy_cpu_to_cuda(crate::Device::Cuda(0))
            .unwrap();
        for _ in 0..12 {
            let out = x.add(&x).unwrap();
            assert_eq!(
                out.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
                vec![2.5; 3079]
            );
        }
        let cache = super::CACHE.lock().unwrap();
        assert!(
            cache
                .entries
                .iter()
                .any(|entry| entry.0.1 == x.data_ptr() as u64)
        );
        assert!(cache.entries.len() <= super::LIMIT && cache.seen.len() <= super::LIMIT);
    }
}

//! A pending driver link belongs to compilation, not the executable cache.
//! The existing module owner takes over before the link-owned cubin expires.
use super::super::Driver;
use super::{CStr, TensorError, c_char, c_int, c_void, invalid};

struct Api {
    create: unsafe extern "C" fn(u32, *mut c_int, *mut *mut c_void, *mut *mut c_void) -> c_int,
    add: unsafe extern "C" fn(
        *mut c_void,
        c_int,
        *mut c_void,
        usize,
        *const c_char,
        u32,
        *mut c_int,
        *mut *mut c_void,
    ) -> c_int,
    complete: unsafe extern "C" fn(*mut c_void, *mut *mut c_void, *mut usize) -> c_int,
    destroy: unsafe extern "C" fn(*mut c_void) -> c_int,
}
impl Api {
    fn load(library: &super::Library) -> Result<Self, TensorError> {
        // SAFETY: documented CUDA driver ABI; the existing driver library owns
        // these symbols. Resolve only for a program needing device linking.
        unsafe {
            (|| -> Result<Self, libloading::Error> {
                Ok(Self {
                    create: *library.get(b"cuLinkCreate_v2\0")?,
                    add: *library.get(b"cuLinkAddData_v2\0")?,
                    complete: *library.get(b"cuLinkComplete\0")?,
                    destroy: *library.get(b"cuLinkDestroy\0")?,
                })
            })()
            .map_err(|error| invalid(format!("cannot load CUDA device linker: {error}")))
        }
    }
}
struct Pending<'a> {
    driver: &'a Driver,
    api: Api,
    state: *mut c_void,
}
impl<'a> Pending<'a> {
    fn new(driver: &'a Driver) -> Result<Self, TensorError> {
        // Ordinary kernels retain this existing owner only for symbol lifetime.
        #[allow(clippy::used_underscore_binding)]
        let api = Api::load(&driver._library)?;
        let mut state = std::ptr::null_mut();
        // SAFETY: no options, writable output, current context held by caller.
        driver.check(
            unsafe {
                (api.create)(
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &raw mut state,
                )
            },
            "cuLinkCreate_v2",
        )?;
        if state.is_null() {
            return Err(invalid("CUDA device linker returned a null state"));
        }
        Ok(Self { driver, api, state })
    }
    fn add_ptx(&self, input: &[u8], name: &CStr) -> Result<(), TensorError> {
        CStr::from_bytes_with_nul(input).map_err(|_| invalid("invalid device-link PTX"))?;
        // The driver's mutable ABI receives owned writable bytes. It retains
        // no input references after returning, so storage can then be released.
        let mut input = input.to_vec();
        // SAFETY: live NUL-terminated PTX, valid state/name, no input options.
        self.driver.check(
            unsafe {
                (self.api.add)(
                    self.state,
                    1, // CU_JIT_INPUT_PTX
                    input.as_mut_ptr().cast(),
                    input.len(),
                    name.as_ptr(),
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            },
            "cuLinkAddData_v2",
        )
    }
    fn load_module(&self, name: &CStr) -> Result<(usize, usize), TensorError> {
        let mut cubin = std::ptr::null_mut();
        let mut size = 0;
        // SAFETY: state is live and output handles are writable.
        self.driver.check(
            unsafe { (self.api.complete)(self.state, &raw mut cubin, &raw mut size) },
            "cuLinkComplete",
        )?;
        if cubin.is_null() || size == 0 {
            return Err(invalid("CUDA device linker returned an empty image"));
        }
        // SAFETY: cubin belongs to this live pending link. Module loading
        // consumes it before Drop destroys the state. No copy survives.
        unsafe { super::load_module(self.driver, cubin, name) }
    }
}
impl Drop for Pending<'_> {
    fn drop(&mut self) {
        // SAFETY: exclusive successfully-created state, context still guarded.
        let status = unsafe { (self.api.destroy)(self.state) };
        #[cfg(test)]
        if status == 0 {
            DESTROYED.with(|count| count.set(count.get() + 1));
        }
        #[cfg(not(test))]
        let _ = status;
    }
}
pub(super) fn load(
    driver: &Driver,
    executor: &[u8],
    provider: &[u8],
    name: &CStr,
) -> Result<(usize, usize), TensorError> {
    let pending = Pending::new(driver)?;
    pending.add_ptx(executor, c"pointwise.ptx")?;
    pending.add_ptx(provider, c"erf_provider.ptx")?;
    pending.load_module(name)
}
#[cfg(test)]
thread_local! {
    static DESTROYED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
#[cfg(test)]
mod tests {
    use super::*;
    // Minimal linker ABI fixtures; actual Erf math is tested through Graph.
    const EXECUTOR: &[u8] = b".version 6.0\n.target sm_50\n.address_size 64\n\
        .extern .func dependency();\n\
        .visible .entry test_kernel() { call dependency, (); ret; }\n\0";
    const PROVIDER: &[u8] = b".version 6.0\n.target sm_50\n.address_size 64\n\
        .visible .func dependency() { ret; }\n\0";

    #[test]
    #[cfg(unix)]
    fn missing_link_symbols_fail_closed() {
        let library: super::super::Library = libloading::os::unix::Library::this().into();
        let error = Api::load(&library).err().expect("missing linker symbol");
        assert!(error.to_string().contains("cuLinkCreate_v2"));
    }

    #[test]
    fn link_failures_release_state_and_success_survives_state_destruction() {
        if crate::cuda::device_count() == 0 {
            eprintln!("skipping driver linker failure/retry: CUDA unavailable");
            return;
        }
        let _guard = super::super::runtime().unwrap().guard(0).unwrap();
        super::super::current_context().unwrap();
        let driver = super::super::driver().unwrap();
        let before = DESTROYED.with(std::cell::Cell::get);
        for (offset, provider, entry, operation) in [
            (1, &b"not PTX\0"[..], c"test_kernel", "cuLinkAddData_v2"),
            (
                2,
                &b".version 6.0\n.target sm_50\n.address_size 64\n\0"[..],
                c"test_kernel",
                "cuLinkComplete",
            ),
            (3, PROVIDER, c"missing_entry", "cuModuleGetFunction"),
            (
                4,
                &b"no terminator"[..],
                c"test_kernel",
                "invalid device-link PTX",
            ),
            (
                5,
                &b".version 999.0\n.target sm_50\n.address_size 64\n\0"[..],
                c"test_kernel",
                "cuLinkAddData_v2",
            ),
        ] {
            let error = load(driver, EXECUTOR, provider, entry).unwrap_err();
            assert!(error.to_string().contains(operation), "{error}");
            assert_eq!(DESTROYED.with(std::cell::Cell::get), before + offset);
        }
        for offset in 6..9 {
            let (module, function) = load(driver, EXECUTOR, PROVIDER, c"test_kernel").unwrap();
            assert_eq!(DESTROYED.with(std::cell::Cell::get), before + offset);
            assert_ne!(function, 0);
            let mut looked_up = std::ptr::null_mut();
            // SAFETY: module owns executable after link-state drop.
            unsafe {
                driver
                    .check(
                        (driver.function)(
                            &raw mut looked_up,
                            module as *mut c_void,
                            c"test_kernel".as_ptr(),
                        ),
                        "cuModuleGetFunction",
                    )
                    .unwrap();
                driver
                    .check((driver.unload)(module as *mut c_void), "cuModuleUnload")
                    .unwrap();
            }
            assert_eq!(looked_up as usize, function);
        }
    }
}

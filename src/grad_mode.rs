use std::cell::Cell;
use std::marker::PhantomData;
use std::rc::Rc;

thread_local! {
    static NO_GRAD_DEPTH: Cell<usize> = const { Cell::new(0) };
}

#[cfg(any(feature = "python-bindings", test))]
thread_local! {
    static MULTITHREADING_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// A thread-local guard which disables eager graph recording until dropped.
///
/// Every live guard contributes one level of suppression, so guards may be
/// dropped in any order without enabling recording prematurely. They are
/// intentionally confined to their creating thread.
pub struct NoGradGuard {
    _not_send: PhantomData<Rc<()>>,
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        exit_no_grad();
    }
}

/// Disables eager graph recording on the current thread for the guard's
/// lifetime.
#[must_use]
pub fn no_grad() -> NoGradGuard {
    enter_no_grad();
    NoGradGuard {
        _not_send: PhantomData,
    }
}

/// Returns whether eager graph recording is enabled on the current thread.
#[must_use]
pub fn is_grad_enabled() -> bool {
    NO_GRAD_DEPTH.get() == 0
}

/// Returns whether multithreaded autograd execution is enabled on the current
/// thread.
#[must_use]
#[cfg(any(feature = "python-bindings", test))]
pub(crate) fn is_multithreading_enabled() -> bool {
    MULTITHREADING_ENABLED.get()
}

/// Sets whether multithreaded autograd execution is enabled on the current
/// thread.
///
/// The current engine executes backward serially in either state. Keeping the
/// control state independent from scheduling preserves `PyTorch`'s public mode
/// semantics without claiming a parallel backward implementation.
#[cfg(any(feature = "python-bindings", test))]
pub(crate) fn set_multithreading_enabled(enabled: bool) {
    MULTITHREADING_ENABLED.set(enabled);
}

pub(crate) fn enter_no_grad() {
    NO_GRAD_DEPTH.set(
        NO_GRAD_DEPTH
            .get()
            .checked_add(1)
            .expect("no-grad nesting depth overflowed usize"),
    );
}

pub(crate) fn exit_no_grad() {
    NO_GRAD_DEPTH.set(
        NO_GRAD_DEPTH
            .get()
            .checked_sub(1)
            .expect("no-grad guard exited without a matching entry"),
    );
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::{is_multithreading_enabled, set_multithreading_enabled};

    #[test]
    fn multithreading_state_defaults_to_true_and_is_mutable() {
        set_multithreading_enabled(true);
        assert!(is_multithreading_enabled());

        set_multithreading_enabled(false);
        assert!(!is_multithreading_enabled());

        set_multithreading_enabled(true);
        assert!(is_multithreading_enabled());
    }

    #[test]
    fn multithreading_state_is_thread_local() {
        set_multithreading_enabled(false);

        let worker_states = thread::spawn(|| {
            let initial = is_multithreading_enabled();
            set_multithreading_enabled(false);
            (initial, is_multithreading_enabled())
        })
        .join()
        .expect("worker should complete");

        assert_eq!(worker_states, (true, false));
        assert!(!is_multithreading_enabled());
        set_multithreading_enabled(true);
    }
}

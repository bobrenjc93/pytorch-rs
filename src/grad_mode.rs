use std::cell::Cell;
use std::marker::PhantomData;
use std::rc::Rc;

thread_local! {
    static NO_GRAD_DEPTH: Cell<usize> = const { Cell::new(0) };
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

/// A thread-local guard which enables eager graph recording until dropped.
///
/// Dropping the guard restores the exact no-grad nesting depth that was active
/// when it was created, so nested no-grad guards surrounding it remain intact.
pub struct EnableGradGuard {
    previous_no_grad_depth: usize,
    _not_send: PhantomData<Rc<()>>,
}

impl Drop for EnableGradGuard {
    fn drop(&mut self) {
        restore_grad_mode(self.previous_no_grad_depth);
    }
}

/// Enables eager graph recording on the current thread for the guard's
/// lifetime.
#[must_use]
pub fn enable_grad() -> EnableGradGuard {
    EnableGradGuard {
        previous_no_grad_depth: enter_enable_grad(),
        _not_send: PhantomData,
    }
}

/// Returns whether eager graph recording is enabled on the current thread.
#[must_use]
pub fn is_grad_enabled() -> bool {
    NO_GRAD_DEPTH.get() == 0
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

pub(crate) fn grad_mode_depth() -> usize {
    NO_GRAD_DEPTH.get()
}

pub(crate) fn enter_enable_grad() -> usize {
    let previous_no_grad_depth = grad_mode_depth();
    NO_GRAD_DEPTH.set(0);
    previous_no_grad_depth
}

pub(crate) fn restore_grad_mode(previous_no_grad_depth: usize) {
    NO_GRAD_DEPTH.set(previous_no_grad_depth);
}

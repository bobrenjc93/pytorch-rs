use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

thread_local! {
    static GRAD_MODE_STATE: RefCell<GradModeState> = const { RefCell::new(GradModeState {
        next_token: 0,
        entries: Vec::new(),
    }) };
}

#[derive(Clone, Copy)]
struct GradModeEntry {
    token: usize,
    enabled: bool,
}

struct GradModeState {
    next_token: usize,
    entries: Vec<GradModeEntry>,
}

impl GradModeState {
    fn push(&mut self, enabled: bool) -> usize {
        let token = self.next_token;
        self.next_token = self
            .next_token
            .checked_add(1)
            .expect("grad-mode guard token overflowed usize");
        self.entries.push(GradModeEntry { token, enabled });
        token
    }

    fn remove(&mut self, token: usize) {
        let position = self
            .entries
            .iter()
            .position(|entry| entry.token == token)
            .expect("grad-mode guard exited without a matching entry");
        self.entries.remove(position);
    }

    fn is_grad_enabled(&self) -> bool {
        self.entries.last().is_none_or(|entry| entry.enabled)
    }
}

/// A thread-local guard which disables eager graph recording until dropped.
///
/// Every live guard owns its own mode entry, so guards may be dropped in any
/// order without disturbing entries owned by other guards. They are
/// intentionally confined to their creating thread.
pub struct NoGradGuard {
    token: usize,
    _not_send: PhantomData<Rc<()>>,
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        exit_no_grad(self.token);
    }
}

/// Disables eager graph recording on the current thread for the guard's
/// lifetime.
#[must_use]
pub fn no_grad() -> NoGradGuard {
    let token = enter_no_grad();
    NoGradGuard {
        token,
        _not_send: PhantomData,
    }
}

/// A thread-local guard which enables eager graph recording until dropped.
///
/// Dropping the guard removes only this guard's enable entry, so outstanding
/// no-grad guards remain responsible for their own entries.
pub struct EnableGradGuard {
    token: usize,
    _not_send: PhantomData<Rc<()>>,
}

impl Drop for EnableGradGuard {
    fn drop(&mut self) {
        exit_enable_grad(self.token);
    }
}

/// Enables eager graph recording on the current thread for the guard's
/// lifetime.
#[must_use]
pub fn enable_grad() -> EnableGradGuard {
    EnableGradGuard {
        token: enter_enable_grad(),
        _not_send: PhantomData,
    }
}

/// Returns whether eager graph recording is enabled on the current thread.
#[must_use]
pub fn is_grad_enabled() -> bool {
    GRAD_MODE_STATE.with(|state| state.borrow().is_grad_enabled())
}

pub(crate) fn enter_no_grad() -> usize {
    GRAD_MODE_STATE.with(|state| state.borrow_mut().push(false))
}

pub(crate) fn exit_no_grad(token: usize) {
    GRAD_MODE_STATE.with(|state| state.borrow_mut().remove(token));
}

pub(crate) fn enter_enable_grad() -> usize {
    GRAD_MODE_STATE.with(|state| state.borrow_mut().push(true))
}

pub(crate) fn exit_enable_grad(token: usize) {
    GRAD_MODE_STATE.with(|state| state.borrow_mut().remove(token));
}

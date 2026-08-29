use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::rc::Rc;

thread_local! {
    static GRAD_MODE_STACK: RefCell<Vec<GradModeEntry>> = const { RefCell::new(Vec::new()) };
    static NEXT_GRAD_MODE_TOKEN: Cell<usize> = const { Cell::new(0) };
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct GradModeToken(usize);

struct GradModeEntry {
    token: GradModeToken,
    enabled: bool,
}

/// A thread-local guard which disables eager graph recording until dropped.
///
/// Guards may be dropped in any order without enabling recording prematurely.
/// They are intentionally confined to their creating thread.
pub struct NoGradGuard {
    token: GradModeToken,
    _not_send: PhantomData<Rc<()>>,
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        exit_grad_mode(self.token);
    }
}

/// A thread-local guard which enables eager graph recording until dropped.
///
/// This temporarily overrides enclosing no-grad guards and restores the
/// previous thread-local state when dropped.
pub struct EnableGradGuard {
    token: GradModeToken,
    _not_send: PhantomData<Rc<()>>,
}

impl Drop for EnableGradGuard {
    fn drop(&mut self) {
        exit_grad_mode(self.token);
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

/// Enables eager graph recording on the current thread for the guard's
/// lifetime.
#[must_use]
pub fn enable_grad() -> EnableGradGuard {
    let token = enter_enable_grad();
    EnableGradGuard {
        token,
        _not_send: PhantomData,
    }
}

/// Returns whether eager graph recording is enabled on the current thread.
#[must_use]
pub fn is_grad_enabled() -> bool {
    GRAD_MODE_STACK.with(|stack| stack.borrow().last().is_none_or(|entry| entry.enabled))
}

pub(crate) fn enter_no_grad() -> GradModeToken {
    push_grad_mode(false)
}

pub(crate) fn enter_enable_grad() -> GradModeToken {
    push_grad_mode(true)
}

pub(crate) fn try_exit_grad_mode(token: GradModeToken) -> bool {
    GRAD_MODE_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        if let Some(index) = stack.iter().rposition(|entry| entry.token == token) {
            stack.remove(index);
            true
        } else {
            false
        }
    })
}

fn exit_grad_mode(token: GradModeToken) {
    assert!(
        try_exit_grad_mode(token),
        "grad-mode guard exited without a matching entry"
    );
}

fn push_grad_mode(enabled: bool) -> GradModeToken {
    let token = NEXT_GRAD_MODE_TOKEN.with(|next| {
        let token = GradModeToken(next.get());
        next.set(
            next.get()
                .checked_add(1)
                .expect("grad-mode nesting token overflowed usize"),
        );
        token
    });
    GRAD_MODE_STACK.with(|stack| stack.borrow_mut().push(GradModeEntry { token, enabled }));
    token
}

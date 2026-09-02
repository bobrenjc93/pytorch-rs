use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::rc::Rc;

thread_local! {
    static GRAD_MODE_STACK: RefCell<Vec<GradModeEntry>> = const { RefCell::new(Vec::new()) };
    static GRAD_MODE_ENABLED: Cell<bool> = const { Cell::new(true) };
    static NEXT_GRAD_MODE_TOKEN: Cell<usize> = const { Cell::new(0) };
}

#[derive(Clone, Copy)]
pub(crate) struct GradModeToken(usize);

struct GradModeEntry {
    token: GradModeToken,
    previous_enabled: bool,
}

/// A thread-local guard which disables eager graph recording until dropped.
///
/// The guard is intentionally confined to its creating thread. When dropped it
/// restores the gradient mode that was effective before the guard was created.
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
/// The guard is intentionally confined to its creating thread. When dropped it
/// restores the gradient mode that was effective before the guard was created.
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
    GRAD_MODE_ENABLED.with(Cell::get)
}

pub(crate) fn enter_no_grad() -> GradModeToken {
    enter_grad_mode(false)
}

pub(crate) fn enter_enable_grad() -> GradModeToken {
    enter_grad_mode(true)
}

pub(crate) fn enter_set_grad_enabled(enabled: bool, previous_enabled: bool) -> GradModeToken {
    enter_grad_mode_with_previous(enabled, previous_enabled)
}

pub(crate) fn exit_grad_mode(token: GradModeToken) {
    let previous_enabled = GRAD_MODE_STACK.with_borrow_mut(|stack| {
        let position = stack
            .iter()
            .rposition(|entry| entry.token.0 == token.0)
            .expect("grad-mode guard exited without a matching entry");
        stack.remove(position).previous_enabled
    });
    GRAD_MODE_ENABLED.with(|enabled| enabled.set(previous_enabled));
}

fn enter_grad_mode(enabled: bool) -> GradModeToken {
    let previous_enabled = GRAD_MODE_ENABLED.with(|current| current.replace(enabled));
    enter_grad_mode_with_previous(enabled, previous_enabled)
}

fn enter_grad_mode_with_previous(enabled: bool, previous_enabled: bool) -> GradModeToken {
    let token = NEXT_GRAD_MODE_TOKEN.with(|next_token| {
        let token = next_token.get();
        next_token.set(
            token
                .checked_add(1)
                .expect("grad-mode nesting token overflowed usize"),
        );
        GradModeToken(token)
    });
    GRAD_MODE_ENABLED.with(|current| current.set(enabled));
    GRAD_MODE_STACK.with_borrow_mut(|stack| {
        stack.push(GradModeEntry {
            token,
            previous_enabled,
        });
    });
    token
}

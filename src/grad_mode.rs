use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::rc::Rc;

thread_local! {
    static BASE_GRAD_MODE_ENABLED: Cell<bool> = const { Cell::new(true) };
    static GRAD_MODE_STACK: RefCell<Vec<GradModeEntry>> = const { RefCell::new(Vec::new()) };
    static NEXT_GRAD_MODE_TOKEN: Cell<usize> = const { Cell::new(0) };
}

#[derive(Clone, Copy)]
pub(crate) struct GradModeToken(usize);

struct GradModeEntry {
    token: GradModeToken,
    enabled: bool,
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
    GRAD_MODE_STACK.with_borrow(|stack| {
        stack
            .last()
            .map_or_else(|| BASE_GRAD_MODE_ENABLED.get(), |entry| entry.enabled)
    })
}

pub(crate) fn enter_no_grad() -> GradModeToken {
    enter_grad_mode(false)
}

pub(crate) fn enter_enable_grad() -> GradModeToken {
    enter_grad_mode(true)
}

#[cfg_attr(not(feature = "python-bindings"), allow(dead_code))]
pub(crate) fn set_grad_enabled(enabled: bool) -> bool {
    let previous_enabled = is_grad_enabled();
    GRAD_MODE_STACK.with_borrow_mut(|stack| {
        if let Some(entry) = stack.last_mut() {
            entry.enabled = enabled;
        } else {
            BASE_GRAD_MODE_ENABLED.set(enabled);
        }
    });
    previous_enabled
}

pub(crate) fn exit_grad_mode(token: GradModeToken) {
    GRAD_MODE_STACK.with_borrow_mut(|stack| {
        let position = stack
            .iter()
            .rposition(|entry| entry.token.0 == token.0)
            .expect("grad-mode guard exited without a matching entry");
        stack.remove(position);
    });
}

fn enter_grad_mode(enabled: bool) -> GradModeToken {
    let token = NEXT_GRAD_MODE_TOKEN.with(|next_token| {
        let token = next_token.get();
        next_token.set(
            token
                .checked_add(1)
                .expect("grad-mode nesting token overflowed usize"),
        );
        GradModeToken(token)
    });
    GRAD_MODE_STACK.with_borrow_mut(|stack| stack.push(GradModeEntry { token, enabled }));
    token
}

#[cfg(test)]
mod tests {
    use super::{
        BASE_GRAD_MODE_ENABLED, GRAD_MODE_STACK, enable_grad, is_grad_enabled, no_grad,
        set_grad_enabled,
    };

    #[test]
    fn set_grad_enabled_updates_current_state_without_pushing_context_entries() {
        BASE_GRAD_MODE_ENABLED.set(true);
        GRAD_MODE_STACK.with_borrow_mut(Vec::clear);
        assert!(is_grad_enabled());

        assert!(set_grad_enabled(false));
        assert!(!is_grad_enabled());
        GRAD_MODE_STACK.with_borrow(|stack| assert!(stack.is_empty()));

        assert!(!set_grad_enabled(true));
        assert!(is_grad_enabled());
        GRAD_MODE_STACK.with_borrow(|stack| assert!(stack.is_empty()));

        {
            let _guard = no_grad();
            assert!(!is_grad_enabled());
            assert!(!set_grad_enabled(true));
            assert!(is_grad_enabled());
            GRAD_MODE_STACK.with_borrow(|stack| assert_eq!(stack.len(), 1));
        }
        assert!(is_grad_enabled());

        set_grad_enabled(false);
        {
            let _guard = enable_grad();
            assert!(is_grad_enabled());
            assert!(set_grad_enabled(false));
            assert!(!is_grad_enabled());
            GRAD_MODE_STACK.with_borrow(|stack| assert_eq!(stack.len(), 1));
        }
        assert!(!is_grad_enabled());

        let outer = no_grad();
        let inner = no_grad();
        drop(outer);
        assert!(!is_grad_enabled());
        drop(inner);
        assert!(!is_grad_enabled());

        BASE_GRAD_MODE_ENABLED.set(true);
        GRAD_MODE_STACK.with_borrow_mut(Vec::clear);
    }
}

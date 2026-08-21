use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

thread_local! {
    static GRAD_MODE_STATE: RefCell<GradModeState> = const {
        RefCell::new(GradModeState {
            enabled: true,
            no_grad_guards: Vec::new(),
        })
    };
}

struct GradModeState {
    enabled: bool,
    no_grad_guards: Vec<NoGradEntry>,
}

struct NoGradEntry {
    previous: bool,
    active: bool,
}

#[derive(Clone, Copy)]
struct NoGradToken {
    index: usize,
}

/// A thread-local guard which disables eager graph recording until dropped.
///
/// Every live guard contributes one level of suppression, so guards may be
/// dropped in any order without enabling recording prematurely. They are
/// intentionally confined to their creating thread.
pub struct NoGradGuard {
    token: Option<NoGradToken>,
    _not_send: PhantomData<Rc<()>>,
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            exit_no_grad(token);
        }
    }
}

/// Disables eager graph recording on the current thread for the guard's
/// lifetime.
#[must_use]
pub fn no_grad() -> NoGradGuard {
    let token = enter_no_grad();
    NoGradGuard {
        token: Some(token),
        _not_send: PhantomData,
    }
}

/// Returns whether eager graph recording is enabled on the current thread.
#[must_use]
pub fn is_grad_enabled() -> bool {
    GRAD_MODE_STATE.with(|state| state.borrow().enabled)
}

#[cfg(feature = "python-bindings")]
pub(crate) fn set_grad_enabled(enabled: bool) {
    GRAD_MODE_STATE.with(|state| state.borrow_mut().enabled = enabled);
}

fn enter_no_grad() -> NoGradToken {
    GRAD_MODE_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let index = state.no_grad_guards.len();
        let previous = state.enabled;
        state.enabled = false;
        state.no_grad_guards.push(NoGradEntry {
            previous,
            active: true,
        });
        NoGradToken { index }
    })
}

fn exit_no_grad(token: NoGradToken) {
    let _ = GRAD_MODE_STATE.try_with(|state| {
        let mut state = state.borrow_mut();
        let entry = state
            .no_grad_guards
            .get_mut(token.index)
            .expect("no-grad guard exited without a matching entry");
        assert!(entry.active, "no-grad guard exited more than once");
        entry.active = false;

        while state
            .no_grad_guards
            .last()
            .is_some_and(|entry| !entry.active)
        {
            let entry = state
                .no_grad_guards
                .pop()
                .expect("inactive no-grad guard disappeared");
            state.enabled = entry.previous;
        }
    });
}

//! Process-global deterministic-algorithm configuration.

use std::sync::atomic::{AtomicU8, Ordering};

const ENABLED_BIT: u8 = 1;
const WARN_ONLY_BIT: u8 = 1 << 1;

static DETERMINISTIC_ALGORITHMS_STATE: AtomicU8 = AtomicU8::new(0);

#[derive(Clone, Copy)]
pub(crate) struct DeterministicAlgorithmsState(u8);

impl DeterministicAlgorithmsState {
    #[must_use]
    pub(crate) const fn enabled(self) -> bool {
        self.0 & ENABLED_BIT != 0
    }

    #[must_use]
    pub(crate) const fn warn_only(self) -> bool {
        self.0 & WARN_ONLY_BIT != 0
    }

    #[must_use]
    pub(crate) const fn debug_mode(self) -> u8 {
        if !self.enabled() {
            0
        } else if self.warn_only() {
            1
        } else {
            2
        }
    }
}

/// Atomically replaces the process-global deterministic-algorithm policy.
///
/// The warning preference remains meaningful state even while deterministic
/// algorithms are disabled, matching `PyTorch`'s independently queryable flags.
pub(crate) fn set_deterministic_algorithms(enabled: bool, warn_only: bool) {
    let state = (u8::from(enabled) * ENABLED_BIT) | (u8::from(warn_only) * WARN_ONLY_BIT);
    DETERMINISTIC_ALGORITHMS_STATE.store(state, Ordering::SeqCst);
}

/// Returns one coherent snapshot of the process-global policy.
#[must_use]
pub(crate) fn deterministic_algorithms_state() -> DeterministicAlgorithmsState {
    DeterministicAlgorithmsState(DETERMINISTIC_ALGORITHMS_STATE.load(Ordering::SeqCst))
}

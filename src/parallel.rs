//! Explicit intra-op budget. The default remains one worker; environment
//! variables and hardware probes never silently change the public setting.
//! The caller participates, so a budget N creates only N-1 background workers.
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::{Arc, RwLock};

static POOL: RwLock<Option<Arc<ThreadPool>>> = RwLock::new(None);

#[must_use]
pub fn get_num_threads() -> usize {
    POOL.read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .map_or(1, |pool| pool.current_num_threads() + 1)
}

/// Sets the process-wide maximum number of intra-op workers. Existing
/// operations retain their pool until they complete; replacement does not
/// invalidate borrowed tensor storage or interrupt reductions.
///
/// # Errors
/// Returns an error for zero workers or when worker creation fails.
pub fn set_num_threads(threads: usize) -> Result<(), String> {
    if threads == 0 {
        return Err("set_num_threads expects a positive integer".into());
    }
    let mut current = POOL
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if current
        .as_ref()
        .map_or(1, |pool| pool.current_num_threads() + 1)
        == threads
    {
        return Ok(());
    }
    let replacement = if threads == 1 {
        None
    } else {
        Some(Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(threads - 1)
                .thread_name(|index| format!("torch-rs-intraop-{index}"))
                .build()
                .map_err(|error| error.to_string())?,
        ))
    };
    // Destroy retired pools outside the configuration lock. Active calls keep
    // an Arc and finish before the last owner releases their workers.
    let old = std::mem::replace(&mut *current, replacement);
    drop(current);
    drop(old);
    Ok(())
}

pub(crate) fn pool() -> Option<Arc<ThreadPool>> {
    POOL.read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

#[cfg(test)]
mod tests {
    #[test]
    fn pool_replacement_keeps_active_workers_alive() {
        super::set_num_threads(4).unwrap();
        let old = super::pool().unwrap();
        let entered = std::sync::Arc::new(std::sync::Barrier::new(2));
        let resume = std::sync::Arc::new(std::sync::Barrier::new(2));
        let worker = {
            let entered = entered.clone();
            let resume = resume.clone();
            std::thread::spawn(move || {
                old.install(|| {
                    entered.wait();
                    resume.wait();
                    assert_eq!(rayon::current_num_threads(), 3);
                });
            })
        };
        entered.wait();
        super::set_num_threads(2).unwrap();
        assert_eq!(super::get_num_threads(), 2);
        resume.wait();
        worker.join().unwrap();
        assert!(super::set_num_threads(0).is_err());
        assert_eq!(super::get_num_threads(), 2);
        super::set_num_threads(1).unwrap();
    }
}

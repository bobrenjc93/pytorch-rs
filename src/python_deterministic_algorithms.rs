//! Process-global state and bindings for deterministic-algorithm configuration.

use std::sync::atomic::{AtomicU8, Ordering};

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyModule};

use crate::python::python_type_name;

const ENABLED_BIT: u8 = 1;
const WARN_ONLY_BIT: u8 = 1 << 1;

// PyTorch retains the warn-only flag even while deterministic algorithms are
// disabled. Keep both flags in one atomic value so readers never observe a
// partially updated pair, including when Python threads update the setting.
static DETERMINISTIC_ALGORITHMS_STATE: AtomicU8 = AtomicU8::new(0);

fn deterministic_algorithms_state() -> u8 {
    DETERMINISTIC_ALGORITHMS_STATE.load(Ordering::SeqCst)
}

fn strict_bool(value: &Bound<'_, PyAny>, argument: &str) -> PyResult<bool> {
    if !value.is_exact_instance_of::<PyBool>() {
        let type_name = python_type_name(value)?;
        let position = if argument == "mode" {
            " (position 1)"
        } else {
            ""
        };
        return Err(PyTypeError::new_err(format!(
            "_set_deterministic_algorithms(): argument '{argument}'{position} must be bool, not {type_name}"
        )));
    }
    value.is_truthy()
}

#[allow(
    clippy::unnecessary_wraps,
    reason = "PyO3 custom argument converters must return PyResult"
)]
fn provided_object(value: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
    Ok(Some(value.clone().unbind()))
}

#[pyfunction(
    name = "_set_deterministic_algorithms",
    signature = (mode, *, warn_only = None),
    text_signature = None
)]
fn set_deterministic_algorithms(
    py: Python<'_>,
    mode: &Bound<'_, PyAny>,
    #[pyo3(from_py_with = provided_object)] warn_only: Option<Py<PyAny>>,
) -> PyResult<()> {
    let mode = strict_bool(mode, "mode")?;
    let warn_only = warn_only.map_or(Ok(false), |value| {
        strict_bool(&value.into_bound(py), "warn_only")
    })?;
    let state = (u8::from(mode) * ENABLED_BIT) | (u8::from(warn_only) * WARN_ONLY_BIT);
    DETERMINISTIC_ALGORITHMS_STATE.store(state, Ordering::SeqCst);
    Ok(())
}

#[pyfunction(
    name = "_get_deterministic_algorithms",
    signature = (),
    text_signature = None
)]
fn get_deterministic_algorithms() -> bool {
    deterministic_algorithms_state() & ENABLED_BIT != 0
}

#[pyfunction(
    name = "_get_deterministic_algorithms_warn_only",
    signature = (),
    text_signature = None
)]
fn get_deterministic_algorithms_warn_only() -> bool {
    deterministic_algorithms_state() & WARN_ONLY_BIT != 0
}

#[pyfunction(
    name = "_get_deterministic_debug_mode",
    signature = (),
    text_signature = None
)]
fn get_deterministic_debug_mode() -> u8 {
    let state = deterministic_algorithms_state();
    if state & ENABLED_BIT == 0 {
        0
    } else if state & WARN_ONLY_BIT != 0 {
        1
    } else {
        2
    }
}

pub(crate) fn add_deterministic_algorithms_builtins(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(set_deterministic_algorithms, module)?)?;
    module.add_function(wrap_pyfunction!(get_deterministic_algorithms, module)?)?;
    module.add_function(wrap_pyfunction!(
        get_deterministic_algorithms_warn_only,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(get_deterministic_debug_mode, module)?)?;

    let exports = module.getattr("__all__")?;
    for name in [
        "_set_deterministic_algorithms",
        "_get_deterministic_algorithms",
        "_get_deterministic_algorithms_warn_only",
        "_get_deterministic_debug_mode",
    ] {
        exports.call_method1("remove", (name,))?;
    }
    Ok(())
}

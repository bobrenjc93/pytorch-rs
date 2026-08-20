//! Process-global deterministic-algorithm state for the Python API.

use std::sync::atomic::{AtomicU8, Ordering};

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyModule};

use crate::python::python_type_name;

const ENABLED: u8 = 1;
const WARN_ONLY: u8 = 1 << 1;

static DETERMINISTIC_ALGORITHM_STATE: AtomicU8 = AtomicU8::new(0);

fn parse_exact_bool(
    value: &Bound<'_, PyAny>,
    argument: &str,
    position: Option<usize>,
) -> PyResult<bool> {
    if !value.is_exact_instance_of::<PyBool>() {
        let position = position.map_or_else(String::new, |value| format!(" (position {value})"));
        let actual = python_type_name(value)?;
        return Err(PyTypeError::new_err(format!(
            "_set_deterministic_algorithms(): argument '{argument}'{position} must be bool, not {actual}"
        )));
    }
    value.is_truthy()
}

#[pyfunction(signature = (mode, *, warn_only))]
fn _set_deterministic_algorithms(
    mode: &Bound<'_, PyAny>,
    warn_only: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let mode = parse_exact_bool(mode, "mode", Some(1))?;
    let warn_only = parse_exact_bool(warn_only, "warn_only", None)?;
    let state = u8::from(mode) | (u8::from(warn_only) << 1);
    DETERMINISTIC_ALGORITHM_STATE.store(state, Ordering::SeqCst);
    Ok(())
}

#[pyfunction]
fn _get_deterministic_algorithms() -> bool {
    DETERMINISTIC_ALGORITHM_STATE.load(Ordering::SeqCst) & ENABLED != 0
}

#[pyfunction]
fn _get_deterministic_algorithms_warn_only() -> bool {
    DETERMINISTIC_ALGORITHM_STATE.load(Ordering::SeqCst) & WARN_ONLY != 0
}

pub(crate) fn add_deterministic_algorithm_state(module: &Bound<'_, PyModule>) -> PyResult<()> {
    for function in [
        wrap_pyfunction!(_set_deterministic_algorithms, module)?,
        wrap_pyfunction!(_get_deterministic_algorithms, module)?,
        wrap_pyfunction!(_get_deterministic_algorithms_warn_only, module)?,
    ] {
        let name = function.getattr("__name__")?;
        module.add_function(function)?;
        module.getattr("__all__")?.call_method1("remove", (name,))?;
    }
    Ok(())
}

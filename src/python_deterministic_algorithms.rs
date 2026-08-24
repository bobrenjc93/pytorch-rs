//! Native storage bindings for Python deterministic-algorithm configuration.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyModule};

use crate::{
    deterministic_algorithms::{deterministic_algorithms_state, set_deterministic_algorithms},
    python::python_type_name,
};

#[derive(Clone, Copy)]
struct DeterministicMode(bool);

impl<'a, 'py> FromPyObject<'a, 'py> for DeterministicMode {
    type Error = PyErr;

    fn extract(object: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let object = object.to_owned();
        if !object.is_exact_instance_of::<PyBool>() {
            let actual = python_type_name(&object)?;
            return Err(PyTypeError::new_err(format!(
                "_set_deterministic_algorithms(): argument 'mode' (position 1) must be bool, not {actual}"
            )));
        }
        object.is_truthy().map(Self)
    }
}

#[derive(Clone, Copy)]
struct DeterministicWarnOnly(bool);

impl<'a, 'py> FromPyObject<'a, 'py> for DeterministicWarnOnly {
    type Error = PyErr;

    fn extract(object: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let object = object.to_owned();
        if !object.is_exact_instance_of::<PyBool>() {
            let actual = python_type_name(&object)?;
            return Err(PyTypeError::new_err(format!(
                "_set_deterministic_algorithms(): argument 'warn_only' must be bool, not {actual}"
            )));
        }
        object.is_truthy().map(Self)
    }
}

#[pyfunction(
    name = "_set_deterministic_algorithms",
    signature = (mode, *, warn_only=DeterministicWarnOnly(false)),
    text_signature = None
)]
fn set_deterministic_algorithms_native(mode: DeterministicMode, warn_only: DeterministicWarnOnly) {
    set_deterministic_algorithms(mode.0, warn_only.0);
}

#[pyfunction(
    name = "_get_deterministic_algorithms",
    signature = (),
    text_signature = None
)]
fn get_deterministic_algorithms_native() -> bool {
    deterministic_algorithms_state().enabled()
}

#[pyfunction(
    name = "_get_deterministic_algorithms_warn_only",
    signature = (),
    text_signature = None
)]
fn get_deterministic_algorithms_warn_only_native() -> bool {
    deterministic_algorithms_state().warn_only()
}

#[pyfunction(
    name = "_get_deterministic_debug_mode",
    signature = (),
    text_signature = None
)]
fn get_deterministic_debug_mode_native() -> u8 {
    deterministic_algorithms_state().debug_mode()
}

pub(crate) fn add_deterministic_algorithms_builtins(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(
        set_deterministic_algorithms_native,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        get_deterministic_algorithms_native,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        get_deterministic_algorithms_warn_only_native,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        get_deterministic_debug_mode_native,
        module
    )?)?;

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

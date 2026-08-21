//! Python bindings for gradient-mode context managers.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::grad_mode::set_grad_enabled as set_core_grad_enabled;

#[pyfunction(signature = (enabled, /))]
fn _set_grad_enabled(enabled: bool) {
    set_core_grad_enabled(enabled);
}

pub(crate) fn add_grad_mode_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_set_grad_enabled, module)?)?;
    module
        .getattr("__all__")?
        .call_method1("remove", ("_set_grad_enabled",))?;
    Ok(())
}

//! Python scalar-conversion descriptors for native tensors.

use std::ffi::CStr;
use std::sync::atomic::AtomicBool;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyInt, PyType};

use crate::{
    TensorError, is_grad_enabled as core_is_grad_enabled,
    python::{PyTensor, PyTensorBase, dispatch_tensorbase_no_argument_mode, warn_once},
    python_tensor_errors::tensor_error,
};

static SCALAR_REQUIRES_GRAD_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);

#[cfg(target_os = "macos")]
const SCALAR_REQUIRES_GRAD_WARNING: &CStr = c"Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.\nConsider using tensor.detach() first. (Triggered internally at /Users/runner/work/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:823.)";
#[cfg(target_os = "linux")]
const SCALAR_REQUIRES_GRAD_WARNING: &CStr = c"Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.\nConsider using tensor.detach() first. (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:822.)";
#[cfg(target_os = "windows")]
const SCALAR_REQUIRES_GRAD_WARNING: &CStr = c"Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.\nConsider using tensor.detach() first. (Triggered internally at C:\\actions-runner\\_work\\pytorch\\pytorch\\torch\\csrc\\autograd\\generated\\python_variable_methods.cpp:823.)";
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const SCALAR_REQUIRES_GRAD_WARNING: &CStr = c"Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.\nConsider using tensor.detach() first.";

#[pymethods]
impl PyTensorBase {
    #[pyo3(text_signature = None)]
    fn int_scalar<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyInt>> {
        let value = slf
            .as_any()
            .cast::<PyTensor>()?
            .try_borrow()?
            .inner()
            .item()
            .map_err(|error| scalar_conversion_error(&error))?;

        // CPython's float-to-int conversion truncates toward zero, produces a
        // PyLong of any required size, and supplies the canonical infinity and
        // NaN errors. Float32 values are represented exactly as Python floats.
        slf.py()
            .get_type::<PyInt>()
            .call1((f64::from(value),))?
            .cast_into::<PyInt>()
            .map_err(Into::into)
    }

    #[pyo3(text_signature = None)]
    fn float_scalar(slf: &Bound<'_, Self>) -> PyResult<f64> {
        let tensor = slf.as_any().cast::<PyTensor>()?.try_borrow()?;
        warn_if_requires_grad(slf.py(), &tensor)?;
        tensor
            .inner()
            .item()
            .map(f64::from)
            .map_err(|error| scalar_conversion_error(&error))
    }

    #[pyo3(text_signature = None)]
    fn complex_scalar(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let tensor = slf.as_any().cast::<PyTensor>()?;
        if let Some(result) = dispatch_tensorbase_no_argument_mode(slf.py(), tensor, "__complex__")?
        {
            return Ok(result);
        }

        let tensor = tensor.try_borrow()?;
        warn_if_requires_grad(slf.py(), &tensor)?;
        let value = tensor
            .inner()
            .item()
            .map_err(|error| scalar_conversion_error(&error))?;
        Ok(PyComplex::from_doubles(slf.py(), f64::from(value), 0.0)
            .into_any()
            .unbind())
    }
}

fn warn_if_requires_grad(py: Python<'_>, tensor: &PyTensor) -> PyResult<()> {
    if core_is_grad_enabled() && tensor.inner().requires_grad() {
        warn_once(
            py,
            &SCALAR_REQUIRES_GRAD_WARNING_EMITTED,
            SCALAR_REQUIRES_GRAD_WARNING,
        )?;
    }
    Ok(())
}

fn scalar_conversion_error(error: &TensorError) -> PyErr {
    if matches!(error, TensorError::ItemRequiresOneElement { .. }) {
        PyValueError::new_err("only one element tensors can be converted to Python scalars")
    } else {
        tensor_error(error)
    }
}

// PyTorch publishes __int__, __float__, and __complex__ as METH_NOARGS methods
// on TensorBase instead of CPython-generated special-method wrappers.
pyo3::inventory::submit! {
    type Inventory = <PyTensorBase as pyo3::impl_::pyclass::PyClassImpl>::Inventory;
    Inventory::new(pyo3::impl_::pyclass::PyClassItems {
        methods: &[
            pyo3::impl_::pymethods::PyMethodDefType::Method(
                pyo3::impl_::pymethods::PyMethodDef::noargs(
                    c"__int__",
                    pyo3::impl_::trampoline::get_trampoline_function!(
                        noargs,
                        PyTensorBase::__pymethod_int_scalar__
                    ),
                    c"",
                ),
            ),
            pyo3::impl_::pymethods::PyMethodDefType::Method(
                pyo3::impl_::pymethods::PyMethodDef::noargs(
                    c"__float__",
                    pyo3::impl_::trampoline::get_trampoline_function!(
                        noargs,
                        PyTensorBase::__pymethod_float_scalar__
                    ),
                    c"",
                ),
            ),
            pyo3::impl_::pymethods::PyMethodDefType::Method(
                pyo3::impl_::pymethods::PyMethodDef::noargs(
                    c"__complex__",
                    pyo3::impl_::trampoline::get_trampoline_function!(
                        noargs,
                        PyTensorBase::__pymethod_complex_scalar__
                    ),
                    c"",
                ),
            ),
        ],
        slots: &[],
    })
}

pub(crate) fn register_scalar_conversions(tensor_base: &Bound<'_, PyType>) -> PyResult<()> {
    // Reassigning the native descriptors refreshes numeric-slot and special
    // lookup while retaining TensorBase descriptor metadata.
    let int_descriptor = tensor_base.getattr("__int__")?;
    tensor_base.setattr("__int__", int_descriptor)?;
    let float_descriptor = tensor_base.getattr("__float__")?;
    tensor_base.setattr("__float__", float_descriptor)?;
    let complex_descriptor = tensor_base.getattr("__complex__")?;
    tensor_base.setattr("__complex__", complex_descriptor)?;

    // The implementation names exist only to let PyO3 generate panic-safe
    // trampolines for the public descriptors above.
    if tensor_base.hasattr("int_scalar")? {
        tensor_base.delattr("int_scalar")?;
    }
    if tensor_base.hasattr("float_scalar")? {
        tensor_base.delattr("float_scalar")?;
    }
    if tensor_base.hasattr("complex_scalar")? {
        tensor_base.delattr("complex_scalar")?;
    }
    Ok(())
}

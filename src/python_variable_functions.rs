//! Stable-ABI construction for PyTorch-style top-level built-in functions.
//!
//! `PyO3`'s `frozen` class option protects Rust instance state, not the Python
//! type dictionary. Its `immutable_type` option needs Python 3.14 when built
//! for the stable ABI. Constructing this zero-state owner with
//! `PyType_FromSpec` keeps the package's abi3-py310 contract while preventing
//! callers from replacing methods that built-in-function pickles depend on.

use std::ffi::c_void;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::{exceptions::PyRuntimeError, ffi};

use crate::python::{
    get_device_variable_function, matmul_variable_function, permute_variable_function,
    positive_variable_function, scalar_tensor_variable_function,
};

const POSITIVE_DOC: &std::ffi::CStr = c"\npositive(input) -> Tensor\n\nReturns :attr:`input`.\nThrows a runtime error if :attr:`input` is a bool tensor.\n\nArgs:\n    input (Tensor): the input tensor.\n\nExample::\n\n    >>> t = torch.randn(5)\n    >>> t\n    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])\n    >>> torch.positive(t)\n    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])\n";

const PERMUTE_DOC: &std::ffi::CStr = c"\npermute(input, dims) -> Tensor\n\nReturns a view of the original tensor :attr:`input` with its dimensions permuted.\n\nArgs:\n    input (Tensor): the input tensor.\n    dims (torch.Size, tuple of int or list of int): the desired ordering of dimensions.\n\nExample:\n    >>> x = torch.randn(2, 3, 5)\n    >>> x.size()\n    torch.Size([2, 3, 5])\n    >>> torch.permute(x, (2, 0, 1)).size()\n    torch.Size([5, 2, 3])\n";

const MATMUL_DOC: &std::ffi::CStr = cr"
matmul(input, other, *, out=None) -> Tensor

Matrix product of two tensors.

The behavior depends on the dimensionality of the tensors as follows:

- If both tensors are 1-dimensional, the dot product (scalar) is returned.
- If both arguments are 2-dimensional, the matrix-matrix product is returned.
- If the first argument is 1-dimensional and the second argument is 2-dimensional,
  a 1 is prepended to its dimension for the purpose of the matrix multiply.
  After the matrix multiply, the prepended dimension is removed.
- If the first argument is 2-dimensional and the second argument is 1-dimensional,
  the matrix-vector product is returned.
- If both arguments are at least 1-dimensional and at least one argument is
  N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
  argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
  batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
  1 is appended to its dimension for the purpose of the batched matrix multiply and removed after.

  The first N-2 dimensions of each argument, the batch dimensions, are
  :ref:`broadcast <broadcasting-semantics>` (and thus must be broadcastable).
  The last 2, the matrix dimensions, are handled as in the matrix-matrix product.

  For example, if :attr:`input` is a
  :math:`(j \times 1 \times n \times m)` tensor and :attr:`other` is a :math:`(k \times m \times p)`
  tensor, the batch dimensions are :math:`(j \times 1)` and :math:`(k)`,
  and the matrix dimensions are :math:`(n \times m)` and :math:`(m \times p)`.
  :attr:`out` will be a :math:`(j \times k \times n \times p)` tensor.

This operation has support for arguments with :ref:`sparse layouts<sparse-docs>`. In particular the
matrix-matrix (both arguments 2-dimensional) supports sparse arguments with the same restrictions
as :func:`torch.mm`


.. warning::
    Sparse support is a beta feature and some layout(s)/dtype/device combinations may not be supported,
    or may not have autograd support. If you notice missing functionality please
    open a feature request.

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

.. note::

    The 1-dimensional dot product version of this function does not support an :attr:`out` parameter.

Arguments:
    input (Tensor): the first tensor to be multiplied
    other (Tensor): the second tensor to be multiplied

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> # vector x vector
    >>> tensor1 = torch.randn(3)
    >>> tensor2 = torch.randn(3)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([])
    >>> # matrix x vector
    >>> tensor1 = torch.randn(3, 4)
    >>> tensor2 = torch.randn(4)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([3])
    >>> # batched matrix x broadcasted vector
    >>> tensor1 = torch.randn(10, 3, 4)
    >>> tensor2 = torch.randn(4)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([10, 3])
    >>> # batched matrix x batched matrix
    >>> tensor1 = torch.randn(10, 3, 4)
    >>> tensor2 = torch.randn(10, 4, 5)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([10, 3, 5])
    >>> # batched matrix x broadcasted matrix
    >>> tensor1 = torch.randn(10, 3, 4)
    >>> tensor2 = torch.randn(4, 5)
    >>> torch.matmul(tensor1, tensor2).size()
    torch.Size([10, 3, 5])

";

#[allow(
    unsafe_code,
    reason = "CPython passes borrowed tuple and dictionary pointers to C method callbacks"
)]
unsafe fn call_arguments(
    py: Python<'_>,
    args: *mut ffi::PyObject,
    kwargs: *mut ffi::PyObject,
) -> PyResult<(Bound<'_, PyTuple>, Option<Bound<'_, PyDict>>)> {
    // SAFETY: the C method callback contract supplies a live positional tuple.
    let args = unsafe { Bound::<PyAny>::from_borrowed_ptr(py, args) }.cast_into::<PyTuple>()?;
    // SAFETY: the keyword pointer is either null or a live dictionary for the
    // duration of the callback.
    let kwargs = unsafe { Bound::<PyAny>::from_borrowed_ptr_or_opt(py, kwargs) }
        .map(Bound::cast_into::<PyDict>)
        .transpose()?;
    Ok((args, kwargs))
}

macro_rules! variable_function_callback {
    ($name:ident, $implementation:ident) => {
        #[allow(
            unsafe_code,
            reason = "the callback is entered through PyO3's panic-safe C trampoline"
        )]
        unsafe fn $name(
            py: Python<'_>,
            _owner: *mut ffi::PyObject,
            args: *mut ffi::PyObject,
            kwargs: *mut ffi::PyObject,
        ) -> PyResult<*mut ffi::PyObject> {
            // SAFETY: PyO3's trampoline forwards CPython's live call arguments.
            let (args, kwargs) = unsafe { call_arguments(py, args, kwargs) }?;
            $implementation(py, &args, kwargs.as_ref()).map(Py::into_ptr)
        }
    };
}

variable_function_callback!(get_device_callback, get_device_variable_function);
variable_function_callback!(scalar_tensor_callback, scalar_tensor_variable_function);
variable_function_callback!(positive_callback, positive_variable_function);
variable_function_callback!(permute_callback, permute_variable_function);
variable_function_callback!(matmul_callback, matmul_variable_function);

/// Creates the immutable owner for exported `_VariableFunctionsClass` methods.
///
/// # Errors
///
/// Returns a Python exception if the stable-ABI type constructor fails.
#[allow(
    unsafe_code,
    reason = "PyType_FromSpec requires an audited raw type specification"
)]
pub(crate) fn create_variable_functions_class(py: Python<'_>) -> PyResult<Py<PyAny>> {
    // CPython descriptors retain pointers to their method definitions. Leak
    // this tiny table deliberately so it remains valid for the type lifetime.
    let methods = Box::leak(Box::new([
        pyo3::impl_::pymethods::PyMethodDef::cfunction_with_keywords(
            c"get_device",
            pyo3::impl_::trampoline::get_trampoline_function!(
                cfunction_with_keywords,
                get_device_callback
            ),
            c"",
        )
        .flags(ffi::METH_STATIC)
        .into_raw(),
        pyo3::impl_::pymethods::PyMethodDef::cfunction_with_keywords(
            c"scalar_tensor",
            pyo3::impl_::trampoline::get_trampoline_function!(
                cfunction_with_keywords,
                scalar_tensor_callback
            ),
            c"",
        )
        .flags(ffi::METH_STATIC)
        .into_raw(),
        pyo3::impl_::pymethods::PyMethodDef::cfunction_with_keywords(
            c"positive",
            pyo3::impl_::trampoline::get_trampoline_function!(
                cfunction_with_keywords,
                positive_callback
            ),
            POSITIVE_DOC,
        )
        .flags(ffi::METH_STATIC)
        .into_raw(),
        pyo3::impl_::pymethods::PyMethodDef::cfunction_with_keywords(
            c"permute",
            pyo3::impl_::trampoline::get_trampoline_function!(
                cfunction_with_keywords,
                permute_callback
            ),
            PERMUTE_DOC,
        )
        .flags(ffi::METH_STATIC)
        .into_raw(),
        pyo3::impl_::pymethods::PyMethodDef::cfunction_with_keywords(
            c"matmul",
            pyo3::impl_::trampoline::get_trampoline_function!(
                cfunction_with_keywords,
                matmul_callback
            ),
            MATMUL_DOC,
        )
        .flags(ffi::METH_STATIC)
        .into_raw(),
        ffi::PyMethodDef::zeroed(),
    ]));
    let mut slots = [
        ffi::PyType_Slot {
            slot: ffi::Py_tp_methods,
            pfunc: methods.as_mut_ptr().cast::<c_void>(),
        },
        ffi::PyType_Slot::default(),
    ];
    let flags = ffi::Py_TPFLAGS_DEFAULT
        | ffi::Py_TPFLAGS_DISALLOW_INSTANTIATION
        | ffi::Py_TPFLAGS_IMMUTABLETYPE;
    let flags = flags
        .try_into()
        .map_err(|_| PyRuntimeError::new_err("variable-function type flags exceed unsigned int"))?;
    let mut specification = ffi::PyType_Spec {
        name: c"torch_rs._C._VariableFunctionsClass".as_ptr(),
        basicsize: 0,
        itemsize: 0,
        flags,
        slots: slots.as_mut_ptr(),
    };

    // SAFETY: the specification and terminated slot and method tables remain
    // live for the call, and CPython returns a new reference or sets an error.
    let owner = unsafe {
        Bound::<PyAny>::from_owned_ptr_or_err(py, ffi::PyType_FromSpec(&raw mut specification))?
    };
    Ok(owner.unbind())
}

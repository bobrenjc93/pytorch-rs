//! Stable-ABI construction for PyTorch-style top-level built-in functions.
//!
//! `PyO3`'s `frozen` class option protects Rust instance state, not the Python
//! type dictionary. Its `immutable_type` option needs Python 3.14 when built
//! for the stable ABI. Constructing this zero-state owner with
//! `PyType_FromSpec` keeps the package's abi3-py310 contract while preventing
//! callers from replacing methods that built-in-function pickles depend on.

use std::ffi::c_void;

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyModule, PyTuple};
use pyo3::{exceptions::PyRuntimeError, ffi};

use crate::python::{
    adjoint_variable_function, can_cast_variable_function, get_device_variable_function,
    is_conj_variable_function, is_inference_variable_function, matmul_variable_function,
    moveaxis_variable_function, movedim_variable_function, mul_variable_function,
    multiply_variable_function, permute_variable_function, positive_variable_function,
    promote_types_variable_function, resolve_conj_variable_function, resolve_neg_variable_function,
    scalar_tensor_variable_function, select_variable_function, unbind_variable_function,
};

static VARIABLE_FUNCTIONS_CLASS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

const VARIABLE_FUNCTION_NAMES: [&str; 18] = [
    "get_device",
    "scalar_tensor",
    "adjoint",
    "positive",
    "is_conj",
    "is_inference",
    "resolve_conj",
    "resolve_neg",
    "unbind",
    "select",
    "permute",
    "movedim",
    "moveaxis",
    "matmul",
    "mul",
    "multiply",
    "can_cast",
    "promote_types",
];

const ADJOINT_DOC: &std::ffi::CStr = cr"
adjoint(input: Tensor) -> Tensor
Returns a view of the tensor conjugated and with the last two dimensions transposed.

``x.adjoint()`` is equivalent to ``x.transpose(-2, -1).conj()`` for complex tensors and
to ``x.transpose(-2, -1)`` for real tensors.

Args:
    {input}

Example::

    >>> x = torch.arange(4, dtype=torch.float)
    >>> A = torch.complex(x, x).reshape(2, 2)
    >>> A
    tensor([[0.+0.j, 1.+1.j],
            [2.+2.j, 3.+3.j]])
    >>> A.adjoint()
    tensor([[0.-0.j, 2.-2.j],
            [1.-1.j, 3.-3.j]])
    >>> (A.adjoint() == A.mH).all()
    tensor(True)
";

const POSITIVE_DOC: &std::ffi::CStr = c"\npositive(input) -> Tensor\n\nReturns :attr:`input`.\nThrows a runtime error if :attr:`input` is a bool tensor.\n\nArgs:\n    input (Tensor): the input tensor.\n\nExample::\n\n    >>> t = torch.randn(5)\n    >>> t\n    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])\n    >>> torch.positive(t)\n    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])\n";

const MUL_DOC: &std::ffi::CStr = cr"
mul(input, other, *, out=None) -> Tensor

Multiplies :attr:`input` by :attr:`other`.


.. math::
    \text{out}_i = \text{input}_i \times \text{other}_i


Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.

Args:
    input (Tensor): the input tensor.
    other (Tensor or Number): the tensor or number to multiply input by.

Keyword args:
    out (Tensor, optional): the output tensor.

Examples::

    >>> a = torch.randn(3)
    >>> a
    tensor([ 0.2015, -0.4255,  2.6087])
    >>> torch.mul(a, 100)
    tensor([  20.1494,  -42.5491,  260.8663])

    >>> b = torch.randn(4, 1)
    >>> b
    tensor([[ 1.1207],
            [-0.3137],
            [ 0.0700],
            [ 0.8378]])
    >>> c = torch.randn(1, 4)
    >>> c
    tensor([[ 0.5146,  0.1216, -0.5244,  2.2382]])
    >>> torch.mul(b, c)
    tensor([[ 0.5767,  0.1363, -0.5877,  2.5083],
            [-0.1614, -0.0382,  0.1645, -0.7021],
            [ 0.0360,  0.0085, -0.0367,  0.1567],
            [ 0.4312,  0.1019, -0.4394,  1.8753]])
";

const MULTIPLY_DOC: &std::ffi::CStr = c"
multiply(input, other, *, out=None)

Alias for :func:`torch.mul`.
";

const PROMOTE_TYPES_DOC: &std::ffi::CStr = c"
promote_types(type1, type2) -> dtype

Returns the :class:`torch.dtype` with the smallest size and scalar kind that is
not smaller nor of lower kind than either `type1` or `type2`. See type promotion
:ref:`documentation <type-promotion-doc>` for more information on the type
promotion logic.

Args:
    type1 (:class:`torch.dtype`)
    type2 (:class:`torch.dtype`)

Example::

    >>> torch.promote_types(torch.int32, torch.float32)
    torch.float32
    >>> torch.promote_types(torch.uint8, torch.long)
    torch.long
";

const CAN_CAST_DOC: &std::ffi::CStr = cr"
can_cast(from_, to) -> bool

Determines if a type conversion is allowed under PyTorch casting rules
described in the type promotion :ref:`documentation <type-promotion-doc>`.

Args:
    from\_ (dtype): The original :class:`torch.dtype`.
    to (dtype): The target :class:`torch.dtype`.

Example::

    >>> torch.can_cast(torch.double, torch.float)
    True
    >>> torch.can_cast(torch.float, torch.int)
    False
";

const IS_CONJ_DOC: &std::ffi::CStr = c"\nis_conj(input) -> (bool)\n\nReturns True if the :attr:`input` is a conjugated tensor, i.e. its conjugate bit is set to `True`.\n\nArgs:\n    input (Tensor): the input tensor.\n";

const IS_INFERENCE_DOC: &std::ffi::CStr = c"\nis_inference(input) -> (bool)\n\nReturns True if :attr:`input` is an inference tensor.\n\nA non-view tensor is an inference tensor if and only if it was\nallocated during inference mode. A view tensor is an inference\ntensor if and only if the tensor it is a view of is an inference tensor.\n\nFor details on inference mode please see\n`Inference Mode <https://pytorch.org/cppdocs/notes/inference_mode.html>`_.\n\nArgs:\n    input (Tensor): the input tensor.\n";

const RESOLVE_CONJ_DOC: &std::ffi::CStr = c"\nresolve_conj(input) -> Tensor\n\nReturns a new tensor with materialized conjugation if :attr:`input`'s conjugate bit is set to `True`,\nelse returns :attr:`input`. The output tensor will always have its conjugate bit set to `False`.\n\nArgs:\n    input (Tensor): the input tensor.\n\nExample::\n\n    >>> x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])\n    >>> y = x.conj()\n    >>> y.is_conj()\n    True\n    >>> z = y.resolve_conj()\n    >>> z\n    tensor([-1 - 1j, -2 - 2j, 3 + 3j])\n    >>> z.is_conj()\n    False\n";

const RESOLVE_NEG_DOC: &std::ffi::CStr = c"\nresolve_neg(input) -> Tensor\n\nReturns a new tensor with materialized negation if :attr:`input`'s negative bit is set to `True`,\nelse returns :attr:`input`. The output tensor will always have its negative bit set to `False`.\n\nArgs:\n    input (Tensor): the input tensor.\n\nExample::\n\n    >>> x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])\n    >>> y = x.conj()\n    >>> z = y.imag\n    >>> z.is_neg()\n    True\n    >>> out = z.resolve_neg()\n    >>> out\n    tensor([-1., -2., 3.])\n    >>> out.is_neg()\n    False\n";

const UNBIND_DOC: &std::ffi::CStr = c"\nunbind(input, dim=0) -> seq\n\nRemoves a tensor dimension.\n\nReturns a tuple of all slices along a given dimension, already without it.\n\nArguments:\n    input (Tensor): the tensor to unbind\n    dim (int): dimension to remove\n\nExample::\n\n    >>> torch.unbind(torch.tensor([[1, 2, 3],\n    >>>                            [4, 5, 6],\n    >>>                            [7, 8, 9]]))\n    (tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9]))\n";

const SELECT_DOC: &std::ffi::CStr = c"\nselect(input, dim, index) -> Tensor\n\nSlices the :attr:`input` tensor along the selected dimension at the given index.\nThis function returns a view of the original tensor with the given dimension removed.\n\n.. note:: If :attr:`input` is a sparse tensor and returning a view of\n          the tensor is not possible, a RuntimeError exception is\n          raised. In this is the case, consider using\n          :func:`torch.select_copy` function.\n\nArgs:\n    input (Tensor): the input tensor.\n    dim (int): the dimension to slice\n    index (int): the index to select with\n\n.. note::\n\n    :meth:`select` is equivalent to slicing. For example,\n    ``tensor.select(0, index)`` is equivalent to ``tensor[index]`` and\n    ``tensor.select(2, index)`` is equivalent to ``tensor[:,:,index]``.\n";

const PERMUTE_DOC: &std::ffi::CStr = c"\npermute(input, dims) -> Tensor\n\nReturns a view of the original tensor :attr:`input` with its dimensions permuted.\n\nArgs:\n    input (Tensor): the input tensor.\n    dims (torch.Size, tuple of int or list of int): the desired ordering of dimensions.\n\nExample:\n    >>> x = torch.randn(2, 3, 5)\n    >>> x.size()\n    torch.Size([2, 3, 5])\n    >>> torch.permute(x, (2, 0, 1)).size()\n    torch.Size([5, 2, 3])\n";

const MOVEDIM_DOC: &std::ffi::CStr = cr"
movedim(input, source, destination) -> Tensor

Moves the dimension(s) of :attr:`input` at the position(s) in :attr:`source`
to the position(s) in :attr:`destination`.

Other dimensions of :attr:`input` that are not explicitly moved remain in
their original order and appear at the positions not specified in :attr:`destination`.

Args:
    input (Tensor): the input tensor.
    source (int or tuple of ints): Original positions of the dims to move. These must be unique.
    destination (int or tuple of ints): Destination positions for each of the original dims. These must also be unique.

Examples::

    >>> t = torch.randn(3,2,1)
    >>> t
    tensor([[[-0.3362],
            [-0.8437]],

            [[-0.9627],
            [ 0.1727]],

            [[ 0.5173],
            [-0.1398]]])
    >>> torch.movedim(t, 1, 0).shape
    torch.Size([2, 3, 1])
    >>> torch.movedim(t, 1, 0)
    tensor([[[-0.3362],
            [-0.9627],
            [ 0.5173]],

            [[-0.8437],
            [ 0.1727],
            [-0.1398]]])
    >>> torch.movedim(t, (1, 2), (0, 1)).shape
    torch.Size([2, 1, 3])
    >>> torch.movedim(t, (1, 2), (0, 1))
    tensor([[[-0.3362, -0.9627,  0.5173]],

            [[-0.8437,  0.1727, -0.1398]]])
";

const MOVEAXIS_DOC: &std::ffi::CStr = cr"
moveaxis(input, source, destination) -> Tensor

Alias for :func:`torch.movedim`.

This function is equivalent to NumPy's moveaxis function.

Examples::

    >>> t = torch.randn(3,2,1)
    >>> t
    tensor([[[-0.3362],
            [-0.8437]],

            [[-0.9627],
            [ 0.1727]],

            [[ 0.5173],
            [-0.1398]]])
    >>> torch.moveaxis(t, 1, 0).shape
    torch.Size([2, 3, 1])
    >>> torch.moveaxis(t, 1, 0)
    tensor([[[-0.3362],
            [-0.9627],
            [ 0.5173]],

            [[-0.8437],
            [ 0.1727],
            [-0.1398]]])
    >>> torch.moveaxis(t, (1, 2), (0, 1)).shape
    torch.Size([2, 1, 3])
    >>> torch.moveaxis(t, (1, 2), (0, 1))
    tensor([[[-0.3362, -0.9627,  0.5173]],

            [[-0.8437,  0.1727, -0.1398]]])
";

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
variable_function_callback!(adjoint_callback, adjoint_variable_function);
variable_function_callback!(positive_callback, positive_variable_function);
variable_function_callback!(mul_callback, mul_variable_function);
variable_function_callback!(multiply_callback, multiply_variable_function);
variable_function_callback!(is_conj_callback, is_conj_variable_function);
variable_function_callback!(is_inference_callback, is_inference_variable_function);
variable_function_callback!(resolve_conj_callback, resolve_conj_variable_function);
variable_function_callback!(resolve_neg_callback, resolve_neg_variable_function);
variable_function_callback!(unbind_callback, unbind_variable_function);
variable_function_callback!(select_callback, select_variable_function);
variable_function_callback!(permute_callback, permute_variable_function);
variable_function_callback!(movedim_callback, movedim_variable_function);
variable_function_callback!(moveaxis_callback, moveaxis_variable_function);
variable_function_callback!(matmul_callback, matmul_variable_function);
variable_function_callback!(can_cast_callback, can_cast_variable_function);
variable_function_callback!(promote_types_callback, promote_types_variable_function);

macro_rules! variable_function_method {
    ($name:expr, $callback:ident, $doc:expr) => {
        pyo3::impl_::pymethods::PyMethodDef::cfunction_with_keywords(
            $name,
            pyo3::impl_::trampoline::get_trampoline_function!(cfunction_with_keywords, $callback),
            $doc,
        )
        .flags(ffi::METH_STATIC)
        .into_raw()
    };
}

#[allow(
    unsafe_code,
    reason = "PyType_FromSpec requires an audited raw type specification"
)]
fn create_variable_functions_class(py: Python<'_>) -> PyResult<Py<PyAny>> {
    // CPython descriptors retain pointers to their method definitions. Leak
    // this tiny table deliberately so it remains valid for the type lifetime.
    let methods = Box::leak(Box::new([
        variable_function_method!(c"get_device", get_device_callback, c""),
        variable_function_method!(c"scalar_tensor", scalar_tensor_callback, c""),
        variable_function_method!(c"adjoint", adjoint_callback, ADJOINT_DOC),
        variable_function_method!(c"positive", positive_callback, POSITIVE_DOC),
        variable_function_method!(c"mul", mul_callback, MUL_DOC),
        variable_function_method!(c"multiply", multiply_callback, MULTIPLY_DOC),
        variable_function_method!(c"is_conj", is_conj_callback, IS_CONJ_DOC),
        variable_function_method!(c"is_inference", is_inference_callback, IS_INFERENCE_DOC),
        variable_function_method!(c"resolve_conj", resolve_conj_callback, RESOLVE_CONJ_DOC),
        variable_function_method!(c"resolve_neg", resolve_neg_callback, RESOLVE_NEG_DOC),
        variable_function_method!(c"unbind", unbind_callback, UNBIND_DOC),
        variable_function_method!(c"select", select_callback, SELECT_DOC),
        variable_function_method!(c"permute", permute_callback, PERMUTE_DOC),
        variable_function_method!(c"movedim", movedim_callback, MOVEDIM_DOC),
        variable_function_method!(c"moveaxis", moveaxis_callback, MOVEAXIS_DOC),
        variable_function_method!(c"matmul", matmul_callback, MATMUL_DOC),
        variable_function_method!(c"can_cast", can_cast_callback, CAN_CAST_DOC),
        variable_function_method!(c"promote_types", promote_types_callback, PROMOTE_TYPES_DOC),
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

/// Adds the immutable `_VariableFunctionsClass` owner and its public callables.
///
/// # Errors
///
/// Returns a Python exception if owner construction or module registration fails.
pub(crate) fn add_variable_functions(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let variable_functions =
        VARIABLE_FUNCTIONS_CLASS.get_or_try_init(py, || create_variable_functions_class(py))?;
    module.add("_VariableFunctionsClass", variable_functions.clone_ref(py))?;
    module
        .getattr("__all__")?
        .call_method1("remove", ("_VariableFunctionsClass",))?;
    let variable_functions = variable_functions.bind(py);
    for name in VARIABLE_FUNCTION_NAMES {
        let function = variable_functions.getattr(name)?;
        function.setattr("__module__", "torch")?;
        module.add(name, function)?;
    }
    Ok(())
}

/// Returns the registered top-level callable used for `__torch_function__` dispatch.
///
/// # Errors
///
/// Returns the operation's initialization error before the owner is registered,
/// or the Python attribute error raised by a failed callable lookup.
pub(crate) fn variable_function(py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
    let variable_functions = VARIABLE_FUNCTIONS_CLASS.get(py).ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "torch.{name} was called before module initialization completed"
        ))
    })?;
    Ok(variable_functions.bind(py).getattr(name)?.unbind())
}

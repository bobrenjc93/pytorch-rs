//! Stable-ABI construction for PyTorch-style top-level built-in functions.
//!
//! `PyO3`'s `frozen` class option protects Rust instance state, not the Python
//! type dictionary. Its `immutable_type` option needs Python 3.14 when built
//! for the stable ABI. Constructing this zero-state owner with
//! `PyType_FromSpec` keeps the package's abi3-py310 contract while preventing
//! callers from replacing methods that built-in-function pickles depend on.

use std::ffi::c_void;

use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyModule, PyTuple};
use pyo3::{exceptions::PyRuntimeError, ffi};

use crate::python::{
    abs_variable_function, absolute_variable_function, add_variable_function,
    adjoint_variable_function, arange_variable_function, as_tensor_variable_function,
    asarray_variable_function, atleast_1d_variable_function, atleast_2d_variable_function,
    atleast_3d_variable_function, broadcast_tensors_variable_function, can_cast_variable_function,
    ceil_variable_function, conj_variable_function, cos_variable_function,
    detach_variable_function, exp_variable_function, fix_variable_function,
    floor_variable_function, get_device_variable_function, imag_variable_function,
    is_conj_variable_function, is_inference_variable_function, log_variable_function,
    matmul_variable_function, mean_variable_function, moveaxis_variable_function,
    movedim_variable_function, mul_variable_function, multiply_variable_function,
    neg_variable_function, negative_variable_function, ones_like_variable_function,
    permute_variable_function, positive_variable_function, promote_types_variable_function,
    ravel_variable_function, real_variable_function, reciprocal_variable_function,
    reshape_variable_function, resolve_conj_variable_function, resolve_neg_variable_function,
    rsqrt_variable_function, scalar_tensor_variable_function, select_variable_function,
    sigmoid_variable_function, sin_variable_function, sqrt_variable_function,
    square_variable_function, sub_variable_function, subtract_variable_function,
    sum_variable_function, tanh_variable_function, trunc_variable_function,
    unbind_variable_function, unsqueeze_variable_function, zeros_like_variable_function,
};

static VARIABLE_FUNCTIONS_CLASS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

const VARIABLE_FUNCTION_NAMES: [&str; 58] = [
    "get_device",
    "as_tensor",
    "asarray",
    "scalar_tensor",
    "arange",
    "ones_like",
    "zeros_like",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "broadcast_tensors",
    "abs",
    "absolute",
    "adjoint",
    "conj",
    "real",
    "imag",
    "positive",
    "detach",
    "ravel",
    "reshape",
    "reciprocal",
    "rsqrt",
    "neg",
    "negative",
    "exp",
    "floor",
    "ceil",
    "trunc",
    "fix",
    "log",
    "sin",
    "cos",
    "sqrt",
    "sigmoid",
    "square",
    "sum",
    "mean",
    "tanh",
    "add",
    "is_vulkan_available",
    "is_conj",
    "is_inference",
    "resolve_conj",
    "resolve_neg",
    "unbind",
    "unsqueeze",
    "select",
    "permute",
    "movedim",
    "moveaxis",
    "matmul",
    "sub",
    "subtract",
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

const ARANGE_DOC: &std::ffi::CStr = cr"
arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor

Returns a 1-D tensor of size :math:`\left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil`
with values from the interval ``[start, end)`` taken with common difference
:attr:`step` beginning from `start`.

Note: When using floating-point dtypes (especially reduced precision types like ``bfloat16``),
the results may be affected by floating-point rounding behavior. Some values in the sequence
might not be exactly representable in certain floating-point formats, which can lead to
repeated values or unexpected rounding. For precise sequences, it is recommended to use
integer dtypes instead of floating-point dtypes.

Note that non-integer :attr:`step` is subject to floating point rounding errors when
comparing against :attr:`end`; to avoid inconsistency, we advise subtracting a small epsilon from :attr:`end`
in such cases.

.. math::
    \text{out}_{{i+1}} = \text{out}_{i} + \text{step}

Args:
    start (Number, optional): the starting value for the set of points. Default: ``0``.
    end (Number): the ending value for the set of points
    step (Number, optional): the gap between each pair of adjacent points. Default: ``1``.

Keyword args:
    out (Tensor, optional): the output tensor.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`). If `dtype` is not given, infer the data type from the other input
        arguments. If any of `start`, `end`, or `stop` are floating-point, the
        `dtype` is inferred to be the default dtype, see
        :meth:`~torch.get_default_dtype`. Otherwise, the `dtype` is inferred to
        be `torch.int64`.
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_device`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.

Example::

    >>> torch.arange(5)
    tensor([ 0,  1,  2,  3,  4])
    >>> torch.arange(1, 4)
    tensor([ 1,  2,  3])
    >>> torch.arange(1, 2.5, 0.5)
    tensor([ 1.0000,  1.5000,  2.0000])
";

const ONES_LIKE_DOC: &std::ffi::CStr = cr"
ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None) -> Tensor

Returns a tensor filled with the scalar value `1`, with the same size as
:attr:`input`.
";

const ZEROS_LIKE_DOC: &std::ffi::CStr = cr"
zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None) -> Tensor

Returns a tensor filled with the scalar value `0`, with the same size as
:attr:`input`.
";

const AS_TENSOR_DOC: &std::ffi::CStr = cr#"
as_tensor(data: Any, *, dtype: Optional[dtype] = None, device: Optional[DeviceLikeType]) -> Tensor

Converts :attr:`data` into a tensor, sharing data and preserving autograd
history if possible.

If :attr:`data` is already a tensor with the requested dtype and device
then :attr:`data` itself is returned, but if :attr:`data` is a
tensor with a different dtype or device then it's copied as if using
`data.to(dtype=dtype, device=device)`.

If :attr:`data` is a NumPy array (an ndarray) with the same dtype and device then a
tensor is constructed using :func:`torch.from_numpy`.

If :attr:`data` is a CuPy array, the returned tensor will be located on the same device as the CuPy array unless
specifically overwritten by :attr:`device` or a default device. The device of the CuPy array is inferred from the
pointer of the array using `cudaPointerGetAttributes` unless :attr:`device` is provided with an explicit device index.

.. seealso::

    :func:`torch.tensor` never shares its data and creates a new "leaf tensor" (see :doc:`/notes/autograd`).


Args:
    data (array_like): Initial data for the tensor. Can be a list, tuple,
        NumPy ``ndarray``, scalar, and other types.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, infers data type from :attr:`data`.
    device (:class:`torch.device`, optional): the device of the constructed tensor. If None and data is a tensor
        then the device of data is used. If None and data is not a tensor then
        the result tensor is constructed on the current device.


Example::

    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.as_tensor(a)
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([-1,  2,  3])

    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.as_tensor(a, device=torch.device('cuda'))
    >>> t
    tensor([ 1,  2,  3])
    >>> t[0] = -1
    >>> a
    array([1,  2,  3])
"#;

const ASARRAY_DOC: &std::ffi::CStr = cr"
asarray(obj: Any, *, dtype: Optional[dtype], device: Optional[DeviceLikeType], copy: Optional[bool] = None, requires_grad: Optional[bool] = None) -> Tensor # noqa: B950

Converts :attr:`obj` to a tensor.

:attr:`obj` can be one of:

1. a tensor
2. a NumPy array or a NumPy scalar
3. a DLPack capsule
4. an object that implements Python's buffer protocol
5. a scalar
6. a sequence of scalars

When :attr:`obj` is a tensor, NumPy array, or DLPack capsule the returned tensor will,
by default, have the same requires_grad as :attr:`obj` (defaulting to False), have the
same datatype, be on the same device, and share memory with it. These properties can be
controlled with the :attr:`dtype`, :attr:`device`, :attr:`copy`, and
:attr:`requires_grad` keyword arguments. If the returned tensor is of a different
datatype, on a different device, or a copy is requested then it will not share its
memory with :attr:`obj`. If :attr:`requires_grad` is ``True`` (or ``None``, and
:attr:`obj` was a tensor with requires_grad set), then the returned tensor will require
a gradient, and if :attr:`obj` is also a tensor with an autograd history then the
returned tensor will have the same history.

When :attr:`obj` is not a tensor, NumPy array, or DLPack capsule but implements Python's
buffer protocol then the buffer is interpreted as an array of bytes grouped according to
the size of the datatype passed to the :attr:`dtype` keyword argument. (If no datatype is
passed then the default floating point datatype is used, instead.) The returned tensor
will have the specified datatype (or default floating point datatype if none is specified)
and, by default, be on the CPU device and share memory with the buffer.

When :attr:`obj` is a NumPy scalar, the returned tensor will be a 0-dimensional tensor on
the CPU and that doesn't share its memory (i.e. ``copy=True``). By default datatype will
be the PyTorch datatype corresponding to the NumPy's scalar's datatype.

When :attr:`obj` is none of the above but a scalar, or a sequence of scalars then the
returned tensor will, by default, infer its datatype from the scalar values, be on the
current default device, and not share its memory.

.. seealso::

    :func:`torch.tensor` creates a tensor that always copies the data from the input object.
    :func:`torch.from_numpy` creates a tensor that always shares memory from NumPy arrays.
    :func:`torch.frombuffer` creates a tensor that always shares memory from objects that
    implement the buffer protocol.
    :func:`torch.from_dlpack` creates a tensor that always shares memory from
    DLPack capsules.

Args:
    obj (object): a tensor, NumPy array, DLPack Capsule, object that implements Python's
           buffer protocol, scalar, or sequence of scalars.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the datatype of the returned tensor.
           Default: ``None``, which causes the datatype of the returned tensor to be
           inferred from :attr:`obj`.
    copy (bool, optional): controls whether the returned tensor shares memory with :attr:`obj`.
           Default: ``None``, which causes the returned tensor to share memory with :attr:`obj`
           whenever possible. If ``True`` then the returned tensor does not share its memory.
           If ``False`` then the returned tensor shares its memory with :attr:`obj` and an
           error is thrown if it cannot.
    device (:class:`torch.device`, optional): the device of the returned tensor.
           Default: ``None``, which causes the device of :attr:`obj` to be used. Or, if
           :attr:`obj` is a Python sequence, the current default device will be used.
    requires_grad (bool, optional): whether the returned tensor requires grad.
           Default: ``None``, which causes requires_grad for the returned tensor to be
           inferred from :attr:`obj`. If ``True``, then the returned tensor will require
           a gradient, and if :attr:`obj` is also a tensor with an autograd history then
           the returned tensor will have the same history.

Example::

    >>> a = torch.tensor([1, 2, 3])
    >>> # Shares memory with tensor 'a'
    >>> b = torch.asarray(a)
    >>> a.data_ptr() == b.data_ptr()
    True
    >>> # Forces memory copy
    >>> c = torch.asarray(a, copy=True)
    >>> a.data_ptr() == c.data_ptr()
    False

    >>> a = torch.tensor([1., 2., 3.], requires_grad=True)
    >>> b = a + 2
    >>> b
    tensor([3., 4., 5.], grad_fn=<AddBackward0>)
    >>> # Shares memory with tensor 'b', with no grad
    >>> c = torch.asarray(b, requires_grad=False)
    >>> c
    tensor([3., 4., 5.])
    >>> # Shares memory with tensor 'b', retaining autograd history
    >>> d = torch.asarray(b, requires_grad=True)
    >>> d
    tensor([3., 4., 5.], grad_fn=<AddBackward0>)
    >>> # Shares memory with tensor 'b', retaining autograd history
    >>> e = torch.asarray(b)
    >>> e
    tensor([3., 4., 5.], grad_fn=<AddBackward0>)

    >>> array = numpy.array([1, 2, 3])
    >>> # Shares memory with array 'array'
    >>> t1 = torch.asarray(array)
    >>> array.__array_interface__['data'][0] == t1.data_ptr()
    True
    >>> # Copies memory due to dtype mismatch
    >>> t2 = torch.asarray(array, dtype=torch.float32)
    >>> array.__array_interface__['data'][0] == t2.data_ptr()
    False

    >>> scalar = numpy.float64(0.5)
    >>> torch.asarray(scalar)
    tensor(0.5000, dtype=torch.float64)
";

const POSITIVE_DOC: &std::ffi::CStr = c"\npositive(input) -> Tensor\n\nReturns :attr:`input`.\nThrows a runtime error if :attr:`input` is a bool tensor.\n\nArgs:\n    input (Tensor): the input tensor.\n\nExample::\n\n    >>> t = torch.randn(5)\n    >>> t\n    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])\n    >>> torch.positive(t)\n    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])\n";

const REAL_DOC: &std::ffi::CStr = c"\nreal(input) -> Tensor\n\nReturns a new tensor containing real values of the :attr:`self` tensor.\nThe returned tensor and :attr:`self` share the same underlying storage.\n\nArgs:\n    input (Tensor): the input tensor.\n\nExample::\n\n    >>> x=torch.randn(4, dtype=torch.cfloat)\n    >>> x\n    tensor([(0.3100+0.3553j), (-0.5445-0.7896j), (-1.6492-0.0633j), (-0.0638-0.8119j)])\n    >>> x.real\n    tensor([ 0.3100, -0.5445, -1.6492, -0.0638])\n\n";

const IMAG_DOC: &std::ffi::CStr = c"\nimag(input) -> Tensor\n\nReturns a new tensor containing imaginary values of the :attr:`self` tensor.\nThe returned tensor and :attr:`self` share the same underlying storage.\n\n.. warning::\n    :func:`imag` is only supported for tensors with complex dtypes.\n\nArgs:\n    input (Tensor): the input tensor.\n\nExample::\n\n    >>> x=torch.randn(4, dtype=torch.cfloat)\n    >>> x\n    tensor([(0.3100+0.3553j), (-0.5445-0.7896j), (-1.6492-0.0633j), (-0.0638-0.8119j)])\n    >>> x.imag\n    tensor([ 0.3553, -0.7896, -0.0633, -0.8119])\n\n";

const ABS_DOC: &std::ffi::CStr = cr"
abs(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Computes the absolute value of each element in :attr:`input`.

.. math::
    \text{out}_{i} = |\text{input}_{i}|

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.abs(torch.tensor([-1, -2, 3]))
    tensor([ 1,  2,  3])
";

const ABSOLUTE_DOC: &std::ffi::CStr = c"
absolute(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Alias for :func:`torch.abs`
";

const RAVEL_DOC: &std::ffi::CStr = c"\nravel(input) -> Tensor\n\nReturn a contiguous flattened tensor. A copy is made only if needed.\n\nArgs:\n    input (Tensor): the input tensor.\n\nExample::\n\n    >>> t = torch.tensor([[[1, 2],\n    ...                    [3, 4]],\n    ...                   [[5, 6],\n    ...                    [7, 8]]])\n    >>> torch.ravel(t)\n    tensor([1, 2, 3, 4, 5, 6, 7, 8])\n";

const RESHAPE_DOC: &std::ffi::CStr = cr"
reshape(input, shape) -> Tensor

Returns a tensor with the same data and number of elements as :attr:`input`,
but with the specified shape. When possible, the returned tensor will be a view
of :attr:`input`. Otherwise, it will be a copy. Contiguous inputs and inputs
with compatible strides can be reshaped without copying, but you should not
depend on the copying vs. viewing behavior.

See :meth:`torch.Tensor.view` on when it is possible to return a view.

A single dimension may be -1, in which case it's inferred from the remaining
dimensions and the number of elements in :attr:`input`.

Args:
    input (Tensor): the tensor to be reshaped
    shape (tuple of int): the new shape

Example::

    >>> a = torch.arange(4.)
    >>> torch.reshape(a, (2, 2))
    tensor([[ 0.,  1.],
            [ 2.,  3.]])
    >>> b = torch.tensor([[0, 1], [2, 3]])
    >>> torch.reshape(b, (-1,))
    tensor([ 0,  1,  2,  3])
";

const RECIPROCAL_DOC: &std::ffi::CStr = cr"
reciprocal(input, *, out=None) -> Tensor

Returns a new tensor with the reciprocal of the elements of :attr:`input`

.. math::
    \text{out}_{i} = \frac{1}{\text{input}_{i}}

.. note::
    Unlike NumPy's reciprocal, torch.reciprocal supports integral inputs. Integral
    inputs to reciprocal are automatically :ref:`promoted <type-promotion-doc>` to
    the default scalar type.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.4595, -2.1219, -1.4314,  0.7298])
    >>> torch.reciprocal(a)
    tensor([-2.1763, -0.4713, -0.6986,  1.3702])
";

const RSQRT_DOC: &std::ffi::CStr = cr"
rsqrt(input, *, out=None) -> Tensor

Returns a new tensor with the reciprocal of the square-root of each of
the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \frac{1}{\sqrt{\text{input}_{i}}}

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.0370,  0.2970,  1.5420, -0.9105])
    >>> torch.rsqrt(a)
    tensor([    nan,  1.8351,  0.8053,     nan])
";

const LOG_DOC: &std::ffi::CStr = cr"
log(input, *, out=None) -> Tensor

Returns a new tensor with the natural logarithm of the elements
of :attr:`input`.

.. math::
    y_{i} = \log_{e}(\text{input}_{i})

The current native implementation supports exact CPU ``float32`` tensors when
autograd recording is inactive or the input does not require gradients.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.log(torch.tensor([1., math.e]))
    tensor([ 0.,  1.])
";

const NEG_DOC: &std::ffi::CStr = cr"
neg(input, *, out=None) -> Tensor

Returns a new tensor with the negative of the elements of :attr:`input`.

.. math::
    \text{out} = -1 \times \text{input}

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(5)
    >>> a
    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])
    >>> torch.neg(a)
    tensor([-0.0090,  0.2262,  0.0682,  0.2866, -0.3940])
";

const NEGATIVE_DOC: &std::ffi::CStr = c"
negative(input, *, out=None) -> Tensor

Alias for :func:`torch.neg`
";

const EXP_DOC: &std::ffi::CStr = cr"
exp(input, *, out=None) -> Tensor

Returns a new tensor with the exponential of the elements
of the input tensor :attr:`input`.

.. math::
    y_{i} = e^{x_{i}}

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.exp(torch.tensor([0, math.log(2.)]))
    tensor([ 1.,  2.])
";

const FLOOR_DOC: &std::ffi::CStr = cr"
floor(input, *, out=None) -> Tensor

Returns a new tensor with the floor of the elements of :attr:`input`,
the largest integer less than or equal to each element.

For integer inputs, follows the array-api convention of returning a
copy of the input tensor.

.. math::
    \text{out}_{i} = \left\lfloor \text{input}_{i} \right\rfloor

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.8166,  1.5308, -0.2530, -0.2091])
    >>> torch.floor(a)
    tensor([-1.,  1., -1., -1.])
";

const CEIL_DOC: &std::ffi::CStr = cr"
ceil(input, *, out=None) -> Tensor

Returns a new tensor with the ceil of the elements of :attr:`input`,
the smallest integer greater than or equal to each element.

For integer inputs, follows the array-api convention of returning a
copy of the input tensor.

.. math::
    \text{out}_{i} = \left\lceil \text{input}_{i} \right\rceil

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.6341, -1.4208, -1.0900,  0.5826])
    >>> torch.ceil(a)
    tensor([-0., -1., -1.,  1.])
";

const TRUNC_DOC: &std::ffi::CStr = cr"
trunc(input, *, out=None) -> Tensor

Returns a new tensor with the truncated integer values of
the elements of :attr:`input`.

For integer inputs, follows the array-api convention of returning a
copy of the input tensor.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 3.4742,  0.5466, -0.8008, -0.9079])
    >>> torch.trunc(a)
    tensor([ 3.,  0., -0., -0.])
";

const FIX_DOC: &std::ffi::CStr = c"
fix(input, *, out=None) -> Tensor

Alias for :func:`torch.trunc`
";

const SIN_DOC: &std::ffi::CStr = cr"
sin(input, *, out=None) -> Tensor

Returns a new tensor with the sine of the elements in the :attr:`input` tensor,
where each value in this input tensor is in radians.

.. math::
    \text{out}_{i} = \sin(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.5461,  0.1347, -2.7266, -0.2746])
    >>> torch.sin(a)
    tensor([-0.5194,  0.1343, -0.4032, -0.2711])
";

const COS_DOC: &std::ffi::CStr = cr"
cos(input, *, out=None) -> Tensor

Returns a new tensor with the cosine of the elements of :attr:`input` given in radians.

.. math::
    \text{out}_{i} = \cos(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
    >>> torch.cos(a)
    tensor([ 0.1395,  0.2957,  0.6553,  0.5574])
";

const SQRT_DOC: &std::ffi::CStr = cr"
sqrt(input, *, out=None) -> Tensor

Returns a new tensor with the square-root of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sqrt{\text{input}_{i}}

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-2.0755,  1.0226,  0.0831,  0.4806])
    >>> torch.sqrt(a)
    tensor([    nan,  1.0112,  0.2883,  0.6933])
";

const SIGMOID_DOC: &std::ffi::CStr = c"
sigmoid(input, *, out=None) -> Tensor

Alias for :func:`torch.special.expit`.
";

const SQUARE_DOC: &std::ffi::CStr = cr"
square(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Returns a new tensor with the square of the elements of :attr:`input`.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-2.0755,  1.0226,  0.0831,  0.4806])
    >>> torch.square(a)
    tensor([ 4.3077,  1.0457,  0.0069,  0.2310])
";

const SUM_DOC: &std::ffi::CStr = c"
sum(input, *, dtype=None) -> Tensor

Returns the sum of all elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

.. note:: Use the `dtype` argument if you need the result in a specific tensor type.
          Otherwise, the result type may be automatically promoted (e.g., from `torch.int32` to `torch.int64`).

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.1133, -0.9567,  0.2958]])
    >>> torch.sum(a)
    tensor(-0.5475)

.. function:: sum(input, dim, keepdim=False, *, dtype=None) -> Tensor
   :noindex:

Returns the sum of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
reduce over all of them.


If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).


Args:
    input (Tensor): the input tensor.\n    \n    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
        If ``None``, all dimensions are reduced.\n\n    \n    keepdim (bool, optional): whether the output tensor has :attr:`dim` retained or not. Default: ``False``.


Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.

Example::

    >>> a = torch.randn(4, 4)
    >>> a
    tensor([[ 0.0569, -0.2475,  0.0737, -0.3429],
            [-0.2993,  0.9138,  0.9337, -1.6864],
            [ 0.1132,  0.7892, -0.1003,  0.5688],
            [ 0.3637, -0.9906, -0.4752, -1.5197]])
    >>> torch.sum(a, 1)
    tensor([-0.4598, -0.1381,  1.3708, -2.6217])
    >>> b = torch.arange(4 * 5 * 6).view(4, 5, 6)
    >>> torch.sum(b, (2, 1))
    tensor([  435.,  1335.,  2235.,  3135.])
";

const MEAN_DOC: &std::ffi::CStr = c"
mean(input, *, dtype=None) -> Tensor

Returns the mean value of all elements in the :attr:`input` tensor.

Args:
    input (Tensor): the input tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. Default: None.

Example::

    >>> a = torch.randn(1, 3)
    >>> a
    tensor([[ 0.1133, -0.9567,  0.2958]])
    >>> torch.mean(a)
    tensor(-0.1825)

.. function:: mean(input, dim, keepdim=False, *, dtype=None, out=None) -> Tensor
   :noindex:

Returns the mean value of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
reduce over all of them.

If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).

Args:
    input (Tensor): the input tensor.
    dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
        If ``None``, all dimensions are reduced.
    keepdim (bool, optional): whether the output tensor has :attr:`dim` retained or not. Default: ``False``.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. Default: None.
    out (Tensor, optional): the output tensor.
";

const TANH_DOC: &std::ffi::CStr = cr"
tanh(input, *, out=None) -> Tensor

Returns a new tensor with the hyperbolic tangent of the elements
of :attr:`input`.

.. math::
    \text{out}_{i} = \tanh(\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.8986, -0.7279,  1.1745,  0.2611])
    >>> torch.tanh(a)
    tensor([ 0.7156, -0.6218,  0.8257,  0.2553])
";

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

const ADD_DOC: &std::ffi::CStr = cr"
add(input, other, *, alpha=1, out=None) -> Tensor

Adds :attr:`other`, scaled by :attr:`alpha`, to :attr:`input`.

The native implementation currently supports only exact native CPU float32
Tensor/Tensor operands with omitted or default-equivalent ``alpha`` and omitted
or ``None`` ``out``. Scalar operands, scalar-only calls, nondefault or boolean
``alpha``, concrete ``out`` tensors, dtype/device extension keywords, tensor
subclasses without ``__torch_function__`` handling, and in-place variants remain
unsupported.
";

const SUB_DOC: &std::ffi::CStr = cr"
sub(input, other, *, alpha=1, out=None) -> Tensor

Subtracts :attr:`other`, scaled by :attr:`alpha`, from :attr:`input`.

.. math::
    \text{{out}}_i = \text{{input}}_i - \text{{alpha}} \times \text{{other}}_i


Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.

Args:
    input (Tensor): the input tensor.
    other (Tensor or Number): the tensor or number to subtract from :attr:`input`.

Keyword args:
    alpha (Number): the multiplier for :attr:`other`.
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.tensor((1, 2))
    >>> b = torch.tensor((0, 1))
    >>> torch.sub(a, b, alpha=2)
    tensor([1, 0])
";

const SUBTRACT_DOC: &std::ffi::CStr = c"
subtract(input, other, *, alpha=1, out=None) -> Tensor

Alias for :func:`torch.sub`.
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

const IS_CONJ_DOC: &std::ffi::CStr = c"\nis_conj(input) -> (bool)\n\nReturns True if the :attr:`input` is a conjugated tensor, i.e. its conjugate bit is set to `True`.\n\nArgs:\n    input (Tensor): the input tensor.\n";

const IS_INFERENCE_DOC: &std::ffi::CStr = c"\nis_inference(input) -> (bool)\n\nReturns True if :attr:`input` is an inference tensor.\n\nA non-view tensor is an inference tensor if and only if it was\nallocated during inference mode. A view tensor is an inference\ntensor if and only if the tensor it is a view of is an inference tensor.\n\nFor details on inference mode please see\n`Inference Mode <https://pytorch.org/cppdocs/notes/inference_mode.html>`_.\n\nArgs:\n    input (Tensor): the input tensor.\n";

const CONJ_DOC: &std::ffi::CStr = c"\nconj(input) -> Tensor\n\nReturns a view of :attr:`input` with a flipped conjugate bit. If :attr:`input` has a non-complex dtype,\nthis function just returns :attr:`input`.\n\n.. note::\n    :func:`torch.conj` performs a lazy conjugation, but the actual conjugated tensor can be materialized\n    at any time using :func:`torch.resolve_conj`.\n\n.. warning:: In the future, :func:`torch.conj` may return a non-writeable view for an :attr:`input` of\n             non-complex dtype. It's recommended that programs not modify the tensor returned by :func:`torch.conj_physical`\n             when :attr:`input` is of non-complex dtype to be compatible with this change.\n\nArgs:\n    input (Tensor): the input tensor.\n\nExample::\n\n    >>> x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])\n    >>> x.is_conj()\n    False\n    >>> y = torch.conj(x)\n    >>> y.is_conj()\n    True\n";

const RESOLVE_CONJ_DOC: &std::ffi::CStr = c"\nresolve_conj(input) -> Tensor\n\nReturns a new tensor with materialized conjugation if :attr:`input`'s conjugate bit is set to `True`,\nelse returns :attr:`input`. The output tensor will always have its conjugate bit set to `False`.\n\nArgs:\n    input (Tensor): the input tensor.\n\nExample::\n\n    >>> x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])\n    >>> y = x.conj()\n    >>> y.is_conj()\n    True\n    >>> z = y.resolve_conj()\n    >>> z\n    tensor([-1 - 1j, -2 - 2j, 3 + 3j])\n    >>> z.is_conj()\n    False\n";

const RESOLVE_NEG_DOC: &std::ffi::CStr = c"\nresolve_neg(input) -> Tensor\n\nReturns a new tensor with materialized negation if :attr:`input`'s negative bit is set to `True`,\nelse returns :attr:`input`. The output tensor will always have its negative bit set to `False`.\n\nArgs:\n    input (Tensor): the input tensor.\n\nExample::\n\n    >>> x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])\n    >>> y = x.conj()\n    >>> z = y.imag\n    >>> z.is_neg()\n    True\n    >>> out = z.resolve_neg()\n    >>> out\n    tensor([-1., -2., 3.])\n    >>> out.is_neg()\n    False\n";

const UNBIND_DOC: &std::ffi::CStr = c"\nunbind(input, dim=0) -> seq\n\nRemoves a tensor dimension.\n\nReturns a tuple of all slices along a given dimension, already without it.\n\nArguments:\n    input (Tensor): the tensor to unbind\n    dim (int): dimension to remove\n\nExample::\n\n    >>> torch.unbind(torch.tensor([[1, 2, 3],\n    >>>                            [4, 5, 6],\n    >>>                            [7, 8, 9]]))\n    (tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9]))\n";

const UNSQUEEZE_DOC: &std::ffi::CStr = c"
unsqueeze(input, dim) -> Tensor

Returns a new tensor with a dimension of size one inserted at the
specified position.

This implementation supports exact native CPU float32 tensors for every valid
insertion dimension. dtype/device extensions, tensor subclasses, broader
``None`` indexing expansion, and ``__torch_function__`` modes remain
unsupported.
";

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

fn is_vulkan_available_variable_function(
    py: Python<'_>,
    _args: &Bound<'_, PyTuple>,
    _kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    // Vulkan execution is not compiled into this native backend. PyTorch's
    // availability query deliberately ignores every argument and keyword.
    false.into_py_any(py)
}

fn nnpack_available_variable_function(
    py: Python<'_>,
    _args: &Bound<'_, PyTuple>,
    _kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    // NNPACK execution is not compiled into this native Cargo backend.
    // PyTorch's private build probe ignores every argument and keyword.
    false.into_py_any(py)
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
variable_function_callback!(as_tensor_callback, as_tensor_variable_function);
variable_function_callback!(asarray_callback, asarray_variable_function);
variable_function_callback!(scalar_tensor_callback, scalar_tensor_variable_function);
variable_function_callback!(arange_callback, arange_variable_function);
variable_function_callback!(ones_like_callback, ones_like_variable_function);
variable_function_callback!(zeros_like_callback, zeros_like_variable_function);
variable_function_callback!(atleast_1d_callback, atleast_1d_variable_function);
variable_function_callback!(atleast_2d_callback, atleast_2d_variable_function);
variable_function_callback!(atleast_3d_callback, atleast_3d_variable_function);
variable_function_callback!(
    broadcast_tensors_callback,
    broadcast_tensors_variable_function
);
variable_function_callback!(abs_callback, abs_variable_function);
variable_function_callback!(absolute_callback, absolute_variable_function);
variable_function_callback!(adjoint_callback, adjoint_variable_function);
variable_function_callback!(positive_callback, positive_variable_function);
variable_function_callback!(detach_callback, detach_variable_function);
variable_function_callback!(ravel_callback, ravel_variable_function);
variable_function_callback!(reshape_callback, reshape_variable_function);
variable_function_callback!(reciprocal_callback, reciprocal_variable_function);
variable_function_callback!(rsqrt_callback, rsqrt_variable_function);
variable_function_callback!(log_callback, log_variable_function);
variable_function_callback!(neg_callback, neg_variable_function);
variable_function_callback!(negative_callback, negative_variable_function);
variable_function_callback!(exp_callback, exp_variable_function);
variable_function_callback!(floor_callback, floor_variable_function);
variable_function_callback!(ceil_callback, ceil_variable_function);
variable_function_callback!(trunc_callback, trunc_variable_function);
variable_function_callback!(fix_callback, fix_variable_function);
variable_function_callback!(sin_callback, sin_variable_function);
variable_function_callback!(cos_callback, cos_variable_function);
variable_function_callback!(sqrt_callback, sqrt_variable_function);
variable_function_callback!(sigmoid_callback, sigmoid_variable_function);
variable_function_callback!(square_callback, square_variable_function);
variable_function_callback!(sum_callback, sum_variable_function);
variable_function_callback!(mean_callback, mean_variable_function);
variable_function_callback!(tanh_callback, tanh_variable_function);
variable_function_callback!(add_callback, add_variable_function);
variable_function_callback!(sub_callback, sub_variable_function);
variable_function_callback!(subtract_callback, subtract_variable_function);
variable_function_callback!(mul_callback, mul_variable_function);
variable_function_callback!(multiply_callback, multiply_variable_function);
variable_function_callback!(
    is_vulkan_available_callback,
    is_vulkan_available_variable_function
);
variable_function_callback!(
    nnpack_available_callback,
    nnpack_available_variable_function
);
variable_function_callback!(is_conj_callback, is_conj_variable_function);
variable_function_callback!(is_inference_callback, is_inference_variable_function);
variable_function_callback!(conj_callback, conj_variable_function);
variable_function_callback!(real_callback, real_variable_function);
variable_function_callback!(imag_callback, imag_variable_function);
variable_function_callback!(resolve_conj_callback, resolve_conj_variable_function);
variable_function_callback!(resolve_neg_callback, resolve_neg_variable_function);
variable_function_callback!(unbind_callback, unbind_variable_function);
variable_function_callback!(unsqueeze_callback, unsqueeze_variable_function);
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
        variable_function_method!(c"as_tensor", as_tensor_callback, AS_TENSOR_DOC),
        variable_function_method!(c"asarray", asarray_callback, ASARRAY_DOC),
        variable_function_method!(c"scalar_tensor", scalar_tensor_callback, c""),
        variable_function_method!(c"arange", arange_callback, ARANGE_DOC),
        variable_function_method!(c"ones_like", ones_like_callback, ONES_LIKE_DOC),
        variable_function_method!(c"zeros_like", zeros_like_callback, ZEROS_LIKE_DOC),
        variable_function_method!(c"atleast_1d", atleast_1d_callback, c""),
        variable_function_method!(c"atleast_2d", atleast_2d_callback, c""),
        variable_function_method!(c"atleast_3d", atleast_3d_callback, c""),
        variable_function_method!(c"broadcast_tensors", broadcast_tensors_callback, c""),
        variable_function_method!(c"abs", abs_callback, ABS_DOC),
        variable_function_method!(c"absolute", absolute_callback, ABSOLUTE_DOC),
        variable_function_method!(c"adjoint", adjoint_callback, ADJOINT_DOC),
        variable_function_method!(c"positive", positive_callback, POSITIVE_DOC),
        variable_function_method!(c"detach", detach_callback, c""),
        variable_function_method!(c"ravel", ravel_callback, RAVEL_DOC),
        variable_function_method!(c"reshape", reshape_callback, RESHAPE_DOC),
        variable_function_method!(c"reciprocal", reciprocal_callback, RECIPROCAL_DOC),
        variable_function_method!(c"rsqrt", rsqrt_callback, RSQRT_DOC),
        variable_function_method!(c"log", log_callback, LOG_DOC),
        variable_function_method!(c"neg", neg_callback, NEG_DOC),
        variable_function_method!(c"negative", negative_callback, NEGATIVE_DOC),
        variable_function_method!(c"exp", exp_callback, EXP_DOC),
        variable_function_method!(c"floor", floor_callback, FLOOR_DOC),
        variable_function_method!(c"ceil", ceil_callback, CEIL_DOC),
        variable_function_method!(c"trunc", trunc_callback, TRUNC_DOC),
        variable_function_method!(c"fix", fix_callback, FIX_DOC),
        variable_function_method!(c"sin", sin_callback, SIN_DOC),
        variable_function_method!(c"cos", cos_callback, COS_DOC),
        variable_function_method!(c"sqrt", sqrt_callback, SQRT_DOC),
        variable_function_method!(c"sigmoid", sigmoid_callback, SIGMOID_DOC),
        variable_function_method!(c"square", square_callback, SQUARE_DOC),
        variable_function_method!(c"sum", sum_callback, SUM_DOC),
        variable_function_method!(c"mean", mean_callback, MEAN_DOC),
        variable_function_method!(c"tanh", tanh_callback, TANH_DOC),
        variable_function_method!(c"add", add_callback, ADD_DOC),
        variable_function_method!(c"sub", sub_callback, SUB_DOC),
        variable_function_method!(c"subtract", subtract_callback, SUBTRACT_DOC),
        variable_function_method!(c"mul", mul_callback, MUL_DOC),
        variable_function_method!(c"multiply", multiply_callback, MULTIPLY_DOC),
        variable_function_method!(c"is_vulkan_available", is_vulkan_available_callback, c""),
        variable_function_method!(c"_nnpack_available", nnpack_available_callback, c""),
        variable_function_method!(c"is_conj", is_conj_callback, IS_CONJ_DOC),
        variable_function_method!(c"is_inference", is_inference_callback, IS_INFERENCE_DOC),
        variable_function_method!(c"conj", conj_callback, CONJ_DOC),
        variable_function_method!(c"real", real_callback, REAL_DOC),
        variable_function_method!(c"imag", imag_callback, IMAG_DOC),
        variable_function_method!(c"resolve_conj", resolve_conj_callback, RESOLVE_CONJ_DOC),
        variable_function_method!(c"resolve_neg", resolve_neg_callback, RESOLVE_NEG_DOC),
        variable_function_method!(c"unbind", unbind_callback, UNBIND_DOC),
        variable_function_method!(c"unsqueeze", unsqueeze_callback, UNSQUEEZE_DOC),
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
    variable_functions
        .getattr("_nnpack_available")?
        .setattr("__module__", "torch")?;
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

import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


MM_DOC = """
mm(input, mat2, *, out=None) -> Tensor

Performs a matrix multiplication of the matrices :attr:`input` and :attr:`mat2`.

If :attr:`input` is a :math:`(n \\times m)` tensor, :attr:`mat2` is a
:math:`(m \\times p)` tensor, :attr:`out` will be a :math:`(n \\times p)` tensor.

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
          For broadcasting matrix products, see :func:`torch.matmul`.

Supports strided and sparse 2-D tensors as inputs, autograd with
respect to strided inputs.

This operation has support for arguments with :ref:`sparse layouts<sparse-docs>`.
If :attr:`out` is provided its layout will be used. Otherwise, the result
layout will be deduced from that of :attr:`input`.


.. warning::
    Sparse support is a beta feature and some layout(s)/dtype/device combinations may not be supported,
    or may not have autograd support. If you notice missing functionality please
    open a feature request.

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

Args:
    input (Tensor): the first matrix to be matrix multiplied
    mat2 (Tensor): the second matrix to be matrix multiplied

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> mat1 = torch.randn(2, 3)
    >>> mat2 = torch.randn(3, 3)
    >>> torch.mm(mat1, mat2)
    tensor([[ 0.4851,  0.5037, -0.3633],
            [-0.0760, -3.6705,  2.4784]])

.. function:: mm(input, mat2, out_dtype, *, out=None) -> Tensor
   :noindex:

Args:
    input (Tensor): the first matrix to be matrix multiplied
    mat2 (Tensor): the second matrix to be matrix multiplied
    out_dtype (dtype): the dtype of the output tensor.
        Supported only on CUDA and for torch.float32 given
        torch.float16/torch.bfloat16 input dtypes.

Keyword args:
    out (Tensor, optional): the output tensor.
"""


def mm_cases(module):
    offset_left = module.tensor(
        np.arange(18, dtype=np.float32).reshape(3, 2, 3).tolist()
    )[1]
    offset_right = module.tensor(
        np.arange(12, dtype=np.float32).reshape(2, 3, 2).tolist()
    )[1]
    noncontiguous_left = module.tensor(
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    ).transpose(0, 1)
    noncontiguous_right = module.tensor(
        [[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]]
    ).transpose(0, 1)
    signed_zero_left = module.tensor([[-0.0, 0.0], [0.0, -0.0]])
    signed_zero_right = module.tensor([[1.0, 1.0], [1.0, -1.0]])
    special_left = module.tensor(
        [[float("inf"), 0.0], [float("-inf"), -1.0], [float("nan"), 2.0]]
    )
    special_right = module.tensor([[1.0, -1.0], [0.5, float("inf")]])

    return (
        (
            "square",
            module.tensor([[1.0, 2.0], [3.0, 4.0]]),
            module.tensor([[5.0, 6.0], [7.0, 8.0]]),
        ),
        (
            "rectangular",
            module.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            module.tensor(
                [[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]]
            ),
        ),
        ("empty rows", module.zeros((0, 3)), module.ones((3, 2))),
        ("empty inner", module.ones((2, 0)), module.zeros((0, 3))),
        ("empty columns", module.ones((2, 3)), module.zeros((3, 0))),
        ("offset", offset_left, offset_right),
        ("noncontiguous", noncontiguous_left, noncontiguous_right),
        ("signed zero", signed_zero_left, signed_zero_right),
        ("nan infinity", special_left, special_right),
    )


class MmTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, expected.dtype)
            self.assertEqual(actual.device, expected.device)
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                np.asarray(expected).reshape(-1).view(np.uint32),
            )

    def test_rank_two_calls_delegate_to_tensor_matmul(self):
        for case, left, right in mm_cases(torch):
            expected = left.matmul(right)
            calls = (
                ("positional", lambda: torch.mm(left, right)),
                ("canonical keywords", lambda: torch.mm(input=left, mat2=right)),
                ("input alias x", lambda: torch.mm(x=left, mat2=right)),
                ("input alias a", lambda: torch.mm(a=left, mat2=right)),
                ("input alias x1", lambda: torch.mm(x1=left, mat2=right)),
                ("out none", lambda: torch.mm(left, right, out=None)),
            )
            for form, call in calls:
                self.assert_tensor_matches(call(), expected, case=(case, form))

    def test_current_autograd_boundary_matches_matmul(self):
        function_left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        function_right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        method_left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        method_right = torch.tensor([[3.0], [4.0]], requires_grad=True)

        result = torch.mm(function_left, function_right)
        expected = method_left.matmul(method_right)
        self.assert_tensor_matches(result, expected, case="requires-grad operands")
        self.assertFalse(result.requires_grad)
        self.assertTrue(result.is_leaf)
        with self.assertRaises(RuntimeError) as raised:
            result.sum().backward()
        self.assertEqual(
            str(raised.exception),
            "element 0 of tensors does not require grad and does not have a grad_fn",
        )
        for operand in (function_left, function_right, method_left, method_right):
            self.assertIsNone(operand.grad)

    def test_unsupported_boundaries_are_explicit(self):
        left = torch.ones((2, 2))
        right = torch.ones((2, 2))
        destination = torch.tensor([[17.0, 18.0], [19.0, 20.0]])

        with self.assertRaisesRegex(
            RuntimeError, r"^mm\(\): the 'out' argument is not supported$"
        ):
            torch.mm(left, right, out=destination)
        self.assertEqual(destination.tolist(), [[17.0, 18.0], [19.0, 20.0]])

        for call, message in (
            (lambda: torch.mm(torch.ones((2,)), right), "self must be a matrix"),
            (lambda: torch.mm(left, torch.ones((2,))), "mat2 must be a matrix"),
            (lambda: torch.mm(torch.ones((1, 2, 2)), right), "self must be a matrix"),
            (lambda: torch.mm(left, torch.ones((1, 2, 2))), "mat2 must be a matrix"),
            (
                lambda: torch.mm(left, right, dtype=torch.float32),
                "mm() got an unexpected keyword argument 'dtype'",
            ),
            (
                lambda: torch.mm(left, right, device=torch.device("cpu")),
                "mm() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: torch.mm(left, right, out_dtype=torch.float32),
                "mm() got an unexpected keyword argument 'out_dtype'",
            ),
            (
                lambda: torch.mm(x1=left, x2=right),
                'mm() missing 1 required positional arguments: "mat2"',
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(Exception, f"^{re.escape(message)}$"):
                    call()

        self.assertFalse(hasattr(torch, "bmm"))
        self.assertFalse(hasattr(torch, "addmm"))
        self.assertFalse(hasattr(torch.Tensor, "mm"))

    def test_shape_errors_reuse_rank_two_matmul_diagnostics(self):
        left = torch.zeros((2, 3))
        right = torch.zeros((4, 2))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^mat1 and mat2 shapes cannot be multiplied \(2x3 and 4x2\)$",
        ):
            torch.mm(left, right)

    def test_torch_function_modes_and_overrides_observe_original_calls(self):
        left = torch.tensor([[1.0]])
        right = torch.tensor([[2.0]])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode()
        with mode:
            self.assertIs(torch.mm(input=left, mat2=right, out=None), marker)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.mm)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(tuple(kwargs), ("input", "mat2", "out"))
        self.assertIs(kwargs["input"], left)
        self.assertIs(kwargs["mat2"], right)
        self.assertIsNone(kwargs["out"])

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                actual = torch.mm(x=left, mat2=right)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_tensor_matches(actual, left.matmul(right), case="forwarded modes")

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.mm(left, Override()), marker)
        function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(function, torch.mm)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], left)
        self.assertIsInstance(args[1], Override)
        self.assertIsNone(kwargs)

    def test_callable_metadata_imports_reload_copy_and_pickle(self):
        function = torch.mm
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "mm")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.mm")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, MM_DOC)
        self.assertRegex(
            repr(function), r"^<built-in method mm of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.mm, function)

        from torch_rs import mm as imported_mm

        self.assertIs(imported_mm, function)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["mm"], function)
        self.assertEqual(torch.__all__.count("mm"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(torch.mm, function)


if __name__ == "__main__":
    unittest.main()

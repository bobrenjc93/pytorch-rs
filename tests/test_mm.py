import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = """
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


def mm_layout_cases(module):
    offset_left = module.tensor(
        np.arange(24, dtype=np.float32).reshape(3, 2, 4).tolist()
    )[1]
    offset_right = module.tensor(
        np.arange(24, dtype=np.float32).reshape(2, 4, 3).tolist()
    )[1]
    strided_left = module.tensor(
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    ).transpose(0, 1)
    strided_right = module.tensor(
        [[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]]
    ).transpose(0, 1)
    return (
        (
            "square",
            module.tensor([[1.0, 2.0], [3.0, 4.0]]),
            module.tensor([[5.0, 6.0], [7.0, 8.0]]),
        ),
        (
            "rectangular",
            module.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            module.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
        ),
        (
            "empty rows",
            module.zeros((0, 3)),
            module.ones((3, 2)),
        ),
        (
            "empty inner",
            module.ones((2, 0)),
            module.zeros((0, 3)),
        ),
        ("offset", offset_left, offset_right),
        ("noncontiguous", strided_left, strided_right),
        (
            "signed zero",
            module.tensor([[-0.0, 0.0], [0.0, -0.0]]),
            module.tensor([[1.0, -1.0], [-1.0, 1.0]]),
        ),
        (
            "nan and infinity",
            module.tensor(
                [[float("inf"), 1.0], [float("-inf"), -1.0], [float("nan"), 2.0]]
            ),
            module.tensor([[1.0, -1.0], [0.5, 1.0]]),
        ),
    )


class TopLevelMmTests(unittest.TestCase):
    def assert_matches_matmul(self, actual, expected, *, case):
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

    def test_rank_two_calls_delegate_to_existing_matmul(self):
        for case, left, right in mm_layout_cases(torch):
            expected = torch.matmul(left, right)
            calls = (
                ("positional", lambda: torch.mm(left, right)),
                ("canonical keywords", lambda: torch.mm(input=left, mat2=right)),
                ("x alias", lambda: torch.mm(x=left, mat2=right)),
                ("a alias", lambda: torch.mm(a=left, mat2=right)),
                ("x1 alias", lambda: torch.mm(x1=left, mat2=right)),
                ("positional mat2 keyword", lambda: torch.mm(left, mat2=right)),
                ("out none", lambda: torch.mm(left, right, out=None)),
            )
            for style, call in calls:
                self.assert_matches_matmul(call(), expected, case=(case, style))

    def test_torch_function_modes_and_operand_overrides(self):
        left = torch.tensor([[1.0]])
        right = torch.tensor([[2.0]])
        function = torch.mm
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
            self.assertIs(torch.mm(x=left, mat2=right, out=None), marker)
        func, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(func, function)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(tuple(kwargs), ("x", "mat2", "out"))
        self.assertIs(kwargs["x"], left)
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
                actual = torch.mm(input=left, mat2=right, out=None)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_matches_matmul(actual, left.matmul(right), case="forwarded modes")

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for case, call, expected_keywords in (
            ("left", lambda value: torch.mm(value, right), None),
            ("right", lambda value: torch.mm(left, value), None),
            ("keyword right", lambda value: torch.mm(input=left, mat2=value), ("input", "mat2")),
            ("out", lambda value: torch.mm(left, right, out=value), ("out",)),
        ):
            value = Override()
            Override.calls.clear()
            self.assertIs(call(value), marker)
            func, dispatch_types, args, kwargs = Override.calls[0]
            with self.subTest(case=case):
                self.assertIs(func, function)
                self.assertEqual(dispatch_types, (Override,))
                if expected_keywords is None:
                    self.assertEqual(len(args), 2)
                    self.assertIsNone(kwargs)
                else:
                    self.assertEqual(tuple(kwargs), expected_keywords)

    def test_autograd_and_unsupported_boundaries_stay_narrow(self):
        left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        result = torch.mm(left, right)
        expected = torch.matmul(left, right)
        self.assert_matches_matmul(result, expected, case="requires-grad operands")
        self.assertFalse(result.requires_grad)
        self.assertTrue(result.is_leaf)

        destination = torch.full((1, 1), 17.0)
        with self.assertRaisesRegex(
            RuntimeError, r"^mm\(\): the 'out' argument is not supported$"
        ):
            torch.mm(left, right, out=destination)
        self.assertEqual(destination.tolist(), [[17.0]])

        with self.assertRaisesRegex(TypeError, "received an invalid combination"):
            torch.mm(left, right, torch.float32)
        self.assertFalse(hasattr(torch, "bmm"))
        self.assertFalse(hasattr(torch, "addmm"))
        self.assertFalse(hasattr(torch, "mv"))

        for input, mat2 in (
            (torch.ones((2,)), torch.ones((2, 2))),
            (torch.ones((1, 2, 2)), torch.ones((2, 2))),
        ):
            with self.subTest(input=input.shape, mat2=mat2.shape):
                with self.assertRaises(RuntimeError) as mm_error:
                    torch.mm(input, mat2)
                with self.assertRaises(RuntimeError) as matmul_error:
                    torch.matmul(input, mat2)
                self.assertEqual(str(mm_error.exception), str(matmul_error.exception))
                self.assertIn("requires two rank-2 tensors", str(mm_error.exception))

    def test_binding_metadata_import_reload_copy_and_pickle(self):
        tensor = torch.tensor([[1.0]])
        cases = (
            (lambda: torch.mm(), "received an invalid combination"),
            (lambda: torch.mm(tensor), "received an invalid combination"),
            (lambda: torch.mm(tensor, tensor, tensor), "received an invalid combination"),
            (
                lambda: torch.mm([], tensor),
                "mm(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.mm(tensor, []),
                "mm(): argument 'mat2' (position 2) must be Tensor, not list",
            ),
            (
                lambda: torch.mm(input=None, mat2=tensor),
                "mm(): argument 'input' must be Tensor, not NoneType",
            ),
            (
                lambda: torch.mm(x1=tensor, x2=tensor),
                'mm() missing 1 required positional arguments: "mat2"',
            ),
            (lambda: torch.mm(tensor, tensor, extra=True), "received an invalid combination"),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, re.escape(message)):
                    call()

        function = torch.mm
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "mm")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.mm")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.mm, function)
        for mutation in (
            lambda: setattr(owner, "mm", None),
            lambda: delattr(owner, "mm"),
        ):
            with self.assertRaises(TypeError):
                mutation()
            self.assertIs(owner.mm, function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("mm"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["mm"], function)

        native = torch._C
        self.assertIs(importlib.import_module("torch_rs").mm, function)
        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.mm, function)
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.mm, function)


if __name__ == "__main__":
    unittest.main()

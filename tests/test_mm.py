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
        (
            "signed zero",
            module.tensor([[-0.0, 0.0]]),
            module.tensor([[-1.0], [1.0]]),
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

    def test_rank_two_calls_delegate_to_matmul(self):
        for case, left, right in mm_layout_cases(torch):
            expected = left.matmul(right)
            calls = (
                ("positional", lambda: torch.mm(left, right)),
                ("canonical keywords", lambda: torch.mm(input=left, mat2=right)),
                ("legacy input alias", lambda: torch.mm(x1=left, mat2=right)),
                ("out none", lambda: torch.mm(left, right, out=None)),
            )
            for style, call in calls:
                self.assert_matches_matmul(call(), expected, case=(case, style))

    def test_autograd_boundary_matches_current_matmul(self):
        mm_left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        mm_right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        matmul_left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        matmul_right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

        mm_output = torch.mm(mm_left, mm_right)
        matmul_output = torch.matmul(matmul_left, matmul_right)
        self.assert_matches_matmul(
            mm_output, matmul_output, case="requires-grad operands"
        )
        self.assertFalse(mm_output.requires_grad)
        self.assertTrue(mm_output.is_leaf)

        with self.assertRaises(RuntimeError) as raised:
            mm_output.sum().backward()
        self.assertEqual(
            str(raised.exception),
            "element 0 of tensors does not require grad and does not have a grad_fn",
        )
        for operand in (mm_left, mm_right, matmul_left, matmul_right):
            self.assertIsNone(operand.grad)

    def test_shape_rank_and_unsupported_surface_stay_narrow(self):
        with self.assertRaisesRegex(
            RuntimeError, r"^mat1 and mat2 shapes cannot be multiplied \(2x3 and 4x2\)$"
        ):
            torch.mm(torch.zeros((2, 3)), torch.zeros((4, 2)))

        for left, right, message in (
            (torch.tensor(1.0), torch.ones((1, 1)), "self must be a matrix"),
            (torch.ones((2,)), torch.ones((2, 2)), "self must be a matrix"),
            (torch.ones((1, 1)), torch.ones((1,)), "mat2 must be a matrix"),
            (torch.ones((1, 2, 2)), torch.ones((2, 2)), "self must be a matrix"),
            (torch.ones((2, 2)), torch.ones((1, 2, 2)), "mat2 must be a matrix"),
        ):
            with self.subTest(left=left.shape, right=right.shape):
                with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                    torch.mm(left, right)

        with self.assertRaisesRegex(
            RuntimeError, r"^mm\(\): the 'out' argument is not supported$"
        ):
            torch.mm(torch.ones((1, 1)), torch.ones((1, 1)), out=torch.zeros((1, 1)))

        for case, call in (
            (
                "positional out_dtype",
                lambda: torch.mm(torch.ones((1, 1)), torch.ones((1, 1)), torch.float32),
            ),
            (
                "keyword out_dtype",
                lambda: torch.mm(
                    torch.ones((1, 1)),
                    torch.ones((1, 1)),
                    out_dtype=torch.float32,
                ),
            ),
            (
                "out_dtype with out none",
                lambda: torch.mm(
                    torch.ones((1, 1)),
                    torch.ones((1, 1)),
                    torch.float32,
                    out=None,
                ),
            ),
        ):
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^mm\(\): out_dtype is not supported for CPU tensors$",
                ):
                    call()

        for case, call in (
            (
                "positional out_dtype none",
                lambda: torch.mm(torch.ones((1, 1)), torch.ones((1, 1)), None),
            ),
            (
                "keyword out_dtype none",
                lambda: torch.mm(
                    torch.ones((1, 1)),
                    torch.ones((1, 1)),
                    out_dtype=None,
                ),
            ),
            (
                "positional out_dtype tensor",
                lambda: torch.mm(
                    torch.ones((1, 1)),
                    torch.ones((1, 1)),
                    torch.ones((1, 1)),
                ),
            ),
        ):
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    TypeError, r"^mm\(\) received an invalid combination"
                ):
                    call()
        self.assertFalse(hasattr(torch, "bmm"))
        self.assertFalse(hasattr(torch, "addmm"))

    def test_torch_function_modes_and_overrides_receive_original_calls(self):
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

        calls = (
            ("positional", lambda: torch.mm(left, right), (left, right), None),
            (
                "canonical keywords",
                lambda: torch.mm(input=left, mat2=right),
                (),
                ("input", "mat2"),
            ),
            ("out none", lambda: torch.mm(left, right, out=None), (left, right), ("out",)),
            (
                "positional out_dtype",
                lambda: torch.mm(left, right, torch.float32),
                (left, right, torch.float32),
                None,
            ),
            (
                "keyword out_dtype",
                lambda: torch.mm(left, right, out_dtype=torch.float32),
                (left, right),
                ("out_dtype",),
            ),
        )
        for case, call, expected_args, keywords in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            with self.subTest(case=case):
                self.assertIs(function, torch.mm)
                self.assertEqual(dispatch_types, ())
                self.assertEqual(args, expected_args)
                if keywords is None:
                    self.assertIsNone(kwargs)
                else:
                    self.assertEqual(tuple(kwargs), keywords)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                actual = torch.mm(input=left, mat2=right)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_matches_matmul(actual, left.matmul(right), case="forwarded modes")

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for case, call, expected_args, expected_keywords in (
            ("left override", lambda value: torch.mm(value, right), 2, None),
            ("right override", lambda value: torch.mm(left, value), 2, None),
            (
                "keyword override",
                lambda value: torch.mm(input=left, mat2=value),
                0,
                ("input", "mat2"),
            ),
            (
                "out_dtype positional override",
                lambda value: torch.mm(left, right, value),
                3,
                None,
            ),
            (
                "out_dtype keyword override",
                lambda value: torch.mm(left, right, out_dtype=value),
                2,
                ("out_dtype",),
            ),
            ("out override", lambda value: torch.mm(left, right, out=value), 2, ("out",)),
        ):
            value = Override()
            Override.calls.clear()
            with self.subTest(case=case):
                self.assertIs(call(value), marker)
                self.assertEqual(len(Override.calls), 1)
                function, dispatch_types, args, kwargs = Override.calls[0]
                self.assertIs(function, torch.mm)
                self.assertEqual(dispatch_types, (Override,))
                self.assertEqual(len(args), expected_args)
                if expected_keywords is None:
                    self.assertIsNone(kwargs)
                else:
                    self.assertEqual(tuple(kwargs), expected_keywords)

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            torch.mm(DecliningOverride(), left)
        self.assertEqual(
            str(raised.exception),
            "Multiple dispatch failed for 'torch.mm'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            f"  - tensor subclass <class '{DecliningOverride.__module__}."
            f"{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented",
        )

        mode = RecordingMode()
        with mode:
            with self.assertRaises(TypeError):
                torch.mm([], right)
        self.assertEqual(mode.calls, [])

        mode = RecordingMode()
        with mode:
            with self.assertRaises(TypeError):
                torch.mm(left, right, out_dtype=None)
        self.assertEqual(mode.calls, [])

    def test_binding_metadata_import_reload_copy_and_pickle(self):
        package = importlib.import_module("torch_rs")
        native = package._C
        function = package.mm
        direct_import = {}
        wildcard_namespace = {}
        exec("from torch_rs import mm", direct_import)
        exec("from torch_rs import *", wildcard_namespace)

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "mm")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.mm")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method mm of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, package._C._VariableFunctionsClass)
        self.assertIs(owner.mm, function)
        self.assertIs(native.mm, function)
        self.assertIs(package.mm, function)
        self.assertIs(direct_import["mm"], function)
        self.assertEqual(package.__all__.count("mm"), 1)
        self.assertNotIn("_VariableFunctionsClass", package.__all__)
        self.assertFalse(hasattr(package, "_VariableFunctionsClass"))
        self.assertIs(wildcard_namespace["mm"], function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.mm, function)
        self.assertIs(importlib.reload(package), package)
        self.assertIs(package.mm, function)
        self.assertEqual(package.__all__.count("mm"), 1)


if __name__ == "__main__":
    unittest.main()

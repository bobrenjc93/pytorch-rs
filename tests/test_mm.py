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
        np.arange(18, dtype=np.float32).reshape(3, 2, 3).tolist(),
        dtype=module.float32,
    )[1]
    offset_right = module.tensor(
        np.arange(12, dtype=np.float32).reshape(2, 3, 2).tolist(),
        dtype=module.float32,
    )[1]
    noncontiguous_left = module.tensor(
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
        dtype=module.float32,
    ).transpose(0, 1)
    noncontiguous_right = module.tensor(
        [[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]],
        dtype=module.float32,
    ).transpose(0, 1)
    return (
        (
            "square",
            module.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=module.float32),
            module.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=module.float32),
        ),
        (
            "rectangular",
            module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=module.float32,
            ),
            module.tensor(
                [[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]],
                dtype=module.float32,
            ),
        ),
        (
            "empty rows",
            module.zeros((0, 3), dtype=module.float32),
            module.ones((3, 2), dtype=module.float32),
        ),
        (
            "empty inner",
            module.ones((2, 0), dtype=module.float32),
            module.zeros((0, 3), dtype=module.float32),
        ),
        ("offset", offset_left, offset_right),
        ("noncontiguous", noncontiguous_left, noncontiguous_right),
        (
            "signed zero",
            module.tensor([[-0.0, 0.0], [1.0, -1.0]], dtype=module.float32),
            module.tensor([[1.0, -2.0], [1.0, 2.0]], dtype=module.float32),
        ),
        (
            "nan and infinity",
            module.tensor(
                [
                    [float("inf"), 1.0],
                    [float("-inf"), -1.0],
                    [float("nan"), 2.0],
                ],
                dtype=module.float32,
            ),
            module.tensor([[1.0, -1.0], [0.5, 1.0]], dtype=module.float32),
        ),
    )


class TopLevelMmTests(unittest.TestCase):
    def assert_matches_tensor_matmul(self, actual, expected, *, case):
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
        for case, left, right in mm_layout_cases(torch):
            expected = left.matmul(right)
            calls = (
                ("positional", lambda: torch.mm(left, right)),
                ("mixed keyword", lambda: torch.mm(left, mat2=right)),
                ("canonical keywords", lambda: torch.mm(input=left, mat2=right)),
                ("out none", lambda: torch.mm(left, right, out=None)),
                (
                    "keyword out none",
                    lambda: torch.mm(input=left, mat2=right, out=None),
                ),
            )
            for style, call in calls:
                self.assert_matches_tensor_matmul(
                    call(),
                    expected,
                    case=(case, style),
                )

    def test_unsupported_boundaries_stay_explicit(self):
        left = torch.ones((1, 1), requires_grad=True)
        right = torch.ones((1, 1), requires_grad=True)
        result = torch.mm(input=left, mat2=right)
        self.assert_matches_tensor_matmul(result, left.matmul(right), case="autograd")
        self.assertFalse(result.requires_grad)
        self.assertTrue(result.is_leaf)
        with self.assertRaisesRegex(
            RuntimeError,
            "^element 0 of tensors does not require grad and does not have a grad_fn$",
        ):
            result.sum().backward()
        self.assertIsNone(left.grad)
        self.assertIsNone(right.grad)

        destination = torch.tensor([[17.0]])
        with self.assertRaisesRegex(
            RuntimeError,
            r"^mm\(\): the 'out' argument is not supported$",
        ):
            torch.mm(left, right, out=destination)
        self.assertEqual(destination.tolist(), [[17.0]])

        rank_cases = (
            (torch.tensor(1.0), torch.ones((1, 1))),
            (torch.ones((2,)), torch.ones((2, 2))),
            (torch.ones((1, 2, 2)), torch.ones((2, 2))),
            (torch.ones((2, 2)), torch.ones((2, 2, 1))),
        )
        for lhs, rhs in rank_cases:
            with self.subTest(left=lhs.shape, right=rhs.shape):
                with self.assertRaisesRegex(RuntimeError, "requires two rank-2 tensors"):
                    torch.mm(lhs, rhs)

        for call in (
            lambda: torch.mm(x1=left, x2=right),
            lambda: torch.mm(input=left, other=right),
            lambda: torch.mm(left, right, dtype=torch.float32),
            lambda: torch.mm(left, right, device=torch.device("cpu")),
            lambda: torch.mm(left, right, out_dtype=torch.float32),
        ):
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

        self.assertFalse(hasattr(torch, "bmm"))
        self.assertFalse(hasattr(torch, "addmm"))
        self.assertEqual(torch.__all__.count("bmm"), 0)
        self.assertEqual(torch.__all__.count("addmm"), 0)

    def test_torch_function_modes_receive_original_calls_and_can_forward(self):
        left = torch.tensor([[1.0]])
        right = torch.tensor([[2.0]])
        destination = torch.zeros((1, 1))
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        calls = (
            ("positional", lambda: torch.mm(left, right), 2, None),
            (
                "canonical",
                lambda: torch.mm(input=left, mat2=right),
                0,
                ("input", "mat2"),
            ),
            (
                "out none",
                lambda: torch.mm(left, right, out=None),
                2,
                ("out",),
            ),
            (
                "concrete out",
                lambda: torch.mm(left, right, out=destination),
                2,
                ("out",),
            ),
        )
        for case, call, expected_arg_count, expected_keywords in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            with self.subTest(case=case):
                self.assertIs(function, torch.mm)
                self.assertEqual(dispatch_types, ())
                self.assertEqual(len(args), expected_arg_count)
                if expected_arg_count:
                    self.assertIs(args[0], left)
                    self.assertIs(args[1], right)
                if expected_keywords is None:
                    self.assertIsNone(kwargs)
                else:
                    self.assertEqual(tuple(kwargs), expected_keywords)

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
        self.assert_matches_tensor_matmul(actual, left.matmul(right), case="forwarded modes")

        for call in (
            lambda: torch.mm([], right),
            lambda: torch.mm(left, []),
            lambda: torch.mm(left, right, unexpected=True),
        ):
            mode = RecordingMode()
            with mode:
                with self.assertRaises(TypeError):
                    call()
            self.assertEqual(mode.calls, [])

    def test_operand_overrides_order_types_and_declining_errors(self):
        native = torch.tensor([[1.0]])
        marker = object()
        events = []

        class LeftOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("left", func, types, args, kwargs))
                return NotImplemented

        class RightOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("right", func, types, args, kwargs))
                return marker

        self.assertIs(torch.mm(LeftOverride(), RightOverride()), marker)
        self.assertEqual([event[0] for event in events], ["left", "right"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, torch.mm)
            self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
            self.assertEqual(len(args), 2)
            self.assertIsNone(kwargs)

        events.clear()
        self.assertIs(torch.mm(input=native, mat2=RightOverride()), marker)
        _, function, dispatch_types, args, kwargs = events[0]
        self.assertIs(function, torch.mm)
        self.assertEqual(dispatch_types, (RightOverride,))
        self.assertEqual(args, ())
        self.assertEqual(tuple(kwargs), ("input", "mat2"))

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            torch.mm(DecliningOverride(), native)
        self.assertEqual(
            str(raised.exception),
            "Multiple dispatch failed for 'torch.mm'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            f"  - tensor subclass <class '{DecliningOverride.__module__}."
            f"{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented",
        )

    def test_callable_metadata_import_reload_copy_pickle_and_exports(self):
        function = torch.mm
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "mm")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.mm")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
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
        for mutation in (
            lambda: setattr(owner, "mm", None),
            lambda: delattr(owner, "mm"),
        ):
            with self.assertRaises(TypeError):
                mutation()
            self.assertIs(owner.mm, function)

        imported = __import__("torch_rs", fromlist=["mm"]).mm
        self.assertIs(imported, function)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["mm"], function)
        self.assertEqual(torch.__all__.count("mm"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        self.assertIsNot(torch.mm, torch.matmul)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        before_all = torch.__all__
        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(torch.mm, function)
        self.assertIsNot(torch.__all__, before_all)
        self.assertEqual(torch.__all__.count("mm"), 1)
        self.assertIs(copy.copy(torch.mm), torch.mm)
        self.assertIs(pickle.loads(pickle.dumps(function)), torch.mm)


if __name__ == "__main__":
    unittest.main()

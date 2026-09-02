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


def mm_layout_cases(module):
    offset_left = module.tensor(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
    )[1]
    offset_mat2 = module.tensor(
        np.arange(40, dtype=np.float32).reshape(2, 4, 5).tolist()
    )[1]

    noncontiguous_left = module.tensor(
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    ).transpose(0, 1)
    noncontiguous_mat2 = module.tensor(
        [[7.0, 10.0, 13.0], [8.0, 11.0, 14.0], [9.0, 12.0, 15.0]]
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
        ("empty columns", module.ones((2, 3)), module.zeros((3, 0))),
        ("empty inner", module.ones((2, 0)), module.zeros((0, 3))),
        ("offset", offset_left, offset_mat2),
        ("noncontiguous", noncontiguous_left, noncontiguous_mat2),
        (
            "signed zero",
            module.tensor([[-0.0, 0.0], [0.0, -0.0]]),
            module.tensor([[1.0, -1.0], [-1.0, 1.0]]),
        ),
        (
            "nan and inf",
            module.tensor(
                [[float("inf"), 1.0], [float("-inf"), -1.0], [float("nan"), 2.0]]
            ),
            module.tensor([[1.0, -1.0], [0.5, 1.0]]),
        ),
    )


def invalid_mm_overload(summary):
    return (
        "mm() received an invalid combination of arguments - got "
        f"({summary}), but expected one of:\n"
        " * (Tensor input, Tensor mat2, *, Tensor out = None)\n"
        " * (Tensor input, Tensor mat2, torch.dtype out_dtype, *, Tensor out = None)\n"
    )


def first_operand_alias_pairs():
    aliases = ("input", "x", "a", "x1")
    for index, left_alias in enumerate(aliases):
        for right_alias in aliases[index + 1 :]:
            yield left_alias, right_alias


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

    def test_rank_two_results_delegate_to_existing_matmul(self):
        for case, left, mat2 in mm_layout_cases(torch):
            expected = torch.matmul(left, mat2)
            calls = (
                ("positional", lambda: torch.mm(left, mat2)),
                ("canonical keywords", lambda: torch.mm(input=left, mat2=mat2)),
                ("x alias", lambda: torch.mm(x=left, mat2=mat2)),
                ("a alias", lambda: torch.mm(a=left, mat2=mat2)),
                ("x1 alias", lambda: torch.mm(x1=left, mat2=mat2)),
                ("out none", lambda: torch.mm(left, mat2, out=None)),
            )
            for style, call in calls:
                self.assert_matches_matmul(call(), expected, case=(case, style))

    def test_existing_matmul_autograd_boundary_is_preserved(self):
        mm_left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        mm_right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        matmul_left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        matmul_right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

        result = torch.mm(mm_left, mm_right)
        expected = torch.matmul(matmul_left, matmul_right)
        self.assert_matches_matmul(result, expected, case="requires-grad operands")
        self.assertFalse(result.requires_grad)
        self.assertTrue(result.is_leaf)

        for output in (result, expected):
            with self.assertRaises(RuntimeError) as raised:
                output.sum().backward()
            self.assertEqual(
                str(raised.exception),
                "element 0 of tensors does not require grad and does not have a grad_fn",
            )
        for operand in (mm_left, mm_right, matmul_left, matmul_right):
            self.assertIsNone(operand.grad)

    def test_shape_rank_out_and_related_aliases_remain_out_of_scope(self):
        left = torch.zeros((2, 3))
        mat2 = torch.zeros((4, 2))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^mat1 and mat2 shapes cannot be multiplied \(2x3 and 4x2\)$",
        ):
            torch.mm(left, mat2)

        for tensor, mat2, message in (
            (torch.tensor(1.0), torch.ones((1, 1)), "self must be a matrix"),
            (torch.ones((2,)), torch.ones((2, 2)), "self must be a matrix"),
            (torch.ones((1, 2, 2)), torch.ones((2, 2)), "self must be a matrix"),
            (torch.ones((1, 1)), torch.ones((1,)), "mat2 must be a matrix"),
            (torch.ones((1, 1)), torch.ones((1, 1, 1)), "mat2 must be a matrix"),
        ):
            with self.subTest(input_shape=tensor.shape, mat2_shape=mat2.shape):
                with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                    torch.mm(tensor, mat2)

        destination = torch.tensor([[17.0]])
        with self.assertRaisesRegex(
            RuntimeError, r"^mm\(\): the 'out' argument is not supported$"
        ):
            torch.mm(torch.ones((1, 1)), torch.ones((1, 1)), out=destination)
        self.assertEqual(destination.tolist(), [[17.0]])

        for name in ("bmm", "addmm"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)

    def test_torch_function_modes_receive_original_calls_and_can_forward(self):
        left = torch.tensor([[1.0]])
        mat2 = torch.tensor([[2.0]])
        out = torch.tensor([[0.0]])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        calls = (
            ("positional", lambda: torch.mm(left, mat2), None, 2),
            (
                "canonical",
                lambda: torch.mm(input=left, mat2=mat2),
                ("input", "mat2"),
                0,
            ),
            ("out none", lambda: torch.mm(left, mat2, out=None), ("out",), 2),
            ("out tensor", lambda: torch.mm(left, mat2, out=out), ("out",), 2),
        )
        for case, call, keywords, positional_count in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            with self.subTest(case=case):
                self.assertIs(function, torch.mm)
                self.assertEqual(dispatch_types, ())
                self.assertEqual(len(args), positional_count)
                if keywords is None:
                    self.assertIs(args[0], left)
                    self.assertIs(args[1], mat2)
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
                actual = torch.mm(input=left, mat2=mat2, out=None)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_matches_matmul(actual, torch.matmul(left, mat2), case="forwarded modes")

        for call in (
            lambda: torch.mm([], mat2),
            lambda: torch.mm(left, []),
            lambda: torch.mm(left, mat2, unexpected=True),
            lambda: torch.mm(input=left, x=mat2, mat2=mat2),
        ):
            mode = RecordingMode()
            with mode:
                with self.assertRaises(TypeError):
                    call()
            self.assertEqual(mode.calls, [])

    def test_duplicate_first_operand_aliases_reject_before_dispatch(self):
        tensor = torch.tensor([[1.0]])
        other = torch.tensor([[2.0]])
        invalid_overload = r"^mm\(\) received an invalid combination of arguments - got "

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return object()

        for left_alias, right_alias in first_operand_alias_pairs():
            with self.subTest(left_alias=left_alias, right_alias=right_alias):
                kwargs = {left_alias: tensor, right_alias: other, "mat2": tensor}
                with self.assertRaisesRegex(TypeError, invalid_overload):
                    torch.mm(**kwargs)

            for override_alias in (left_alias, right_alias):
                with self.subTest(
                    left_alias=left_alias,
                    right_alias=right_alias,
                    override_alias=override_alias,
                ):
                    Override.calls.clear()
                    mode = RecordingMode()
                    kwargs = {left_alias: tensor, right_alias: other, "mat2": tensor}
                    kwargs[override_alias] = Override()
                    with mode:
                        with self.assertRaisesRegex(TypeError, invalid_overload):
                            torch.mm(**kwargs)
                    self.assertEqual(mode.calls, [])
                    self.assertEqual(Override.calls, [])

    def test_operand_override_order_types_and_declining_errors(self):
        native = torch.tensor([[1.0]])
        events = []
        marker = object()

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
        self.assertIs(kwargs["input"], native)
        self.assertIsInstance(kwargs["mat2"], RightOverride)

        out_events = []

        class OutOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                out_events.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.mm(native, native, out=OutOverride()), marker)
        function, dispatch_types, args, kwargs = out_events[0]
        self.assertIs(function, torch.mm)
        self.assertEqual(dispatch_types, (OutOverride,))
        self.assertEqual(len(args), 2)
        self.assertEqual(tuple(kwargs), ("out",))
        self.assertIsInstance(kwargs["out"], OutOverride)

        subclass_events = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_events.append("base")
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_events.append(("derived", types))
                return marker

        self.assertIs(torch.mm(BaseOverride(), DerivedOverride()), marker)
        self.assertEqual(subclass_events, [("derived", (DerivedOverride, BaseOverride))])

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

    def test_binding_metadata_import_reload_copy_and_pickle(self):
        tensor = torch.tensor([[1.0]])
        cases = (
            (
                lambda: torch.mm(),
                'mm() missing 2 required positional argument: "input", "mat2"',
            ),
            (
                lambda: torch.mm(tensor),
                'mm() missing 1 required positional arguments: "mat2"',
            ),
            (
                lambda: torch.mm(tensor, tensor, tensor),
                invalid_mm_overload("Tensor, Tensor, Tensor"),
            ),
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
                lambda: torch.mm(input=tensor, other=tensor),
                'mm() missing 1 required positional arguments: "mat2"',
            ),
            (
                lambda: torch.mm(tensor, tensor, input=tensor),
                invalid_mm_overload("Tensor, Tensor, input=Tensor"),
            ),
            (
                lambda: torch.mm(tensor, tensor, mat2=tensor),
                invalid_mm_overload("Tensor, Tensor, mat2=Tensor"),
            ),
            (
                lambda: torch.mm(tensor, tensor, out_dtype=torch.float32),
                invalid_mm_overload("Tensor, Tensor, out_dtype=torch.dtype"),
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

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
        direct_namespace = {}
        exec("from torch_rs import mm", direct_namespace)
        self.assertIs(direct_namespace["mm"], function)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["mm"], function)

        native = torch._C
        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.mm, function)
        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(torch.mm, function)


if __name__ == "__main__":
    unittest.main()

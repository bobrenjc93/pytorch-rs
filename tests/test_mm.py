import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


def mm_layout_cases(module):
    offset_left = module.tensor(
        np.arange(24, dtype=np.float32).reshape(4, 2, 3).tolist()
    )[1]
    offset_right = module.tensor(
        np.arange(18, dtype=np.float32).reshape(3, 3, 2).tolist()
    )[1]
    strided_left = module.tensor(
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    ).transpose(0, 1)
    strided_right = module.tensor(
        [[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]]
    ).transpose(0, 1)

    return (
        (
            "contiguous",
            module.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            module.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
        ),
        ("offset contiguous", offset_left, offset_right),
        ("strided", strided_left, strided_right),
        ("empty rows", module.zeros((0, 3)), module.ones((3, 4))),
        ("empty inner", module.ones((2, 0)), module.zeros((0, 3))),
        ("empty columns", module.ones((2, 3)), module.zeros((3, 0))),
    )


class MmTests(unittest.TestCase):
    def assert_same_tensor(self, actual, expected, *, case):
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

    def test_positional_and_canonical_keyword_calls_reuse_rank_two_engine(self):
        for case, left, right in mm_layout_cases(torch):
            expected = left @ right
            for style, actual in (
                ("positional", torch.mm(left, right)),
                ("canonical keywords", torch.mm(input=left, mat2=right)),
                ("x alias", torch.mm(x=left, mat2=right)),
                ("a alias", torch.mm(a=left, mat2=right)),
                ("x1 alias", torch.mm(x1=left, mat2=right)),
            ):
                self.assert_same_tensor(actual, expected, case=(case, style))

    def test_shape_and_rank_errors_are_specific(self):
        for left_shape, right_shape in (
            ((2, 3), (4, 2)),
            ((0, 3), (4, 0)),
        ):
            message = (
                "mat1 and mat2 shapes cannot be multiplied "
                f"({left_shape[0]}x{left_shape[1]} and "
                f"{right_shape[0]}x{right_shape[1]})"
            )
            for call in (
                lambda: torch.mm(torch.zeros(left_shape), torch.zeros(right_shape)),
                lambda: torch.mm(
                    input=torch.zeros(left_shape), mat2=torch.zeros(right_shape)
                ),
            ):
                with self.subTest(left=left_shape, right=right_shape):
                    with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                        call()

        rank_cases = (
            (torch.tensor(1.0), torch.ones((1, 1)), "self must be a matrix"),
            (torch.ones((2,)), torch.ones((2, 2)), "self must be a matrix"),
            (torch.ones((2, 2)), torch.ones((2,)), "mat2 must be a matrix"),
            (
                torch.ones((1, 2, 2)),
                torch.ones((2, 2)),
                "self must be a matrix",
            ),
            (
                torch.ones((2, 2)),
                torch.ones((1, 2, 2)),
                "mat2 must be a matrix",
            ),
        )
        for left, right, message in rank_cases:
            with self.subTest(left=left.shape, right=right.shape):
                with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                    torch.mm(left, right)

    def test_autograd_out_and_out_dtype_stay_explicitly_unsupported(self):
        plain = torch.ones((2, 2))
        tracked = torch.ones((2, 2), requires_grad=True)
        message = "mm(): operands that require grad are not supported"
        for left, right in (
            (tracked, plain),
            (plain, tracked),
            (tracked, tracked),
        ):
            with self.subTest(left=left.requires_grad, right=right.requires_grad):
                with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                    torch.mm(left, right)
                with torch.no_grad():
                    with self.assertRaisesRegex(
                        RuntimeError, f"^{re.escape(message)}$"
                    ):
                        torch.mm(input=left, mat2=right)

        detached = torch.mm(tracked.detach(), tracked.detach())
        self.assertFalse(detached.requires_grad)
        self.assertTrue(detached.is_leaf)
        np.testing.assert_array_equal(
            np.asarray(detached), np.full((2, 2), 2.0, dtype=np.float32)
        )

        for out in (None, torch.zeros((2, 2))):
            with self.subTest(out=out):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^mm\(\): the 'out' argument is not supported$",
                ):
                    torch.mm(plain, plain, out=out)
        for call in (
            lambda: torch.mm(plain, plain, torch.float32),
            lambda: torch.mm(plain, plain, out_dtype=torch.float32),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                r"^mm\(\): the 'out_dtype' argument is not supported$",
            ):
                call()

    def test_modes_receive_original_calls_and_can_forward(self):
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

        calls = (
            ("positional", lambda: function(left, right), None),
            (
                "canonical keywords",
                lambda: function(input=left, mat2=right),
                ("input", "mat2"),
            ),
            ("input alias", lambda: function(x=left, mat2=right), ("x", "mat2")),
        )
        for case, call, keyword_names in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            self.assertEqual(len(mode.calls), 1)
            func, dispatch_types, args, kwargs = mode.calls[0]
            with self.subTest(case=case):
                self.assertIs(func, function)
                self.assertEqual(dispatch_types, ())
                if keyword_names is None:
                    self.assertEqual(len(args), 2)
                    self.assertIs(args[0], left)
                    self.assertIs(args[1], right)
                    self.assertIsNone(kwargs)
                else:
                    self.assertEqual(args, ())
                    self.assertEqual(tuple(kwargs), keyword_names)
                    self.assertIs(kwargs[keyword_names[0]], left)
                    self.assertIs(kwargs[keyword_names[1]], right)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                actual = function(input=left, mat2=right)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_same_tensor(actual, left @ right, case="forwarded modes")

        invalid_calls = (
            lambda: function([], right),
            lambda: function(left),
            lambda: function(left, right, extra=True),
        )
        for call in invalid_calls:
            mode = RecordingMode()
            with mode:
                with self.assertRaises(TypeError):
                    call()
            self.assertEqual(mode.calls, [])

    def test_operand_and_out_overrides_follow_variable_function_dispatch(self):
        left = torch.tensor([[1.0]])
        right = torch.tensor([[2.0]])
        function = torch.mm
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        calls = (
            (lambda value: function(value, right), None),
            (lambda value: function(left, value), None),
            (lambda value: function(input=left, mat2=value), "mat2"),
            (lambda value: function(left, right, out=value), "out"),
        )
        for call, keyword in calls:
            value = Override()
            Override.calls.clear()
            self.assertIs(call(value), marker)
            self.assertEqual(len(Override.calls), 1)
            func, dispatch_types, args, kwargs = Override.calls[0]
            self.assertIs(func, function)
            self.assertEqual(dispatch_types, (Override,))
            if keyword is None:
                self.assertEqual(len(args), 2)
                self.assertIsNone(kwargs)
            else:
                self.assertIs(kwargs[keyword], value)

        order = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                order.append(("base", types))
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                order.append(("derived", types))
                return marker

        self.assertIs(function(BaseOverride(), DerivedOverride()), marker)
        self.assertEqual(order, [("derived", (DerivedOverride, BaseOverride))])

        mode_calls = []

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                mode_calls.append((func, types, args, kwargs))
                return NotImplemented

        Override.calls.clear()
        value = Override()
        with DecliningMode():
            self.assertIs(function(input=left, mat2=value), marker)
        self.assertEqual(len(mode_calls), 1)
        self.assertEqual(len(Override.calls), 1)
        self.assertIs(mode_calls[0][0], function)
        self.assertIs(Override.calls[0][0], function)

    def test_out_dtype_keyword_dispatch_and_duplicate_binding(self):
        left = torch.tensor([[1.0]])
        right = torch.tensor([[2.0]])
        function = torch.mm
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        for call, positional_count, keyword_names in (
            (
                lambda: function(left, right, out_dtype=torch.float32),
                2,
                ("out_dtype",),
            ),
            (
                lambda: function(
                    input=left, mat2=right, out_dtype=torch.float32
                ),
                0,
                ("input", "mat2", "out_dtype"),
            ),
        ):
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            self.assertEqual(len(mode.calls), 1)
            func, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(func, function)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(len(args), positional_count)
            self.assertEqual(tuple(kwargs), keyword_names)
            self.assertIs(kwargs["out_dtype"], torch.float32)

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for call in (
            lambda: function(Override(), right, out_dtype=torch.float32),
            lambda: function(left, right, out_dtype=Override()),
        ):
            Override.calls.clear()
            self.assertIs(call(), marker)
            self.assertEqual(len(Override.calls), 1)
            self.assertIs(Override.calls[0][0], function)
            self.assertEqual(Override.calls[0][1], (Override,))
            self.assertEqual(tuple(Override.calls[0][3]), ("out_dtype",))

        mode = RecordingMode()
        with mode:
            with self.assertRaisesRegex(
                TypeError,
                r"^mm\(\) got multiple values for argument 'out_dtype'$",
            ):
                function(
                    left,
                    right,
                    torch.float32,
                    out_dtype=torch.float32,
                )
        self.assertEqual(mode.calls, [])

    def test_callable_metadata_pickling_exports_and_tensor_boundary(self):
        function = torch.mm
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "mm")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.mm")
        self.assertEqual(function.__module__, "torch")
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
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.mm, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(function, protocol)), function)

        self.assertEqual(torch.__all__.count("mm"), 1)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["mm"], function)
        self.assertFalse(hasattr(torch.Tensor, "mm"))
        with self.assertRaises(AttributeError):
            inspect.getattr_static(torch.Tensor, "mm")


if __name__ == "__main__":
    unittest.main()

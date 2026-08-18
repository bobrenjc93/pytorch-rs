import copy
import inspect
import pickle
import re
import types
import unittest

import numpy as np

import torch_rs as torch


FUNCTION_DOC = """
result_type(tensor1, tensor2) -> dtype

Returns the :class:`torch.dtype` that would result from performing an arithmetic
operation on the provided input tensors. See type promotion :ref:`documentation <type-promotion-doc>`
for more information on the type promotion logic.

Args:
    tensor1 (Tensor or Number): an input tensor or number
    tensor2 (Tensor or Number): an input tensor or number

Example::

    >>> torch.result_type(torch.tensor([1, 2], dtype=torch.int), 1.0)
    torch.float32
    >>> torch.result_type(torch.tensor([1, 2], dtype=torch.uint8), torch.tensor(1))
    torch.uint8
"""


class ResultTypeTests(unittest.TestCase):
    def assert_error(self, exception_type, message, call):
        with self.assertRaisesRegex(exception_type, f"^{re.escape(message)}$"):
            call()

    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        tracked.sum().backward()
        base = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        )
        offset = base[1]
        strided = base.transpose(0, 1)

        self.assertGreater(offset.storage_offset(), 0)
        self.assertFalse(strided.is_contiguous())
        return (
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3))),
            ("offset view", offset),
            ("strided view", strided),
            ("autograd leaf", leaf),
            ("autograd non-leaf view", tracked),
            ("accumulated gradient", leaf.grad),
        )

    def tensor_state(self, tensor):
        return (
            tensor.tolist(),
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.dtype,
            tensor.device,
            tensor.requires_grad,
            tensor.is_leaf,
        )

    def test_tensor_tensor_forms_promote_metadata_to_the_canonical_singleton(self):
        cases = self.tensor_cases()
        for left_name, left in cases:
            for right_name, right in cases:
                left_state = self.tensor_state(left)
                right_state = self.tensor_state(right)
                calls = (
                    ("positional", lambda: torch.result_type(left, right)),
                    ("mixed", lambda: torch.result_type(left, other=right)),
                    (
                        "keywords",
                        lambda: torch.result_type(tensor=left, other=right),
                    ),
                    (
                        "reversed keywords",
                        lambda: torch.result_type(other=right, tensor=left),
                    ),
                )
                for form, call in calls:
                    with self.subTest(
                        left=left_name, right=right_name, form=form
                    ):
                        self.assertIs(call(), torch.float32)
                self.assertEqual(self.tensor_state(left), left_state)
                self.assertEqual(self.tensor_state(right), right_state)

    def test_torch_function_modes_receive_original_calls_and_can_forward(self):
        left = torch.tensor([1.0])
        right = torch.tensor([2.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        cases = (
            (
                "positional",
                lambda: torch.result_type(left, right),
                (left, right),
                None,
            ),
            (
                "mixed",
                lambda: torch.result_type(left, other=right),
                (left,),
                {"other": right},
            ),
            (
                "keywords",
                lambda: torch.result_type(tensor=left, other=right),
                (),
                {"tensor": left, "other": right},
            ),
            (
                "reversed keywords",
                lambda: torch.result_type(other=right, tensor=left),
                (),
                {"other": right, "tensor": left},
            ),
            (
                "scalar overload",
                lambda: torch.result_type(1, right),
                (1, right),
                None,
            ),
        )
        for case, call, expected_args, expected_kwargs in cases:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            with self.subTest(case=case):
                self.assertEqual(len(mode.calls), 1)
                function, dispatch_types, args, kwargs = mode.calls[0]
                self.assertIs(function, torch.result_type)
                self.assertEqual(dispatch_types, ())
                self.assertEqual(args, expected_args)
                self.assertEqual(kwargs, expected_kwargs)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                result = torch.result_type(other=right, tensor=left)

        self.assertIs(result, torch.float32)
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, torch.result_type)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, ())
            self.assertEqual(tuple(kwargs), ("other", "tensor"))
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_operand_overrides_preserve_forms_order_and_scalar_subclasses(self):
        tensor = torch.tensor([1.0])
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        forms = (
            (
                "first positional",
                lambda value: torch.result_type(value, tensor),
                ("override", "tensor"),
                None,
            ),
            (
                "second positional",
                lambda value: torch.result_type(tensor, value),
                ("tensor", "override"),
                None,
            ),
            (
                "first keyword",
                lambda value: torch.result_type(tensor=value, other=tensor),
                (),
                (("tensor", "override"), ("other", "tensor")),
            ),
            (
                "second keyword",
                lambda value: torch.result_type(tensor=tensor, other=value),
                (),
                (("tensor", "tensor"), ("other", "override")),
            ),
        )
        for case, call, expected_args, expected_kwargs in forms:
            value = Override()
            Override.calls = []
            self.assertIs(call(value), marker)
            self.assertEqual(len(Override.calls), 1)
            function, dispatch_types, args, kwargs = Override.calls[0]
            observed_args = tuple(
                "override" if argument is value else "tensor"
                for argument in args
            )
            observed_kwargs = (
                None
                if kwargs is None
                else tuple(
                    (
                        key,
                        "override" if argument is value else "tensor",
                    )
                    for key, argument in kwargs.items()
                )
            )
            with self.subTest(case=case):
                self.assertIs(function, torch.result_type)
                self.assertEqual(dispatch_types, (Override,))
                self.assertEqual(observed_args, expected_args)
                self.assertEqual(observed_kwargs, expected_kwargs)

        events = []

        class Base:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("base", types))
                return marker

        class Derived(Base):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("derived", types))
                return marker

        self.assertIs(torch.result_type(Base(), Derived()), marker)
        self.assertEqual(events, [("derived", (Derived, Base))])

        class ScalarOverride(int):
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        scalar = ScalarOverride(1)
        self.assertIs(torch.result_type(scalar, tensor), marker)
        self.assertEqual(len(ScalarOverride.calls), 1)
        function, dispatch_types, args, kwargs = ScalarOverride.calls[0]
        self.assertIs(function, torch.result_type)
        self.assertEqual(dispatch_types, (ScalarOverride,))
        self.assertEqual(args, (scalar, tensor))
        self.assertIsNone(kwargs)

    def test_modes_run_after_validation_and_declines_report_all_handlers(self):
        tensor = torch.tensor([1.0])

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return torch.float32

        invalid_calls = (
            lambda: torch.result_type(),
            lambda: torch.result_type([], tensor),
            lambda: torch.result_type(tensor, tensor, extra=True),
            lambda: torch.result_type(1, tensor, extra=True),
        )
        for call in invalid_calls:
            mode = RecordingMode()
            with mode:
                with self.assertRaises(TypeError):
                    call()
            self.assertEqual(mode.calls, [])

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        mode = DecliningMode()
        message = (
            "Multiple dispatch failed for 'torch.result_type'; all "
            "__torch_function__ handlers returned NotImplemented:\n\n"
            f"  - mode object {mode!r}\n\n"
            "For more information, try re-running with "
            "TORCH_LOGS=not_implemented"
        )
        with mode:
            self.assert_error(
                TypeError,
                message,
                lambda: torch.result_type(tensor, tensor),
            )

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            torch.result_type(DecliningOverride(), tensor)
        self.assertEqual(
            str(raised.exception),
            "Multiple dispatch failed for 'torch.result_type'; all "
            "__torch_function__ handlers returned NotImplemented:\n\n"
            f"  - tensor subclass <class '{DecliningOverride.__module__}."
            f"{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with "
            "TORCH_LOGS=not_implemented",
        )

    def test_scalar_overloads_are_explicitly_unsupported_after_binding(self):
        tensor = torch.tensor([1.0])
        message = (
            "result_type(): scalar operands are not supported; "
            "both operands must be Tensor"
        )
        scalar_calls = (
            lambda: torch.result_type(1, tensor),
            lambda: torch.result_type(scalar=1, tensor=tensor),
            lambda: torch.result_type(tensor, 1.0),
            lambda: torch.result_type(tensor=tensor, other=1.0),
            lambda: torch.result_type(True, False),
            lambda: torch.result_type(1 + 2j, tensor),
            lambda: torch.result_type(np.int32(1), tensor),
            lambda: torch.result_type(np.complex64(1), tensor),
            lambda: torch.result_type(scalar1=1, scalar2=2),
        )
        for call in scalar_calls:
            with self.subTest(call=call):
                self.assert_error(NotImplementedError, message, call)

        overloads = (
            "but expected one of:\n"
            " * (Tensor tensor, Tensor other)\n"
            " * (Number scalar, Tensor tensor)\n"
            " * (Tensor tensor, Number other)\n"
            " * (Number scalar1, Number scalar2)\n"
        )
        self.assert_error(
            TypeError,
            "result_type() received an invalid combination of arguments - "
            f"got (), {overloads}",
            lambda: torch.result_type(),
        )
        self.assert_error(
            TypeError,
            "result_type() received an invalid combination of arguments - got "
            f"(Tensor, Tensor, extra=bool), {overloads}",
            lambda: torch.result_type(tensor, tensor, extra=True),
        )
        self.assert_error(
            TypeError,
            "result_type() received an invalid combination of arguments - got "
            f"(int, Tensor, extra=bool), {overloads}",
            lambda: torch.result_type(1, tensor, extra=True),
        )
        with self.assertRaisesRegex(
            TypeError,
            r"^result_type\(\) received an invalid combination of arguments - "
            r"got \(torch\.dtype, Tensor\),",
        ):
            torch.result_type(torch.float32, tensor)

    def test_callable_metadata_exports_and_pickling_match_pytorch(self):
        function = torch.result_type
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "result_type")
        self.assertEqual(
            function.__qualname__, "_VariableFunctionsClass.result_type"
        )
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method result_type of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        reducer, (owner, name) = function.__reduce__()
        self.assertIs(reducer, getattr)
        self.assertEqual(name, "result_type")
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.result_type, function)
        self.assertIsNone(function.__self__)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("result_type"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["result_type"], function)


if __name__ == "__main__":
    unittest.main()

import inspect
import re
import sys
import types
import unittest

import numpy as np

import torch_rs as torch


METHOD_DOC = (
    "\nnarrow(dimension, start, length) -> Tensor\n\n"
    "See :func:`torch.narrow`.\n"
)
SIGNATURES = (
    "\n * (int dim, Tensor start, int length)"
    "\n * (int dim, int start, int length)\n"
)


def offset_noncontiguous_source(*, requires_grad=False):
    values = [float(value) for value in range(48)]
    return torch.tensor(values, requires_grad=requires_grad).reshape(2, 2, 3, 4)[
        1
    ].transpose(0, 1)


class TensorNarrowTests(unittest.TestCase):
    def assert_same_view(self, actual, expected):
        self.assertEqual(actual.tolist(), expected.tolist())
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.data_ptr(), expected.data_ptr())
        self.assertTrue(actual.is_set_to(expected))
        self.assertIs(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)

    def test_call_forms_negative_start_and_noncontiguous_offset_views(self):
        source = offset_noncontiguous_source()
        self.assertEqual(source.shape, (3, 2, 4))
        self.assertEqual(source.stride(), (4, 12, 1))
        self.assertEqual(source.storage_offset(), 24)

        expected = source.narrow(0, 1, 2)
        calls = (
            ("positional", lambda: source.narrow(0, 1, 2)),
            ("mixed", lambda: source.narrow(0, start=1, length=2)),
            ("keywords", lambda: source.narrow(dim=0, start=1, length=2)),
            (
                "reordered keywords",
                lambda: source.narrow(length=2, start=1, dim=0),
            ),
            ("negative dimension", lambda: source.narrow(-3, 1, 2)),
            ("negative start", lambda: source.narrow(0, -2, 2)),
        )
        for case, call in calls:
            with self.subTest(case=case):
                narrowed = call()
                self.assert_same_view(narrowed, expected)
                self.assertEqual(
                    narrowed.tolist(),
                    [
                        [[28.0, 29.0, 30.0, 31.0], [40.0, 41.0, 42.0, 43.0]],
                        [[32.0, 33.0, 34.0, 35.0], [44.0, 45.0, 46.0, 47.0]],
                    ],
                )

        self.assertEqual(
            expected.data_ptr(),
            source.data_ptr() + source.stride()[0] * source.element_size(),
        )

    def test_zero_length_and_empty_views_preserve_strides_and_offsets(self):
        source = offset_noncontiguous_source()
        for start, expected_offset in ((0, 24), (1, 28), (-1, 32), (3, 36)):
            with self.subTest(start=start):
                narrowed = source.narrow(0, start, 0)
                self.assertEqual(narrowed.shape, (0, 2, 4))
                self.assertEqual(narrowed.stride(), (4, 12, 1))
                self.assertEqual(narrowed.storage_offset(), expected_offset)
                self.assertEqual(narrowed.data_ptr(), 0)
                self.assertEqual(narrowed.tolist(), [])
                self.assertTrue(narrowed.is_set_to(source.narrow(0, start, 0)))

        inner_empty = torch.zeros((4, 0, 3))
        inner_empty_view = inner_empty.narrow(0, -1, 1)
        self.assertEqual(inner_empty_view.shape, (1, 0, 3))
        self.assertEqual(inner_empty_view.stride(), (3, 3, 1))
        self.assertEqual(inner_empty_view.storage_offset(), 9)
        self.assertEqual(inner_empty_view.data_ptr(), 0)
        self.assertEqual(inner_empty_view.tolist(), [[]])

        leading_empty = torch.zeros((0, 3))
        leading_empty_view = leading_empty.narrow(-2, 0, 0)
        self.assertEqual(leading_empty_view.shape, (0, 3))
        self.assertEqual(leading_empty_view.stride(), (3, 1))
        self.assertEqual(leading_empty_view.storage_offset(), 0)
        self.assertTrue(leading_empty_view.is_set_to(leading_empty.narrow(0, 0, 0)))

    def test_range_errors_and_deliberate_surface_limits(self):
        tensor = torch.zeros((4, 2, 3))
        scalar = torch.tensor(1.0)
        cases = (
            (
                lambda: scalar.narrow(0, 0, 0),
                RuntimeError,
                "narrow() cannot be applied to a 0-dim tensor.",
            ),
            (
                lambda: scalar.narrow(99, 0, -1),
                RuntimeError,
                "narrow() cannot be applied to a 0-dim tensor.",
            ),
            (
                lambda: tensor.narrow(99, 5, -1),
                RuntimeError,
                "narrow(): length must be non-negative.",
            ),
            (
                lambda: tensor.narrow(3, 0, 1),
                IndexError,
                "Dimension out of range (expected to be in range of [-3, 2], but got 3)",
            ),
            (
                lambda: tensor.narrow(-4, 0, 1),
                IndexError,
                "Dimension out of range (expected to be in range of [-3, 2], but got -4)",
            ),
            (
                lambda: tensor.narrow(1, 0, 1),
                RuntimeError,
                "Tensor.narrow only supports dimension 0",
            ),
            (
                lambda: tensor.narrow(-2, 0, 1),
                RuntimeError,
                "Tensor.narrow only supports dimension 0",
            ),
            (
                lambda: tensor.narrow(0, -5, 0),
                IndexError,
                "start out of range (expected to be in range of [-4, 4], but got -5)",
            ),
            (
                lambda: tensor.narrow(0, 5, 0),
                IndexError,
                "start out of range (expected to be in range of [-4, 4], but got 5)",
            ),
            (
                lambda: tensor.narrow(0, -1, 2),
                RuntimeError,
                "start (3) + length (2) exceeds dimension size (4).",
            ),
            (
                lambda: torch.zeros((0, 3)).narrow(0, 0, 1),
                RuntimeError,
                "start (0) + length (1) exceeds dimension size (0).",
            ),
            (
                lambda: tensor.narrow(0, torch.tensor(1), 1),
                RuntimeError,
                "Tensor.narrow only supports integer start values",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        self.assertFalse(hasattr(torch, "narrow"))
        self.assertNotIn("narrow", torch.__all__)

    def test_integer_index_binding_and_conversion_order(self):
        tensor = torch.zeros((5, 2))

        class IntegerSubclass(int):
            pass

        self.assertEqual(
            tensor.narrow(np.int8(-2), IntegerSubclass(-4), np.uint32(2)).shape,
            (2, 2),
        )

        calls = []

        class StatefulIndex:
            def __init__(self, label, values):
                self.label = label
                self.values = iter(values)

            def __index__(self):
                calls.append(self.label)
                return next(self.values)

        start = StatefulIndex("start", (1, 2, 3))
        length = StatefulIndex("length", (2, 1, 1))
        narrowed = tensor.narrow(0, start, length)
        self.assertEqual(
            calls,
            ["start", "length", "length", "length", "start", "start"],
        )
        self.assertEqual(narrowed.shape, (1, 2))
        self.assertEqual(narrowed.storage_offset(), 6)

        for call in (
            lambda: tensor.narrow(2**100, 0, 0),
            lambda: tensor.narrow(0, 2**100, 0),
            lambda: tensor.narrow(0, 0, 2**100),
        ):
            with self.assertRaises(ValueError) as raised:
                call()
            self.assertEqual(str(raised.exception), "Overflow when unpacking long long")

    def test_binding_errors_match_the_generated_overloads(self):
        tensor = torch.zeros((4, 3))
        no_mismatch = {
            (): "",
            (0,): "int",
            (0, 0): "int, int",
            (0, 0, 0, 0): "int, int, int, int",
        }
        for arguments, summary in no_mismatch.items():
            with self.subTest(arguments=arguments):
                with self.assertRaises(TypeError) as raised:
                    tensor.narrow(*arguments)
                self.assertEqual(
                    str(raised.exception),
                    "narrow() received an invalid combination of arguments - got "
                    f"({summary}), but expected one of:{SIGNATURES}",
                )

        cases = (
            (
                lambda: tensor.narrow(None, 0, 0),
                "NoneType, int, int",
                "(!NoneType!, !int!, int)",
                "(!NoneType!, int, int)",
            ),
            (
                lambda: tensor.narrow(0, None, 0),
                "int, NoneType, int",
                "(int, !NoneType!, int)",
                "(int, !NoneType!, int)",
            ),
            (
                lambda: tensor.narrow(0, 0, None),
                "int, int, NoneType",
                "(int, !int!, !NoneType!)",
                "(int, int, !NoneType!)",
            ),
            (
                lambda: tensor.narrow(True, 0, 0),
                "bool, int, int",
                "(!bool!, !int!, int)",
                "(!bool!, int, int)",
            ),
        )
        for call, summary, tensor_detail, integer_detail in cases:
            with self.subTest(summary=summary):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(
                    str(raised.exception),
                    "narrow() received an invalid combination of arguments - got "
                    f"({summary}), but expected one of:\n"
                    " * (int dim, Tensor start, int length)\n"
                    "      didn't match because some of the arguments have invalid "
                    f"types: {tensor_detail}\n"
                    " * (int dim, int start, int length)\n"
                    "      didn't match because some of the arguments have invalid "
                    f"types: {integer_detail}\n",
                )

    def test_tensorbase_descriptor_metadata_and_unbound_behavior(self):
        tensor = torch.zeros((4, 3))
        descriptor = inspect.getattr_static(torch.Tensor, "narrow")
        bound = tensor.narrow

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "narrow")
        self.assertEqual(descriptor.__qualname__, "TensorBase.narrow")
        self.assertEqual(bound.__qualname__, "Tensor.narrow")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        self.assertEqual(
            repr(descriptor),
            "<method 'narrow' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.narrow, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertTrue(
            descriptor(tensor, 0, 1, 2).is_set_to(tensor.narrow(0, 1, 2))
        )
        for callable_object in (descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        cases = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.narrow() needs an argument",
            ),
            (
                lambda: descriptor(1, 0, 0, 1),
                "descriptor 'narrow' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor, dim=0, start=0, length=1),
                "unbound method TensorBase.narrow() needs an argument",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_autograd_no_grad_and_empty_ranges(self):
        leaf = torch.tensor([float(value) for value in range(48)], requires_grad=True)
        source = (leaf * 2.0).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        narrowed = source.narrow(-3, -2, 2)
        self.assertTrue(narrowed.requires_grad)
        self.assertFalse(narrowed.is_leaf)
        self.assertEqual(narrowed.output_nr, 0)

        diagnostic = torch.tensor([2.0], requires_grad=True).narrow(0, 0, 1)
        with self.assertRaisesRegex(
            ValueError,
            r"^dropout probability has to be between 0 and 1, but got "
            r"tensor\(\[2\.\], grad_fn=<SliceBackward0>\)$",
        ):
            torch.nn.functional.dropout(None, p=diagnostic, training=False)

        weights = torch.tensor(
            [float(value) for value in range(1, 17)]
        ).reshape(2, 2, 4)
        (narrowed * weights).sum().backward()
        expected = [0.0] * 48
        for index, weight in zip(
            (
                28,
                29,
                30,
                31,
                40,
                41,
                42,
                43,
                32,
                33,
                34,
                35,
                44,
                45,
                46,
                47,
            ),
            range(1, 17),
        ):
            expected[index] = 2.0 * weight
        self.assertEqual(leaf.grad.tolist(), expected)

        no_grad_source = torch.zeros((4, 3), requires_grad=True)
        with torch.no_grad():
            untracked = no_grad_source.narrow(dim=0, start=1, length=2)
        self.assertTrue(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertEqual(untracked.output_nr, 0)
        self.assertTrue(untracked.is_set_to(no_grad_source.narrow(0, 1, 2)))

        empty_leaf = torch.zeros((4, 2), requires_grad=True)
        empty_view = empty_leaf.narrow(0, 2, 0)
        empty_view.sum().backward()
        self.assertEqual(empty_leaf.grad.tolist(), [[0.0, 0.0]] * 4)

    def test_torch_function_modes_receive_original_calls_and_forward(self):
        tensor = torch.zeros((4, 3))
        tensor_start = torch.tensor(1)
        descriptor = inspect.getattr_static(torch.Tensor, "narrow")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        positional = RecordingMode(marker)
        with positional:
            positional_result = tensor.narrow(0, 1, 2)
        self.assertIs(positional_result, marker)
        function, dispatch_types, args, kwargs = positional.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (tensor, 0, 1, 2))
        self.assertIsNone(kwargs)

        keyword = RecordingMode(marker)
        with keyword:
            keyword_result = tensor.narrow(length=2, start=1, dim=0)
        self.assertIs(keyword_result, marker)
        function, dispatch_types, args, kwargs = keyword.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (tensor,))
        self.assertEqual(kwargs, {"length": 2, "start": 1, "dim": 0})

        tensor_mode = RecordingMode(marker)
        with tensor_mode:
            tensor_result = tensor.narrow(0, tensor_start, 1)
        self.assertIs(tensor_result, marker)
        self.assertEqual(tensor_mode.calls[0][2], (tensor, 0, tensor_start, 1))

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.narrow(dim=0, start=-3, length=2)
        self.assertEqual(order, ["upper", "lower"])
        self.assertTrue(forwarded.is_set_to(tensor.narrow(0, 1, 2)))

        index_calls = []

        class CustomIndex:
            def __init__(self, label):
                self.label = label

            def __index__(self):
                index_calls.append(self.label)
                return 1

        deferred = RecordingMode(marker)
        with deferred:
            deferred_result = tensor.narrow(
                2**100,
                CustomIndex("start"),
                CustomIndex("length"),
            )
        self.assertIs(deferred_result, marker)
        self.assertEqual(index_calls, ["start", "length"])

        invalid = RecordingMode(marker)
        with self.assertRaises(TypeError):
            with invalid:
                tensor.narrow(True, 0, 1)
        self.assertEqual(invalid.calls, [])

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        with self.assertRaises(TypeError) as raised:
            with lower:
                with declining:
                    tensor.narrow(0, 1, 2)
        self.assertRegex(
            str(raised.exception),
            re.compile(
                r"^Multiple dispatch failed for 'torch\.Tensor\.narrow'; all "
                r"__torch_function__ handlers returned NotImplemented:\n\n"
                r"  - mode object <.*RecordingMode object at 0x[0-9a-f]+>\n\n"
                r"For more information, try re-running with "
                r"TORCH_LOGS=not_implemented$"
            ),
        )
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(lower.calls, [])

    @unittest.skipUnless(
        sys.maxsize == (1 << 63) - 1,
        "checked offset boundary requires a 64-bit platform",
    )
    def test_extreme_empty_offsets_are_checked(self):
        maximum = sys.maxsize
        with self.assertRaises(RuntimeError) as raised:
            torch.zeros((maximum, 0)).narrow(0, maximum, maximum)
        self.assertEqual(
            str(raised.exception),
            f"start ({maximum}) + length ({maximum}) exceeds dimension size ({maximum}).",
        )

        source = torch.zeros((2, 0, maximum))
        valid = source.narrow(0, 1, 0)
        self.assertEqual(valid.shape, (0, 0, maximum))
        self.assertEqual(valid.stride(), (maximum, maximum, 1))
        self.assertEqual(valid.storage_offset(), maximum)

        with self.assertRaises(RuntimeError) as raised:
            source.narrow(0, 2, 0)
        self.assertEqual(str(raised.exception), "Tensor: invalid storage offset -2")

        offset = torch.zeros((maximum, 0))[maximum - 1].reshape((maximum, 0))
        with self.assertRaises(RuntimeError) as raised:
            offset.narrow(0, maximum, 0)
        self.assertEqual(str(raised.exception), "Tensor: invalid storage offset -3")


if __name__ == "__main__":
    unittest.main()

import inspect
import re
import types
import unittest

import numpy as np

import torch_rs as torch


METHOD_DOC = "\nselect(dim, index) -> Tensor\n\nSee :func:`torch.select`\n"


def offset_noncontiguous_source(*, requires_grad=False):
    values = [float(value) for value in range(48)]
    return torch.tensor(values, requires_grad=requires_grad).reshape(2, 2, 3, 4)[
        1
    ].transpose(0, 1)


class TensorSelectTests(unittest.TestCase):
    def assert_same_view(self, actual, expected):
        self.assertEqual(actual.tolist(), expected.tolist())
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.data_ptr(), expected.data_ptr())
        self.assertTrue(actual.is_set_to(expected))
        self.assertIs(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)

    def test_call_forms_reuse_native_leading_integer_index_views(self):
        source = offset_noncontiguous_source()
        self.assertEqual(source.shape, (3, 2, 4))
        self.assertEqual(source.stride(), (4, 12, 1))
        self.assertEqual(source.storage_offset(), 24)

        expected = source[1]
        calls = (
            ("positional", lambda: source.select(0, 1)),
            ("mixed", lambda: source.select(0, index=1)),
            ("keywords", lambda: source.select(dim=0, index=1)),
            ("reordered keywords", lambda: source.select(index=1, dim=0)),
            ("normalized negative dimension", lambda: source.select(-3, 1)),
            ("negative index", lambda: source.select(dim=-3, index=-2)),
        )
        for case, call in calls:
            with self.subTest(case=case):
                selected = call()
                self.assert_same_view(selected, expected)
                self.assertEqual(
                    selected.tolist(),
                    [[28.0, 29.0, 30.0, 31.0], [40.0, 41.0, 42.0, 43.0]],
                )

        vector = torch.tensor([1.0, 2.0, 3.0])
        scalar = vector.select(-1, -1)
        self.assertEqual(scalar.shape, ())
        self.assertEqual(scalar.stride(), ())
        self.assertEqual(scalar.storage_offset(), 2)
        self.assertEqual(scalar.item(), 3.0)
        self.assertTrue(scalar.is_set_to(vector[-1]))

    def test_empty_views_bounds_and_deliberate_dimension_limits(self):
        empty = torch.zeros((2, 0, 3), requires_grad=True)
        selected = empty.select(0, 1)
        self.assertEqual(selected.shape, (0, 3))
        self.assertEqual(selected.stride(), (3, 1))
        self.assertEqual(selected.storage_offset(), 3)
        self.assertEqual(selected.data_ptr(), 0)
        self.assertEqual(selected.tolist(), [])
        self.assertTrue(selected.is_set_to(empty[1]))

        cases = (
            (
                lambda: torch.zeros((0, 2)).select(0, 0),
                IndexError,
                "select(): index 0 out of range for tensor of size [0, 2] at dimension 0",
            ),
            (
                lambda: torch.zeros((0, 2)).select(-2, -1),
                IndexError,
                "select(): index -1 out of range for tensor of size [0, 2] at dimension 0",
            ),
            (
                lambda: torch.zeros((2, 3, 4)).select(0, 2),
                IndexError,
                "select(): index 2 out of range for tensor of size [2, 3, 4] at dimension 0",
            ),
            (
                lambda: torch.zeros((2, 3, 4)).select(-3, -3),
                IndexError,
                "select(): index -3 out of range for tensor of size [2, 3, 4] at dimension 0",
            ),
            (
                lambda: torch.tensor(1.0).select(0, 0),
                IndexError,
                "select() cannot be applied to a 0-dim tensor.",
            ),
            (
                lambda: torch.tensor(1.0).select(-2, 99),
                IndexError,
                "select() cannot be applied to a 0-dim tensor.",
            ),
            (
                lambda: torch.zeros((2, 3, 4)).select(3, 0),
                IndexError,
                "Dimension out of range (expected to be in range of [-3, 2], but got 3)",
            ),
            (
                lambda: torch.zeros((2, 3, 4)).select(-4, 0),
                IndexError,
                "Dimension out of range (expected to be in range of [-3, 2], but got -4)",
            ),
            (
                lambda: torch.zeros((2, 3, 4)).select(1, 0),
                RuntimeError,
                "Tensor.select only supports dimension 0",
            ),
            (
                lambda: torch.zeros((2, 3, 4)).select(-2, 0),
                RuntimeError,
                "Tensor.select only supports dimension 0",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        self.assertFalse(hasattr(torch, "select"))
        self.assertNotIn("select", torch.__all__)

    def test_autograd_no_grad_and_downstream_operations(self):
        leaf = torch.tensor([float(value) for value in range(48)], requires_grad=True)
        source = (leaf * 2.0).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        selected = source.select(-3, 1)

        self.assertTrue(selected.requires_grad)
        self.assertFalse(selected.is_leaf)
        self.assertEqual(selected.output_nr, 0)
        self.assert_same_view(selected, source[1])

        (selected.transpose(0, 1) * 3.0).sum().backward()
        expected_gradient = [0.0] * 48
        for index in (*range(28, 32), *range(40, 44)):
            expected_gradient[index] = 6.0
        self.assertEqual(leaf.grad.tolist(), expected_gradient)

        no_grad_source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        with torch.no_grad():
            untracked = no_grad_source.select(dim=0, index=1)
        self.assertTrue(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertEqual(untracked.output_nr, 0)
        self.assertTrue(untracked.is_set_to(no_grad_source[1]))

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty.select(0, 1).sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))
        self.assertEqual(empty.grad.tolist(), [[], []])

    def test_integer_types_binding_and_conversion_order(self):
        tensor = torch.zeros((2, 3))

        class IntegerSubclass(int):
            pass

        self.assertEqual(tensor.select(IntegerSubclass(0), np.int64(1)).shape, (3,))
        self.assertEqual(tensor.select(np.int8(-2), np.uint32(0)).shape, (3,))

        calls = []

        class StatefulIndex:
            def __index__(self):
                calls.append("index")
                return (0, 1, 0)[len(calls) - 1]

        selected = tensor.select(0, StatefulIndex())
        self.assertEqual(calls, ["index", "index", "index"])
        self.assertEqual(selected.storage_offset(), 0)

        cases = (
            (
                lambda: tensor.select(),
                'select() missing 2 required positional argument: "dim", "index"',
            ),
            (
                lambda: tensor.select(0),
                'select() missing 1 required positional arguments: "index"',
            ),
            (
                lambda: tensor.select(index=1),
                'select() missing 2 required positional argument: "dim", "index"',
            ),
            (
                lambda: tensor.select(0, 1, 2),
                "select() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: tensor.select(0, 1, dim=0),
                "select() got multiple values for argument 'dim'",
            ),
            (
                lambda: tensor.select(0, 1, index=0),
                "select() got multiple values for argument 'index'",
            ),
            (
                lambda: tensor.select(0, 1, extra=0),
                "select() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: tensor.select(None, 0),
                "select(): argument 'dim' (position 1) must be int, not NoneType",
            ),
            (
                lambda: tensor.select(dim="0", index=0),
                "select(): argument 'dim' must be int, not str",
            ),
            (
                lambda: tensor.select(0, True),
                "select(): argument 'index' (position 2) must be int, not bool",
            ),
            (
                lambda: tensor.select(dim=0, index=1.0),
                "select(): argument 'index' must be int, not float",
            ),
            (
                lambda: tensor.select(2**100, "bad"),
                "select(): argument 'index' (position 2) must be int, not str",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        for call in (
            lambda: tensor.select(2**100, 0),
            lambda: tensor.select(0, 2**100),
            lambda: tensor.select(dim=0, index=-(2**100)),
        ):
            with self.assertRaises(ValueError) as raised:
                call()
            self.assertEqual(str(raised.exception), "Overflow when unpacking long long")

    def test_tensorbase_descriptor_metadata_and_unbound_behavior(self):
        tensor = torch.zeros((2, 3))
        descriptor = inspect.getattr_static(torch.Tensor, "select")
        bound = tensor.select

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "select")
        self.assertEqual(descriptor.__qualname__, "TensorBase.select")
        self.assertEqual(bound.__qualname__, "Tensor.select")
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
            "<method 'select' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.select, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertTrue(descriptor(tensor, 0, 1).is_set_to(tensor[1]))
        for callable_object in (descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        cases = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.select() needs an argument",
            ),
            (
                lambda: descriptor(1, 0, 0),
                "descriptor 'select' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor, dim=0, index=1),
                "unbound method TensorBase.select() needs an argument",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_receive_original_calls_and_forward(self):
        tensor = torch.zeros((2, 3, 4))
        descriptor = inspect.getattr_static(torch.Tensor, "select")
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
            positional_result = tensor.select(0, 1)
        self.assertIs(positional_result, marker)
        function, dispatch_types, args, kwargs = positional.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (tensor, 0, 1))
        self.assertIsNone(kwargs)

        keyword = RecordingMode(marker)
        with keyword:
            keyword_result = tensor.select(index=1, dim=0)
        self.assertIs(keyword_result, marker)
        function, dispatch_types, args, kwargs = keyword.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (tensor,))
        self.assertEqual(kwargs, {"index": 1, "dim": 0})

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.select(dim=-3, index=1)
        self.assertEqual(order, ["upper", "lower"])
        self.assertTrue(forwarded.is_set_to(tensor[1]))

        index_calls = []

        class CustomIndex:
            def __index__(self):
                index_calls.append("index")
                return 1

        deferred = RecordingMode(marker)
        with deferred:
            deferred_result = tensor.select(2**100, CustomIndex())
        self.assertIs(deferred_result, marker)
        self.assertEqual(index_calls, ["index"])
        self.assertEqual(len(deferred.calls), 1)

        invalid = RecordingMode(marker)
        with self.assertRaises(TypeError):
            with invalid:
                tensor.select(True, 0)
        self.assertEqual(invalid.calls, [])

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        with self.assertRaises(TypeError) as raised:
            with lower:
                with declining:
                    tensor.select(0, 1)
        self.assertRegex(
            str(raised.exception),
            re.compile(
                r"^Multiple dispatch failed for 'torch\.Tensor\.select'; all "
                r"__torch_function__ handlers returned NotImplemented:\n\n"
                r"  - mode object <.*RecordingMode object at 0x[0-9a-f]+>\n\n"
                r"For more information, try re-running with "
                r"TORCH_LOGS=not_implemented$"
            ),
        )
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(lower.calls, [])


if __name__ == "__main__":
    unittest.main()

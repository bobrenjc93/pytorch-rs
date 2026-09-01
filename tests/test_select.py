import inspect
import pickle
import re
import types
import unittest

import numpy as np

import torch_rs as torch


METHOD_DOC = "\nselect(dim, index) -> Tensor\n\nSee :func:`torch.select`\n"
FUNCTION_DOC = (
    "\nselect(input, dim, index) -> Tensor\n\n"
    "Slices the :attr:`input` tensor along the selected dimension at the given index.\n"
    "This function returns a view of the original tensor with the given dimension removed.\n\n"
    ".. note:: If :attr:`input` is a sparse tensor and returning a view of\n"
    "          the tensor is not possible, a RuntimeError exception is\n"
    "          raised. In this is the case, consider using\n"
    "          :func:`torch.select_copy` function.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n"
    "    dim (int): the dimension to slice\n"
    "    index (int): the index to select with\n\n"
    ".. note::\n\n"
    "    :meth:`select` is equivalent to slicing. For example,\n"
    "    ``tensor.select(0, index)`` is equivalent to ``tensor[index]`` and\n"
    "    ``tensor.select(2, index)`` is equivalent to ``tensor[:,:,index]``.\n"
)


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

    def test_call_forms_reuse_native_select_views(self):
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

    def test_arbitrary_dimension_views_match_indexing(self):
        contiguous = torch.tensor([float(value) for value in range(24)]).reshape(
            2, 3, 4
        )
        offset = torch.tensor([float(value) for value in range(120)]).reshape(
            2, 3, 4, 5
        )[1]
        noncontiguous = offset_noncontiguous_source()
        empty = torch.zeros((2, 3, 0, 4))

        cases = (
            ("contiguous middle", contiguous, 1, 2, contiguous.transpose(0, 1)[2]),
            (
                "contiguous trailing",
                contiguous,
                2,
                1,
                contiguous.transpose(0, 2)[1].transpose(0, 1),
            ),
            ("offset middle", offset, 1, 2, offset.transpose(0, 1)[2]),
            (
                "noncontiguous middle",
                noncontiguous,
                1,
                1,
                noncontiguous.transpose(0, 1)[1],
            ),
            (
                "empty middle",
                empty,
                1,
                2,
                empty.transpose(0, 1)[2],
            ),
            ("negative dim and index", contiguous, -2, -1, contiguous.transpose(0, 1)[2]),
        )
        for case, source, dimension, index, expected in cases:
            with self.subTest(case=case, surface="method"):
                self.assert_same_view(source.select(dimension, index), expected)
            with self.subTest(case=case, surface="top-level"):
                self.assert_same_view(torch.select(source, dimension, index), expected)

    def test_empty_views_bounds_and_index_errors(self):
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
                lambda: torch.zeros((2, 3, 4)).select(1, 3),
                IndexError,
                "select(): index 3 out of range for tensor of size [2, 3, 4] at dimension 1",
            ),
            (
                lambda: torch.zeros((2, 3, 4)).select(-2, -4),
                IndexError,
                "select(): index -4 out of range for tensor of size [2, 3, 4] at dimension 1",
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
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        self.assertTrue(hasattr(torch, "select"))
        self.assertIn("select", torch.__all__)

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

        full_sum_leaf = torch.tensor(
            [float(value) for value in range(48)], requires_grad=True
        )
        full_sum_source = (full_sum_leaf * 2.0).reshape(2, 2, 3, 4)[
            1
        ].transpose(0, 1)
        full_sum_selected = full_sum_source.select(1, -1)
        self.assert_same_view(full_sum_selected, full_sum_source.transpose(0, 1)[1])
        full_sum_selected.sum().backward()
        expected_full_sum_gradient = [0.0] * 48
        for index in range(36, 48):
            expected_full_sum_gradient[index] = 2.0
        self.assertEqual(full_sum_leaf.grad.tolist(), expected_full_sum_gradient)

        no_grad_source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        with torch.no_grad():
            untracked = no_grad_source.select(dim=1, index=1)
        self.assertTrue(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertEqual(untracked.output_nr, 0)
        self.assertTrue(untracked.is_set_to(no_grad_source.transpose(0, 1)[1]))

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

    def test_top_level_forms_reuse_native_select_views(self):
        source = offset_noncontiguous_source()
        expected = source[1]
        calls = (
            ("positional", lambda: torch.select(source, 0, 1)),
            ("mixed", lambda: torch.select(source, 0, index=1)),
            ("keywords", lambda: torch.select(source, dim=0, index=1)),
            (
                "all keywords",
                lambda: torch.select(input=source, dim=0, index=1),
            ),
            (
                "reordered keywords",
                lambda: torch.select(index=1, input=source, dim=0),
            ),
            ("input alias x", lambda: torch.select(x=source, dim=0, index=1)),
            ("input alias a", lambda: torch.select(a=source, dim=0, index=1)),
            ("input alias x1", lambda: torch.select(x1=source, dim=0, index=1)),
            ("normalized negative dimension", lambda: torch.select(source, -3, 1)),
            ("negative index", lambda: torch.select(source, -3, -2)),
        )
        for case, call in calls:
            with self.subTest(case=case):
                selected = call()
                self.assert_same_view(selected, expected)

        vector = torch.tensor([1.0, 2.0, 3.0])
        scalar = torch.select(vector, -1, -1)
        self.assertEqual(scalar.shape, ())
        self.assertEqual(scalar.stride(), ())
        self.assertEqual(scalar.storage_offset(), 2)
        self.assertEqual(scalar.item(), 3.0)
        self.assertTrue(scalar.is_set_to(vector[-1]))

    def test_top_level_autograd_no_grad_output_number_and_empty_view(self):
        leaf = torch.tensor([float(value) for value in range(48)], requires_grad=True)
        source = (leaf * 2.0).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        selected = torch.select(source, -3, 1)

        self.assertTrue(selected.requires_grad)
        self.assertFalse(selected.is_leaf)
        self.assertEqual(selected.output_nr, 0)
        self.assert_same_view(selected, source[1])
        (selected.transpose(0, 1) * 3.0).sum().backward()
        expected_gradient = [0.0] * 48
        for index in (*range(28, 32), *range(40, 44)):
            expected_gradient[index] = 6.0
        self.assertEqual(leaf.grad.tolist(), expected_gradient)

        full_sum_leaf = torch.tensor(
            [float(value) for value in range(48)], requires_grad=True
        )
        full_sum_source = (full_sum_leaf * 2.0).reshape(2, 2, 3, 4)[
            1
        ].transpose(0, 1)
        full_sum_selected = torch.select(full_sum_source, 1, -1)
        self.assert_same_view(full_sum_selected, full_sum_source.transpose(0, 1)[1])
        full_sum_selected.sum().backward()
        expected_full_sum_gradient = [0.0] * 48
        for index in range(36, 48):
            expected_full_sum_gradient[index] = 2.0
        self.assertEqual(full_sum_leaf.grad.tolist(), expected_full_sum_gradient)

        no_grad_source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        with torch.no_grad():
            untracked = torch.select(input=no_grad_source, dim=1, index=1)
        self.assertTrue(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertEqual(untracked.output_nr, 0)
        self.assertTrue(untracked.is_set_to(no_grad_source.transpose(0, 1)[1]))

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        selected_empty = torch.select(empty, 0, 1)
        self.assertEqual(selected_empty.shape, (0, 3))
        self.assertEqual(selected_empty.stride(), (3, 1))
        self.assertEqual(selected_empty.storage_offset(), 3)
        self.assertEqual(selected_empty.data_ptr(), 0)
        self.assertTrue(selected_empty.is_set_to(empty[1]))
        selected_empty.sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))
        self.assertEqual(empty.grad.tolist(), [[], []])

    def test_top_level_binding_errors_and_conversion_order(self):
        tensor = torch.zeros((2, 3, 4))
        scalar = torch.tensor(1.0)
        cases = (
            (
                lambda: torch.select(),
                TypeError,
                'select() missing 3 required positional argument: "input", "dim", "index"',
            ),
            (
                lambda: torch.select(tensor),
                TypeError,
                'select() missing 2 required positional argument: "dim", "index"',
            ),
            (
                lambda: torch.select(tensor, 0),
                TypeError,
                'select() missing 1 required positional arguments: "index"',
            ),
            (
                lambda: torch.select(dim=0, index=1),
                TypeError,
                'select() missing 3 required positional argument: "input", "dim", "index"',
            ),
            (
                lambda: torch.select(tensor, 0, 1, 2),
                TypeError,
                "select() takes 3 positional arguments but 4 were given",
            ),
            (
                lambda: torch.select(tensor, 0, 1, input=tensor),
                TypeError,
                "select() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.select(tensor, 0, 1, dim=0),
                TypeError,
                "select() got multiple values for argument 'dim'",
            ),
            (
                lambda: torch.select(tensor, 0, 1, index=0),
                TypeError,
                "select() got multiple values for argument 'index'",
            ),
            (
                lambda: torch.select(tensor, 0, 1, extra=0),
                TypeError,
                "select() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.select(x=tensor, dim=0, index=1, extra=0),
                TypeError,
                "select() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.select([], 0, 1),
                TypeError,
                "select(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.select(input=[], dim=0, index=1),
                TypeError,
                "select(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.select(tensor, None, 0),
                TypeError,
                "select(): argument 'dim' (position 2) must be int, not NoneType",
            ),
            (
                lambda: torch.select(tensor, dim="0", index=0),
                TypeError,
                "select(): argument 'dim' must be int, not str",
            ),
            (
                lambda: torch.select(tensor, 0, True),
                TypeError,
                "select(): argument 'index' (position 3) must be int, not bool",
            ),
            (
                lambda: torch.select(tensor, dim=0, index=1.0),
                TypeError,
                "select(): argument 'index' must be int, not float",
            ),
            (
                lambda: torch.select(tensor, 2**100, "bad"),
                TypeError,
                "select(): argument 'index' (position 3) must be int, not str",
            ),
            (
                lambda: torch.select(tensor, 0, 2),
                IndexError,
                "select(): index 2 out of range for tensor of size [2, 3, 4] at dimension 0",
            ),
            (
                lambda: torch.select(tensor, -3, -3),
                IndexError,
                "select(): index -3 out of range for tensor of size [2, 3, 4] at dimension 0",
            ),
            (
                lambda: torch.select(tensor, 1, 3),
                IndexError,
                "select(): index 3 out of range for tensor of size [2, 3, 4] at dimension 1",
            ),
            (
                lambda: torch.select(tensor, -2, -4),
                IndexError,
                "select(): index -4 out of range for tensor of size [2, 3, 4] at dimension 1",
            ),
            (
                lambda: torch.select(tensor, 3, 0),
                IndexError,
                "Dimension out of range (expected to be in range of [-3, 2], but got 3)",
            ),
            (
                lambda: torch.select(scalar, 0, 0),
                IndexError,
                "select() cannot be applied to a 0-dim tensor.",
            ),
            (
                lambda: torch.select(torch.zeros((0, 2)), 0, 0),
                IndexError,
                "select(): index 0 out of range for tensor of size [0, 2] at dimension 0",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message), self.assertRaises(error_type) as raised:
                call()
            self.assertEqual(str(raised.exception), message)

        for call in (
            lambda: torch.select(tensor, 2**100, 0),
            lambda: torch.select(tensor, 0, 2**100),
        ):
            with self.assertRaisesRegex(ValueError, "^Overflow when unpacking long long$"):
                call()

        calls = []

        class StatefulIndex:
            def __index__(self):
                calls.append("index")
                return (0, 1, 0)[len(calls) - 1]

        selected = torch.select(tensor, np.int64(0), StatefulIndex())
        self.assertEqual(calls, ["index", "index", "index"])
        self.assertEqual(selected.storage_offset(), 0)

    def test_top_level_torch_function_modes_receive_original_calls_and_forward(self):
        tensor = torch.zeros((2, 3, 4))
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        cases = (
            (lambda: torch.select(tensor, 0, 1), (tensor, 0, 1), None),
            (lambda: torch.select(tensor, 0, index=1), (tensor, 0), {"index": 1}),
            (
                lambda: torch.select(input=tensor, dim=0, index=1),
                (),
                {"input": tensor, "dim": 0, "index": 1},
            ),
            (
                lambda: torch.select(x=tensor, dim=0, index=1),
                (),
                {"x": tensor, "dim": 0, "index": 1},
            ),
            (lambda: torch.select(tensor, 1, 0), (tensor, 1, 0), None),
            (lambda: torch.select(tensor, 2**100, 0), (tensor, 2**100, 0), None),
        )
        for call, expected_args, expected_kwargs in cases:
            mode = RecordingMode(marker)
            with mode:
                result = call()
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, torch.select)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, expected_args)
            self.assertEqual(kwargs, expected_kwargs)

        index_calls = []

        class CustomIndex:
            def __index__(self):
                index_calls.append("index")
                return 1

        deferred = RecordingMode(marker)
        with deferred:
            self.assertIs(torch.select(tensor, 2**100, CustomIndex()), marker)
        self.assertEqual(index_calls, ["index"])

        invalid = RecordingMode(marker)
        with invalid, self.assertRaises(TypeError):
            torch.select(tensor, "0", 1)
        self.assertEqual(invalid.calls, [])

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.select(input=tensor, dim=0, index=1)
        self.assertTrue(forwarded.is_set_to(tensor[1]))
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        self.assertTrue(all(entry[1] is torch.select for entry in order))

        declining = RecordingMode(NotImplemented)
        with declining, self.assertRaisesRegex(
            TypeError,
            "^Multiple dispatch failed for 'torch\\.select'; all "
            "__torch_function__ handlers returned NotImplemented:",
        ):
            torch.select(tensor, 0, 1)
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

    def test_top_level_tensor_like_overrides_use_public_function(self):
        marker = object()
        calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        cases = (
            (lambda: torch.select(value, 0, 1), (value, 0, 1), None),
            (
                lambda: torch.select(input=value, dim=0, index=1),
                (),
                {"input": value, "dim": 0, "index": 1},
            ),
            (
                lambda: torch.select(x=value, dim=0, index=1),
                (),
                {"x": value, "dim": 0, "index": 1},
            ),
            (lambda: torch.select(value, 1, 0), (value, 1, 0), None),
            (lambda: torch.select(value, 2**100, 2**100), (value, 2**100, 2**100), None),
        )
        for call, expected_args, expected_kwargs in cases:
            self.assertIs(call(), marker)
            function, dispatch_types, args, kwargs = calls[-1]
            self.assertIs(function, torch.select)
            self.assertEqual(dispatch_types, (Override,))
            self.assertEqual(args, expected_args)
            self.assertEqual(kwargs, expected_kwargs)

        call_count = len(calls)
        with self.assertRaises(TypeError):
            torch.select(value, 0, "1")
        self.assertEqual(len(calls), call_count)

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError,
            "^Multiple dispatch failed for 'torch\\.select'; all "
            "__torch_function__ handlers returned NotImplemented:",
        ):
            torch.select(DecliningOverride(), 0, 1)

    def test_top_level_callable_metadata_documentation_and_exports(self):
        function = torch.select
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "select")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.select")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method select of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.select, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("select"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["select"], function)


if __name__ == "__main__":
    unittest.main()

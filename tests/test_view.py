import inspect
import operator
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


INCOMPATIBLE_LAYOUT = (
    "view size is not compatible with input tensor's size and stride "
    "(at least one dimension spans across two contiguous subspaces). "
    "Use .reshape(...) instead."
)


class IntSubclass(int):
    pass


class IndexDimension:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


class StatefulIndexDimension:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def __index__(self):
        value = self.values[self.calls]
        self.calls += 1
        return value


class TensorViewTests(unittest.TestCase):
    def shape_forms(self, shape):
        forms = (
            ("tuple", tuple(shape), False),
            ("list", list(shape), False),
            ("Size", torch.Size(shape), False),
            ("keyword", tuple(shape), True),
        )
        if len(shape) > 1:
            forms += (("variadic", tuple(shape), False),)
        return forms

    def call_view(self, source, form, argument, keyword):
        if form == "variadic":
            return source.view(*argument)
        if keyword:
            return source.view(size=argument)
        return source.view(argument)

    def assert_view_result(
        self,
        result,
        source,
        *,
        expected_shape,
        expected_stride,
        expected_offset,
    ):
        direct = source.reshape(expected_shape)
        self.assertIsNot(result, source)
        self.assertEqual(result.shape, expected_shape)
        self.assertEqual(result.stride(), expected_stride)
        self.assertEqual(result.storage_offset(), expected_offset)
        self.assertEqual(result.is_contiguous(), direct.is_contiguous())
        self.assertEqual(result.requires_grad, direct.requires_grad)
        self.assertEqual(result.is_leaf, direct.is_leaf)
        self.assertIs(result.dtype, torch.float32)
        self.assertEqual(result.device, torch.device("cpu"))
        np.testing.assert_array_equal(np.asarray(result), np.asarray(direct))
        self.assertEqual(result.data_ptr(), source.data_ptr())
        self.assertTrue(result.is_set_to(direct))

    def make_layout_cases(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist())
        noncontiguous = base.transpose(0, 1)
        return (
            ("scalar", torch.tensor(-0.0), (), (), (), 0),
            (
                "empty-offset",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                (2, 0),
                (2, 0),
                (1, 1),
                1,
            ),
            (
                "empty-same-shape",
                torch.zeros((0, 1)) + 1,
                (0, 1),
                (0, 1),
                (1, 0),
                0,
            ),
            (
                "contiguous",
                base,
                (6, 4),
                (6, 4),
                (4, 1),
                0,
            ),
            (
                "contiguous-offset",
                base[1],
                (2, 6),
                (2, 6),
                (6, 1),
                12,
            ),
            (
                "noncontiguous-same-shape",
                noncontiguous,
                (3, 2, 4),
                (3, 2, 4),
                (4, 12, 1),
                0,
            ),
            (
                "noncontiguous-compatible-split",
                noncontiguous,
                (3, 2, 2, 2),
                (3, 2, 2, 2),
                (4, 12, 2, 1),
                0,
            ),
        )

    def test_all_shape_forms_delegate_to_native_view(self):
        for (
            case,
            source,
            shape,
            expected_shape,
            expected_stride,
            expected_offset,
        ) in self.make_layout_cases():
            for form, argument, keyword in self.shape_forms(shape):
                with self.subTest(case=case, form=form):
                    result = self.call_view(source, form, argument, keyword)
                    self.assert_view_result(
                        result,
                        source,
                        expected_shape=expected_shape,
                        expected_stride=expected_stride,
                        expected_offset=expected_offset,
                    )

    def test_single_integer_and_index_delegate_to_native_view(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        cases = (
            ("scalar", torch.tensor(-0.0), 1, (1,), (1,), 0),
            ("inferred", base, -1, (24,), (1,), 0),
            ("offset", base[1], IntSubclass(12), (12,), (1,), 12),
            (
                "empty-offset",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                np.int64(-1),
                (0,),
                (1,),
                1,
            ),
            (
                "compatible-noncontiguous",
                torch.tensor(
                    np.arange(6, dtype=np.float32).reshape(2, 3).tolist()
                ).transpose(0, 1)[0],
                IndexDimension(2),
                (2,),
                (3,),
                0,
            ),
        )
        for case, source, dimension, shape, stride, offset in cases:
            with self.subTest(case=case):
                result = source.view(dimension)
                self.assert_view_result(
                    result,
                    source,
                    expected_shape=shape,
                    expected_stride=stride,
                    expected_offset=offset,
                )

        stateful = StatefulIndexDimension((24, 1, 24))
        result = base.view(stateful)
        self.assertEqual(result.shape, (24,))
        self.assertEqual(stateful.calls, 3)

        first = StatefulIndexDimension((2, 1, 2))
        second = StatefulIndexDimension((3,))
        result = torch.zeros((6,)).view(first, second)
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.stride(), (3, 1))
        self.assertEqual((first.calls, second.calls), (3, 1))

    def test_inferred_and_extreme_empty_shapes_preserve_aliasing(self):
        source = torch.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist())
        for form, argument, keyword in self.shape_forms((2, -1, 2)):
            with self.subTest(kind="inferred", form=form):
                result = self.call_view(source, form, argument, keyword)
                self.assertEqual(result.shape, (2, 6, 2))
                self.assertEqual(result.stride(), (12, 2, 1))
                self.assertEqual(result.data_ptr(), source.data_ptr())
                np.testing.assert_array_equal(
                    np.asarray(result), np.asarray(source).reshape(2, 6, 2)
                )

        maximum = sys.maxsize
        empty = torch.zeros((0,))
        for form, argument, keyword in self.shape_forms((0, maximum, maximum)):
            with self.subTest(kind="extreme-empty", form=form):
                result = self.call_view(empty, form, argument, keyword)
                self.assertEqual(result.shape, (0, maximum, maximum))
                self.assertEqual(result.stride(), (1, maximum, 1))
                self.assertEqual(result.storage_offset(), 0)
                self.assertEqual(result.numel(), 0)
                self.assertEqual(result.tolist(), [])
                self.assertEqual(result.data_ptr(), empty.data_ptr())

        inferred_empty = empty.view((-1,))
        self.assertEqual(inferred_empty.shape, (0,))
        self.assertEqual(inferred_empty.data_ptr(), empty.data_ptr())

    def test_incompatible_layout_and_shape_errors_never_copy(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        source = torch.tensor(values.tolist()).transpose(0, 1)

        for shape in ((6, 4), [6, 4], torch.Size((6, 4))):
            with self.subTest(shape_type=type(shape).__name__):
                with self.assertRaisesRegex(
                    RuntimeError, f"^{re.escape(INCOMPATIBLE_LAYOUT)}$"
                ):
                    source.view(shape)
        with self.assertRaisesRegex(
            RuntimeError, f"^{re.escape(INCOMPATIBLE_LAYOUT)}$"
        ):
            source.view(6, 4)

        reshaped = source.reshape((6, 4))
        self.assertNotEqual(reshaped.data_ptr(), source.data_ptr())
        self.assertFalse(reshaped.is_set_to(source))

        cases = (
            ((2, 2), "shape '[2, 2]' is invalid for input of size 6"),
            ((-1, -1), "only one dimension can be inferred"),
            ((2, -2), "invalid shape dimension -2 at index 1 of shape [2, -2]"),
        )
        for shape, message in cases:
            for form in ("sequence", "variadic"):
                with self.subTest(shape=shape, form=form):
                    with self.assertRaisesRegex(
                        RuntimeError, f"^{re.escape(message)}$"
                    ):
                        if form == "variadic":
                            torch.zeros((6,)).view(*shape)
                        else:
                            torch.zeros((6,)).view(shape)

        ambiguous = (
            "cannot reshape tensor of 0 elements into shape [0, -1] because the "
            "unspecified dimension size -1 can be any value and is ambiguous"
        )
        for form in ("sequence", "variadic"):
            with self.subTest(ambiguous=form), self.assertRaisesRegex(
                RuntimeError, f"^{re.escape(ambiguous)}$"
            ):
                if form == "variadic":
                    torch.zeros((0,)).view(0, -1)
                else:
                    torch.zeros((0,)).view((0, -1))

        with self.assertRaisesRegex(
            RuntimeError, r"^shape '\[5\]' is invalid for input of size 6$"
        ):
            torch.zeros((6,)).view(5)
        with self.assertRaisesRegex(
            RuntimeError, f"^{re.escape(INCOMPATIBLE_LAYOUT)}$"
        ):
            source.view(-1)

    def test_dimension_conversion_matches_all_integer_shape_forms(self):
        tensor = torch.zeros((6,))
        shapes = (
            (IntSubclass(2), np.int64(3)),
            [IndexDimension(2), np.uint32(3)],
            torch.Size((2, 3)),
            (1, True, 6),
        )
        for shape in shapes:
            with self.subTest(shape=shape):
                result = tensor.view(shape)
                self.assertEqual(result.numel(), 6)
                self.assertEqual(result.data_ptr(), tensor.data_ptr())
                variadic = tensor.view(*shape)
                self.assertEqual(
                    variadic.shape, tuple(operator.index(value) for value in shape)
                )
                self.assertEqual(variadic.data_ptr(), tensor.data_ptr())

        with self.assertRaises(TypeError):
            tensor.view((True, 6))
        with self.assertRaisesRegex(
            TypeError,
            r"^view\(\): argument 'size' failed to unpack the object at pos 2 "
            r'with error "type must be tuple of ints,but got float"$',
        ):
            tensor.view((2, 3.0))
        with self.assertRaisesRegex(TypeError, "Overflow when unpacking long long"):
            tensor.view((2**63, 1))

    def test_operator_index_poisoning_cannot_change_shape_parsing(self):
        tensor = torch.zeros((6,))
        original_index = operator.index
        try:
            operator.index = lambda value: {2: 1, 3: 6}.get(value, value)

            result = tensor.view((2, 3))
            self.assertEqual(result.shape, (2, 3))
            self.assertEqual(result.stride(), (3, 1))
            self.assertEqual(result.data_ptr(), tensor.data_ptr())
            variadic = tensor.view(2, 3)
            self.assertEqual(variadic.shape, (2, 3))
            self.assertEqual(variadic.stride(), (3, 1))
            self.assertEqual(variadic.data_ptr(), tensor.data_ptr())
            flattened = tensor.view(-1)
            self.assertEqual(flattened.shape, (6,))
            self.assertEqual(flattened.stride(), (1,))
            self.assertEqual(flattened.data_ptr(), tensor.data_ptr())
            with self.assertRaises(TypeError):
                tensor.view((2, 3.0))
        finally:
            operator.index = original_index

    def test_autograd_repeated_backward_and_no_grad_use_view_semantics(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)
        result = source.view(3, -1)

        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.stride(), (1, 3))
        self.assertEqual(result.storage_offset(), 0)
        self.assertTrue(result.requires_grad)
        self.assertFalse(result.is_leaf)
        self.assertEqual(result.data_ptr(), source.data_ptr())

        weights = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        (result * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad),
            np.asarray([[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]]),
        )

        flat_leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        flat = flat_leaf.view(-1)
        self.assertEqual(flat.shape, (6,))
        self.assertEqual(flat.stride(), (1,))
        self.assertFalse(flat.is_leaf)
        weights = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        (flat * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(flat_leaf.grad),
            np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
        )

        repeated_leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        repeated_loss = repeated_leaf.transpose(0, 1).view([3, 2]).sum()
        repeated_loss.backward()
        repeated_loss.backward()
        np.testing.assert_array_equal(
            np.asarray(repeated_leaf.grad),
            np.full((2, 3), 2.0, dtype=np.float32),
        )

        no_grad_leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        no_grad_source = no_grad_leaf.transpose(0, 1)
        with torch.no_grad():
            no_grad_result = no_grad_source.view(3, 2)

        self.assertTrue(no_grad_result.requires_grad)
        self.assertTrue(no_grad_result.is_leaf)
        self.assertEqual(no_grad_result.stride(), (1, 3))
        self.assertEqual(no_grad_result.data_ptr(), no_grad_source.data_ptr())
        self.assertIsNone(no_grad_leaf.grad)

    def test_tensorbase_descriptor_metadata_matches_the_native_method(self):
        tensor = torch.tensor([1.0, 2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "view")
        bound = tensor.view

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "view")
        self.assertEqual(descriptor.__qualname__, "TensorBase.view")
        self.assertEqual(bound.__name__, "view")
        self.assertEqual(bound.__qualname__, "Tensor.view")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertEqual(
            repr(descriptor), "<method 'view' of 'torch._C.TensorBase' objects>"
        )
        self.assertIs(torch.Tensor.view, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        for callable_object in (descriptor, bound):
            self.assertTrue(callable_object.__doc__.startswith("\nview(*shape) -> Tensor\n"))
            self.assertIn(".. method:: view(dtype) -> Tensor", callable_object.__doc__)
            self.assertTrue(
                callable_object.__doc__.endswith(
                    ">>> x.view(torch.uint8).size()\n    torch.Size([4, 16])\n"
                )
            )
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertEqual(descriptor(tensor, (2, 1)).shape, (2, 1))
        self.assertEqual(descriptor(tensor, -1).shape, (2,))
        self.assertEqual(descriptor(tensor, 2, 1).shape, (2, 1))
        self.assertEqual(descriptor(tensor, size=[2, 1]).shape, (2, 1))

    def test_torch_function_modes_receive_original_calls_and_forward(self):
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        descriptor = inspect.getattr_static(torch.Tensor, "view")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        size = torch.Size((2, 3))
        cases = (
            ("tuple", lambda: tensor.view((2, 3)), (tensor, (2, 3)), None),
            ("list", lambda: tensor.view([2, 3]), (tensor, [2, 3]), None),
            ("Size", lambda: tensor.view(size), (tensor, size), None),
            ("integer", lambda: tensor.view(-1), (tensor, -1), None),
            ("variadic", lambda: tensor.view(2, 3), (tensor, 2, 3), None),
            (
                "keyword",
                lambda: tensor.view(size=(2, 3)),
                (tensor,),
                {"size": (2, 3)},
            ),
        )
        for case, call, expected_args, expected_kwargs in cases:
            mode = RecordingMode(marker)
            with self.subTest(case=case), mode:
                result = call()
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, expected_args)
            self.assertEqual(kwargs, expected_kwargs)

        for form, call in (
            ("sequence", lambda: tensor.view((2, 3.0))),
            ("variadic", lambda: tensor.view(2, 3.0)),
        ):
            deferred = RecordingMode(marker)
            with self.subTest(deferred=form), deferred:
                self.assertIs(call(), marker)
            self.assertEqual(len(deferred.calls), 1)

        invalid = RecordingMode(marker)
        with invalid, self.assertRaises(TypeError):
            tensor.view(range(2))
        self.assertEqual(invalid.calls, [])

        invalid_variadic = RecordingMode(marker)
        with invalid_variadic, self.assertRaises(TypeError):
            tensor.view(2.0, 3)
        self.assertEqual(invalid_variadic.calls, [])

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append((self.label, func, dispatch_types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.view(size=[2, 3])
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor,))
            self.assertEqual(kwargs, {"size": [2, 3]})
        self.assertEqual(forwarded.shape, (2, 3))
        self.assertEqual(forwarded.data_ptr(), tensor.data_ptr())

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.view(-1)
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor, -1))
            self.assertIsNone(kwargs)
        self.assertEqual(forwarded.shape, (6,))
        self.assertEqual(forwarded.data_ptr(), tensor.data_ptr())

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.view(2, 3)
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor, 2, 3))
            self.assertIsNone(kwargs)
        self.assertEqual(forwarded.shape, (2, 3))
        self.assertEqual(forwarded.data_ptr(), tensor.data_ptr())

        declining = RecordingMode(NotImplemented)
        with self.assertRaises(TypeError) as raised:
            with declining:
                tensor.view((2, 3))
        self.assertTrue(
            str(raised.exception).startswith(
                "Multiple dispatch failed for 'torch.Tensor.view'; all "
                "__torch_function__ handlers returned NotImplemented:"
            )
        )
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

    def test_variadic_mixed_argument_errors_match_the_native_parser(self):
        tensor = torch.zeros((6,))
        overloads = (
            "but expected one of:\n"
            " * (torch.dtype dtype)\n"
            " * (tuple of ints size)\n"
        )
        binding_cases = (
            (lambda: tensor.view(2.0, 3), "float, int"),
            (lambda: tensor.view((2,), 3), "tuple, int"),
            (lambda: tensor.view(True, 6), "bool, int"),
            (lambda: tensor.view(torch.float32, 6), "torch.dtype, int"),
            (lambda: tensor.view(2, 3, size=(6,)), "int, int, size=tuple"),
        )
        for call, summary in binding_cases:
            with self.subTest(summary=summary), self.assertRaises(TypeError) as raised:
                call()
            self.assertEqual(
                str(raised.exception),
                f"view() received an invalid combination of arguments - got ({summary}), {overloads}",
            )

        unpack_cases = (
            (lambda: tensor.view(2, 3.0), 2, "float"),
            (lambda: tensor.view(2, (3,)), 2, "tuple"),
            (lambda: tensor.view(2, torch.float32), 2, "torch.dtype"),
        )
        for call, position, actual in unpack_cases:
            with self.subTest(actual=actual), self.assertRaises(TypeError) as raised:
                call()
            self.assertEqual(
                str(raised.exception),
                f"view(): argument 'size' failed to unpack the object at pos {position} "
                f'with error "type must be tuple of ints,but got {actual}"',
            )

    def test_keyword_integer_and_dtype_overloads_remain_unsupported(self):
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        original = np.asarray(tensor).copy()
        calls = (
            lambda: tensor.view(size=-1),
            lambda: tensor.view(torch.float32),
            lambda: tensor.view(dtype=torch.float32),
            lambda: tensor.view(True),
        )
        for call in calls:
            with self.subTest(call=call), self.assertRaises(TypeError):
                call()
        np.testing.assert_array_equal(np.asarray(tensor), original)
        self.assertEqual(tensor.shape, (6,))
        self.assertEqual(tensor.stride(), (1,))


if __name__ == "__main__":
    unittest.main()

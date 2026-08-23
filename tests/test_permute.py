import inspect
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


PERMUTE_DOC = (
    "\npermute(*dims) -> Tensor\n\n"
    "Returns a view of the tensor with its dimensions permuted.\n\n"
    "Args:\n"
    "    dims (torch.Size, int..., tuple of int or list of int): the desired "
    "ordering of dimensions.\n\n"
    "Example:\n"
    "    >>> x = torch.randn(2, 3, 5)\n"
    "    >>> x.size()\n"
    "    torch.Size([2, 3, 5])\n"
    "    >>> x.permute(2, 0, 1).size()\n"
    "    torch.Size([5, 2, 3])\n"
)


class TensorPermuteTests(unittest.TestCase):
    def assert_tensor(self, actual, expected, source, *, dimensions):
        normalized = tuple(axis % len(source.shape) for axis in dimensions)
        self.assertEqual(actual.shape, tuple(source.shape[axis] for axis in normalized))
        self.assertEqual(
            actual.stride(), tuple(source.stride()[axis] for axis in normalized)
        )
        self.assertEqual(actual.storage_offset(), source.storage_offset())
        self.assertEqual(actual.data_ptr(), source.data_ptr())
        self.assertIs(actual.dtype, source.dtype)
        self.assertEqual(actual.device, source.device)
        self.assertIsNot(actual, source)
        np.testing.assert_array_equal(
            np.asarray(actual), np.asarray(expected, dtype=np.float32)
        )

    def assert_error(self, exception_type, message, call):
        with self.assertRaises(exception_type) as raised:
            call()
        self.assertEqual(str(raised.exception), message)

    def test_variadic_sequence_and_keyword_forms_are_shared_storage_views(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist())
        source = base.transpose(0, 2)[1]
        expected_source = values.transpose(2, 1, 0)[1]
        dimensions = (-1, -2)
        expected = expected_source.transpose(1, 0)

        for view in (
            source.permute(*dimensions),
            source.permute(dimensions),
            source.permute(list(dimensions)),
            source.permute(dims=dimensions),
            source.permute(dims=list(dimensions)),
        ):
            with self.subTest(view=view):
                self.assert_tensor(
                    view,
                    expected,
                    source,
                    dimensions=dimensions,
                )

    def test_scalar_and_empty_views_normalize_negative_axes(self):
        scalar = torch.tensor([2.5, 3.5])[1]
        for view in (
            scalar.permute(()),
            scalar.permute([]),
            scalar.permute(dims=()),
            scalar.permute(dims=[]),
        ):
            with self.subTest(kind="scalar", view=view):
                self.assertEqual(view.shape, ())
                self.assertEqual(view.stride(), ())
                self.assertEqual(view.storage_offset(), 1)
                self.assertEqual(view.data_ptr(), scalar.data_ptr())
                self.assertEqual(view.item(), 3.5)
                self.assertIsNot(view, scalar)

        empty = torch.zeros((4, 2, 0, 3)).transpose(0, 3)[2]
        dimensions = (-1, -3, -2)
        for view in (
            empty.permute(*dimensions),
            empty.permute(dimensions),
            empty.permute(list(dimensions)),
            empty.permute(dims=dimensions),
            empty.permute(dims=list(dimensions)),
        ):
            with self.subTest(kind="empty", view=view):
                self.assertEqual(view.shape, (4, 2, 0))
                self.assertEqual(
                    view.stride(),
                    (empty.stride()[2], empty.stride()[0], empty.stride()[1]),
                )
                self.assertEqual(view.storage_offset(), empty.storage_offset())
                self.assertEqual(view.data_ptr(), empty.data_ptr())
                self.assertEqual(view.numel(), 0)

    def test_autograd_and_no_grad_delegate_to_the_native_permutation_view(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        weights = np.linspace(-2.0, 3.0, num=24, dtype=np.float32).reshape(4, 2, 3)
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        view = leaf.permute(-1, 0, 1)

        self.assertTrue(view.requires_grad)
        self.assertFalse(view.is_leaf)
        self.assertEqual(view.data_ptr(), leaf.data_ptr())
        (view * torch.tensor(weights.tolist())).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad), weights.transpose(1, 2, 0)
        )

        with torch.no_grad():
            untracked = leaf.permute(dims=(1, 2, 0))
        self.assertTrue(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertEqual(untracked.data_ptr(), leaf.data_ptr())
        self.assertEqual(untracked.shape, (3, 4, 2))
        self.assertEqual(untracked.stride(), (4, 1, 12))

    def test_rank_duplicate_and_range_errors_match_pytorch(self):
        tensor = torch.zeros((2, 3, 4))
        rank_message = (
            "permute(sparse_coo): number of dimensions in the tensor input does "
            "not match the length of the desired ordering of dimensions i.e. "
            "input.dim() = 3 is not equal to len(dims) = 2"
        )
        for call in (
            lambda: tensor.permute(0, 1),
            lambda: tensor.permute((0, 1)),
            lambda: tensor.permute(dims=[0, 1]),
        ):
            self.assert_error(RuntimeError, rank_message, call)

        scalar_rank_message = (
            "permute(sparse_coo): number of dimensions in the tensor input does "
            "not match the length of the desired ordering of dimensions i.e. "
            "input.dim() = 0 is not equal to len(dims) = 1"
        )
        self.assert_error(
            RuntimeError,
            scalar_rank_message,
            lambda: torch.tensor(1.0).permute(-1),
        )

        for dimensions in ((0, 1, 1), (0, 1, -2), (-1, 2, 0)):
            with self.subTest(dimensions=dimensions):
                self.assert_error(
                    RuntimeError,
                    "permute(): duplicate dims are not allowed.",
                    lambda dimensions=dimensions: tensor.permute(dimensions),
                )

        for dimensions in ((0, 0, 3), (0, -3, 3), (-3, 0, 3)):
            with self.subTest(mixed_duplicate=dimensions):
                self.assert_error(
                    RuntimeError,
                    "permute(): duplicate dims are not allowed.",
                    lambda dimensions=dimensions: tensor.permute(dimensions),
                )

        self.assert_error(
            IndexError,
            "Dimension out of range (expected to be in range of [-3, 2], but got 3)",
            lambda: tensor.permute(0, 3, 0),
        )

        for dimension in (-4, 3):
            with self.subTest(dimension=dimension):
                self.assert_error(
                    IndexError,
                    "Dimension out of range (expected to be in range of [-3, 2], "
                    f"but got {dimension})",
                    lambda dimension=dimension: tensor.permute(0, 1, dimension),
                )

    def test_extreme_empty_reordered_element_count_overflow_matches_pytorch(self):
        source = torch.zeros((3, 0, 1, sys.maxsize))
        for call in (
            lambda: source.permute(3, 0, 1, 2),
            lambda: source.permute((0, 3, 1, 2)),
            lambda: source.permute(dims=[2, 3, 0, 1]),
        ):
            self.assert_error(
                RuntimeError,
                "numel: integer multiplication overflow",
                call,
            )

        for dimensions in ((3, 1, 0, 2), (1, 3, 0, 2)):
            with self.subTest(dimensions=dimensions):
                view = source.permute(dimensions)
                self.assertEqual(
                    view.shape,
                    tuple(source.shape[axis] for axis in dimensions),
                )
                self.assertEqual(
                    view.stride(),
                    tuple(source.stride()[axis] for axis in dimensions),
                )
                self.assertEqual(view.storage_offset(), source.storage_offset())
                self.assertEqual(view.data_ptr(), source.data_ptr())

    def test_dimension_types_and_binding_errors_match_pytorch(self):
        class IntSubclass(int):
            pass

        class IndexOnly:
            def __index__(self):
                return 1

        tensor = torch.zeros((2, 3, 4))
        self.assertEqual(
            tensor.permute(IntSubclass(2), np.int64(0), IndexOnly()).shape,
            (4, 2, 3),
        )
        self.assertEqual(tensor.permute([0, True, 2]).shape, (2, 3, 4))
        self.assertEqual(tensor.permute(dims=(0, True, 2)).shape, (2, 3, 4))

        for dimensions, message in (
            (
                [True, 0, 2],
                "permute(): argument 'dims' (position 1) must be tuple of ints, "
                "but found element of type bool at pos 0",
            ),
            (
                (True, 0, 2),
                "permute(): argument 'dims' (position 1) must be tuple of ints, "
                "but found element of type bool at pos 0",
            ),
        ):
            with self.subTest(form="positional", dimensions=dimensions):
                self.assert_error(
                    TypeError,
                    message,
                    lambda dimensions=dimensions: tensor.permute(dimensions),
                )

        for dimensions, outer_type in (([True, 0, 2], "list"), ((True, 0, 2), "tuple")):
            with self.subTest(form="keyword", dimensions=dimensions):
                self.assert_error(
                    TypeError,
                    "permute(): argument 'dims' must be tuple of ints, not "
                    f"{outer_type}",
                    lambda dimensions=dimensions: tensor.permute(dims=dimensions),
                )

        self.assert_error(
            TypeError,
            "permute(): argument 'dims' (position 1) must be tuple of ints, not float",
            lambda: tensor.permute(1.5),
        )
        self.assert_error(
            TypeError,
            "permute(): argument 'dims' must be tuple of ints, not int",
            lambda: tensor.permute(dims=1),
        )
        self.assert_error(
            TypeError,
            "permute(): argument 'dims' failed to unpack the object at pos 2 with "
            'error "type must be tuple of ints,but got float"',
            lambda: tensor.permute(0, 1.5, 2),
        )
        self.assert_error(
            TypeError,
            "permute(): argument 'dims' failed to unpack the object at pos 2 with "
            'error "type must be tuple of ints,but got numpy.bool"',
            lambda: tensor.permute([0, np.bool_(True), 2]),
        )
        self.assert_error(
            TypeError,
            "permute(): argument 'dims' (position 1) must be tuple of ints, but "
            "found element of type float at pos 0",
            lambda: tensor.permute([1.5, 0, 2]),
        )
        self.assert_error(
            TypeError,
            "permute(): argument 'dims' must be tuple of ints, not list",
            lambda: tensor.permute(dims=[1.5, 0, 2]),
        )

        binding_cases = (
            (
                lambda: tensor.permute(),
                'permute() missing 1 required positional arguments: "dims"',
            ),
            (
                lambda: tensor.permute(unexpected=None),
                'permute() missing 1 required positional arguments: "dims"',
            ),
            (
                lambda: tensor.permute(2, 0, 1, unexpected=None),
                "permute() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: tensor.permute(2, 0, 1, dims=(2, 0, 1)),
                "permute() got multiple values for argument 'dims'",
            ),
            (
                lambda: tensor.permute((2, 0, 1), (0, 1, 2)),
                "permute() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: tensor.permute(1.5, 0, 2),
                "permute() takes 1 positional argument but 3 were given",
            ),
            (
                lambda: tensor.permute([0, 1.5, 2], unexpected=None),
                "permute() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: tensor.permute(dims=1, unexpected=None),
                "permute(): argument 'dims' must be tuple of ints, not int",
            ),
        )
        for call, message in binding_cases:
            with self.subTest(message=message):
                self.assert_error(TypeError, message, call)

    def test_index_conversion_order_matches_the_legacy_binding(self):
        class StatefulIndex:
            def __init__(self, name, calls, value):
                self.name = name
                self.calls = calls
                self.value = value

            def __index__(self):
                self.calls.append(self.name)
                return self.value

        tensor = torch.zeros((2, 3, 4))
        variadic_calls = []
        tensor.permute(
            StatefulIndex("first", variadic_calls, 2),
            StatefulIndex("second", variadic_calls, 0),
            StatefulIndex("third", variadic_calls, 1),
        )
        self.assertEqual(
            variadic_calls,
            ["first", "first", "first", "second", "third"],
        )

        sequence_calls = []
        tensor.permute(
            [
                StatefulIndex("first", sequence_calls, 2),
                StatefulIndex("second", sequence_calls, 0),
                StatefulIndex("third", sequence_calls, 1),
            ]
        )
        self.assertEqual(
            sequence_calls,
            ["first", "first", "second", "third"],
        )

    def test_torch_function_modes_receive_original_calls_before_native_validation(self):
        tensor = torch.zeros((2, 3, 4))
        descriptor = inspect.getattr_static(torch.Tensor, "permute")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        dimensions_tuple = (2, 0, 1)
        dimensions_list = [2, 0, 1]
        cases = (
            ("variadic", lambda: tensor.permute(2, 0, 1), (2, 0, 1), None),
            (
                "tuple",
                lambda: tensor.permute(dimensions_tuple),
                (dimensions_tuple,),
                None,
            ),
            (
                "list",
                lambda: tensor.permute(dimensions_list),
                (dimensions_list,),
                None,
            ),
            (
                "keyword tuple",
                lambda: tensor.permute(dims=dimensions_tuple),
                (),
                {"dims": dimensions_tuple},
            ),
            (
                "keyword list",
                lambda: tensor.permute(dims=dimensions_list),
                (),
                {"dims": dimensions_list},
            ),
            ("rank mismatch", lambda: tensor.permute(0, 1), (0, 1), None),
            ("duplicate", lambda: tensor.permute(0, 1, 1), (0, 1, 1), None),
            ("out of range", lambda: tensor.permute(0, 1, 3), (0, 1, 3), None),
            (
                "deferred element type",
                lambda: tensor.permute(0, 1.5, 2),
                (0, 1.5, 2),
                None,
            ),
        )
        for case, call, expected_arguments, expected_kwargs in cases:
            mode = RecordingMode(marker)
            with self.subTest(case=case), mode:
                result = call()
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(), [mode]
                )
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertIs(args[0], tensor)
            self.assertEqual(args[1:], expected_arguments)
            self.assertEqual(kwargs, expected_kwargs)
            self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        invalid = RecordingMode(marker)
        with invalid, self.assertRaises(TypeError):
            tensor.permute([1.5, 0, 2])
        self.assertEqual(invalid.calls, [])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_dimension_overrides_are_ordered_and_follow_a_declining_mode(self):
        tensor = torch.zeros((2, 3, 4))
        descriptor = inspect.getattr_static(torch.Tensor, "permute")
        marker = object()
        override_calls = []

        class BaseDimension(int):
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                override_calls.append((cls, func, dispatch_types, args, kwargs))
                return NotImplemented

        class DerivedDimension(BaseDimension):
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                override_calls.append((cls, func, dispatch_types, args, kwargs))
                return NotImplemented

        class AcceptingDimension(int):
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                override_calls.append((cls, func, dispatch_types, args, kwargs))
                return marker

        dimensions = (
            BaseDimension(2),
            DerivedDimension(0),
            AcceptingDimension(1),
        )
        result = tensor.permute(*dimensions)
        self.assertIs(result, marker)
        self.assertEqual(
            [entry[0] for entry in override_calls],
            [DerivedDimension, BaseDimension, AcceptingDimension],
        )
        for _, function, dispatch_types, args, kwargs in override_calls:
            self.assertIs(function, descriptor)
            self.assertEqual(
                dispatch_types,
                (DerivedDimension, BaseDimension, AcceptingDimension),
            )
            self.assertIs(args[0], tensor)
            self.assertTrue(
                all(
                    actual is expected
                    for actual, expected in zip(args[1:], dimensions)
                )
            )
            self.assertIsNone(kwargs)

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        override_calls.clear()
        accepting_mode = RecordingMode(object())
        with accepting_mode:
            result = tensor.permute(dimensions)
        self.assertIs(result, accepting_mode.result)
        self.assertEqual(override_calls, [])
        function, dispatch_types, args, kwargs = accepting_mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(
            dispatch_types,
            (DerivedDimension, BaseDimension, AcceptingDimension),
        )
        self.assertIs(args[0], tensor)
        self.assertIs(args[1], dimensions)
        self.assertIsNone(kwargs)

        override_calls.clear()
        declining_mode = RecordingMode(NotImplemented)
        keyword_dimensions = list(dimensions)
        with declining_mode:
            result = tensor.permute(dims=keyword_dimensions)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [declining_mode]
            )
        self.assertIs(result, marker)
        self.assertEqual(
            [entry[0] for entry in override_calls],
            [DerivedDimension, BaseDimension, AcceptingDimension],
        )
        function, dispatch_types, args, kwargs = declining_mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(
            dispatch_types,
            (DerivedDimension, BaseDimension, AcceptingDimension),
        )
        self.assertEqual(args, (tensor,))
        self.assertEqual(kwargs, {"dims": keyword_dimensions})
        self.assertIs(kwargs["dims"], keyword_dimensions)
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        index_calls = []

        class IndexDimension:
            def __index__(self):
                index_calls.append("index")
                return 2

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                return marker

        index_dimension = IndexDimension()
        intercepting_mode = RecordingMode(marker)
        with intercepting_mode:
            self.assertIs(tensor.permute(index_dimension, 0, 1), marker)
        self.assertEqual(index_calls, ["index"])
        self.assertEqual(intercepting_mode.calls[0][1], (IndexDimension,))

    def test_torch_function_mode_forwarding_declining_raising_and_restoration(self):
        tensor = torch.zeros((2, 3, 4))
        descriptor = inspect.getattr_static(torch.Tensor, "permute")
        dimensions = [2, 0, 1]
        events = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                events.append(
                    (
                        self.label,
                        func,
                        dispatch_types,
                        args,
                        kwargs,
                        tuple(torch.overrides._get_current_function_mode_stack()),
                    )
                )
                return func(*args, **(kwargs or {}))

        lower = ForwardingMode("lower")
        upper = ForwardingMode("upper")
        with lower:
            with upper:
                forwarded = tensor.permute(dims=dimensions)
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(), [lower, upper]
                )
        self.assertEqual([event[0] for event in events], ["upper", "lower"])
        self.assertEqual(events[0][5], (lower,))
        self.assertEqual(events[1][5], ())
        for _, function, dispatch_types, args, kwargs, _ in events:
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor,))
            self.assertEqual(kwargs, {"dims": dimensions})
        self.assertEqual(forwarded.shape, (4, 2, 3))
        self.assertEqual(forwarded.stride(), (1, 12, 4))
        self.assertEqual(forwarded.data_ptr(), tensor.data_ptr())
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        declining = RecordingMode(NotImplemented)
        bypassed_lower = RecordingMode(object())
        with bypassed_lower:
            with declining:
                with self.assertRaises(TypeError) as raised:
                    tensor.permute(0, 1, 2)
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(),
                    [bypassed_lower, declining],
                )
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [bypassed_lower]
            )
        self.assertTrue(
            str(raised.exception).startswith(
                "Multiple dispatch failed for 'torch.Tensor.permute'; all "
                "__torch_function__ handlers returned NotImplemented:"
            )
        )
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(bypassed_lower.calls, [])

        expected_error = ValueError("permute mode failed")

        class RaisingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append(
                    (
                        func,
                        dispatch_types,
                        args,
                        kwargs,
                        tuple(torch.overrides._get_current_function_mode_stack()),
                    )
                )
                raise expected_error

        raising = RaisingMode()
        with raising:
            try:
                tensor.permute(2, 0, 1)
            except ValueError as error:
                self.assertIs(error, expected_error)
            else:
                self.fail("Tensor.permute accepted a raising mode")
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [raising]
            )
        self.assertEqual(len(raising.calls), 1)
        function, dispatch_types, args, kwargs, callback_stack = raising.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (tensor, 2, 0, 1))
        self.assertIsNone(kwargs)
        self.assertEqual(callback_stack, ())
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        recovered = tensor.permute(2, 0, 1)
        self.assertEqual(recovered.shape, (4, 2, 3))
        self.assertEqual(recovered.data_ptr(), tensor.data_ptr())

    def test_descriptor_metadata_matches_pytorch_shape(self):
        descriptor = inspect.getattr_static(torch.Tensor, "permute")
        tensor = torch.zeros((2, 3, 4))
        bound = tensor.permute

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "permute")
        self.assertEqual(descriptor.__qualname__, "TensorBase.permute")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        self.assertEqual(descriptor.__doc__, PERMUTE_DOC)
        self.assertEqual(bound.__doc__, PERMUTE_DOC)
        for callable_object in (descriptor, bound):
            with self.subTest(callable_object=callable_object):
                with self.assertRaises(ValueError):
                    inspect.signature(callable_object)

        result = descriptor(tensor, 2, 0, 1)
        self.assertEqual(result.shape, (4, 2, 3))
        with self.assertRaisesRegex(
            TypeError, "^unbound method TensorBase.permute\\(\\) needs an argument$"
        ):
            descriptor()


if __name__ == "__main__":
    unittest.main()

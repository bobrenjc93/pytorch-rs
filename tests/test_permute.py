import inspect
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

        for dimension in (-4, 3):
            with self.subTest(dimension=dimension):
                self.assert_error(
                    IndexError,
                    "Dimension out of range (expected to be in range of [-3, 2], "
                    f"but got {dimension})",
                    lambda dimension=dimension: tensor.permute(0, 1, dimension),
                )

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

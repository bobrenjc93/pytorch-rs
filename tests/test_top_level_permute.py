import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = (
    "\npermute(input, dims) -> Tensor\n\n"
    "Returns a view of the original tensor :attr:`input` with its dimensions "
    "permuted.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n"
    "    dims (torch.Size, tuple of int or list of int): the desired ordering "
    "of dimensions.\n\n"
    "Example:\n"
    "    >>> x = torch.randn(2, 3, 5)\n"
    "    >>> x.size()\n"
    "    torch.Size([2, 3, 5])\n"
    "    >>> torch.permute(x, (2, 0, 1)).size()\n"
    "    torch.Size([5, 2, 3])\n"
)


class TopLevelPermuteTests(unittest.TestCase):
    def assert_error(self, exception_type, message, call):
        with self.assertRaises(exception_type) as raised:
            call()
        self.assertEqual(str(raised.exception), message)

    def assert_view(self, actual, expected, source, dimensions):
        normalized = tuple(axis % len(source.shape) for axis in dimensions)
        self.assertEqual(
            actual.shape, tuple(source.shape[axis] for axis in normalized)
        )
        self.assertEqual(
            actual.stride(), tuple(source.stride()[axis] for axis in normalized)
        )
        self.assertEqual(actual.storage_offset(), source.storage_offset())
        self.assertEqual(actual.data_ptr(), source.data_ptr())
        self.assertIs(actual.dtype, source.dtype)
        self.assertEqual(actual.device, source.device)
        self.assertIsNot(actual, source)
        np.testing.assert_array_equal(np.asarray(actual), expected)

    def test_tuple_list_keyword_and_legacy_input_forms_are_native_views(self):
        values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        base = torch.tensor(values.tolist())
        source = base.transpose(0, 2)[1]
        expected_source = values.transpose(2, 1, 0, 3)[1]
        dimensions = (-1, 0, 1)
        expected = expected_source.transpose(2, 0, 1)

        for view in (
            torch.permute(source, dimensions),
            torch.permute(source, list(dimensions)),
            torch.permute(input=source, dims=dimensions),
            torch.permute(dims=list(dimensions), input=source),
            torch.permute(x=source, dims=dimensions),
            torch.permute(a=source, dims=list(dimensions)),
            torch.permute(x1=source, dims=dimensions),
        ):
            with self.subTest(view=view):
                self.assert_view(view, expected, source, dimensions)

    def test_scalars_empties_and_extreme_metadata_preserve_aliasing(self):
        scalar = torch.tensor([2.5, 3.5])[1]
        for view in (
            torch.permute(scalar, ()),
            torch.permute(scalar, []),
            torch.permute(input=scalar, dims=()),
            torch.permute(input=scalar, dims=[]),
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
            torch.permute(empty, dimensions),
            torch.permute(input=empty, dims=list(dimensions)),
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

        extreme = torch.zeros((3, 0, 1, sys.maxsize))
        self.assert_error(
            RuntimeError,
            "numel: integer multiplication overflow",
            lambda: torch.permute(extreme, (3, 0, 1, 2)),
        )
        reordered = torch.permute(input=extreme, dims=[3, 1, 0, 2])
        self.assertEqual(reordered.shape, (sys.maxsize, 0, 3, 1))
        self.assertEqual(
            reordered.stride(), tuple(extreme.stride()[axis] for axis in (3, 1, 0, 2))
        )
        self.assertEqual(reordered.data_ptr(), extreme.data_ptr())

    def test_autograd_empty_backward_and_no_grad_reuse_method_semantics(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        weights = np.linspace(-2.0, 3.0, num=24, dtype=np.float32).reshape(4, 2, 3)
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        view = torch.permute(input=leaf, dims=(-1, 0, 1))

        self.assertTrue(view.requires_grad)
        self.assertFalse(view.is_leaf)
        self.assertEqual(view.data_ptr(), leaf.data_ptr())
        (view * torch.tensor(weights.tolist())).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad), weights.transpose(1, 2, 0)
        )

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        torch.permute(empty, [-1, 0, 1]).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(empty.grad), np.zeros((2, 0, 3), dtype=np.float32)
        )

        with torch.no_grad():
            untracked = torch.permute(leaf, (1, 2, 0))
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
            lambda: torch.permute(tensor, (0, 1)),
            lambda: torch.permute(input=tensor, dims=[0, 1]),
        ):
            self.assert_error(RuntimeError, rank_message, call)

        self.assert_error(
            RuntimeError,
            "permute(sparse_coo): number of dimensions in the tensor input does "
            "not match the length of the desired ordering of dimensions i.e. "
            "input.dim() = 0 is not equal to len(dims) = 1",
            lambda: torch.permute(torch.tensor(1.0), [-1]),
        )

        for dimensions in ((0, 1, 1), (0, 1, -2), (0, 0, 3)):
            with self.subTest(dimensions=dimensions):
                self.assert_error(
                    RuntimeError,
                    "permute(): duplicate dims are not allowed.",
                    lambda dimensions=dimensions: torch.permute(tensor, dimensions),
                )

        for dimensions, bad_dimension in (((0, 1, 3), 3), ((0, 1, -4), -4)):
            with self.subTest(dimensions=dimensions):
                self.assert_error(
                    IndexError,
                    "Dimension out of range (expected to be in range of [-3, 2], "
                    f"but got {bad_dimension})",
                    lambda dimensions=dimensions: torch.permute(tensor, dimensions),
                )

        self.assert_error(
            IndexError,
            "Dimension out of range (expected to be in range of [-3, 2], but got 3)",
            lambda: torch.permute(tensor, (0, 3, 0)),
        )

    def test_dimension_types_and_binding_error_precedence_match_pytorch(self):
        class IntSubclass(int):
            pass

        class IndexOnly:
            def __index__(self):
                return 1

        tensor = torch.zeros((2, 3, 4))
        self.assertEqual(
            torch.permute(tensor, (IntSubclass(2), np.int64(0), IndexOnly())).shape,
            (4, 2, 3),
        )
        self.assertEqual(torch.permute(tensor, [0, True, 2]).shape, (2, 3, 4))

        cases = (
            (
                lambda: torch.permute(),
                'permute() missing 2 required positional argument: "input", "dims"',
            ),
            (
                lambda: torch.permute(tensor),
                'permute() missing 1 required positional arguments: "dims"',
            ),
            (
                lambda: torch.permute(dims=(2, 0, 1)),
                'permute() missing 2 required positional argument: "input", "dims"',
            ),
            (
                lambda: torch.permute(tensor, 2, 0, 1),
                "permute() takes 2 positional arguments but 4 were given",
            ),
            (
                lambda: torch.permute(1, (0,)),
                "permute(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.permute(input=[], dims=(0,)),
                "permute(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.permute(tensor, 1),
                "permute(): argument 'dims' (position 2) must be tuple of ints, not int",
            ),
            (
                lambda: torch.permute(input=tensor, dims=1),
                "permute(): argument 'dims' must be tuple of ints, not int",
            ),
            (
                lambda: torch.permute(tensor, [1.5, 0, 2]),
                "permute(): argument 'dims' (position 2) must be tuple of ints, "
                "but found element of type float at pos 0",
            ),
            (
                lambda: torch.permute(input=tensor, dims=[1.5, 0, 2]),
                "permute(): argument 'dims' must be tuple of ints, not list",
            ),
            (
                lambda: torch.permute(tensor, [0, np.bool_(True), 2]),
                "permute(): argument 'dims' failed to unpack the object at pos 2 "
                'with error "type must be tuple of ints,but got numpy.bool"',
            ),
            (
                lambda: torch.permute(tensor, (2, 0, 1), input=tensor),
                "permute() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.permute(tensor, (2, 0, 1), dims=(2, 0, 1)),
                "permute() got multiple values for argument 'dims'",
            ),
            (
                lambda: torch.permute(tensor, (0, 1), extra=True),
                "permute() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.permute(tensor, [0, 1.5, 2], extra=True),
                "permute() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.permute(tensor, [1.5, 0, 2], extra=True),
                "permute(): argument 'dims' (position 2) must be tuple of ints, "
                "but found element of type float at pos 0",
            ),
            (
                lambda: torch.permute(x=tensor, dims=(2, 0, 1), extra=True),
                "permute() got an unexpected keyword argument 'x'",
            ),
        )
        for call, message in cases:
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
        for style in ("positional", "keyword"):
            calls = []
            dimensions = [
                StatefulIndex("first", calls, 2),
                StatefulIndex("second", calls, 0),
                StatefulIndex("third", calls, 1),
            ]
            if style == "positional":
                torch.permute(tensor, dimensions)
            else:
                torch.permute(input=tensor, dims=dimensions)
            self.assertEqual(calls, ["first", "first", "second", "third"])

        calls = []
        dimensions = [
            StatefulIndex("first", calls, 2),
            StatefulIndex("second", calls, 0),
            StatefulIndex("third", calls, 1),
        ]
        self.assert_error(
            TypeError,
            "permute() got an unexpected keyword argument 'extra'",
            lambda: torch.permute(tensor, dimensions, extra=True),
        )
        self.assertEqual(calls, ["first"])

    def test_callable_metadata_documentation_and_exports_match_pytorch(self):
        function = torch.permute
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "permute")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.permute")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertRegex(
            repr(function),
            r"^<built-in method permute of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.permute, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                self.assertIs(restored, function)

        self.assertEqual(torch.__all__.count("permute"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["permute"], function)


if __name__ == "__main__":
    unittest.main()

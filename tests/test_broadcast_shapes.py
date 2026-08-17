import copy
import importlib
import inspect
import pickle
import re
import sys
import types
import unittest
from contextlib import ExitStack
from unittest import mock

import numpy as np
import torch_rs as torch


FUNCTION_DOC = (
    r"""broadcast_shapes(*shapes) -> Size

    Similar to :func:`broadcast_tensors` but for shapes.

    This is equivalent to
    ``torch.broadcast_tensors(*map(torch.empty, shapes))[0].shape``
    but avoids the need to create intermediate tensors. This is useful for
    broadcasting tensors of common batch shape but different rightmost shape,
    e.g. to broadcast mean vectors with covariance matrices.

    Example::

        >>> torch.broadcast_shapes((2,), (3, 1), (1, 1, 1))
        torch.Size([1, 3, 2])

    Args:
        \*shapes (torch.Size): Shapes of tensors.

    Returns:
        shape (torch.Size): A shape compatible with all input shapes.

    Raises:
        RuntimeError: If shapes are incompatible.
    """
)

if sys.version_info >= (3, 13):
    # CPython 3.13+ cleans function docstring indentation while preserving
    # the terminating newline; PyTorch's source docstring follows that rule.
    FUNCTION_DOC = inspect.cleandoc(FUNCTION_DOC) + "\n"


class BroadcastShapesTests(unittest.TestCase):
    def assert_size(self, expected, call):
        result = call()
        self.assertIs(type(result), torch.Size)
        self.assertEqual(tuple(result), expected)
        if all(-(2**63) <= dimension < 2**63 for dimension in expected):
            self.assertEqual(repr(result), f"torch.Size({list(expected)})")

    def assert_error(self, exception_type, message, call):
        with self.assertRaisesRegex(exception_type, f"^{re.escape(message)}$"):
            call()

    def test_empty_integer_sequence_size_and_zero_dimension_inputs(self):
        cases = (
            ("no shapes", (), lambda: torch.broadcast_shapes()),
            ("None is ignored", (), lambda: torch.broadcast_shapes(None)),
            ("empty tuple", (), lambda: torch.broadcast_shapes(())),
            ("empty list", (), lambda: torch.broadcast_shapes([])),
            ("empty Size", (), lambda: torch.broadcast_shapes(torch.Size())),
            ("integer", (7,), lambda: torch.broadcast_shapes(7)),
            ("zero integer", (0,), lambda: torch.broadcast_shapes(0)),
            ("boolean true", (1,), lambda: torch.broadcast_shapes(True)),
            ("boolean false", (0,), lambda: torch.broadcast_shapes(False)),
            (
                "tuple list and Size",
                (5, 3, 2),
                lambda: torch.broadcast_shapes(
                    (2,), [3, 1], torch.Size([5, 1, 1])
                ),
            ),
            (
                "zero broadcasts with one",
                (2, 0, 3),
                lambda: torch.broadcast_shapes((2, 0, 3), [1, 1, 3]),
            ),
            (
                "leading singleton",
                (1, 3, 2),
                lambda: torch.broadcast_shapes((2,), (3, 1), (1, 1, 1)),
            ),
            (
                "ignored None among shapes",
                (2, 3),
                lambda: torch.broadcast_shapes(None, [2, 3], None),
            ),
        )
        for case, expected, call in cases:
            with self.subTest(case=case):
                self.assert_size(expected, call)

        source_list = [3, 1]
        source_size = torch.Size([5, 1, 1])
        self.assert_size(
            (5, 3, 2),
            lambda: torch.broadcast_shapes((2,), source_list, source_size),
        )
        self.assertEqual(source_list, [3, 1])
        self.assertEqual(source_size, torch.Size([5, 1, 1]))

    def test_large_python_dimensions_and_ranks_do_not_allocate_tensors(self):
        huge = 2**100
        rank = 256
        expected = (huge, *(1 for _ in range(rank - 2)), 0)

        with ExitStack() as stack:
            factories = [
                stack.enter_context(
                    mock.patch.object(
                        torch,
                        name,
                        side_effect=AssertionError(f"{name} must not be called"),
                        create=True,
                    )
                )
                for name in ("empty", "zeros", "ones", "tensor", "broadcast_tensors")
            ]
            self.assert_size(
                expected,
                lambda: torch.broadcast_shapes(
                    (huge, *((1,) * (rank - 2)), 0),
                    (1, *((1,) * (rank - 2)), 1),
                ),
            )

        for factory in factories:
            factory.assert_not_called()

    def test_incompatible_negative_and_invalid_elements_match_public_errors(self):
        errors = (
            (
                RuntimeError,
                "Attempting to broadcast a dimension of length 3 at -1! "
                "Mismatching argument at index 1 had (3,); but expected shape "
                "should be broadcastable to [2]",
                lambda: torch.broadcast_shapes((2,), (3,)),
            ),
            (
                RuntimeError,
                "Attempting to broadcast a dimension of length 4 at -2! "
                "Mismatching argument at index 1 had [4, 1]; but expected shape "
                "should be broadcastable to [2, 3]",
                lambda: torch.broadcast_shapes((2, 3), [4, 1]),
            ),
            (
                ValueError,
                "Attempting to broadcast a dimension with negative length!",
                lambda: torch.broadcast_shapes((-1,)),
            ),
            (
                ValueError,
                "Attempting to broadcast a dimension with negative length!",
                lambda: torch.broadcast_shapes(torch.Size([2, -1])),
            ),
            (
                RuntimeError,
                "Attempting to broadcast a dimension of length -1 at -1! "
                "Mismatching argument at index 1 had (-1,); but expected shape "
                "should be broadcastable to [2]",
                lambda: torch.broadcast_shapes((2,), (-1,)),
            ),
            (
                RuntimeError,
                "Input shapes should be of type ints, a tuple of ints, or a "
                "list of ints, got 3.0",
                lambda: torch.broadcast_shapes(3.0),
            ),
            (
                TypeError,
                "torch.Size() takes an iterable of 'int' (item 0 is 'float')",
                lambda: torch.broadcast_shapes((2.0,)),
            ),
            (
                TypeError,
                "'<' not supported between instances of 'str' and 'int'",
                lambda: torch.broadcast_shapes(("2",)),
            ),
            (
                AssertionError,
                "Expected bool, got <class 'numpy.bool'>",
                lambda: torch.broadcast_shapes((np.int64(2),)),
            ),
            (
                TypeError,
                "broadcast_shapes() got an unexpected keyword argument 'shapes'",
                lambda: torch.broadcast_shapes(shapes=(2, 3)),
            ),
        )
        for exception_type, message, call in errors:
            with self.subTest(exception_type=exception_type, message=message):
                self.assert_error(exception_type, message, call)

        # PyTorch's shape merge short-circuits equal dimensions before the
        # final torch.Size validation.
        self.assert_size((1,), lambda: torch.broadcast_shapes((1.0,)))
        self.assert_size((2,), lambda: torch.broadcast_shapes((2,), (2.0,)))

    def test_functional_ownership_signature_documentation_exports_and_pickle(self):
        functional = importlib.import_module("torch_rs.functional")
        function = functional.broadcast_shapes

        self.assertIs(torch.functional, functional)
        self.assertIs(torch.broadcast_shapes, function)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "broadcast_shapes")
        self.assertEqual(function.__qualname__, "broadcast_shapes")
        self.assertEqual(function.__module__, "torch_rs.functional")
        self.assertEqual(str(inspect.signature(function)), "(*shapes)")
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertRegex(
            repr(function), r"^<function broadcast_shapes at 0x[0-9a-f]+>$"
        )

        self.assertEqual(functional.__all__.count("broadcast_shapes"), 1)
        self.assertEqual(torch.__all__.count("broadcast_shapes"), 0)
        functional_wildcard = {}
        exec("from torch_rs.functional import *", functional_wildcard)
        self.assertIs(functional_wildcard["broadcast_shapes"], function)
        top_level_wildcard = {}
        exec("from torch_rs import *", top_level_wildcard)
        self.assertNotIn("broadcast_shapes", top_level_wildcard)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)

        self.assertFalse(hasattr(torch, "broadcast_tensors"))
        self.assertFalse(hasattr(functional, "broadcast_tensors"))
        self.assertFalse(hasattr(torch, "SymInt"))


if __name__ == "__main__":
    unittest.main()


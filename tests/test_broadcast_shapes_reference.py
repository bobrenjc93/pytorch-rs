import importlib
import inspect
import pickle
import types
import unittest
from contextlib import ExitStack
from unittest import mock

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class BroadcastShapesReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "broadcast_shapes differentials require pinned PyTorch 2.13.0"
            )

    def outcome(self, call):
        try:
            result = call()
        except Exception as error:
            return ("error", type(error).__name__, str(error))
        return (
            "value",
            type(result).__module__.replace("torch_rs", "torch"),
            type(result).__name__,
            tuple(result),
        )

    def assert_calls_match(self, actual_call, expected_call):
        self.assertEqual(self.outcome(actual_call), self.outcome(expected_call))

    def test_empty_integer_tuple_list_size_zero_and_large_shapes_match(self):
        huge = 2**100
        cases = (
            (lambda: torch.broadcast_shapes(), lambda: reference_torch.broadcast_shapes()),
            (
                lambda: torch.broadcast_shapes(None),
                lambda: reference_torch.broadcast_shapes(None),
            ),
            (
                lambda: torch.broadcast_shapes(()),
                lambda: reference_torch.broadcast_shapes(()),
            ),
            (
                lambda: torch.broadcast_shapes([]),
                lambda: reference_torch.broadcast_shapes([]),
            ),
            (
                lambda: torch.broadcast_shapes(torch.Size()),
                lambda: reference_torch.broadcast_shapes(reference_torch.Size()),
            ),
            (lambda: torch.broadcast_shapes(7), lambda: reference_torch.broadcast_shapes(7)),
            (lambda: torch.broadcast_shapes(0), lambda: reference_torch.broadcast_shapes(0)),
            (
                lambda: torch.broadcast_shapes(True),
                lambda: reference_torch.broadcast_shapes(True),
            ),
            (
                lambda: torch.broadcast_shapes(False),
                lambda: reference_torch.broadcast_shapes(False),
            ),
            (
                lambda: torch.broadcast_shapes(
                    (2,), [3, 1], torch.Size([5, 1, 1])
                ),
                lambda: reference_torch.broadcast_shapes(
                    (2,), [3, 1], reference_torch.Size([5, 1, 1])
                ),
            ),
            (
                lambda: torch.broadcast_shapes((2, 0, 3), [1, 1, 3]),
                lambda: reference_torch.broadcast_shapes((2, 0, 3), [1, 1, 3]),
            ),
            (
                lambda: torch.broadcast_shapes((2,), (3, 1), (1, 1, 1)),
                lambda: reference_torch.broadcast_shapes(
                    (2,), (3, 1), (1, 1, 1)
                ),
            ),
            (
                lambda: torch.broadcast_shapes(None, [2, 3], None),
                lambda: reference_torch.broadcast_shapes(None, [2, 3], None),
            ),
            (
                lambda: torch.broadcast_shapes((huge, 0), (1, 1)),
                lambda: reference_torch.broadcast_shapes((huge, 0), (1, 1)),
            ),
            (
                lambda: torch.broadcast_shapes((1.0,)),
                lambda: reference_torch.broadcast_shapes((1.0,)),
            ),
            (
                lambda: torch.broadcast_shapes((2,), (2.0,)),
                lambda: reference_torch.broadcast_shapes((2,), (2.0,)),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_calls_match(actual_call, expected_call)

    def test_incompatibility_negative_invalid_element_and_binding_errors_match(self):
        cases = (
            (
                lambda: torch.broadcast_shapes((2,), (3,)),
                lambda: reference_torch.broadcast_shapes((2,), (3,)),
            ),
            (
                lambda: torch.broadcast_shapes((2, 3), [4, 1]),
                lambda: reference_torch.broadcast_shapes((2, 3), [4, 1]),
            ),
            (
                lambda: torch.broadcast_shapes((-1,)),
                lambda: reference_torch.broadcast_shapes((-1,)),
            ),
            (
                lambda: torch.broadcast_shapes(torch.Size([2, -1])),
                lambda: reference_torch.broadcast_shapes(
                    reference_torch.Size([2, -1])
                ),
            ),
            (
                lambda: torch.broadcast_shapes((2,), (-1,)),
                lambda: reference_torch.broadcast_shapes((2,), (-1,)),
            ),
            (
                lambda: torch.broadcast_shapes(3.0),
                lambda: reference_torch.broadcast_shapes(3.0),
            ),
            (
                lambda: torch.broadcast_shapes((2.0,)),
                lambda: reference_torch.broadcast_shapes((2.0,)),
            ),
            (
                lambda: torch.broadcast_shapes(("2",)),
                lambda: reference_torch.broadcast_shapes(("2",)),
            ),
            (
                lambda: torch.broadcast_shapes((None,)),
                lambda: reference_torch.broadcast_shapes((None,)),
            ),
            (
                lambda: torch.broadcast_shapes((np.int64(2),)),
                lambda: reference_torch.broadcast_shapes((np.int64(2),)),
            ),
            (
                lambda: torch.broadcast_shapes(np.int64(2)),
                lambda: reference_torch.broadcast_shapes(np.int64(2)),
            ),
            (
                lambda: torch.broadcast_shapes(shapes=(2, 3)),
                lambda: reference_torch.broadcast_shapes(shapes=(2, 3)),
            ),
            (
                lambda: torch.broadcast_shapes((2,), shape=(2,)),
                lambda: reference_torch.broadcast_shapes((2,), shape=(2,)),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_calls_match(actual_call, expected_call)

    def test_functional_alias_metadata_signature_docs_exports_and_pickle_match(self):
        actual_functional = importlib.import_module("torch_rs.functional")
        expected_functional = importlib.import_module("torch.functional")
        actual = torch.broadcast_shapes
        expected = reference_torch.broadcast_shapes

        self.assertIs(torch.functional, actual_functional)
        self.assertIs(reference_torch.functional, expected_functional)
        self.assertIs(actual, actual_functional.broadcast_shapes)
        self.assertIs(expected, expected_functional.broadcast_shapes)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(
            actual_functional.__all__.count("broadcast_shapes"),
            expected_functional.__all__.count("broadcast_shapes"),
        )
        self.assertEqual(
            torch.__all__.count("broadcast_shapes"),
            reference_torch.__all__.count("broadcast_shapes"),
        )

        actual_top_wildcard = {}
        expected_top_wildcard = {}
        exec("from torch_rs import *", actual_top_wildcard)
        exec("from torch import *", expected_top_wildcard)
        self.assertEqual(
            "broadcast_shapes" in actual_top_wildcard,
            "broadcast_shapes" in expected_top_wildcard,
        )
        actual_functional_wildcard = {}
        expected_functional_wildcard = {}
        exec("from torch_rs.functional import *", actual_functional_wildcard)
        exec("from torch.functional import *", expected_functional_wildcard)
        self.assertIs(actual_functional_wildcard["broadcast_shapes"], actual)
        self.assertIs(expected_functional_wildcard["broadcast_shapes"], expected)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(actual, protocol=protocol)), actual
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(expected, protocol=protocol)), expected
                )

    def test_both_implementations_avoid_tensor_factories(self):
        # Warm the reference helper's lazy imports before replacing public
        # factories with sentinels.
        self.assertEqual(reference_torch.broadcast_shapes((2,), (1,)), (2,))
        self.assertEqual(torch.broadcast_shapes((2,), (1,)), (2,))

        for module in (torch, reference_torch):
            with self.subTest(module=module.__name__), ExitStack() as stack:
                factories = [
                    stack.enter_context(
                        mock.patch.object(
                            module,
                            name,
                            side_effect=AssertionError(f"{name} must not be called"),
                            create=True,
                        )
                    )
                    for name in (
                        "empty",
                        "zeros",
                        "ones",
                        "tensor",
                        "broadcast_tensors",
                    )
                ]
                self.assertEqual(
                    tuple(module.broadcast_shapes((2**100, 0), (1, 1))),
                    (2**100, 0),
                )
                for factory in factories:
                    factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()


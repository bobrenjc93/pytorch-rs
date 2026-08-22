import collections
import copy
import importlib
import inspect
import pickle
import pickletools
import types
import unittest
from collections import namedtuple
from types import MappingProxyType

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class DictSubclass(dict):
    pass


class ListSubclass(list):
    pass


class TupleSubclass(tuple):
    pass


class CopyFailingDict(dict):
    def __copy__(self):
        raise TypeError("copy disabled")


class CopyFailingList(list):
    def __copy__(self):
        raise TypeError("copy disabled")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DefaultConvertReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "default_convert differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def test_leaf_identity_and_recursive_builtin_behavior_match_pytorch_2_13(self):
        Point = namedtuple("Point", ("x", "y"))
        actual_tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        expected_tensor = reference_torch.tensor([1.0, 2.0], requires_grad=True)
        text = "".join(("same", "-text"))
        byte_string = b"same-bytes"

        actual_source = {
            "tensor": actual_tensor,
            "nested": [
                (text, 10**40),
                Point({"value": actual_tensor}, [None, byte_string]),
            ],
        }
        expected_source = {
            "tensor": expected_tensor,
            "nested": [
                (text, 10**40),
                Point({"value": expected_tensor}, [None, byte_string]),
            ],
        }
        actual = torch.utils.data.default_convert(actual_source)
        expected = reference_torch.utils.data.default_convert(expected_source)

        self.assertIs(type(actual), type(expected))
        self.assertIsNot(actual, actual_source)
        self.assertIsNot(expected, expected_source)
        self.assertEqual(list(actual), list(expected))
        self.assertIs(actual["tensor"], actual_tensor)
        self.assertIs(expected["tensor"], expected_tensor)
        np.testing.assert_array_equal(
            np.asarray(actual["tensor"]), expected["tensor"].detach().cpu().numpy()
        )

        actual_nested = actual["nested"]
        expected_nested = expected["nested"]
        self.assertIs(type(actual_nested), type(expected_nested))
        self.assertIsNot(actual_nested, actual_source["nested"])
        self.assertIsNot(expected_nested, expected_source["nested"])
        self.assertIs(type(actual_nested[0]), type(expected_nested[0]))
        self.assertIs(type(actual_nested[0]), list)
        for actual_leaf, expected_leaf, actual_leaf_source, expected_leaf_source in zip(
            actual_nested[0],
            expected_nested[0],
            actual_source["nested"][0],
            expected_source["nested"][0],
            strict=True,
        ):
            self.assertIs(actual_leaf, actual_leaf_source)
            self.assertIs(expected_leaf, expected_leaf_source)

        actual_point = actual_nested[1]
        expected_point = expected_nested[1]
        self.assertIs(type(actual_point), type(expected_point))
        self.assertIs(type(actual_point), Point)
        self.assertIsNot(actual_point, actual_source["nested"][1])
        self.assertIsNot(expected_point, expected_source["nested"][1])
        self.assertIs(type(actual_point.x), type(expected_point.x))
        self.assertIs(type(actual_point.y), type(expected_point.y))
        self.assertIsNot(actual_point.x, actual_source["nested"][1].x)
        self.assertIsNot(actual_point.y, actual_source["nested"][1].y)
        self.assertIs(actual_point.x["value"], actual_tensor)
        self.assertIs(expected_point.x["value"], expected_tensor)
        self.assertEqual(actual_point.y, expected_point.y)

    def test_mutable_copy_and_repeated_reference_behavior_match_pytorch_2_13(self):
        actual_child = [1, {"value": 2}]
        expected_child = [1, {"value": 2}]
        actual_source = [actual_child, actual_child]
        expected_source = [expected_child, expected_child]

        actual = torch.utils.data.default_convert(actual_source)
        expected = reference_torch.utils.data.default_convert(expected_source)

        self.assertEqual(actual, expected)
        for output, source in ((actual, actual_source), (expected, expected_source)):
            self.assertIsNot(output, source)
            self.assertIsNot(output[0], source[0])
            self.assertIsNot(output[1], source[1])
            self.assertIsNot(output[0], output[1])
            self.assertIsNot(output[0][1], source[0][1])
            self.assertIsNot(output[0][1], output[1][1])

    def test_metadata_exports_and_function_copy_match_pytorch_2_13(self):
        actual_data = importlib.import_module("torch_rs.utils.data")
        expected_data = importlib.import_module("torch.utils.data")
        actual_collate = importlib.import_module(
            "torch_rs.utils.data._utils.collate"
        )
        expected_collate = importlib.import_module("torch.utils.data._utils.collate")
        actual = actual_data.default_convert
        expected = expected_data.default_convert

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)

        self.assertIs(actual_data.default_convert, actual_collate.default_convert)
        self.assertIs(expected_data.default_convert, expected_collate.default_convert)
        self.assertIn("default_convert", actual_data.__all__)
        self.assertIn("default_convert", expected_data.__all__)
        self.assertEqual(actual_data.__all__.count("default_convert"), 1)
        self.assertEqual(expected_data.__all__.count("default_convert"), 1)
        self.assertEqual(
            hasattr(actual_collate, "__all__"), hasattr(expected_collate, "__all__")
        )

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(actual, protocol=protocol)), actual
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.utils.data.default_convert
        expected = reference_torch.utils.data.default_convert
        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual(1, 2), lambda: expected(1, 2)),
            (lambda: actual(value=1), lambda: expected(value=1)),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_generic_mapping_and_sequence_behavior_matches_pytorch_2_13(self):
        actual_mapping = DictSubclass(value=(1, 2))
        expected_mapping = DictSubclass(value=(1, 2))
        actual_mapping.marker = object()
        expected_mapping.marker = object()
        actual_sequence = ListSubclass(((1, 2),))
        expected_sequence = ListSubclass(((1, 2),))
        actual_sequence.marker = object()
        expected_sequence.marker = object()

        cases = (
            (
                torch.utils.data.DataChunk([(1, 2), {"value": (3, 4)}]),
                reference_torch.utils.data.DataChunk(
                    [(1, 2), {"value": (3, 4)}]
                ),
            ),
            (actual_mapping, expected_mapping),
            (
                collections.OrderedDict(
                    (("first", (1, 2)), ("second", [3]))
                ),
                collections.OrderedDict(
                    (("first", (1, 2)), ("second", [3]))
                ),
            ),
            (
                MappingProxyType({"value": (1, 2)}),
                MappingProxyType({"value": (1, 2)}),
            ),
            (actual_sequence, expected_sequence),
            (TupleSubclass((1, 2)), TupleSubclass((1, 2))),
            (range(3), range(3)),
            (
                CopyFailingDict(value=(1, 2)),
                CopyFailingDict(value=(1, 2)),
            ),
            (CopyFailingList(((1, 2),)), CopyFailingList(((1, 2),))),
        )

        for actual_source, expected_source in cases:
            with self.subTest(value_type=type(actual_source).__name__):
                actual = torch.utils.data.default_convert(actual_source)
                expected = reference_torch.utils.data.default_convert(expected_source)
                self.assertEqual(actual, expected)
                self.assertEqual(
                    f"{type(actual).__module__}.{type(actual).__qualname__}".replace(
                        "torch_rs", "torch"
                    ),
                    f"{type(expected).__module__}.{type(expected).__qualname__}",
                )
                self.assertIsNot(actual, actual_source)
                self.assertIsNot(expected, expected_source)

                if hasattr(actual_source, "marker"):
                    self.assertIs(actual.marker, actual_source.marker)
                    self.assertIs(expected.marker, expected_source.marker)
                if hasattr(actual, "raw_iterator"):
                    self.assertEqual(
                        list(actual.raw_iterator()), list(expected.raw_iterator())
                    )

    def test_numpy_boundary_and_unsupported_neighbors_remain_deliberate(self):
        for value in (np.array([1, 2]), np.int64(3), np.str_("text")):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaisesRegex(NotImplementedError, "NumPy"):
                    torch.utils.data.default_convert(value)
                reference_torch.utils.data.default_convert(value)

        actual_data = importlib.import_module("torch_rs.utils.data")
        actual_collate = importlib.import_module(
            "torch_rs.utils.data._utils.collate"
        )
        self.assertFalse(hasattr(actual_data, "default_collate"))
        self.assertFalse(hasattr(actual_data, "DataLoader"))
        self.assertFalse(hasattr(actual_collate, "default_collate"))


if __name__ == "__main__":
    unittest.main()

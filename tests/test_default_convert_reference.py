import copy
import importlib
import inspect
import pickle
import pickletools
import types
import unittest
from collections import OrderedDict, UserDict, UserList, namedtuple

import numpy as np

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


Point = namedtuple("Point", ("x", "y"))


class DictSubclass(dict):
    pass


class ListSubclass(list):
    pass


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

    def outcome_shape(self, value, tensor):
        if value is tensor:
            return ("tensor",)
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return (
                "namedtuple",
                type(value).__name__,
                tuple(self.outcome_shape(item, tensor) for item in value),
            )
        if type(value) is list:
            return ("list", tuple(self.outcome_shape(item, tensor) for item in value))
        if type(value) is dict:
            return (
                "dict",
                tuple(
                    (key, self.outcome_shape(item, tensor))
                    for key, item in value.items()
                ),
            )
        return ("leaf", type(value).__name__, value)

    def make_supported_case(self, module):
        tensor = module.tensor([1.0, 2.0])
        shared = [tensor, "leaf"]
        point = Point({"tensor": tensor}, (True, b"bytes"))
        source = {
            "shared": [shared, shared],
            "tuple": (1, "two", tensor),
            "point": point,
            "empty_list": [],
            "empty_dict": {},
            "empty_tuple": (),
        }
        return source, tensor

    def test_supported_recursive_behavior_matches_pytorch_2_13(self):
        actual_source, actual_tensor = self.make_supported_case(torch)
        expected_source, expected_tensor = self.make_supported_case(reference_torch)

        actual = torch.utils.data.default_convert(actual_source)
        expected = reference_torch.utils.data.default_convert(expected_source)

        self.assertEqual(
            self.outcome_shape(actual, actual_tensor),
            self.outcome_shape(expected, expected_tensor),
        )
        self.assertIs(actual["shared"][0][0], actual_tensor)
        self.assertIs(expected["shared"][0][0], expected_tensor)
        self.assertIsNot(actual, actual_source)
        self.assertIsNot(expected, expected_source)
        self.assertIsNot(actual["shared"], actual_source["shared"])
        self.assertIsNot(expected["shared"], expected_source["shared"])
        self.assertIsNot(actual["shared"][0], actual["shared"][1])
        self.assertIsNot(expected["shared"][0], expected["shared"][1])
        self.assertIs(type(actual["point"]), type(expected["point"]))
        self.assertIs(type(actual["tuple"]), type(expected["tuple"]))

        actual_leaves = (actual_tensor, None, True, 257, -3.5, "text", b"bytes")
        expected_leaves = (
            expected_tensor,
            None,
            True,
            257,
            -3.5,
            "text",
            b"bytes",
        )
        for actual_leaf, expected_leaf in zip(
            actual_leaves, expected_leaves, strict=True
        ):
            with self.subTest(leaf_type=type(actual_leaf).__name__):
                self.assertIs(
                    torch.utils.data.default_convert(actual_leaf), actual_leaf
                )
                self.assertIs(
                    reference_torch.utils.data.default_convert(expected_leaf),
                    expected_leaf,
                )

    def test_signature_annotations_documentation_and_metadata_match(self):
        actual = torch.utils.data.default_convert
        expected = reference_torch.utils.data.default_convert

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
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
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

    def test_argument_forms_and_errors_match_pytorch_2_13(self):
        actual = torch.utils.data.default_convert
        expected = reference_torch.utils.data.default_convert
        actual_leaf = object()
        expected_leaf = object()
        self.assertIs(actual(data=actual_leaf), actual_leaf)
        self.assertIs(expected(data=expected_leaf), expected_leaf)

        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(value=None), lambda: expected(value=None)),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_exports_and_intentionally_unsupported_neighbors(self):
        actual_data = importlib.import_module("torch_rs.utils.data")
        expected_data = importlib.import_module("torch.utils.data")
        actual_collate = importlib.import_module(
            "torch_rs.utils.data._utils.collate"
        )
        expected_collate = importlib.import_module("torch.utils.data._utils.collate")

        self.assertIs(actual_data.default_convert, actual_collate.default_convert)
        self.assertIs(expected_data.default_convert, expected_collate.default_convert)
        self.assertEqual(
            actual_data.default_convert.__module__.replace("torch_rs", "torch"),
            expected_data.default_convert.__module__,
        )
        self.assertEqual(actual_data.__all__.count("default_convert"), 1)
        self.assertEqual(expected_data.__all__.count("default_convert"), 1)

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.utils.data import *", actual_namespace)
        exec("from torch.utils.data import *", expected_namespace)
        self.assertIs(
            actual_namespace["default_convert"], actual_data.default_convert
        )
        self.assertIs(
            expected_namespace["default_convert"], expected_data.default_convert
        )

        self.assertFalse(hasattr(actual_data, "default_collate"))
        self.assertFalse(hasattr(actual_data, "DataLoader"))
        self.assertFalse(hasattr(actual_collate, "default_collate"))
        self.assertFalse(hasattr(actual_collate, "collate"))
        self.assertTrue(hasattr(expected_data, "default_collate"))
        self.assertTrue(hasattr(expected_data, "DataLoader"))
        self.assertTrue(hasattr(expected_collate, "default_collate"))
        self.assertTrue(hasattr(expected_collate, "collate"))

    def test_numpy_behavior_is_an_explicit_supported_scope_difference(self):
        actual = torch.utils.data.default_convert
        expected = reference_torch.utils.data.default_convert
        values = (
            np.array([1, 2]),
            np.array(["a", "b"]),
            np.array([object()], dtype=object),
            np.int64(1),
            np.float32(1.5),
            np.str_("text"),
        )
        for value in values:
            with self.subTest(value_type=type(value)):
                with self.assertRaisesRegex(
                    TypeError,
                    "^default_convert does not support NumPy arrays or scalars$",
                ):
                    actual(value)
                expected(value)

    def test_exotic_collections_remain_outside_the_supported_scope(self):
        actual = torch.utils.data.default_convert
        expected = reference_torch.utils.data.default_convert
        values = (
            DictSubclass(value=1),
            OrderedDict(value=1),
            UserDict(value=1),
            ListSubclass([1]),
            UserList([1]),
            range(2),
            bytearray(b"a"),
        )
        for value in values:
            with self.subTest(value_type=type(value)):
                with self.assertRaisesRegex(
                    TypeError,
                    "^default_convert only supports built-in dict, list, tuple, "
                    "and namedtuple containers$",
                ):
                    actual(value)
                expected_result = expected(value)
                self.assertIsNot(expected_result, value)

    def test_pickle_and_copy_behavior_matches_pytorch_2_13(self):
        actual = torch.utils.data.default_convert
        expected = reference_torch.utils.data.default_convert

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(actual, protocol=protocol)), actual
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(expected, protocol=protocol)), expected
                )
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )


if __name__ == "__main__":
    unittest.main()

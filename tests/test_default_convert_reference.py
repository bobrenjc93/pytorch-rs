import copy
import importlib
import inspect
import pickle
import pickletools
import types
import unittest
from collections import OrderedDict, UserDict, UserList, deque, namedtuple
from collections.abc import Mapping, Sequence
from types import MappingProxyType

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


class TupleSubclass(tuple):
    pass


class ConstructibleSequence(Sequence):
    def __init__(self, values):
        self.values = tuple(values)

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)


class ConstructorRejectingMapping(Mapping):
    def __init__(self):
        self.data = {"value": (1, 2)}

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class ConstructorRejectingSequence(Sequence):
    def __init__(self):
        self.values = ((1, 2),)

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)


class CopyRejectingMutableMapping(UserDict):
    def __copy__(self):
        raise TypeError("copy disabled")


class CopyRejectingMutableSequence(UserList):
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

    def make_collection_cases(self, module):
        dict_subclass = DictSubclass(value=(1, 2))
        dict_subclass.marker = object()
        list_subclass = ListSubclass([(1, 2)])
        list_subclass.marker = object()
        return {
            "data_chunk": module.utils.data.DataChunk([(1, 2)]),
            "dict_subclass": dict_subclass,
            "ordered_dict": OrderedDict(value=(1, 2)),
            "user_dict": UserDict(value=(1, 2)),
            "mapping_proxy": MappingProxyType({"value": (1, 2)}),
            "list_subclass": list_subclass,
            "user_list": UserList([(1, 2)]),
            "constructible_sequence": ConstructibleSequence([(1, 2)]),
            "tuple_subclass": TupleSubclass(((1, 2),)),
            "range_fallback": range(2),
            "deque": deque([(1, 2)]),
            "bytearray": bytearray(b"a"),
            "mapping_copy_fallback": CopyRejectingMutableMapping(
                value=(1, 2)
            ),
            "mapping_constructor_fallback": ConstructorRejectingMapping(),
            "sequence_copy_fallback": CopyRejectingMutableSequence([(1, 2)]),
            "sequence_constructor_fallback": ConstructorRejectingSequence(),
        }

    def collection_shape(self, value):
        value_type = type(value)
        type_name = (
            value_type.__module__.replace("torch_rs", "torch"),
            value_type.__name__,
        )
        if isinstance(value, Mapping):
            return (
                "mapping",
                type_name,
                tuple(
                    (key, self.collection_shape(item))
                    for key, item in value.items()
                ),
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return (
                "sequence",
                type_name,
                tuple(self.collection_shape(item) for item in value),
            )
        return ("leaf", type_name, value)

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

    def test_mapping_and_sequence_copy_fallbacks_match_pytorch_2_13(self):
        actual = torch.utils.data.default_convert
        expected = reference_torch.utils.data.default_convert
        actual_cases = self.make_collection_cases(torch)
        expected_cases = self.make_collection_cases(reference_torch)

        actual_results = {}
        expected_results = {}
        for name in actual_cases:
            with self.subTest(name=name):
                actual_result = actual(actual_cases[name])
                expected_result = expected(expected_cases[name])
                actual_results[name] = actual_result
                expected_results[name] = expected_result
                self.assertEqual(
                    self.collection_shape(actual_result),
                    self.collection_shape(expected_result),
                )
                self.assertIsNot(actual_result, actual_cases[name])
                self.assertIsNot(expected_result, expected_cases[name])

        for name in ("dict_subclass", "list_subclass"):
            self.assertIs(
                actual_results[name].marker, actual_cases[name].marker
            )
            self.assertIs(
                expected_results[name].marker, expected_cases[name].marker
            )

        actual_chunk = actual_results["data_chunk"]
        expected_chunk = expected_results["data_chunk"]
        self.assertIs(actual_chunk.items, actual_cases["data_chunk"].items)
        self.assertIs(expected_chunk.items, expected_cases["data_chunk"].items)
        self.assertEqual(
            self.collection_shape(list(actual_chunk.raw_iterator())),
            self.collection_shape(list(expected_chunk.raw_iterator())),
        )

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

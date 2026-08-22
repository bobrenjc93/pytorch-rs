import copy
import importlib
import inspect
import pickle
import types
import unittest
from collections import OrderedDict, UserDict, UserList, namedtuple
from collections.abc import Mapping

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


Point = namedtuple("Point", ("x", "y"))


class FancyDict(dict):
    pass


class FancyList(list):
    pass


class FancyTuple(tuple):
    pass


class CopyRejectingDict(dict):
    def __copy__(self):
        raise TypeError("copy is unavailable")


class CopyRejectingList(list):
    def __copy__(self):
        raise TypeError("copy is unavailable")


class ConstructorRejectingMapping(Mapping):
    def __init__(self, values, *, label):
        self.values = dict(values)
        self.label = label

    def __getitem__(self, key):
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


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

    def test_leaf_identity_and_recursive_builtin_copying_match(self):
        actual_tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        expected_tensor = reference_torch.tensor([1.0, 2.0], requires_grad=True)
        marker = object()
        text = "".join(("torch", "_rs"))
        leaves = (None, True, 12345678901234567890, 1.25, 2.0 - 3.0j, text, marker)

        for leaf in leaves:
            with self.subTest(type=type(leaf).__name__):
                self.assertIs(torch.utils.data.default_convert(leaf), leaf)
                self.assertIs(reference_torch.utils.data.default_convert(leaf), leaf)

        self.assertIs(torch.utils.data.default_convert(actual_tensor), actual_tensor)
        self.assertIs(
            reference_torch.utils.data.default_convert(expected_tensor),
            expected_tensor,
        )

        actual_point = Point(actual_tensor, {"marker": marker})
        expected_point = Point(expected_tensor, {"marker": marker})
        actual_source = {
            "items": [actual_tensor, (marker, text)],
            "point": actual_point,
            "empty": (),
        }
        expected_source = {
            "items": [expected_tensor, (marker, text)],
            "point": expected_point,
            "empty": (),
        }
        actual = torch.utils.data.default_convert(actual_source)
        expected = reference_torch.utils.data.default_convert(expected_source)

        self.assertIs(type(actual), type(expected))
        self.assertIsNot(actual, actual_source)
        self.assertIsNot(expected, expected_source)
        self.assertIs(type(actual["items"]), type(expected["items"]))
        self.assertIsNot(actual["items"], actual_source["items"])
        self.assertIsNot(expected["items"], expected_source["items"])
        self.assertIs(actual["items"][0], actual_tensor)
        self.assertIs(expected["items"][0], expected_tensor)
        self.assertIs(type(actual["items"][1]), type(expected["items"][1]))
        self.assertEqual(type(actual["items"][1]), list)
        self.assertIs(actual["items"][1][0], marker)
        self.assertIs(expected["items"][1][0], marker)
        self.assertIs(actual["items"][1][1], text)
        self.assertIs(expected["items"][1][1], text)

        self.assertIs(type(actual["point"]), type(expected["point"]))
        self.assertIs(type(actual["point"]), Point)
        self.assertIsNot(actual["point"], actual_point)
        self.assertIsNot(expected["point"], expected_point)
        self.assertIs(actual["point"].x, actual_tensor)
        self.assertIs(expected["point"].x, expected_tensor)
        self.assertIsNot(actual["point"].y, actual_point.y)
        self.assertIsNot(expected["point"].y, expected_point.y)
        self.assertIs(actual["point"].y["marker"], marker)
        self.assertIs(expected["point"].y["marker"], marker)
        self.assertEqual(actual["empty"], expected["empty"])
        self.assertIs(type(actual["empty"]), list)

    def test_signature_annotations_documentation_metadata_and_exports_match(self):
        actual_data = importlib.import_module("torch_rs.utils.data")
        expected_data = importlib.import_module("torch.utils.data")
        actual_collate = importlib.import_module("torch_rs.utils.data._utils.collate")
        expected_collate = importlib.import_module("torch.utils.data._utils.collate")
        actual = actual_data.default_convert
        expected = expected_data.default_convert

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
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

        self.assertIs(actual_data.default_convert, actual_collate.default_convert)
        self.assertIs(expected_data.default_convert, expected_collate.default_convert)
        self.assertFalse(hasattr(actual_collate, "__all__"))
        self.assertEqual(
            hasattr(actual_collate, "__all__"),
            hasattr(expected_collate, "__all__"),
        )

        supported = {
            "BatchSampler",
            "ChainDataset",
            "ConcatDataset",
            "DataChunk",
            "Dataset",
            "DistributedSampler",
            "IterableDataset",
            "Sampler",
            "SequentialSampler",
            "StackDataset",
            "Subset",
            "TensorDataset",
            "default_convert",
            "get_worker_info",
        }
        self.assertEqual(
            actual_data.__all__,
            [name for name in expected_data.__all__ if name in supported],
        )
        actual_namespace = {}
        exec("from torch_rs.utils.data import *", actual_namespace)
        self.assertIs(actual_namespace["default_convert"], actual)

        for unsupported in ("DataLoader", "default_collate"):
            with self.subTest(unsupported=unsupported):
                self.assertFalse(hasattr(actual_data, unsupported))
        self.assertFalse(hasattr(actual_collate, "default_collate"))

    def test_call_diagnostics_copy_and_pickle_match(self):
        actual = torch.utils.data.default_convert
        expected = reference_torch.utils.data.default_convert
        marker = object()
        self.assertIs(actual(data=marker), marker)
        self.assertIs(expected(data=marker), marker)

        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(value=None), lambda: expected(value=None)),
        )
        for actual_call, expected_call in cases:
            self.assert_error_matches(actual_call, expected_call)

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)

    def test_numpy_rejection_is_the_deliberate_non_numpy_boundary(self):
        actual = torch.utils.data.default_convert
        expected = reference_torch.utils.data.default_convert
        numeric_array = np.arange(3, dtype=np.float32)
        numeric_scalar = np.int64(4)
        string_scalar = np.str_("value")

        expected_array = expected(numeric_array)
        expected_scalar = expected(numeric_scalar)
        self.assertIsInstance(expected_array, reference_torch.Tensor)
        self.assertIsInstance(expected_scalar, reference_torch.Tensor)
        self.assertIs(expected(string_scalar), string_scalar)

        for value in (numeric_array, numeric_scalar, string_scalar):
            with self.subTest(type=type(value).__name__):
                with self.assertRaisesRegex(
                    TypeError,
                    "^default_convert\\(\\): NumPy arrays and scalars are not "
                    "supported$",
                ):
                    actual(value)

    def test_generic_mapping_copy_constructor_and_fallback_behavior_matches(self):
        actual = torch.utils.data.default_convert
        expected = reference_torch.utils.data.default_convert

        pairs = (
            (
                OrderedDict(first=(1, 2), second=[3, 4]),
                OrderedDict(first=(1, 2), second=[3, 4]),
            ),
            (UserDict(value=(5, 6)), UserDict(value=(5, 6))),
            (FancyDict(value=(7, 8)), FancyDict(value=(7, 8))),
            (
                types.MappingProxyType({"value": (9, 10)}),
                types.MappingProxyType({"value": (9, 10)}),
            ),
            (
                CopyRejectingDict(value=(11, 12)),
                CopyRejectingDict(value=(11, 12)),
            ),
            (
                ConstructorRejectingMapping({"value": (13, 14)}, label="actual"),
                ConstructorRejectingMapping({"value": (13, 14)}, label="expected"),
            ),
        )
        for actual_source, expected_source in pairs:
            with self.subTest(type=type(actual_source).__name__):
                actual_result = actual(actual_source)
                expected_result = expected(expected_source)
                self.assertIs(type(actual_result), type(expected_result))
                self.assertEqual(dict(actual_result), dict(expected_result))
                self.assertEqual(
                    actual_result is actual_source,
                    expected_result is expected_source,
                )

    def test_generic_sequence_copy_constructor_and_fallback_behavior_matches(self):
        actual = torch.utils.data.default_convert
        expected = reference_torch.utils.data.default_convert
        actual_chunk = torch.utils.data.DataChunk([7, (8, 9)])
        expected_chunk = reference_torch.utils.data.DataChunk([7, (8, 9)])

        pairs = (
            (UserList([1, (2, 3)]), UserList([1, (2, 3)])),
            (FancyList([4, (5, 6)]), FancyList([4, (5, 6)])),
            (actual_chunk, expected_chunk),
            (FancyTuple((10, (11, 12))), FancyTuple((10, (11, 12)))),
            (range(3), range(3)),
            (
                CopyRejectingList([13, (14, 15)]),
                CopyRejectingList([13, (14, 15)]),
            ),
        )
        for actual_source, expected_source in pairs:
            with self.subTest(type=type(actual_source).__name__):
                actual_result = actual(actual_source)
                expected_result = expected(expected_source)
                self.assertEqual(
                    type(actual_result).__name__, type(expected_result).__name__
                )
                self.assertEqual(list(actual_result), list(expected_result))
                self.assertEqual(
                    actual_result is actual_source,
                    expected_result is expected_source,
                )

        actual_converted_chunk = actual(actual_chunk)
        expected_converted_chunk = expected(expected_chunk)
        self.assertEqual(
            list(actual_converted_chunk.raw_iterator()),
            list(expected_converted_chunk.raw_iterator()),
        )
        self.assertEqual(
            actual_converted_chunk.items is actual_chunk.items,
            expected_converted_chunk.items is expected_chunk.items,
        )

    def test_numpy_rejection_reaches_generic_mapping_and_sequence_contents(self):
        actual = torch.utils.data.default_convert
        expected = reference_torch.utils.data.default_convert
        array = np.arange(3, dtype=np.float32)
        cases = (
            (
                OrderedDict(payload=array),
                OrderedDict(payload=array),
                lambda result: result["payload"],
            ),
            (
                UserDict(payload=array),
                UserDict(payload=array),
                lambda result: result["payload"],
            ),
            (
                UserList([array]),
                UserList([array]),
                lambda result: result[0],
            ),
            (
                torch.utils.data.DataChunk([array]),
                reference_torch.utils.data.DataChunk([array]),
                lambda result: result[0],
            ),
            (
                types.MappingProxyType({"payload": array}),
                types.MappingProxyType({"payload": array}),
                lambda result: result["payload"],
            ),
        )

        for actual_source, expected_source, extract in cases:
            with self.subTest(type=type(actual_source).__name__):
                expected_leaf = extract(expected(expected_source))
                self.assertIsInstance(expected_leaf, reference_torch.Tensor)
                with self.assertRaisesRegex(
                    TypeError,
                    "^default_convert\\(\\): NumPy arrays and scalars are not "
                    "supported$",
                ):
                    actual(actual_source)


if __name__ == "__main__":
    unittest.main()

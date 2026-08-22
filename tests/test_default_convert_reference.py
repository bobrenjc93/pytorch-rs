import copy
import importlib
import inspect
import pickle
import types
import unittest
from collections import namedtuple

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


Point = namedtuple("Point", ("x", "y"))
NUMPY_ERROR = (
    "torch_rs.utils.data.default_convert does not support NumPy arrays or scalars"
)


class ListSubclass(list):
    pass


class DictSubclass(dict):
    pass


class TupleSubclass(tuple):
    pass


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DefaultConvertReferenceTests(unittest.TestCase):
    @staticmethod
    def conversion_shape(module):
        tensor = module.tensor([1.0, 2.0], requires_grad=True)
        marker = object()
        key = object()
        source = {
            key: [tensor, marker],
            "tuple": (marker, [tensor]),
            "point": Point((tensor,), {"marker": marker}),
            "empty": (),
        }
        converted = module.utils.data.default_convert(source)
        point = converted["point"]
        converted_key = next(item for item in converted if item is key)
        return {
            "root_type": type(converted).__name__,
            "root_is_copy": converted is not source,
            "key_identity": converted_key is key,
            "list_type": type(converted[key]).__name__,
            "list_is_copy": converted[key] is not source[key],
            "tensor_identity": converted[key][0] is tensor,
            "marker_identity": converted[key][1] is marker,
            "tuple_type": type(converted["tuple"]).__name__,
            "nested_list_is_copy": converted["tuple"][1]
            is not source["tuple"][1],
            "namedtuple_type": type(point).__name__,
            "namedtuple_is_copy": point is not source["point"],
            "namedtuple_tuple_type": type(point.x).__name__,
            "namedtuple_dict_type": type(point.y).__name__,
            "empty_tuple_type": type(converted["empty"]).__name__,
            "source_tuple_unchanged": type(source["tuple"]).__name__,
        }

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def test_builtin_recursive_and_identity_behavior_matches_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        self.assertEqual(
            self.conversion_shape(torch), self.conversion_shape(reference_torch)
        )

        for module in (torch, reference_torch):
            tensor = module.tensor(1.0)
            leaves = (tensor, None, True, 17, 2.5, 3 + 4j, "text", b"bytes")
            for value in leaves:
                with self.subTest(module=module.__name__, value=type(value).__name__):
                    self.assertIs(module.utils.data.default_convert(value), value)

    def test_signature_documentation_metadata_and_exports_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_data = importlib.import_module("torch_rs.utils.data")
        expected_data = importlib.import_module("torch.utils.data")
        actual_module = importlib.import_module(
            "torch_rs.utils.data._utils.collate"
        )
        expected_module = importlib.import_module("torch.utils.data._utils.collate")
        actual = actual_data.default_convert
        expected = expected_data.default_convert
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

        self.assertIs(actual, actual_module.default_convert)
        self.assertIs(expected, expected_module.default_convert)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual_module.__doc__, expected_module.__doc__)
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(
            actual_data.__all__,
            [name for name in expected_data.__all__ if name in supported],
        )
        self.assertEqual(
            hasattr(actual_module, "__all__"), hasattr(expected_module, "__all__")
        )

        namespace = {}
        exec("from torch_rs.utils.data import *", namespace)
        self.assertIs(namespace["default_convert"], actual)

        self.assertFalse(hasattr(actual_data, "default_collate"))
        self.assertFalse(hasattr(actual_data, "DataLoader"))
        self.assertFalse(hasattr(actual_module, "default_collate"))

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.utils.data.default_convert
        expected = reference_torch.utils.data.default_convert
        marker = object()
        self.assertIs(actual(data=marker), marker)
        self.assertIs(expected(data=marker), marker)

        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (
                lambda: actual(value=None),
                lambda: expected(value=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_copy_and_pickle_behavior_matches_pytorch_2_13(self):
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

    def test_numpy_conversion_is_the_deliberately_unsupported_boundary(self):
        actual = torch.utils.data.default_convert
        expected = reference_torch.utils.data.default_convert
        numeric_values = (
            np.array([1, 2], dtype=np.int64),
            np.array(3.5, dtype=np.float32),
            np.int64(7),
            np.float32(2.5),
        )
        for value in numeric_values:
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(TypeError) as raised:
                    actual(value)
                self.assertEqual(raised.exception.args, (NUMPY_ERROR,))
                self.assertIsInstance(expected(value), reference_torch.Tensor)

        passthrough_values = (
            np.array(["text"], dtype=object),
            np.str_("text"),
        )
        for value in passthrough_values:
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(TypeError) as raised:
                    actual(value)
                self.assertEqual(raised.exception.args, (NUMPY_ERROR,))
                self.assertIs(expected(value), value)

    def test_exotic_collection_subclasses_are_deliberately_unsupported(self):
        actual = torch.utils.data.default_convert
        expected = reference_torch.utils.data.default_convert
        factories = (
            lambda: ListSubclass([(1, 2)]),
            lambda: DictSubclass(value=(1, 2)),
            lambda: TupleSubclass(((1, 2),)),
        )

        for factory in factories:
            actual_value = factory()
            expected_value = factory()
            with self.subTest(value_type=type(actual_value).__name__):
                self.assertIs(actual(actual_value), actual_value)
                self.assertIsNot(expected(expected_value), expected_value)


if __name__ == "__main__":
    unittest.main()

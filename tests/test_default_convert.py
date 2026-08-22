import collections
import copy
import importlib
import inspect
import pickle
import types
import unittest
from collections import namedtuple
from types import MappingProxyType

import numpy as np
import torch_rs as torch

from torch_rs.utils.data import DataChunk, default_convert


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


class ArraySubclass(np.ndarray):
    pass


class DefaultConvertTests(unittest.TestCase):
    def test_tensors_scalars_strings_and_other_leaves_preserve_identity(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        tensor_view = tensor[1]
        leaves = (
            tensor,
            tensor_view,
            None,
            True,
            17,
            2.5,
            3 + 4j,
            "text",
            b"bytes",
            bytearray(b"mutable bytes"),
            range(4),
            object(),
        )

        for value in leaves:
            with self.subTest(value_type=type(value).__name__):
                self.assertIs(default_convert(value), value)

    def test_builtin_containers_are_recursively_converted_without_mutation(self):
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        marker = object()
        key = object()
        shared = [tensor, marker]
        source = {
            key: shared,
            "tuple": (marker, [tensor]),
            "point": Point((tensor,), {"marker": marker}),
            "empty_dict": {},
            "empty_list": [],
            "empty_tuple": (),
            "repeated": [shared, shared],
        }

        converted = default_convert(source)

        self.assertIs(type(converted), dict)
        self.assertIsNot(converted, source)
        self.assertIn(key, converted)
        self.assertIs(next(item for item in converted if item is key), key)

        converted_shared = converted[key]
        self.assertIs(type(converted_shared), list)
        self.assertIsNot(converted_shared, shared)
        self.assertIs(converted_shared[0], tensor)
        self.assertIs(converted_shared[1], marker)

        self.assertIs(type(converted["tuple"]), list)
        self.assertIs(converted["tuple"][0], marker)
        self.assertIs(type(converted["tuple"][1]), list)
        self.assertIsNot(converted["tuple"][1], source["tuple"][1])
        self.assertIs(converted["tuple"][1][0], tensor)

        point = converted["point"]
        self.assertIs(type(point), Point)
        self.assertIsNot(point, source["point"])
        self.assertIs(type(point.x), list)
        self.assertIs(point.x[0], tensor)
        self.assertIs(type(point.y), dict)
        self.assertIsNot(point.y, source["point"].y)
        self.assertIs(point.y["marker"], marker)

        self.assertEqual(converted["empty_dict"], {})
        self.assertIsNot(converted["empty_dict"], source["empty_dict"])
        self.assertEqual(converted["empty_list"], [])
        self.assertIsNot(converted["empty_list"], source["empty_list"])
        self.assertEqual(converted["empty_tuple"], [])
        self.assertIs(type(converted["empty_tuple"]), list)

        repeated = converted["repeated"]
        self.assertIsNot(repeated, source["repeated"])
        self.assertIsNot(repeated[0], shared)
        self.assertIsNot(repeated[1], shared)
        self.assertIsNot(repeated[0], repeated[1])
        self.assertIs(repeated[0][0], tensor)
        self.assertIs(repeated[1][1], marker)

        self.assertIs(type(source["tuple"]), tuple)
        self.assertIs(type(source["point"].x), tuple)
        self.assertIs(source[key], shared)
        self.assertIs(source["repeated"][0], shared)
        self.assertIs(source["repeated"][1], shared)

    def test_numpy_arrays_and_scalars_are_rejected_without_importing_numpy(self):
        values = (
            np.array([1, 2], dtype=np.int64),
            np.array(3.5, dtype=np.float32),
            np.array(["text"], dtype=object),
            np.int64(7),
            np.float32(2.5),
            np.bool_(True),
            np.str_("text"),
            np.bytes_("bytes"),
            np.arange(3).view(ArraySubclass),
        )

        for value in values:
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(TypeError) as raised:
                    default_convert(value)
                self.assertEqual(raised.exception.args, (NUMPY_ERROR,))

        nested = {"values": [np.int64(1)]}
        with self.assertRaises(TypeError) as raised:
            default_convert(nested)
        self.assertEqual(raised.exception.args, (NUMPY_ERROR,))
        self.assertIs(type(nested["values"]), list)
        self.assertIs(type(nested["values"][0]), np.int64)

        dtype = np.dtype("float32")
        self.assertIs(default_convert(dtype), dtype)

    def test_exotic_collection_subclasses_remain_unsupported_and_unchanged(self):
        nested_tuple = (1, 2)
        values = (
            ListSubclass([nested_tuple]),
            DictSubclass(value=nested_tuple),
            TupleSubclass((nested_tuple,)),
            collections.OrderedDict(value=nested_tuple),
            collections.defaultdict(list, value=nested_tuple),
            collections.UserList([nested_tuple]),
            collections.UserDict(value=nested_tuple),
            collections.deque([nested_tuple]),
            MappingProxyType({"value": nested_tuple}),
            DataChunk([nested_tuple]),
        )

        for value in values:
            with self.subTest(value_type=type(value).__name__):
                self.assertIs(default_convert(value), value)

    def test_signature_metadata_exports_and_unsupported_neighbors(self):
        data_module = importlib.import_module("torch_rs.utils.data")
        collate_module = importlib.import_module(
            "torch_rs.utils.data._utils.collate"
        )

        self.assertIs(torch.utils.data, data_module)
        self.assertIs(data_module.default_convert, default_convert)
        self.assertIs(collate_module.default_convert, default_convert)
        self.assertIs(type(default_convert), types.FunctionType)
        self.assertEqual(str(inspect.signature(default_convert)), "(data)")
        self.assertEqual(default_convert.__annotations__, {})
        self.assertEqual(
            default_convert.__module__, "torch_rs.utils.data._utils.collate"
        )
        self.assertEqual(default_convert.__name__, "default_convert")
        self.assertEqual(default_convert.__qualname__, "default_convert")
        self.assertIsNone(default_convert.__defaults__)
        self.assertIsNone(default_convert.__kwdefaults__)
        self.assertEqual(default_convert.__dict__, {})
        self.assertFalse(hasattr(default_convert, "__text_signature__"))
        self.assertFalse(hasattr(collate_module, "__all__"))
        self.assertEqual(
            data_module.__all__,
            [
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
            ],
        )
        self.assertEqual(data_module.__all__.count("default_convert"), 1)

        namespace = {}
        exec("from torch_rs.utils.data import *", namespace)
        self.assertIs(namespace["default_convert"], default_convert)

        self.assertFalse(hasattr(data_module, "default_collate"))
        self.assertFalse(hasattr(data_module, "DataLoader"))
        self.assertFalse(hasattr(collate_module, "default_collate"))

    def test_argument_errors_and_keyword_form(self):
        marker = object()
        self.assertIs(default_convert(data=marker), marker)

        cases = (
            (
                lambda: default_convert(),
                "default_convert() missing 1 required positional argument: 'data'",
            ),
            (
                lambda: default_convert(None, None),
                "default_convert() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: default_convert(value=None),
                "default_convert() got an unexpected keyword argument 'value'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(raised.exception.args, (message,))

    def test_copy_and_pickle_preserve_the_canonical_function(self):
        self.assertIs(copy.copy(default_convert), default_convert)
        self.assertIs(copy.deepcopy(default_convert), default_convert)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(
                    pickle.dumps(default_convert, protocol=protocol)
                )
                self.assertIs(restored, default_convert)


if __name__ == "__main__":
    unittest.main()

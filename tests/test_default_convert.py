import collections
import copy
import importlib
import inspect
import pickle
import types
import unittest
from collections import namedtuple
from collections.abc import Mapping, Sequence
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


class CopyFailingDict(dict):
    def __copy__(self):
        raise TypeError("copy is unavailable")


class CopyFailingList(list):
    def __copy__(self):
        raise TypeError("copy is unavailable")


class UpdateFailingDict(dict):
    def update(self, *args, **kwargs):
        raise TypeError("update is unavailable")


class SetItemFailingList(list):
    def __setitem__(self, key, value):
        raise TypeError("item replacement is unavailable")


class ConstructorFailingMapping(Mapping):
    def __init__(self):
        self.values = {"value": (1, 2)}

    def __getitem__(self, key):
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


class ConstructorFailingSequence(Sequence):
    def __init__(self):
        self.values = ((1, 2),)

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)


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

        wrapped_values = (
            ListSubclass([np.int64(1)]),
            DictSubclass(value=np.float32(2.5)),
            collections.UserList([np.array([1, 2])]),
            collections.UserDict(value=np.array(3.5)),
            DataChunk([np.bool_(True)]),
        )
        for value in wrapped_values:
            with self.subTest(wrapper_type=type(value).__name__):
                with self.assertRaises(TypeError) as raised:
                    default_convert(value)
                self.assertEqual(raised.exception.args, (NUMPY_ERROR,))

        dtype = np.dtype("float32")
        self.assertIs(default_convert(dtype), dtype)

    def test_mapping_and_sequence_types_follow_pytorch_copying_rules(self):
        nested_tuple = (1, 2)
        marker = object()

        list_subclass = ListSubclass([nested_tuple])
        list_subclass.marker = marker
        converted_list_subclass = default_convert(list_subclass)
        self.assertIs(type(converted_list_subclass), ListSubclass)
        self.assertIsNot(converted_list_subclass, list_subclass)
        self.assertEqual(converted_list_subclass, [[1, 2]])
        self.assertEqual(list_subclass, [nested_tuple])
        self.assertIs(converted_list_subclass.marker, marker)

        dict_subclass = DictSubclass(value=nested_tuple)
        dict_subclass.marker = marker
        converted_dict_subclass = default_convert(dict_subclass)
        self.assertIs(type(converted_dict_subclass), DictSubclass)
        self.assertIsNot(converted_dict_subclass, dict_subclass)
        self.assertEqual(converted_dict_subclass, {"value": [1, 2]})
        self.assertEqual(dict_subclass, {"value": nested_tuple})
        self.assertIs(converted_dict_subclass.marker, marker)

        ordered = collections.OrderedDict(
            (("first", nested_tuple), ("last", (3,)))
        )
        converted_ordered = default_convert(ordered)
        self.assertIs(type(converted_ordered), collections.OrderedDict)
        self.assertIsNot(converted_ordered, ordered)
        self.assertEqual(
            list(converted_ordered.items()),
            [("first", [1, 2]), ("last", [3])],
        )
        self.assertEqual(
            list(ordered.items()), [("first", nested_tuple), ("last", (3,))]
        )

        defaulted = collections.defaultdict(list, value=nested_tuple)
        converted_defaulted = default_convert(defaulted)
        self.assertIs(type(converted_defaulted), collections.defaultdict)
        self.assertIsNot(converted_defaulted, defaulted)
        self.assertIs(converted_defaulted.default_factory, list)
        self.assertEqual(converted_defaulted, {"value": [1, 2]})
        self.assertEqual(defaulted, {"value": nested_tuple})

        user_list = collections.UserList([nested_tuple])
        converted_user_list = default_convert(user_list)
        self.assertIs(type(converted_user_list), collections.UserList)
        self.assertIsNot(converted_user_list, user_list)
        self.assertIsNot(converted_user_list.data, user_list.data)
        self.assertEqual(converted_user_list, [[1, 2]])
        self.assertEqual(user_list, [nested_tuple])

        user_dict = collections.UserDict(value=nested_tuple)
        converted_user_dict = default_convert(user_dict)
        self.assertIs(type(converted_user_dict), collections.UserDict)
        self.assertIsNot(converted_user_dict, user_dict)
        self.assertIsNot(converted_user_dict.data, user_dict.data)
        self.assertEqual(converted_user_dict, {"value": [1, 2]})
        self.assertEqual(user_dict, {"value": nested_tuple})

        deque = collections.deque([nested_tuple])
        converted_deque = default_convert(deque)
        self.assertIs(type(converted_deque), collections.deque)
        self.assertIsNot(converted_deque, deque)
        self.assertEqual(converted_deque, collections.deque([[1, 2]]))
        self.assertEqual(deque, collections.deque([nested_tuple]))

        proxy = MappingProxyType({"value": nested_tuple})
        converted_proxy = default_convert(proxy)
        self.assertIs(type(converted_proxy), MappingProxyType)
        self.assertIsNot(converted_proxy, proxy)
        self.assertEqual(dict(converted_proxy), {"value": [1, 2]})
        self.assertEqual(dict(proxy), {"value": nested_tuple})

        tuple_subclass = TupleSubclass((nested_tuple,))
        converted_tuple_subclass = default_convert(tuple_subclass)
        self.assertIs(type(converted_tuple_subclass), list)
        self.assertEqual(converted_tuple_subclass, [[1, 2]])
        self.assertEqual(tuple_subclass, (nested_tuple,))

        byte_array = bytearray(b"ab")
        converted_byte_array = default_convert(byte_array)
        self.assertIs(type(converted_byte_array), bytearray)
        self.assertIsNot(converted_byte_array, byte_array)
        self.assertEqual(converted_byte_array, byte_array)

        self.assertEqual(default_convert(range(3)), [0, 1, 2])
        self.assertEqual(default_convert(memoryview(b"ab")), [97, 98])

        chunk = DataChunk([nested_tuple])
        converted_chunk = default_convert(chunk)
        self.assertIs(type(converted_chunk), DataChunk)
        self.assertIsNot(converted_chunk, chunk)
        self.assertEqual(list(converted_chunk), [[1, 2]])
        self.assertEqual(list(chunk), [nested_tuple])
        self.assertIs(converted_chunk.items, chunk.items)
        self.assertEqual(list(converted_chunk.raw_iterator()), [nested_tuple])

    def test_mapping_and_sequence_construction_failures_fall_back_to_builtins(self):
        mapping_cases = (
            CopyFailingDict(value=(1, 2)),
            UpdateFailingDict(value=(1, 2)),
            ConstructorFailingMapping(),
        )
        for value in mapping_cases:
            with self.subTest(value_type=type(value).__name__):
                converted = default_convert(value)
                self.assertIs(type(converted), dict)
                self.assertEqual(converted, {"value": [1, 2]})

        sequence_cases = (
            CopyFailingList([(1, 2)]),
            SetItemFailingList([(1, 2)]),
            ConstructorFailingSequence(),
        )
        for value in sequence_cases:
            with self.subTest(value_type=type(value).__name__):
                converted = default_convert(value)
                self.assertIs(type(converted), list)
                self.assertEqual(converted, [[1, 2]])

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

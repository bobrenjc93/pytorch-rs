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


class DefaultConvertTests(unittest.TestCase):
    def test_leaf_values_are_returned_by_identity(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        text = "".join(("identity", "-text"))
        byte_string = b"identity-bytes"
        values = (
            tensor,
            None,
            True,
            10**40,
            1.25,
            2.0 + 3.0j,
            text,
            byte_string,
            Ellipsis,
        )

        for value in values:
            with self.subTest(value_type=type(value).__name__):
                self.assertIs(default_convert(value), value)

        self.assertIs(default_convert(data=tensor), tensor)
        self.assertTrue(tensor.requires_grad)
        self.assertTrue(tensor.is_leaf)

    def test_builtin_containers_recurse_without_mutating_the_input(self):
        Point = namedtuple("Point", ("x", "y"))
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        key = object()
        source = {
            key: [tensor, ("text", 7)],
            "point": Point({"tensor": tensor}, [None, b"bytes"]),
        }

        converted = default_convert(source)

        self.assertIs(type(converted), dict)
        self.assertIsNot(converted, source)
        self.assertEqual(list(converted), [key, "point"])
        self.assertIs(next(iter(converted)), key)

        converted_list = converted[key]
        self.assertIs(type(converted_list), list)
        self.assertIsNot(converted_list, source[key])
        self.assertIs(converted_list[0], tensor)
        self.assertIs(type(converted_list[1]), list)
        self.assertEqual(converted_list[1], ["text", 7])
        self.assertIs(converted_list[1][0], source[key][1][0])
        self.assertIs(converted_list[1][1], source[key][1][1])

        converted_point = converted["point"]
        self.assertIs(type(converted_point), Point)
        self.assertIsNot(converted_point, source["point"])
        self.assertIs(type(converted_point.x), dict)
        self.assertIsNot(converted_point.x, source["point"].x)
        self.assertIs(converted_point.x["tensor"], tensor)
        self.assertIs(type(converted_point.y), list)
        self.assertIsNot(converted_point.y, source["point"].y)
        self.assertEqual(converted_point.y, [None, b"bytes"])

        self.assertIs(type(source[key][1]), tuple)
        self.assertIs(type(source["point"]), Point)
        self.assertIs(type(source["point"].x), dict)
        self.assertIs(type(source["point"].y), list)

    def test_each_mutable_container_occurrence_is_copied(self):
        child = [1, {"value": 2}]
        source = [child, child]
        converted = default_convert(source)

        self.assertIsNot(converted, source)
        self.assertIsNot(converted[0], child)
        self.assertIsNot(converted[1], child)
        self.assertIsNot(converted[0], converted[1])
        self.assertIsNot(converted[0][1], child[1])
        self.assertIsNot(converted[0][1], converted[1][1])
        self.assertEqual(converted, source)

    def test_numpy_arrays_and_scalars_are_explicitly_unsupported(self):
        values = (
            np.array([1, 2], dtype=np.int64),
            np.array("text"),
            np.array(object(), dtype=object),
            np.int64(3),
            np.float32(1.5),
            np.bool_(True),
            np.str_("text"),
        )

        for value in values:
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    "^default_convert does not support NumPy arrays or scalars$",
                ):
                    default_convert(value)

        with self.assertRaisesRegex(NotImplementedError, "NumPy"):
            default_convert({"nested": [np.int32(1)]})

    def test_mapping_and_sequence_subclasses_use_copy_or_fallback(self):
        tensor = torch.tensor([1.0, 2.0])

        mapping = DictSubclass(value=(tensor, "text"))
        mapping.marker = object()
        converted_mapping = default_convert(mapping)
        self.assertIs(type(converted_mapping), DictSubclass)
        self.assertIsNot(converted_mapping, mapping)
        self.assertIs(converted_mapping.marker, mapping.marker)
        self.assertIs(converted_mapping["value"][0], tensor)
        self.assertEqual(converted_mapping["value"][1], "text")
        self.assertIs(type(converted_mapping["value"]), list)

        ordered = collections.OrderedDict(
            (("first", (1, 2)), ("second", [3]))
        )
        converted_ordered = default_convert(ordered)
        self.assertIs(type(converted_ordered), collections.OrderedDict)
        self.assertIsNot(converted_ordered, ordered)
        self.assertEqual(list(converted_ordered), ["first", "second"])
        self.assertEqual(converted_ordered, {"first": [1, 2], "second": [3]})

        proxy = MappingProxyType({"value": (1, 2)})
        converted_proxy = default_convert(proxy)
        self.assertIs(type(converted_proxy), type(proxy))
        self.assertIsNot(converted_proxy, proxy)
        self.assertEqual(converted_proxy, {"value": [1, 2]})

        sequence = ListSubclass([(tensor, "text")])
        sequence.marker = object()
        converted_sequence = default_convert(sequence)
        self.assertIs(type(converted_sequence), ListSubclass)
        self.assertIsNot(converted_sequence, sequence)
        self.assertIs(converted_sequence.marker, sequence.marker)
        self.assertIs(converted_sequence[0][0], tensor)
        self.assertEqual(converted_sequence[0][1], "text")
        self.assertIs(type(converted_sequence[0]), list)

        chunk = DataChunk([(tensor, "text"), {"value": (1, 2)}])
        converted_chunk = default_convert(chunk)
        self.assertIs(type(converted_chunk), DataChunk)
        self.assertIsNot(converted_chunk, chunk)
        self.assertIs(converted_chunk[0][0], tensor)
        self.assertEqual(converted_chunk[0][1], "text")
        self.assertEqual(converted_chunk[1], {"value": [1, 2]})
        self.assertEqual(list(converted_chunk.raw_iterator()), list(chunk))

        self.assertEqual(default_convert(TupleSubclass((1, 2))), [1, 2])
        self.assertEqual(default_convert(range(3)), [0, 1, 2])
        self.assertIs(
            type(default_convert(CopyFailingDict(value=(1, 2)))), dict
        )
        self.assertIs(
            type(default_convert(CopyFailingList(((1, 2),)))), list
        )

    def test_metadata_exports_and_unsupported_neighbors(self):
        data_module = importlib.import_module("torch_rs.utils.data")
        collate_module = importlib.import_module(
            "torch_rs.utils.data._utils.collate"
        )

        self.assertIs(type(default_convert), types.FunctionType)
        self.assertEqual(str(inspect.signature(default_convert)), "(data)")
        self.assertEqual(default_convert.__annotations__, {})
        self.assertIsNone(default_convert.__defaults__)
        self.assertIsNone(default_convert.__kwdefaults__)
        self.assertEqual(default_convert.__dict__, {})
        self.assertEqual(
            default_convert.__module__, "torch_rs.utils.data._utils.collate"
        )
        self.assertEqual(default_convert.__name__, "default_convert")
        self.assertEqual(default_convert.__qualname__, "default_convert")
        self.assertIn(
            "Convert each NumPy array element into a :class:`torch.Tensor`.",
            default_convert.__doc__,
        )

        self.assertIs(torch.utils.data.default_convert, default_convert)
        self.assertIs(data_module.default_convert, default_convert)
        self.assertIs(collate_module.default_convert, default_convert)
        self.assertEqual(data_module.__all__.count("default_convert"), 1)
        self.assertFalse(hasattr(collate_module, "__all__"))

        namespace = {}
        exec("from torch_rs.utils.data import *", namespace)
        self.assertIs(namespace["default_convert"], default_convert)

        self.assertFalse(hasattr(data_module, "default_collate"))
        self.assertFalse(hasattr(data_module, "DataLoader"))
        self.assertFalse(hasattr(collate_module, "default_collate"))

    def test_function_copy_pickle_and_argument_handling(self):
        self.assertIs(copy.copy(default_convert), default_convert)
        self.assertIs(copy.deepcopy(default_convert), default_convert)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(
                    pickle.dumps(default_convert, protocol=protocol)
                )
                self.assertIs(restored, default_convert)

        with self.assertRaises(TypeError):
            default_convert()
        with self.assertRaises(TypeError):
            default_convert(1, 2)
        with self.assertRaises(TypeError):
            default_convert(value=1)


if __name__ == "__main__":
    unittest.main()

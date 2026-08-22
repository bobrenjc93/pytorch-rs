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

from torch_rs.utils.data import DataChunk, default_convert


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


class DefaultConvertTests(unittest.TestCase):
    def test_tensor_scalars_strings_and_other_leaves_preserve_identity(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        leaves = (
            tensor,
            None,
            True,
            int("123456789012345678901234567890"),
            float("1.25"),
            complex(2.0, -3.0),
            "".join(("torch", "_rs")),
            bytes((0, 1, 2, 255)),
            object(),
        )

        for leaf in leaves:
            with self.subTest(type=type(leaf).__name__):
                self.assertIs(default_convert(leaf), leaf)

        converted_tensor = default_convert(tensor)
        self.assertIs(converted_tensor, tensor)
        self.assertTrue(converted_tensor.requires_grad)
        self.assertEqual(converted_tensor.data_ptr(), tensor.data_ptr())

    def test_builtin_containers_recurse_and_copy_like_pytorch(self):
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        marker = object()
        point = Point(tensor, {"marker": marker})
        source = {
            "list": [tensor, (marker, "text")],
            "point": point,
            "empty_dict": {},
            "empty_list": [],
            "empty_tuple": (),
        }

        converted = default_convert(source)

        self.assertIs(type(converted), dict)
        self.assertIsNot(converted, source)
        self.assertIs(type(converted["list"]), list)
        self.assertIsNot(converted["list"], source["list"])
        self.assertIs(converted["list"][0], tensor)
        self.assertIs(type(converted["list"][1]), list)
        self.assertIs(converted["list"][1][0], marker)
        self.assertIs(converted["list"][1][1], source["list"][1][1])

        converted_point = converted["point"]
        self.assertIs(type(converted_point), Point)
        self.assertIsNot(converted_point, point)
        self.assertIs(converted_point.x, tensor)
        self.assertIs(type(converted_point.y), dict)
        self.assertIsNot(converted_point.y, point.y)
        self.assertIs(converted_point.y["marker"], marker)

        self.assertEqual(converted["empty_dict"], {})
        self.assertIsNot(converted["empty_dict"], source["empty_dict"])
        self.assertEqual(converted["empty_list"], [])
        self.assertIsNot(converted["empty_list"], source["empty_list"])
        self.assertEqual(converted["empty_tuple"], [])
        self.assertIs(type(converted["empty_tuple"]), list)

    def test_numpy_arrays_and_scalars_are_rejected_without_mutation(self):
        numpy_values = (
            np.arange(4, dtype=np.float32),
            np.array(7, dtype=np.int64),
            np.int64(3),
            np.float32(1.5),
            np.bool_(True),
            np.str_("value"),
        )
        for value in numpy_values:
            with self.subTest(type=type(value).__name__):
                with self.assertRaisesRegex(
                    TypeError,
                    "^default_convert\\(\\): NumPy arrays and scalars are not "
                    "supported$",
                ):
                    default_convert(value)

        array = np.arange(3, dtype=np.int64)
        wrapped_values = (
            ["before", {"array": array}, "after"],
            OrderedDict(array=array),
            UserDict(array=array),
            UserList([array]),
            DataChunk([array]),
            types.MappingProxyType({"array": array}),
        )
        for source in wrapped_values:
            with self.subTest(type=type(source).__name__):
                with self.assertRaisesRegex(TypeError, "NumPy arrays and scalars"):
                    default_convert(source)

        self.assertIs(wrapped_values[0][1]["array"], array)
        self.assertIs(wrapped_values[1]["array"], array)
        self.assertIs(wrapped_values[2]["array"], array)
        self.assertIs(wrapped_values[3][0], array)
        self.assertIs(wrapped_values[4][0], array)
        self.assertIs(wrapped_values[5]["array"], array)

    def test_generic_mappings_use_copy_constructor_and_fallback_paths(self):
        ordered = OrderedDict(first=(1, 2), second=[3, 4])
        user_dict = UserDict(value=(5, 6))
        fancy = FancyDict(value=(7, 8))
        fancy.label = object()

        for source in (ordered, user_dict, fancy):
            with self.subTest(type=type(source).__name__):
                converted = default_convert(source)
                self.assertIs(type(converted), type(source))
                self.assertIsNot(converted, source)
                for key in source:
                    self.assertIsNot(converted[key], source[key])
                    self.assertEqual(converted[key], list(source[key]))

        self.assertEqual(tuple(default_convert(ordered)), ("first", "second"))
        self.assertIs(default_convert(fancy).label, fancy.label)

        proxy = types.MappingProxyType({"value": (9, 10)})
        converted_proxy = default_convert(proxy)
        self.assertIs(type(converted_proxy), type(proxy))
        self.assertIsNot(converted_proxy, proxy)
        self.assertEqual(converted_proxy["value"], [9, 10])

        copy_rejecting = CopyRejectingDict(value=(11, 12))
        converted_copy_rejecting = default_convert(copy_rejecting)
        self.assertIs(type(converted_copy_rejecting), dict)
        self.assertEqual(converted_copy_rejecting, {"value": [11, 12]})

        constructor_rejecting = ConstructorRejectingMapping(
            {"value": (13, 14)}, label="metadata"
        )
        converted_constructor_rejecting = default_convert(constructor_rejecting)
        self.assertIs(type(converted_constructor_rejecting), dict)
        self.assertEqual(converted_constructor_rejecting, {"value": [13, 14]})

    def test_generic_sequences_use_copy_constructor_and_fallback_paths(self):
        user_list = UserList([1, (2, 3)])
        fancy = FancyList([4, (5, 6)])
        fancy.label = object()
        chunk = DataChunk([7, (8, 9)])

        for source in (user_list, fancy, chunk):
            with self.subTest(type=type(source).__name__):
                converted = default_convert(source)
                self.assertIs(type(converted), type(source))
                self.assertIsNot(converted, source)
                self.assertIs(type(converted[1]), list)
                self.assertEqual(converted[1], list(source[1]))

        self.assertIs(default_convert(fancy).label, fancy.label)
        converted_chunk = default_convert(chunk)
        self.assertIs(converted_chunk.items, chunk.items)
        self.assertEqual(list(converted_chunk.raw_iterator()), [7, (8, 9)])

        tuple_subclass = FancyTuple((10, (11, 12)))
        self.assertEqual(default_convert(tuple_subclass), [10, [11, 12]])
        self.assertIs(type(default_convert(tuple_subclass)), list)
        self.assertEqual(default_convert(range(3)), [0, 1, 2])
        self.assertIs(type(default_convert(range(3))), list)

        copy_rejecting = CopyRejectingList([13, (14, 15)])
        converted_copy_rejecting = default_convert(copy_rejecting)
        self.assertIs(type(converted_copy_rejecting), list)
        self.assertEqual(converted_copy_rejecting, [13, [14, 15]])

    def test_signature_documentation_metadata_exports_and_unsupported_neighbors(self):
        data_module = importlib.import_module("torch_rs.utils.data")
        collate_module = importlib.import_module("torch_rs.utils.data._utils.collate")
        utils_module = importlib.import_module("torch_rs.utils.data._utils")
        dataset_module = importlib.import_module("torch_rs.utils.data.dataset")

        self.assertIs(type(default_convert), types.FunctionType)
        self.assertEqual(str(inspect.signature(default_convert)), "(data)")
        self.assertEqual(default_convert.__annotations__, {})
        self.assertEqual(
            default_convert.__module__, "torch_rs.utils.data._utils.collate"
        )
        self.assertEqual(default_convert.__name__, "default_convert")
        self.assertEqual(default_convert.__qualname__, "default_convert")
        self.assertIn("Convert each NumPy array element", default_convert.__doc__)
        self.assertIn("a single data point to be converted", default_convert.__doc__)
        self.assertIsNone(default_convert.__defaults__)
        self.assertIsNone(default_convert.__kwdefaults__)
        self.assertEqual(default_convert.__dict__, {})
        self.assertFalse(hasattr(default_convert, "__text_signature__"))

        self.assertIs(torch.utils.data.default_convert, default_convert)
        self.assertIs(data_module.default_convert, default_convert)
        self.assertIs(collate_module.default_convert, default_convert)
        self.assertNotIn("default_convert", utils_module.__dict__)
        self.assertNotIn("default_convert", dataset_module.__dict__)
        self.assertEqual(data_module.__all__.count("default_convert"), 1)
        self.assertFalse(hasattr(collate_module, "__all__"))

        wildcard_namespace = {}
        exec("from torch_rs.utils.data import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["default_convert"], default_convert)

        self.assertFalse(hasattr(data_module, "default_collate"))
        self.assertFalse(hasattr(data_module, "DataLoader"))
        self.assertFalse(hasattr(collate_module, "default_collate"))

    def test_call_forms_copy_and_pickle(self):
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
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

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

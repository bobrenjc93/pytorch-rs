import copy
import importlib
import inspect
import pickle
import types
import unittest
from collections import OrderedDict, UserDict, UserList, deque, namedtuple
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import numpy as np

import torch_rs as torch

from torch_rs.utils.data import DataChunk, default_convert


Point = namedtuple("Point", ("x", "y"))

FUNCTION_DOC = """Convert each NumPy array element into a :class:`torch.Tensor`.

    If the input is a `Sequence`, `Collection`, or `Mapping`, it tries to convert each element inside to a :class:`torch.Tensor`.
    If the input is not a NumPy array, it is left unchanged.
    This is used as the default function for collation when both `batch_sampler` and `batch_size`
    are NOT defined in :class:`~torch.utils.data.DataLoader`.

    The general input type to output type mapping is similar to that
    of :func:`~torch.utils.data.default_collate`. See the description there for more details.

    Args:
        data: a single data point to be converted

    Examples:
        >>> # xdoctest: +SKIP
        >>> # Example with `int`
        >>> default_convert(0)
        0
        >>> # Example with NumPy array
        >>> default_convert(np.array([0, 1]))
        tensor([0, 1])
        >>> # Example with NamedTuple
        >>> Point = namedtuple("Point", ["x", "y"])
        >>> default_convert(Point(0, 0))
        Point(x=0, y=0)
        >>> default_convert(Point(np.array(0), np.array(0)))
        Point(x=tensor(0), y=tensor(0))
        >>> # Example with List
        >>> default_convert([np.array([0, 1]), np.array([2, 3])])
        [tensor([0, 1]), tensor([2, 3])]
    """

NUMPY_ERROR = "default_convert does not support NumPy arrays or scalars"


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


class IterableOnly:
    def __iter__(self):
        return iter((1, 2))


class DefaultConvertTests(unittest.TestCase):
    def test_tensor_scalars_strings_and_other_leaves_preserve_identity(self):
        tensor = torch.tensor([1.0, 2.0])
        leaves = (
            tensor,
            None,
            True,
            False,
            0,
            257,
            -3.5,
            2 + 4j,
            "sample",
            b"sample",
            Ellipsis,
            object(),
        )

        for leaf in leaves:
            with self.subTest(leaf_type=type(leaf).__name__):
                self.assertIs(default_convert(leaf), leaf)

        iterable = IterableOnly()
        unordered = {1, 2}
        self.assertIs(default_convert(iterable), iterable)
        self.assertIs(default_convert(unordered), unordered)

    def test_builtin_containers_recurse_with_pytorch_copying_behavior(self):
        tensor = torch.tensor([1.0])
        key = object()
        shared = [tensor, "leaf"]
        point = Point({"tensor": tensor}, (True, b"bytes"))
        source = {
            key: [shared, shared],
            "tuple": (1, "two", tensor),
            "point": point,
            "empty_list": [],
            "empty_dict": {},
            "empty_tuple": (),
        }

        converted = default_convert(source)

        self.assertIs(type(converted), dict)
        self.assertIsNot(converted, source)
        self.assertIs(next(iter(converted)), key)

        converted_shared = converted[key]
        self.assertIs(type(converted_shared), list)
        self.assertIsNot(converted_shared, source[key])
        self.assertIsNot(converted_shared[0], shared)
        self.assertIsNot(converted_shared[1], shared)
        self.assertIsNot(converted_shared[0], converted_shared[1])
        self.assertIs(converted_shared[0][0], tensor)
        self.assertIs(converted_shared[0][1], shared[1])

        self.assertIs(type(converted["tuple"]), list)
        self.assertEqual(converted["tuple"][:2], [1, "two"])
        self.assertIs(converted["tuple"][2], tensor)

        converted_point = converted["point"]
        self.assertIs(type(converted_point), Point)
        self.assertIsNot(converted_point, point)
        self.assertIs(type(converted_point.x), dict)
        self.assertIsNot(converted_point.x, point.x)
        self.assertIs(converted_point.x["tensor"], tensor)
        self.assertIs(type(converted_point.y), list)
        self.assertEqual(converted_point.y, [True, b"bytes"])

        self.assertEqual(converted["empty_list"], [])
        self.assertIsNot(converted["empty_list"], source["empty_list"])
        self.assertEqual(converted["empty_dict"], {})
        self.assertIsNot(converted["empty_dict"], source["empty_dict"])
        self.assertEqual(converted["empty_tuple"], [])
        self.assertEqual(source["tuple"], (1, "two", tensor))
        self.assertIs(source["point"], point)

    def test_numpy_arrays_and_scalars_are_rejected_at_any_supported_depth(self):
        class ArraySubclass(np.ndarray):
            pass

        values = (
            np.array([1, 2]),
            np.array(["a", "b"]),
            np.array([object()], dtype=object),
            np.int64(1),
            np.float32(1.5),
            np.bool_(True),
            np.str_("text"),
            np.array([1]).view(ArraySubclass),
        )
        for value in values:
            for wrapped in (value, [value], {"value": value}, Point(value, 1)):
                with self.subTest(value_type=type(value), wrapped_type=type(wrapped)):
                    with self.assertRaisesRegex(TypeError, f"^{NUMPY_ERROR}$"):
                        default_convert(wrapped)

        numpy_key = np.int64(3)
        converted = default_convert({numpy_key: "value"})
        self.assertIs(next(iter(converted)), numpy_key)

    def test_mapping_and_sequence_subclasses_use_pytorch_copy_fallbacks(self):
        chunk = DataChunk([(1, 2)])
        dict_subclass = DictSubclass(value=(1, 2))
        dict_marker = object()
        dict_subclass.marker = dict_marker
        list_subclass = ListSubclass([(1, 2)])
        list_marker = object()
        list_subclass.marker = list_marker

        cases = {
            "data_chunk": (chunk, DataChunk, [[1, 2]]),
            "dict_subclass": (
                dict_subclass,
                DictSubclass,
                {"value": [1, 2]},
            ),
            "ordered_dict": (
                OrderedDict(value=(1, 2)),
                OrderedDict,
                OrderedDict(value=[1, 2]),
            ),
            "user_dict": (UserDict(value=(1, 2)), UserDict, {"value": [1, 2]}),
            "mapping_proxy": (
                MappingProxyType({"value": (1, 2)}),
                type(MappingProxyType({})),
                {"value": [1, 2]},
            ),
            "list_subclass": (list_subclass, ListSubclass, [[1, 2]]),
            "user_list": (UserList([(1, 2)]), UserList, [[1, 2]]),
            "constructible_sequence": (
                ConstructibleSequence([(1, 2)]),
                ConstructibleSequence,
                [[1, 2]],
            ),
            "tuple_subclass": (TupleSubclass(((1, 2),)), list, [[1, 2]]),
            "range_fallback": (range(2), list, [0, 1]),
            "deque": (deque([(1, 2)]), deque, deque([[1, 2]])),
            "bytearray": (bytearray(b"a"), bytearray, bytearray(b"a")),
            "mapping_copy_fallback": (
                CopyRejectingMutableMapping(value=(1, 2)),
                dict,
                {"value": [1, 2]},
            ),
            "mapping_constructor_fallback": (
                ConstructorRejectingMapping(),
                dict,
                {"value": [1, 2]},
            ),
            "sequence_copy_fallback": (
                CopyRejectingMutableSequence([(1, 2)]),
                list,
                [[1, 2]],
            ),
            "sequence_constructor_fallback": (
                ConstructorRejectingSequence(),
                list,
                [[1, 2]],
            ),
        }
        converted = {}
        for name, (value, expected_type, expected_value) in cases.items():
            with self.subTest(name=name):
                result = default_convert(value)
                converted[name] = result
                self.assertIs(type(result), expected_type)
                self.assertIsNot(result, value)
                if isinstance(result, Mapping):
                    self.assertEqual(dict(result), dict(expected_value))
                else:
                    self.assertEqual(list(result), list(expected_value))

        self.assertIs(converted["dict_subclass"].marker, dict_marker)
        self.assertIs(converted["list_subclass"].marker, list_marker)
        self.assertIs(converted["data_chunk"].items, chunk.items)
        self.assertEqual(list(converted["data_chunk"].raw_iterator()), [(1, 2)])

    def test_signature_annotations_documentation_and_metadata(self):
        self.assertIs(type(default_convert), types.FunctionType)
        self.assertEqual(str(inspect.signature(default_convert)), "(data)")
        self.assertEqual(default_convert.__annotations__, {})
        self.assertEqual(
            default_convert.__module__, "torch_rs.utils.data._utils.collate"
        )
        self.assertEqual(default_convert.__name__, "default_convert")
        self.assertEqual(default_convert.__qualname__, "default_convert")
        self.assertEqual(
            inspect.cleandoc(default_convert.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(default_convert.__defaults__)
        self.assertIsNone(default_convert.__kwdefaults__)
        self.assertEqual(default_convert.__dict__, {})
        self.assertFalse(hasattr(default_convert, "__text_signature__"))
        self.assertEqual(inspect.get_annotations(default_convert), {})

    def test_argument_forms_and_errors(self):
        leaf = object()
        self.assertIs(default_convert(data=leaf), leaf)

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

    def test_exports_and_unsupported_neighbors(self):
        data_module = importlib.import_module("torch_rs.utils.data")
        collate_module = importlib.import_module(
            "torch_rs.utils.data._utils.collate"
        )
        dataset_module = importlib.import_module("torch_rs.utils.data.dataset")

        self.assertIs(torch.utils.data.default_convert, default_convert)
        self.assertIs(data_module.default_convert, default_convert)
        self.assertIs(collate_module.default_convert, default_convert)
        self.assertNotIn("default_convert", dataset_module.__dict__)
        self.assertEqual(data_module.__all__.count("default_convert"), 1)
        self.assertFalse(hasattr(collate_module, "__all__"))

        wildcard_namespace = {}
        exec("from torch_rs.utils.data import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["default_convert"], default_convert)

        for unsupported in ("default_collate", "DataLoader"):
            with self.subTest(unsupported=unsupported):
                self.assertFalse(hasattr(data_module, unsupported))
                self.assertNotIn(unsupported, wildcard_namespace)
        self.assertFalse(hasattr(collate_module, "default_collate"))
        self.assertFalse(hasattr(collate_module, "collate"))

    def test_pickle_and_copy_preserve_the_global_function(self):
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

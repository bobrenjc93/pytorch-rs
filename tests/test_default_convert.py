import copy
import importlib
import inspect
import pickle
import types
import unittest
from collections import UserDict, UserList, namedtuple
from collections.abc import Mapping, Sequence

import numpy as np
import torch_rs as torch

from torch_rs.utils.data import DataChunk, default_convert


NUMPY_ERROR = "default_convert(): NumPy arrays and scalars are not supported"
CONTAINER_ERROR = (
    "default_convert(): recursive Mapping, sequence, and named-tuple inputs "
    "are not supported"
)
Point = namedtuple("Point", ("x", "y"))


class ArraySubclass(np.ndarray):
    pass


class MappingProbe(Mapping):
    def __getitem__(self, key):
        raise AssertionError("mapping contents must not be read")

    def __iter__(self):
        raise AssertionError("mapping contents must not be traversed")

    def __len__(self):
        raise AssertionError("mapping length must not be read")


class SequenceProbe(Sequence):
    def __getitem__(self, index):
        raise AssertionError("sequence contents must not be read")

    def __len__(self):
        raise AssertionError("sequence length must not be read")


class IterableLeaf:
    def __iter__(self):
        raise AssertionError("leaf iterators must not be consumed")


class Text(str):
    pass


class Blob(bytes):
    pass


class DefaultConvertTests(unittest.TestCase):
    def test_exact_tensors_and_non_recursive_leaves_preserve_identity(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        text = Text("torch_rs")
        blob = Blob(b"torch_rs")
        leaves = (
            tensor,
            None,
            True,
            int("123456789012345678901234567890"),
            float("1.25"),
            complex(2.0, -3.0),
            text,
            blob,
            object(),
            IterableLeaf(),
        )

        for leaf in leaves:
            with self.subTest(type=type(leaf).__name__):
                self.assertIs(default_convert(leaf), leaf)

    def test_tensor_storage_layout_and_autograd_state_are_unchanged(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        view = leaf.transpose(0, 1)[1]
        before = (
            view.data_ptr(),
            view.storage_offset(),
            view.shape,
            view.stride(),
            view.requires_grad,
            view.is_leaf,
        )

        converted = default_convert(view)

        self.assertIs(converted, view)
        self.assertEqual(
            (
                converted.data_ptr(),
                converted.storage_offset(),
                converted.shape,
                converted.stride(),
                converted.requires_grad,
                converted.is_leaf,
            ),
            before,
        )
        (converted * torch.tensor([2.0, 3.0])).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad),
            np.array([[0.0, 2.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float32),
        )

    def test_numpy_arrays_scalars_and_subclasses_are_rejected(self):
        values = (
            np.arange(4, dtype=np.float32),
            np.array(7, dtype=np.int64),
            np.int64(3),
            np.float32(1.5),
            np.bool_(True),
            np.str_("value"),
            np.bytes_("value"),
            np.arange(2).view(ArraySubclass),
            np.ma.array([1.0, 2.0]),
        )

        for value in values:
            with self.subTest(type=type(value).__name__):
                with self.assertRaises(TypeError) as raised:
                    default_convert(value)
                self.assertEqual(raised.exception.args, (NUMPY_ERROR,))

    def test_recursive_container_inputs_are_rejected_without_traversal(self):
        values = (
            {},
            UserDict(value=1),
            MappingProbe(),
            Point(1, 2),
            (),
            [1, 2],
            UserList([1, 2]),
            range(2),
            bytearray(b"ab"),
            memoryview(b"ab"),
            SequenceProbe(),
            DataChunk([1, 2]),
        )

        for value in values:
            with self.subTest(type=type(value).__name__):
                with self.assertRaises(TypeError) as raised:
                    default_convert(value)
                self.assertEqual(raised.exception.args, (CONTAINER_ERROR,))

        text = "text"
        blob = b"bytes"
        self.assertIs(default_convert(text), text)
        self.assertIs(default_convert(blob), blob)

    def test_metadata_exports_copying_pickling_and_unsupported_neighbors(self):
        data_module = importlib.import_module("torch_rs.utils.data")
        collate_module = importlib.import_module("torch_rs.utils.data._utils.collate")
        private_package = importlib.import_module("torch_rs.utils.data._utils")
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
        self.assertIsNone(default_convert.__defaults__)
        self.assertIsNone(default_convert.__kwdefaults__)
        self.assertEqual(default_convert.__dict__, {})
        self.assertFalse(hasattr(default_convert, "__text_signature__"))

        self.assertIs(torch.utils.data.default_convert, default_convert)
        self.assertIs(data_module.default_convert, default_convert)
        self.assertIs(collate_module.default_convert, default_convert)
        self.assertNotIn("default_convert", private_package.__dict__)
        self.assertNotIn("default_convert", dataset_module.__dict__)
        self.assertEqual(data_module.__all__.count("default_convert"), 1)
        self.assertFalse(hasattr(collate_module, "__all__"))
        self.assertNotIn("default_convert", torch.__all__)

        namespace = {}
        exec("from torch_rs.utils.data import *", namespace)
        self.assertIs(namespace["default_convert"], default_convert)

        self.assertFalse(hasattr(data_module, "default_collate"))
        self.assertFalse(hasattr(data_module, "DataLoader"))
        self.assertFalse(hasattr(collate_module, "default_collate"))

        marker = object()
        self.assertIs(default_convert(data=marker), marker)
        for call in (
            lambda: default_convert(),
            lambda: default_convert(None, None),
            lambda: default_convert(value=None),
        ):
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

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

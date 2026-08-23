import copy
import importlib
import inspect
import pickle
import pickletools
import types
import unittest
from collections import namedtuple

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


NUMPY_ERROR = "default_convert(): NumPy arrays and scalars are not supported"
CONTAINER_ERROR = (
    "default_convert(): recursive Mapping, sequence, and named-tuple inputs "
    "are not supported"
)
Point = namedtuple("Point", ("x", "y"))


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
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    @staticmethod
    def pickle_shape(function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            elif opcode.name == "FRAME":
                argument = None
            shape.append((opcode.name, argument))
        return shape

    def test_leaf_identity_matches_pytorch_2_13(self):
        actual = torch.utils.data.default_convert
        expected = reference_torch.utils.data.default_convert
        text = "".join(("torch", "_rs"))
        blob = bytes((0, 1, 2, 255))
        marker = object()
        leaves = (
            None,
            True,
            int("123456789012345678901234567890"),
            float("1.25"),
            complex(2.0, -3.0),
            text,
            blob,
            marker,
        )

        for leaf in leaves:
            with self.subTest(type=type(leaf).__name__):
                self.assertIs(actual(leaf), leaf)
                self.assertIs(expected(leaf), leaf)

        actual_tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        expected_tensor = reference_torch.tensor([1.0, 2.0], requires_grad=True)
        self.assertIs(actual(actual_tensor), actual_tensor)
        self.assertIs(expected(expected_tensor), expected_tensor)

    def test_tensor_storage_layout_and_autograd_match_pytorch_2_13(self):
        actual_leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        actual_view = actual_leaf.transpose(0, 1)[1]
        expected_view = expected_leaf.transpose(0, 1)[1]

        actual = torch.utils.data.default_convert(actual_view)
        expected = reference_torch.utils.data.default_convert(expected_view)

        self.assertIs(actual, actual_view)
        self.assertIs(expected, expected_view)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        self.assertEqual(actual.data_ptr(), actual_view.data_ptr())
        self.assertEqual(expected.data_ptr(), expected_view.data_ptr())
        self.assertEqual(
            actual.data_ptr() - actual_leaf.data_ptr(),
            expected.data_ptr() - expected_leaf.data_ptr(),
        )

        actual_weights = torch.tensor([2.0, 3.0])
        expected_weights = reference_torch.tensor([2.0, 3.0])
        (actual * actual_weights).sum().backward()
        (expected * expected_weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), expected_leaf.grad.detach().cpu().numpy()
        )

    def test_metadata_exports_copying_and_pickling_match_pytorch_2_13(self):
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
        self.assertEqual(actual_collate.__doc__, expected_collate.__doc__)
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
        namespace = {}
        exec("from torch_rs.utils.data import *", namespace)
        self.assertIs(namespace["default_convert"], actual)

        for unsupported in ("DataLoader", "default_collate"):
            with self.subTest(unsupported=unsupported):
                self.assertFalse(hasattr(actual_data, unsupported))
        self.assertFalse(hasattr(actual_collate, "default_collate"))

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_call_diagnostics_match_pytorch_2_13(self):
        actual = torch.utils.data.default_convert
        expected = reference_torch.utils.data.default_convert
        marker = object()
        self.assertIs(actual(data=marker), marker)
        self.assertIs(expected(data=marker), marker)

        for actual_call, expected_call in (
            (lambda: actual(), lambda: expected()),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(value=None), lambda: expected(value=None)),
        ):
            self.assert_error_matches(actual_call, expected_call)

    def test_numpy_and_recursive_conversion_boundaries_are_explicit(self):
        actual = torch.utils.data.default_convert
        expected = reference_torch.utils.data.default_convert

        array = np.arange(3, dtype=np.float32)
        scalar = np.int64(4)
        string = np.str_("value")
        self.assertIsInstance(expected(array), reference_torch.Tensor)
        self.assertIsInstance(expected(scalar), reference_torch.Tensor)
        self.assertIs(expected(string), string)
        for value in (array, scalar, string):
            with self.subTest(type=type(value).__name__):
                with self.assertRaises(TypeError) as raised:
                    actual(value)
                self.assertEqual(raised.exception.args, (NUMPY_ERROR,))

        actual_point = Point(1, 2)
        expected_point = Point(1, 2)
        container_pairs = (
            ({"value": 1}, {"value": 1}),
            ([1, 2], [1, 2]),
            ((1, 2), (1, 2)),
            (actual_point, expected_point),
        )
        for actual_value, expected_value in container_pairs:
            with self.subTest(type=type(actual_value).__name__):
                with self.assertRaises(TypeError) as raised:
                    actual(actual_value)
                self.assertEqual(raised.exception.args, (CONTAINER_ERROR,))
                self.assertIsNot(expected(expected_value), expected_value)


if __name__ == "__main__":
    unittest.main()

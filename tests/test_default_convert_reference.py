import importlib
import inspect
import unittest
from collections import OrderedDict, namedtuple

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


Point = namedtuple("Point", ["x", "y"])


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DefaultConvertReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "default_convert differentials require pinned PyTorch 2.13.0"
            )

    def assert_tensor_matches(self, actual, expected):
        with self.subTest(metadata=True):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(str(actual.layout), str(expected.layout))
        np.testing.assert_array_equal(
            np.asarray(actual), expected.detach().cpu().numpy()
        )

    def test_tensor_identity_and_no_batching_match_pytorch_2_13(self):
        actual_left = torch.tensor([1.0, -0.0], requires_grad=True)
        actual_right = torch.tensor([2.0, 3.0], requires_grad=True)
        expected_left = reference_torch.tensor(
            [1.0, -0.0], dtype=reference_torch.float32, requires_grad=True
        )
        expected_right = reference_torch.tensor(
            [2.0, 3.0], dtype=reference_torch.float32, requires_grad=True
        )

        actual = torch.utils.data.default_convert([actual_left, actual_right])
        expected = reference_torch.utils.data.default_convert(
            [expected_left, expected_right]
        )

        self.assertIs(type(actual), type(expected))
        self.assertIs(actual[0], actual_left)
        self.assertIs(actual[1], actual_right)
        self.assertIs(expected[0], expected_left)
        self.assertIs(expected[1], expected_right)
        self.assertNotIsInstance(actual, torch.Tensor)
        for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
            self.assert_tensor_matches(actual_leaf, expected_leaf)

        (actual[0] * torch.tensor([5.0, 7.0])).sum().backward()
        (
            expected[0]
            * reference_torch.tensor([5.0, 7.0], dtype=reference_torch.float32)
        ).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_left.grad), expected_left.grad.detach().cpu().numpy()
        )

    def test_recursive_containers_strings_and_bytes_match_pytorch_2_13(self):
        label = "label"
        payload = b"payload"
        actual_x = torch.tensor([1.0, 2.0])
        actual_y = torch.tensor(3.0)
        actual_z = torch.tensor([4.0])
        expected_x = reference_torch.tensor(
            [1.0, 2.0], dtype=reference_torch.float32
        )
        expected_y = reference_torch.tensor(3.0, dtype=reference_torch.float32)
        expected_z = reference_torch.tensor([4.0], dtype=reference_torch.float32)
        actual_data = OrderedDict(
            [
                ("z", [actual_x, (actual_y, Point(actual_z, label))]),
                ("a", {"payload": payload}),
            ]
        )
        expected_data = OrderedDict(
            [
                ("z", [expected_x, (expected_y, Point(expected_z, label))]),
                ("a", {"payload": payload}),
            ]
        )

        actual = torch.utils.data.default_convert(actual_data)
        expected = reference_torch.utils.data.default_convert(expected_data)

        self.assertIs(type(actual), type(expected))
        self.assertEqual(list(actual), list(expected))
        self.assertIs(type(actual["z"]), type(expected["z"]))
        self.assertIs(type(actual["z"][1]), type(expected["z"][1]))
        self.assertIs(type(actual["z"][1][1]), type(expected["z"][1][1]))
        self.assertIs(type(actual["a"]), type(expected["a"]))
        self.assertIs(actual["z"][0], actual_x)
        self.assertIs(actual["z"][1][0], actual_y)
        self.assertIs(actual["z"][1][1].x, actual_z)
        self.assertIs(actual["z"][1][1].y, label)
        self.assertIs(actual["a"]["payload"], payload)
        self.assertIs(expected["z"][0], expected_x)
        self.assertIs(expected["z"][1][0], expected_y)
        self.assertIs(expected["z"][1][1].x, expected_z)
        self.assertIs(expected["z"][1][1].y, label)
        self.assertIs(expected["a"]["payload"], payload)
        self.assert_tensor_matches(actual["z"][0], expected["z"][0])
        self.assert_tensor_matches(actual["z"][1][0], expected["z"][1][0])
        self.assert_tensor_matches(actual["z"][1][1].x, expected["z"][1][1].x)

    def test_pytorch_supported_conversion_paths_fail_closed(self):
        unsupported_actual_values = (
            np.asarray([1.0], dtype=np.float32),
            np.float32(1.0),
            np.int64(1),
            1.0,
            1,
            object(),
            None,
        )
        unsupported_expected_values = (
            np.asarray([1.0], dtype=np.float32),
            np.float32(1.0),
            np.int64(1),
            1.0,
            1,
            object(),
            None,
        )

        for actual_value, expected_value in zip(
            unsupported_actual_values, unsupported_expected_values, strict=True
        ):
            with self.subTest(value_type=type(actual_value).__name__):
                with self.assertRaises(TypeError):
                    torch.utils.data.default_convert(actual_value)
                reference_torch.utils.data.default_convert(expected_value)

        foreign = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        with self.assertRaisesRegex(TypeError, "exact native tensors"):
            torch.utils.data.default_convert(foreign)
        self.assertIs(reference_torch.utils.data.default_convert(foreign), foreign)

    def test_default_collate_boundary_remains_fail_closed(self):
        with self.assertRaises(TypeError):
            torch.utils.data.default_collate(["a", "b"])
        reference_torch.utils.data.default_collate(["a", "b"])

    def test_imports_exports_signature_and_loader_boundary_match(self):
        actual_data = importlib.import_module("torch_rs.utils.data")
        expected_data = importlib.import_module("torch.utils.data")
        actual_module = importlib.import_module("torch_rs.utils.data._utils.collate")
        expected_module = importlib.import_module("torch.utils.data._utils.collate")
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
            "default_collate",
            "default_convert",
            "get_worker_info",
        }

        self.assertIs(actual_data.default_convert, actual_module.default_convert)
        self.assertIs(expected_data.default_convert, expected_module.default_convert)
        self.assertEqual(
            actual_module.default_convert.__module__.replace("torch_rs", "torch"),
            expected_module.default_convert.__module__,
        )
        self.assertEqual(
            actual_module.default_convert.__name__,
            expected_module.default_convert.__name__,
        )
        self.assertEqual(
            actual_module.default_convert.__qualname__,
            expected_module.default_convert.__qualname__,
        )
        self.assertEqual(
            str(inspect.signature(actual_module.default_convert)),
            str(inspect.signature(expected_module.default_convert)),
        )
        self.assertEqual(
            actual_data.__all__,
            [name for name in expected_data.__all__ if name in supported],
        )
        self.assertFalse(hasattr(actual_data, "DataLoader"))


if __name__ == "__main__":
    unittest.main()

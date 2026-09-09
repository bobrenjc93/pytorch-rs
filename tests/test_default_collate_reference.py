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
class DefaultCollateReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "default_collate differentials require pinned PyTorch 2.13.0"
            )

    @staticmethod
    def tensor_from_array(module, array, *, requires_grad=False):
        kwargs = {"dtype": module.float32, "requires_grad": requires_grad}
        if array.shape == ():
            return module.tensor(float(array.reshape(()).item()), **kwargs)
        if any(dimension == 0 for dimension in array.shape):
            return module.zeros(tuple(array.shape), **kwargs)
        return module.tensor(array.tolist(), **kwargs)

    @staticmethod
    def generated_array(shape, seed):
        if shape == ():
            return np.asarray((seed % 17) - 8.5, dtype=np.float32)
        elements = int(np.prod(shape))
        values = np.arange(elements, dtype=np.float32).reshape(shape)
        return values * np.float32(0.25) + np.float32((seed % 11) - 5)

    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(str(actual.layout), str(expected.layout))
        with self.subTest(case=case, values=True):
            actual_bits = np.asarray(actual).reshape(-1).view(np.uint32)
            expected_bits = expected.detach().cpu().numpy().reshape(-1).view(np.uint32)
            np.testing.assert_array_equal(actual_bits, expected_bits)

    def assert_default_collate_tensor_matches(self, actual_batch, expected_batch, *, case):
        actual = torch.utils.data.default_collate(actual_batch)
        expected = reference_torch.utils.data.default_collate(expected_batch)
        self.assert_tensor_matches(actual, expected, case=case)
        for source in actual_batch:
            if source.numel():
                self.assertFalse(actual.is_set_to(source))
                self.assertNotEqual(actual.data_ptr(), source.data_ptr())

    def test_tensor_values_metadata_and_autograd_match_pytorch_2_13(self):
        cases = (
            ("scalar", (), 3),
            ("rank1", (4,), 2),
            ("rank2", (2, 3), 2),
            ("empty", (2, 0, 3), 2),
        )
        for case, shape, count in cases:
            arrays = [
                self.generated_array(shape, 20260908 + index * 17 + len(shape))
                for index in range(count)
            ]
            actual_batch = [self.tensor_from_array(torch, array) for array in arrays]
            expected_batch = [
                self.tensor_from_array(reference_torch, array) for array in arrays
            ]
            with self.subTest(case=case):
                self.assert_default_collate_tensor_matches(
                    actual_batch, expected_batch, case=case
                )

        actual_base = self.tensor_from_array(
            torch, self.generated_array((3, 4), 20261001)
        )
        expected_base = self.tensor_from_array(
            reference_torch, self.generated_array((3, 4), 20261001)
        )
        self.assert_default_collate_tensor_matches(
            [actual_base.transpose(0, 1)[0], actual_base.transpose(0, 1)[2]],
            [
                expected_base.transpose(0, 1)[0],
                expected_base.transpose(0, 1)[2],
            ],
            case="noncontiguous views",
        )

        actual_left = torch.tensor([1.0, 2.0], requires_grad=True)
        actual_right = torch.tensor([3.0, 4.0], requires_grad=True)
        expected_left = reference_torch.tensor(
            [1.0, 2.0], dtype=reference_torch.float32, requires_grad=True
        )
        expected_right = reference_torch.tensor(
            [3.0, 4.0], dtype=reference_torch.float32, requires_grad=True
        )
        actual = torch.utils.data.default_collate([actual_left, actual_right])
        expected = reference_torch.utils.data.default_collate(
            [expected_left, expected_right]
        )
        self.assert_tensor_matches(actual, expected, case="autograd output")
        weights = [[5.0, 7.0], [11.0, 13.0]]
        (actual * torch.tensor(weights)).sum().backward()
        (
            expected
            * reference_torch.tensor(weights, dtype=reference_torch.float32)
        ).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_left.grad), expected_left.grad.detach().numpy()
        )
        np.testing.assert_array_equal(
            np.asarray(actual_right.grad), expected_right.grad.detach().numpy()
        )

    def test_recursive_list_namedtuple_and_dict_outputs_match_pytorch_2_13(self):
        actual_batch = [
            OrderedDict(
                [
                    ("b", [torch.tensor([1.0]), Point(torch.tensor(2.0), torch.tensor([3.0]))]),
                    ("a", [torch.tensor([4.0])]),
                ]
            ),
            OrderedDict(
                [
                    (
                        "b",
                        [torch.tensor([10.0]), Point(torch.tensor(20.0), torch.tensor([30.0]))],
                    ),
                    ("a", [torch.tensor([40.0])]),
                ]
            ),
        ]
        expected_batch = [
            OrderedDict(
                [
                    (
                        "b",
                        [
                            reference_torch.tensor([1.0], dtype=reference_torch.float32),
                            Point(
                                reference_torch.tensor(2.0, dtype=reference_torch.float32),
                                reference_torch.tensor([3.0], dtype=reference_torch.float32),
                            ),
                        ],
                    ),
                    ("a", [reference_torch.tensor([4.0], dtype=reference_torch.float32)]),
                ]
            ),
            OrderedDict(
                [
                    (
                        "b",
                        [
                            reference_torch.tensor([10.0], dtype=reference_torch.float32),
                            Point(
                                reference_torch.tensor(20.0, dtype=reference_torch.float32),
                                reference_torch.tensor([30.0], dtype=reference_torch.float32),
                            ),
                        ],
                    ),
                    (
                        "a",
                        [reference_torch.tensor([40.0], dtype=reference_torch.float32)],
                    ),
                ]
            ),
        ]

        actual = torch.utils.data.default_collate(actual_batch)
        expected = reference_torch.utils.data.default_collate(expected_batch)

        self.assertIs(type(actual), type(expected))
        self.assertEqual(list(actual), list(expected))
        self.assertIs(type(actual["b"]), type(expected["b"]))
        self.assertIs(type(actual["b"][1]), type(expected["b"][1]))
        self.assert_tensor_matches(actual["b"][0], expected["b"][0], case="list leaf")
        self.assert_tensor_matches(
            actual["b"][1].x, expected["b"][1].x, case="namedtuple scalar leaf"
        )
        self.assert_tensor_matches(
            actual["b"][1].y, expected["b"][1].y, case="namedtuple tensor leaf"
        )
        self.assert_tensor_matches(actual["a"][0], expected["a"][0], case="dict leaf")

    def test_plain_tuple_returns_pytorch_compatible_list(self):
        actual = torch.utils.data.default_collate(
            [
                (torch.tensor([1.0]), torch.tensor([2.0])),
                (torch.tensor([3.0]), torch.tensor([4.0])),
            ]
        )
        expected = reference_torch.utils.data.default_collate(
            [
                (
                    reference_torch.tensor([1.0], dtype=reference_torch.float32),
                    reference_torch.tensor([2.0], dtype=reference_torch.float32),
                ),
                (
                    reference_torch.tensor([3.0], dtype=reference_torch.float32),
                    reference_torch.tensor([4.0], dtype=reference_torch.float32),
                ),
            ]
        )

        self.assertIs(type(actual), list)
        self.assertIs(type(expected), list)
        self.assertIs(type(actual), type(expected))
        self.assertEqual(len(actual), len(expected))
        for index, (actual_leaf, expected_leaf) in enumerate(zip(actual, expected)):
            self.assert_tensor_matches(
                actual_leaf, expected_leaf, case=("plain tuple leaf", index)
            )

    def test_pytorch_supported_non_tensor_collation_paths_fail_closed(self):
        unsupported_actual_batches = (
            [np.asarray([1.0], dtype=np.float32), np.asarray([2.0], dtype=np.float32)],
            [np.float32(1.0), np.float32(2.0)],
            [1.0, 2.0],
            [1, 2],
            ["a", "b"],
            [b"a", b"b"],
        )
        unsupported_expected_batches = (
            [np.asarray([1.0], dtype=np.float32), np.asarray([2.0], dtype=np.float32)],
            [np.float32(1.0), np.float32(2.0)],
            [1.0, 2.0],
            [1, 2],
            ["a", "b"],
            [b"a", b"b"],
        )

        for actual_batch, expected_batch in zip(
            unsupported_actual_batches, unsupported_expected_batches, strict=True
        ):
            with self.subTest(batch_type=type(actual_batch[0]).__name__):
                with self.assertRaises(TypeError):
                    torch.utils.data.default_collate(actual_batch)
                reference_torch.utils.data.default_collate(expected_batch)

    def test_arbitrary_objects_and_foreign_tensors_fail_closed(self):
        for batch in ([object(), object()], [None, None]):
            with self.subTest(batch_type=type(batch[0]).__name__):
                with self.assertRaises(TypeError):
                    torch.utils.data.default_collate(batch)
                with self.assertRaises(TypeError):
                    reference_torch.utils.data.default_collate(batch)

        foreign_batches = (
            [
                reference_torch.tensor([1.0], dtype=reference_torch.float32),
                reference_torch.tensor([2.0], dtype=reference_torch.float32),
            ],
            [
                reference_torch.tensor([1], dtype=reference_torch.int64),
                reference_torch.tensor([2], dtype=reference_torch.int64),
            ],
        )
        for batch in foreign_batches:
            with self.subTest(dtype=batch[0].dtype):
                with self.assertRaisesRegex(TypeError, "exact native CPU float32"):
                    torch.utils.data.default_collate(batch)
                reference_torch.utils.data.default_collate(batch)

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

        self.assertIs(actual_data.default_collate, actual_module.default_collate)
        self.assertIs(expected_data.default_collate, expected_module.default_collate)
        self.assertEqual(
            actual_module.default_collate.__module__.replace("torch_rs", "torch"),
            expected_module.default_collate.__module__,
        )
        self.assertEqual(
            actual_module.default_collate.__name__,
            expected_module.default_collate.__name__,
        )
        self.assertEqual(
            actual_module.default_collate.__qualname__,
            expected_module.default_collate.__qualname__,
        )
        self.assertEqual(
            str(inspect.signature(actual_module.default_collate)),
            str(inspect.signature(expected_module.default_collate)),
        )
        self.assertEqual(
            actual_data.__all__,
            [name for name in expected_data.__all__ if name in supported],
        )
        self.assertFalse(hasattr(actual_data, "DataLoader"))


if __name__ == "__main__":
    unittest.main()

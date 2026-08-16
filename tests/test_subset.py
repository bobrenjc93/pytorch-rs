from collections.abc import Sequence
import importlib
import inspect
import re
from typing import get_args, get_origin
import unittest

import numpy as np
import torch_rs as torch

from torch_rs.utils.data import Dataset, Subset, TensorDataset


class RecordingDataset(Dataset):
    def __init__(self):
        self.calls = []

    def __getitem__(self, index):
        self.calls.append(index)
        if isinstance(index, list):
            return [f"item-{item}" for item in index]
        return f"item-{index}"


class BatchedRecordingDataset(RecordingDataset):
    def __getitems__(self, indices):
        self.calls.append(("batch", indices))
        return [f"batch-{item}" for item in indices]


class NonCallableBatchedDataset(RecordingDataset):
    __getitems__ = None


class SubsetTests(unittest.TestCase):
    def test_length_integer_indexing_tensor_views_and_autograd_lineage(self):
        source = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            requires_grad=True,
        )
        dataset = TensorDataset(source)
        indices = [2, 0, 1]
        subset = Subset(dataset, indices)

        self.assertEqual(len(subset), 3)
        self.assertIs(subset.dataset, dataset)
        self.assertIs(subset.indices, indices)

        for index, source_index in (
            (0, 2),
            (1, 0),
            (2, 1),
            (-1, 1),
            (-3, 2),
            (np.int64(1), 0),
        ):
            with self.subTest(index=index):
                sample = subset[index][0]
                self.assertEqual(sample.shape, (3,))
                self.assertEqual(sample.stride(), (1,))
                self.assertEqual(sample.storage_offset(), source_index * 3)
                self.assertTrue(sample.requires_grad)
                self.assertFalse(sample.is_leaf)
                self.assertEqual(
                    sample.data_ptr() - source.data_ptr(),
                    sample.storage_offset() * source.element_size(),
                )
                np.testing.assert_array_equal(
                    np.asarray(sample.detach()),
                    np.asarray(source.detach())[source_index],
                )

        selected = subset[1][0]
        (selected * torch.tensor([2.0, 3.0, 5.0])).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(source.grad),
            [[2.0, 3.0, 5.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )

    def test_list_indexing_maps_once_and_delegates_to_dataset_getitem(self):
        dataset = RecordingDataset()
        subset = Subset(dataset, (4, 1, 3))

        self.assertEqual(
            subset[[2, 0, -1]], ["item-3", "item-4", "item-3"]
        )
        self.assertEqual(dataset.calls, [[3, 4, 3]])

        dataset.calls.clear()
        self.assertEqual(subset[[]], [])
        self.assertEqual(dataset.calls, [[]])

        with self.assertRaisesRegex(
            TypeError, "^list indices must be integers or slices, not list$"
        ):
            Subset(["zero", "one"], [1])[[0]]

    def test_getitems_delegation_and_fallback(self):
        delegated_dataset = BatchedRecordingDataset()
        delegated = Subset(delegated_dataset, [4, 1, 3])
        self.assertEqual(
            delegated.__getitems__([2, 0, -1]),
            ["batch-3", "batch-4", "batch-3"],
        )
        self.assertEqual(delegated_dataset.calls, [("batch", [3, 4, 3])])

        for dataset in (RecordingDataset(), NonCallableBatchedDataset()):
            with self.subTest(dataset=type(dataset).__name__):
                subset = Subset(dataset, [4, 1, 3])
                self.assertEqual(
                    subset.__getitems__([2, 0, -1]),
                    ["item-3", "item-4", "item-3"],
                )
                self.assertEqual(dataset.calls, [3, 4, 3])

    def test_subclass_override_guard(self):
        class IncompleteSubset(Subset):
            def __getitem__(self, idx):
                return super().__getitem__(idx)

        message = (
            "IncompleteSubset overrides __getitem__ but not __getitems__. "
            "When subclassing Subset and overriding __getitem__, you must also override "
            "__getitems__ to ensure DataLoader works correctly with your custom logic. "
            "A simple implementation:\n\n"
            "def __getitems__(self, indices):\n"
            "    return [self.__getitem__(idx) for idx in indices]"
        )
        with self.assertRaisesRegex(
            NotImplementedError, f"^{re.escape(message)}$"
        ):
            IncompleteSubset([10], [0])

        class CompleteSubset(Subset):
            def __getitem__(self, idx):
                return super().__getitem__(idx)

            def __getitems__(self, indices):
                return [self.__getitem__(idx) for idx in indices]

        complete = CompleteSubset([10, 20], [1, 0])
        self.assertEqual(complete[0], 20)
        self.assertEqual(complete.__getitems__([0, 1]), [20, 10])

        class DisabledBatchSubset(Subset):
            __getitems__ = None

            def __getitem__(self, idx):
                return super().__getitem__(idx)

        self.assertEqual(DisabledBatchSubset([10], [0])[0], 10)

    def test_imports_exports_annotations_signatures_and_documentation(self):
        data_module = importlib.import_module("torch_rs.utils.data")
        dataset_module = importlib.import_module("torch_rs.utils.data.dataset")

        self.assertIs(torch.utils.data, data_module)
        self.assertIs(data_module.Subset, Subset)
        self.assertIs(dataset_module.Subset, Subset)
        self.assertTrue(issubclass(Subset, Dataset))
        self.assertIsInstance(Subset([1], [0]), Dataset)
        self.assertEqual(Subset.__module__, "torch_rs.utils.data.dataset")
        self.assertEqual(
            data_module.__all__,
            [
                "ConcatDataset",
                "Dataset",
                "Sampler",
                "StackDataset",
                "Subset",
                "TensorDataset",
            ],
        )
        self.assertEqual(
            dataset_module.__all__,
            ["Dataset", "TensorDataset", "StackDataset", "ConcatDataset", "Subset"],
        )

        wildcard_namespace = {}
        exec("from torch_rs.utils.data import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["Subset"], Subset)

        signature = inspect.signature(Subset)
        self.assertEqual(tuple(signature.parameters), ("dataset", "indices"))
        self.assertIs(
            signature.parameters["dataset"].annotation,
            Subset.__annotations__["dataset"],
        )
        self.assertEqual(
            signature.parameters["indices"].annotation, Sequence[int]
        )
        self.assertIs(signature.return_annotation, None)
        self.assertIs(
            get_origin(Subset.__annotations__["dataset"]), Dataset
        )
        self.assertEqual(Subset.__annotations__["indices"], Sequence[int])
        self.assertEqual(
            Subset.__getitems__.__annotations__["indices"], list[int]
        )
        self.assertIs(
            get_args(Subset.__annotations__["dataset"])[0],
            get_args(Subset.__getitems__.__annotations__["return"])[0],
        )
        self.assertEqual(str(inspect.signature(Subset.__getitem__)), "(self, idx)")
        self.assertEqual(
            str(inspect.signature(Subset.__getitems__)),
            "(self, indices: list[int]) -> list[+_T_co]",
        )
        self.assertEqual(str(inspect.signature(Subset.__len__)), "(self) -> int")
        self.assertIn(
            "When subclassing `Subset` and overriding `__getitem__`",
            Subset.__doc__,
        )
        self.assertIn("dataset (Dataset): The whole Dataset", Subset.__doc__)

    def test_errors_and_method_diagnostics(self):
        subset = Subset([10, 20], [1])
        for index in (1, -2):
            with self.subTest(index=index):
                with self.assertRaisesRegex(IndexError, "^list index out of range$"):
                    subset[index]

        with self.assertRaisesRegex(
            TypeError, "^list indices must be integers or slices, not float$"
        ):
            subset[0.0]
        with self.assertRaisesRegex(
            TypeError,
            "^Subset.__init__\\(\\) missing 2 required positional arguments: "
            "'dataset' and 'indices'$",
        ):
            Subset()
        with self.assertRaisesRegex(
            TypeError,
            "^Subset.__getitem__\\(\\) missing 1 required positional argument: 'idx'$",
        ):
            subset.__getitem__()
        with self.assertRaisesRegex(
            TypeError,
            "^Subset.__getitems__\\(\\) takes 2 positional arguments but 3 were given$",
        ):
            subset.__getitems__([], [])
        with self.assertRaisesRegex(
            TypeError,
            "^Subset.__len__\\(\\) takes 1 positional argument but 2 were given$",
        ):
            subset.__len__(1)


if __name__ == "__main__":
    unittest.main()

import importlib
import inspect
from typing import Iterable, get_args, get_origin
import unittest
import warnings

import numpy as np
import torch_rs as torch

from torch_rs.utils.data import ConcatDataset, Dataset, Subset, TensorDataset


class ListDataset(Dataset):
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)


class ConcatDatasetTests(unittest.TestCase):
    def test_dataset_addition_creates_and_chains_concat_datasets(self):
        left = TensorDataset(torch.tensor([[1.0], [2.0]]))
        right = Subset(TensorDataset(torch.tensor([[3.0], [4.0], [5.0]])), [2, 0])
        tail = ListDataset(["tail"])

        direct = left + right
        self.assertIs(type(direct), ConcatDataset)
        self.assertEqual(direct.datasets, [left, right])
        self.assertIs(direct.datasets[0], left)
        self.assertIs(direct.datasets[1], right)
        self.assertEqual(direct.cumulative_sizes, [2, 4])
        self.assertEqual(
            [direct[index][0].item() for index in range(4)], [1, 2, 5, 3]
        )

        chained = direct + tail
        self.assertIs(type(chained), ConcatDataset)
        self.assertEqual(chained.datasets, [direct, tail])
        self.assertIs(chained.datasets[0], direct)
        self.assertIs(chained.datasets[1], tail)
        self.assertEqual(chained.cumulative_sizes, [4, 5])
        self.assertEqual(chained[0][0].item(), 1)
        self.assertEqual(chained[-2][0].item(), 3)
        self.assertEqual(chained[-1], "tail")
        self.assertIs(type(chained.datasets[0]), ConcatDataset)

        for owner in (right, tail):
            with self.subTest(owner=type(owner).__name__):
                combined = owner + left
                self.assertIs(type(combined), ConcatDataset)
                self.assertIs(combined.datasets[0], owner)
                self.assertIs(combined.datasets[1], left)

    def test_construction_materializes_iterable_and_computes_cumulative_sizes(self):
        children = [
            ListDataset([]),
            ListDataset(["zero", "one"]),
            ListDataset([]),
            ListDataset(["two", "three", "four"]),
        ]
        yielded = []

        def child_iterable():
            for child in children:
                yielded.append(child)
                yield child

        dataset = ConcatDataset(child_iterable())

        self.assertIs(type(dataset.datasets), list)
        self.assertEqual(yielded, children)
        self.assertEqual(len(dataset.datasets), len(children))
        for stored, child in zip(dataset.datasets, children):
            self.assertIs(stored, child)
        self.assertEqual(dataset.cumulative_sizes, [0, 2, 2, 5])
        self.assertEqual(ConcatDataset.cumsum(children), [0, 2, 2, 5])
        self.assertEqual(len(dataset), 5)

        original_list = children.copy()
        copied = ConcatDataset(original_list)
        self.assertIsNot(copied.datasets, original_list)
        original_list.clear()
        self.assertEqual(copied.datasets, children)

        for empty in ([], iter(())):
            with self.subTest(empty=type(empty).__name__):
                with self.assertRaisesRegex(
                    AssertionError, "^datasets should not be an empty iterable$"
                ):
                    ConcatDataset(empty)

    def test_indexing_across_empty_children_preserves_views_and_autograd(self):
        direct_source = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        subset_source = torch.tensor(
            [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]],
            requires_grad=True,
        )
        empty = TensorDataset(torch.zeros((0, 3)))
        direct = TensorDataset(direct_source)
        subset = Subset(TensorDataset(subset_source), [2, 0])
        dataset = ConcatDataset([empty, direct, Subset(direct, []), subset, empty])

        self.assertEqual(dataset.cumulative_sizes, [0, 2, 2, 4, 4])
        self.assertEqual(len(dataset), 4)
        cases = (
            (0, direct_source, 0),
            (1, direct_source, 1),
            (2, subset_source, 2),
            (3, subset_source, 0),
            (-1, subset_source, 0),
            (-2, subset_source, 2),
            (-3, direct_source, 1),
            (-4, direct_source, 0),
            (np.int64(2), subset_source, 2),
        )
        for index, source, source_index in cases:
            with self.subTest(index=index):
                sample = dataset[index][0]
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

        (dataset[1][0] * torch.tensor([2.0, 3.0, 5.0])).sum().backward()
        (dataset[-2][0] * torch.tensor([7.0, 11.0, 13.0])).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(direct_source.grad),
            [[0.0, 0.0, 0.0], [2.0, 3.0, 5.0]],
        )
        np.testing.assert_array_equal(
            np.asarray(subset_source.grad),
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [7.0, 11.0, 13.0]],
        )

    def test_index_construction_and_method_errors(self):
        dataset = ConcatDataset([[], [10], []])
        all_empty = ConcatDataset([[], []])

        self.assertEqual(dataset[0], 10)
        self.assertEqual(dataset[-1], 10)
        self.assertEqual(dataset[False], 10)
        with self.assertRaisesRegex(IndexError, "^list index out of range$"):
            dataset[1]
        with self.assertRaisesRegex(
            ValueError, "^absolute value of index should not exceed dataset length$"
        ):
            dataset[-2]
        with self.assertRaisesRegex(IndexError, "^list index out of range$"):
            all_empty[0]
        with self.assertRaisesRegex(
            ValueError, "^absolute value of index should not exceed dataset length$"
        ):
            all_empty[-1]
        with self.assertRaisesRegex(
            TypeError, "^list indices must be integers or slices, not float$"
        ):
            dataset[0.5]
        with self.assertRaisesRegex(
            TypeError,
            "^'<' not supported between instances of 'slice' and 'int'$",
        ):
            dataset[:]

        with self.assertRaisesRegex(TypeError, "^'NoneType' object is not iterable$"):
            ConcatDataset(None)
        with self.assertRaisesRegex(
            TypeError, r"^object of type 'object' has no len\(\)$"
        ):
            ConcatDataset([object()])
        with self.assertRaisesRegex(
            TypeError,
            r"^ConcatDataset.__init__\(\) missing 1 required positional argument: "
            "'datasets'$",
        ):
            ConcatDataset()
        with self.assertRaisesRegex(
            TypeError,
            r"^ConcatDataset.__init__\(\) takes 2 positional arguments but 3 "
            "were given$",
        ):
            ConcatDataset([], [])
        with self.assertRaisesRegex(
            TypeError,
            r"^ConcatDataset.__init__\(\) got an unexpected keyword argument 'foo'$",
        ):
            ConcatDataset(foo=[])
        with self.assertRaisesRegex(
            TypeError,
            r"^ConcatDataset.__getitem__\(\) missing 1 required positional argument: "
            "'idx'$",
        ):
            dataset.__getitem__()
        with self.assertRaisesRegex(
            TypeError,
            r"^ConcatDataset.__len__\(\) takes 1 positional argument but 2 were given$",
        ):
            dataset.__len__(1)

    def test_imports_metadata_and_deprecated_alias(self):
        data_module = importlib.import_module("torch_rs.utils.data")
        dataset_module = importlib.import_module("torch_rs.utils.data.dataset")

        self.assertIs(data_module.ConcatDataset, ConcatDataset)
        self.assertIs(dataset_module.ConcatDataset, ConcatDataset)
        self.assertTrue(issubclass(ConcatDataset, Dataset))
        self.assertIsInstance(ConcatDataset([[1]]), Dataset)
        self.assertEqual(ConcatDataset.__module__, "torch_rs.utils.data.dataset")
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
                "default_collate",
                "get_worker_info",
            ],
        )
        self.assertEqual(
            dataset_module.__all__,
            [
                "Dataset",
                "IterableDataset",
                "TensorDataset",
                "StackDataset",
                "ConcatDataset",
                "ChainDataset",
                "Subset",
            ],
        )
        for unsupported in ("DataLoader",):
            self.assertFalse(hasattr(data_module, unsupported))
            self.assertFalse(hasattr(dataset_module, unsupported))

        wildcard_namespace = {}
        exec("from torch_rs.utils.data import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["ConcatDataset"], ConcatDataset)
        self.assertNotIn("DataLoader", wildcard_namespace)

        signature = inspect.signature(ConcatDataset)
        self.assertEqual(tuple(signature.parameters), ("datasets",))
        self.assertEqual(signature.parameters["datasets"].annotation, Iterable[Dataset])
        self.assertIs(signature.return_annotation, None)
        self.assertEqual(
            ConcatDataset.__init__.__annotations__,
            {"datasets": Iterable[Dataset], "return": None},
        )
        self.assertIs(get_origin(ConcatDataset.__annotations__["datasets"]), list)
        dataset_annotation = get_args(ConcatDataset.__annotations__["datasets"])[0]
        self.assertIs(get_origin(dataset_annotation), Dataset)
        self.assertIs(
            get_args(dataset_annotation)[0],
            get_args(Subset.__annotations__["dataset"])[0],
        )
        self.assertEqual(ConcatDataset.__annotations__["cumulative_sizes"], list[int])
        self.assertEqual(str(inspect.signature(ConcatDataset.cumsum)), "(sequence)")
        self.assertEqual(str(inspect.signature(ConcatDataset.__getitem__)), "(self, idx)")
        self.assertEqual(str(inspect.signature(ConcatDataset.__len__)), "(self) -> int")
        self.assertEqual(
            str(inspect.signature(Dataset.__add__)),
            "(self, other: 'Dataset[_T_co]') -> 'ConcatDataset[_T_co]'",
        )
        self.assertEqual(
            Dataset.__add__.__annotations__,
            {"other": "Dataset[_T_co]", "return": "ConcatDataset[_T_co]"},
        )
        self.assertIsNone(Dataset.__add__.__doc__)
        self.assertIn(
            "Dataset as a concatenation of multiple datasets", ConcatDataset.__doc__
        )

        deprecated_property = inspect.getattr_static(
            ConcatDataset, "cummulative_sizes"
        )
        self.assertIs(type(deprecated_property), property)
        self.assertEqual(
            deprecated_property.fget.__deprecated__,
            "`cummulative_sizes` attribute is renamed to `cumulative_sizes`",
        )
        self.assertEqual(
            str(inspect.signature(deprecated_property.fget)), "(self)"
        )
        dataset = ConcatDataset([[1], [2, 3]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy_sizes = dataset.cummulative_sizes
        self.assertIs(legacy_sizes, dataset.cumulative_sizes)
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, FutureWarning)
        self.assertEqual(
            str(caught[0].message),
            "`cummulative_sizes` attribute is renamed to `cumulative_sizes`",
        )


if __name__ == "__main__":
    unittest.main()

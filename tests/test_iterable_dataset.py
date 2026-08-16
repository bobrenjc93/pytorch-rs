import importlib
import inspect
import operator
import unittest
from collections.abc import Iterable as IterableABC
from typing import Iterable, get_args, get_origin

import torch_rs as torch

from torch_rs.utils.data import Dataset, IterableDataset


class FiniteStream(IterableDataset[int]):
    def __init__(self, values):
        self.values = values
        self.iteration_calls = 0

    def __iter__(self):
        self.iteration_calls += 1
        yield from self.values


class SizedFiniteStream(FiniteStream):
    def __init__(self, values):
        super().__init__(values)
        self.length_calls = 0

    def __len__(self):
        self.length_calls += 1
        return len(self.values)


class IterableDatasetTests(unittest.TestCase):
    def test_abstract_iteration_and_subclass_iteration(self):
        self.assertTrue(inspect.isabstract(IterableDataset))
        self.assertEqual(IterableDataset.__abstractmethods__, frozenset({"__iter__"}))
        self.assertNotIn("__iter__", IterableDataset.__dict__)

        with self.assertRaises(TypeError) as raised:
            IterableDataset()
        self.assertIn("abstract class IterableDataset", str(raised.exception))
        self.assertIn("'__iter__'", str(raised.exception))

        empty = FiniteStream([])
        finite = FiniteStream([1, 2, 3])
        self.assertEqual(list(empty), [])
        self.assertEqual(list(finite), [1, 2, 3])
        self.assertEqual(list(finite), [1, 2, 3])
        self.assertEqual(empty.iteration_calls, 1)
        self.assertEqual(finite.iteration_calls, 2)
        self.assertIsInstance(finite, Dataset)
        self.assertIsInstance(finite, IterableABC)

    def test_no_default_length_and_optional_subclass_length(self):
        stream = FiniteStream([10, 20])
        self.assertNotIn("__len__", IterableDataset.__dict__)
        self.assertFalse(hasattr(IterableDataset, "__len__"))

        with self.assertRaisesRegex(
            TypeError, r"^object of type 'FiniteStream' has no len\(\)$"
        ):
            len(stream)
        self.assertEqual(operator.length_hint(stream), 0)
        self.assertEqual(list(stream), [10, 20])

        sized = SizedFiniteStream([1, 2])
        self.assertEqual(len(sized), 2)
        self.assertEqual(sized.length_calls, 1)
        sized.values.append(3)
        self.assertEqual(operator.length_hint(sized), 3)
        self.assertEqual(sized.length_calls, 2)

    def test_generic_inheritance_signature_and_metadata(self):
        type_parameter = Dataset.__parameters__[0]

        self.assertEqual(inspect.signature(IterableDataset), inspect.Signature())
        self.assertEqual(IterableDataset.__annotations__, {})
        self.assertEqual(IterableDataset.__parameters__, (type_parameter,))
        self.assertIs(IterableDataset.__bases__[0], Dataset)
        self.assertIn(IterableABC, IterableDataset.__bases__)
        self.assertTrue(issubclass(IterableDataset, Dataset))
        self.assertTrue(issubclass(IterableDataset, IterableABC))

        self.assertEqual(len(IterableDataset.__orig_bases__), 2)
        dataset_base, iterable_base = IterableDataset.__orig_bases__
        self.assertIs(get_origin(dataset_base), Dataset)
        self.assertEqual(get_args(dataset_base), (type_parameter,))
        self.assertIs(get_origin(iterable_base), IterableABC)
        self.assertEqual(get_args(iterable_base), (type_parameter,))
        self.assertEqual(FiniteStream.__orig_bases__, (IterableDataset[int],))

        add_signature = inspect.signature(IterableDataset.__add__)
        self.assertEqual(tuple(add_signature.parameters), ("self", "other"))
        self.assertIs(
            get_origin(add_signature.parameters["other"].annotation), Dataset
        )
        self.assertEqual(
            get_args(add_signature.parameters["other"].annotation), (type_parameter,)
        )
        self.assertIs(add_signature.return_annotation, inspect.Signature.empty)
        self.assertEqual(
            IterableDataset.__add__.__annotations__,
            {"other": Dataset[type_parameter]},
        )
        self.assertIsNone(IterableDataset.__add__.__doc__)

        self.assertEqual(
            IterableDataset.__module__, "torch_rs.utils.data.dataset"
        )
        self.assertEqual(IterableDataset.__name__, "IterableDataset")
        self.assertEqual(IterableDataset.__qualname__, "IterableDataset")
        self.assertIn("An iterable Dataset", IterableDataset.__doc__)
        self.assertIn("All subclasses should overwrite", IterableDataset.__doc__)

    def test_imports_exports_and_out_of_scope_neighbors(self):
        data_module = importlib.import_module("torch_rs.utils.data")
        dataset_module = importlib.import_module("torch_rs.utils.data.dataset")

        self.assertIs(torch.utils.data.IterableDataset, IterableDataset)
        self.assertIs(data_module.IterableDataset, IterableDataset)
        self.assertIs(dataset_module.IterableDataset, IterableDataset)
        self.assertEqual(
            data_module.__all__,
            [
                "BatchSampler",
                "ChainDataset",
                "ConcatDataset",
                "Dataset",
                "IterableDataset",
                "Sampler",
                "SequentialSampler",
                "StackDataset",
                "Subset",
                "TensorDataset",
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

        wildcard_namespace = {}
        exec("from torch_rs.utils.data import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["IterableDataset"], IterableDataset)

        for unsupported in (
            "DataLoader",
            "RandomSampler",
            "default_collate",
            "get_worker_info",
        ):
            with self.subTest(unsupported=unsupported):
                self.assertFalse(hasattr(data_module, unsupported))
                self.assertFalse(hasattr(dataset_module, unsupported))
                self.assertNotIn(unsupported, wildcard_namespace)


if __name__ == "__main__":
    unittest.main()

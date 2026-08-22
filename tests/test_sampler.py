import importlib
import inspect
import operator
import unittest
from collections.abc import Iterator
from typing import Generic, get_args, get_origin

import torch_rs as torch

from torch_rs.utils.data import BatchSampler, Sampler


class IterOnlySampler(Sampler[int]):
    def __init__(self, values):
        self.values = values

    def __iter__(self) -> Iterator[int]:
        return iter(self.values)


class SizedSampler(IterOnlySampler):
    def __len__(self) -> int:
        return len(self.values)


class SamplerTests(unittest.TestCase):
    def test_base_iterator_is_deliberately_unimplemented(self):
        sampler = Sampler()

        for call in (lambda: iter(sampler), sampler.__iter__):
            with self.subTest(call=call):
                with self.assertRaises(NotImplementedError) as raised:
                    call()
                self.assertEqual(raised.exception.args, ())

        self.assertFalse(inspect.isabstract(Sampler))
        self.assertNotIn("__abstractmethods__", Sampler.__dict__)
        self.assertIs(Sampler.__init__, object.__init__)
        with self.assertRaisesRegex(TypeError, r"^Sampler\(\) takes no arguments$"):
            Sampler([0, 1])
        with self.assertRaisesRegex(TypeError, r"^Sampler\(\) takes no arguments$"):
            Sampler(data_source=[0, 1])

    def test_missing_length_preserves_iteration_fallbacks(self):
        sampler = IterOnlySampler([2, 0, 1])

        self.assertNotIn("__len__", Sampler.__dict__)
        self.assertNotIn("__len__", IterOnlySampler.__dict__)
        self.assertEqual(operator.length_hint(sampler), 0)
        self.assertEqual(list(sampler), [2, 0, 1])
        self.assertEqual(tuple(sampler), (2, 0, 1))
        with self.assertRaisesRegex(
            TypeError, r"^object of type 'IterOnlySampler' has no len\(\)$"
        ):
            len(sampler)

        sized = SizedSampler([2, 0, 1])
        self.assertEqual(len(sized), 3)
        self.assertEqual(operator.length_hint(sized), 3)
        self.assertEqual(list(sized), [2, 0, 1])

    def test_generic_covariance_signatures_and_metadata(self):
        (parameter,) = Sampler.__parameters__

        self.assertEqual(parameter.__name__, "_T_co")
        self.assertTrue(parameter.__covariant__)
        self.assertFalse(parameter.__contravariant__)
        self.assertIsNone(parameter.__bound__)
        self.assertEqual(parameter.__constraints__, ())
        self.assertEqual(len(Sampler.__orig_bases__), 1)
        self.assertIs(get_origin(Sampler.__orig_bases__[0]), Generic)
        self.assertIs(get_args(Sampler.__orig_bases__[0])[0], parameter)
        self.assertIs(get_origin(Sampler[int]), Sampler)
        self.assertEqual(get_args(Sampler[int]), (int,))

        self.assertEqual(str(inspect.signature(Sampler)), "()")
        self.assertEqual(
            str(inspect.signature(Sampler.__iter__)),
            "(self) -> collections.abc.Iterator[+_T_co]",
        )
        self.assertEqual(Sampler.__annotations__, {})
        self.assertEqual(tuple(Sampler.__iter__.__annotations__), ("return",))
        return_annotation = Sampler.__iter__.__annotations__["return"]
        self.assertIs(get_origin(return_annotation), Iterator)
        self.assertIs(get_args(return_annotation)[0], parameter)
        self.assertIsNone(Sampler.__iter__.__doc__)

        self.assertEqual(IterOnlySampler.__orig_bases__, (Sampler[int],))
        self.assertIsInstance(IterOnlySampler([]), Sampler)
        self.assertTrue(issubclass(IterOnlySampler, Sampler))
        self.assertIn("Base class for all Samplers.", Sampler.__doc__)
        self.assertIn("isn't strictly required", Sampler.__doc__)

    def test_imports_exports_and_unsupported_surface(self):
        data_module = importlib.import_module("torch_rs.utils.data")
        sampler_module = importlib.import_module("torch_rs.utils.data.sampler")

        self.assertIs(torch.utils.data, data_module)
        self.assertIs(data_module.Sampler, Sampler)
        self.assertIs(sampler_module.Sampler, Sampler)
        self.assertEqual(Sampler.__module__, "torch_rs.utils.data.sampler")
        self.assertEqual(Sampler.__name__, "Sampler")
        self.assertEqual(Sampler.__qualname__, "Sampler")
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
                "default_convert",
                "get_worker_info",
            ],
        )
        self.assertEqual(
            sampler_module.__all__,
            ["BatchSampler", "Sampler", "SequentialSampler"],
        )

        data_namespace = {}
        sampler_namespace = {}
        exec("from torch_rs.utils.data import *", data_namespace)
        exec("from torch_rs.utils.data.sampler import *", sampler_namespace)
        self.assertIs(data_namespace["BatchSampler"], BatchSampler)
        self.assertIs(sampler_namespace["BatchSampler"], BatchSampler)
        self.assertIs(data_namespace["Sampler"], Sampler)
        self.assertIs(sampler_namespace["Sampler"], Sampler)

        unsupported = (
            "DataLoader",
            "RandomSampler",
            "SubsetRandomSampler",
            "WeightedRandomSampler",
        )
        for name in unsupported:
            with self.subTest(name=name):
                self.assertFalse(hasattr(data_module, name))
                self.assertFalse(hasattr(sampler_module, name))
                self.assertNotIn(name, data_namespace)
                self.assertNotIn(name, sampler_namespace)


if __name__ == "__main__":
    unittest.main()

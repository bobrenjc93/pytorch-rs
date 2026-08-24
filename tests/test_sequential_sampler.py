import importlib
import inspect
import operator
import unittest
from collections.abc import Iterator, Sized
from typing import get_args, get_origin

import torch_rs as torch

from torch_rs.utils.data import Sampler, SequentialSampler


class MutableSizedSource:
    def __init__(self, size):
        self.size = size
        self.length_calls = 0

    def __len__(self):
        self.length_calls += 1
        return self.size


class SourceLengthError(Exception):
    pass


class FailingSizedSource:
    def __len__(self):
        raise SourceLengthError("source length is unavailable")


class SequentialSamplerTests(unittest.TestCase):
    def test_iteration_uses_current_data_source_length(self):
        empty = []
        empty_sampler = SequentialSampler(empty)
        self.assertIs(empty_sampler.data_source, empty)
        self.assertEqual(list(empty_sampler), [])

        mutable = [object(), object()]
        mutable_sampler = SequentialSampler(mutable)
        self.assertEqual(list(mutable_sampler), [0, 1])
        mutable.extend((object(), object()))
        self.assertEqual(list(mutable_sampler), [0, 1, 2, 3])
        mutable.clear()
        self.assertEqual(list(mutable_sampler), [])

        custom = MutableSizedSource(3)
        custom_sampler = SequentialSampler(custom)
        self.assertEqual(list(custom_sampler), list(range(len(custom))))
        custom.size = 5
        self.assertEqual(list(custom_sampler), list(range(len(custom))))

    def test_length_delegates_without_caching_and_preserves_errors(self):
        source = MutableSizedSource(2)
        sampler = SequentialSampler(source)

        self.assertEqual(source.length_calls, 0)
        self.assertEqual(len(sampler), 2)
        self.assertEqual(source.length_calls, 1)
        source.size = 6
        self.assertEqual(operator.length_hint(sampler), 6)
        self.assertEqual(source.length_calls, 2)

        failing = SequentialSampler(FailingSizedSource())
        for call in (lambda: len(failing), lambda: iter(failing)):
            with self.subTest(call=call):
                with self.assertRaises(SourceLengthError) as raised:
                    call()
                self.assertEqual(raised.exception.args, ("source length is unavailable",))

    def test_signature_annotations_inheritance_documentation_and_metadata(self):
        signature = inspect.signature(SequentialSampler)
        self.assertEqual(tuple(signature.parameters), ("data_source",))
        parameter = signature.parameters["data_source"]
        self.assertIs(parameter.kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        self.assertIs(parameter.default, inspect.Parameter.empty)
        self.assertIs(parameter.annotation, Sized)
        self.assertIs(signature.return_annotation, None)

        self.assertEqual(SequentialSampler.__annotations__, {"data_source": Sized})
        self.assertEqual(SequentialSampler.__orig_bases__, (Sampler[int],))
        self.assertEqual(SequentialSampler.__parameters__, ())
        self.assertIs(SequentialSampler.__bases__[0], Sampler)
        self.assertIsInstance(SequentialSampler([]), Sampler)

        self.assertEqual(
            SequentialSampler.__init__.__annotations__,
            {"data_source": Sized, "return": None},
        )
        self.assertEqual(
            SequentialSampler.__iter__.__annotations__,
            {"return": Iterator[int]},
        )
        self.assertEqual(SequentialSampler.__len__.__annotations__, {"return": int})
        self.assertIsNone(SequentialSampler.__init__.__doc__)
        self.assertIsNone(SequentialSampler.__iter__.__doc__)
        self.assertIsNone(SequentialSampler.__len__.__doc__)
        self.assertIn("Samples elements sequentially", SequentialSampler.__doc__)
        self.assertIn("Must implement __len__", SequentialSampler.__doc__)

        self.assertIs(get_origin(SequentialSampler.__orig_bases__[0]), Sampler)
        self.assertEqual(get_args(SequentialSampler.__orig_bases__[0]), (int,))

    def test_imports_exports_and_unsupported_neighbors(self):
        data_module = importlib.import_module("torch_rs.utils.data")
        sampler_module = importlib.import_module("torch_rs.utils.data.sampler")

        self.assertIs(torch.utils.data.SequentialSampler, SequentialSampler)
        self.assertIs(data_module.SequentialSampler, SequentialSampler)
        self.assertIs(sampler_module.SequentialSampler, SequentialSampler)
        self.assertEqual(
            SequentialSampler.__module__, "torch_rs.utils.data.sampler"
        )
        self.assertEqual(SequentialSampler.__name__, "SequentialSampler")
        self.assertEqual(SequentialSampler.__qualname__, "SequentialSampler")
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
            sampler_module.__all__,
            ["BatchSampler", "Sampler", "SequentialSampler"],
        )

        data_namespace = {}
        sampler_namespace = {}
        exec("from torch_rs.utils.data import *", data_namespace)
        exec("from torch_rs.utils.data.sampler import *", sampler_namespace)
        self.assertIs(data_namespace["SequentialSampler"], SequentialSampler)
        self.assertIs(sampler_namespace["SequentialSampler"], SequentialSampler)

        for name in (
            "DataLoader",
            "RandomSampler",
            "SubsetRandomSampler",
            "WeightedRandomSampler",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(data_module, name))
                self.assertFalse(hasattr(sampler_module, name))
                self.assertNotIn(name, data_namespace)
                self.assertNotIn(name, sampler_namespace)


if __name__ == "__main__":
    unittest.main()

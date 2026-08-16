import importlib
import inspect
import operator
import unittest
from collections.abc import Iterable, Iterator
from typing import get_args, get_origin

import torch_rs as torch

from torch_rs.utils.data import BatchSampler, Sampler


class LegacyIterable:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return self.values[index]


class IterOnlySampler(Sampler[int]):
    def __init__(self, values):
        self.values = values

    def __iter__(self) -> Iterator[int]:
        return iter(self.values)


class SizedSampler(IterOnlySampler):
    def __len__(self) -> int:
        return len(self.values)


class BatchSamplerTests(unittest.TestCase):
    def test_batches_finite_iterables_with_and_without_remainders(self):
        cases = (
            ([], 3, False, []),
            ([], 3, True, []),
            ([0, 1], 3, False, [[0, 1]]),
            ([0, 1], 3, True, []),
            ([0, 1, 2], 3, False, [[0, 1, 2]]),
            ([0, 1, 2], 3, True, [[0, 1, 2]]),
            (range(7), 3, False, [[0, 1, 2], [3, 4, 5], [6]]),
            (range(7), 3, True, [[0, 1, 2], [3, 4, 5]]),
            (LegacyIterable([5, 2, 9, 1]), 2, False, [[5, 2], [9, 1]]),
            (LegacyIterable([5, 2, 9, 1]), 2, True, [[5, 2], [9, 1]]),
        )
        for sampler, batch_size, drop_last, expected in cases:
            with self.subTest(
                sampler=type(sampler).__name__,
                size=len(sampler) if hasattr(sampler, "__len__") else None,
                batch_size=batch_size,
                drop_last=drop_last,
            ):
                self.assertEqual(
                    list(BatchSampler(sampler, batch_size, drop_last)), expected
                )

        one_shot = iter([10, 11, 12, 13, 14])
        batches = BatchSampler(one_shot, 2, False)
        self.assertEqual(list(batches), [[10, 11], [12, 13], [14]])
        self.assertEqual(list(batches), [])

    def test_construction_validation_and_attributes(self):
        sampler = object()
        batch_sampler = BatchSampler(sampler, 4, False)

        self.assertIs(batch_sampler.sampler, sampler)
        self.assertEqual(batch_sampler.batch_size, 4)
        self.assertIs(batch_sampler.drop_last, False)

        for batch_size in (0, -1, True, False, 1.0, "2", None):
            with self.subTest(batch_size=batch_size):
                with self.assertRaises(ValueError) as raised:
                    BatchSampler([], batch_size, False)
                self.assertEqual(
                    str(raised.exception),
                    "batch_size should be a positive integer value, but got "
                    f"batch_size={batch_size}",
                )

        for drop_last in (0, 1, None, "False", []):
            with self.subTest(drop_last=drop_last):
                with self.assertRaises(ValueError) as raised:
                    BatchSampler([], 1, drop_last)
                self.assertEqual(
                    str(raised.exception),
                    "drop_last should be a boolean value, but got "
                    f"drop_last={drop_last}",
                )

    def test_length_delegates_to_sized_sampler(self):
        sized = SizedSampler(range(7))

        keep_remainder = BatchSampler(sized, 3, False)
        drop_remainder = BatchSampler(sized, 3, True)
        self.assertEqual(len(keep_remainder), 3)
        self.assertEqual(operator.length_hint(keep_remainder), 3)
        self.assertEqual(len(drop_remainder), 2)
        self.assertEqual(operator.length_hint(drop_remainder), 2)

        sized.values = range(8)
        self.assertEqual(len(keep_remainder), 3)
        self.assertEqual(len(drop_remainder), 2)

        for drop_last in (False, True):
            unsized = BatchSampler(IterOnlySampler(range(7)), 3, drop_last)
            with self.subTest(drop_last=drop_last):
                with self.assertRaisesRegex(
                    TypeError, r"^object of type 'IterOnlySampler' has no len\(\)$"
                ):
                    len(unsized)

    def test_signature_annotations_inheritance_and_metadata(self):
        self.assertEqual(
            str(inspect.signature(BatchSampler)),
            "(sampler: Union[torch_rs.utils.data.sampler.Sampler[int], "
            "collections.abc.Iterable[int]], batch_size: int, drop_last: bool) "
            "-> None",
        )
        self.assertEqual(
            tuple(
                name
                for name in BatchSampler.__dict__
                if name != "__annotations__"
            ),
            (
                "__module__",
                "__doc__",
                "__init__",
                "__iter__",
                "__len__",
                "__orig_bases__",
                "__parameters__",
            ),
        )
        self.assertEqual(BatchSampler.__annotations__, {})
        self.assertEqual(BatchSampler.__orig_bases__, (Sampler[list[int]],))
        self.assertIs(BatchSampler.__bases__[0], Sampler)
        self.assertIsInstance(BatchSampler([], 1, False), Sampler)

        init_annotations = BatchSampler.__init__.__annotations__
        self.assertEqual(
            tuple(init_annotations),
            ("sampler", "batch_size", "drop_last", "return"),
        )
        sampler_annotation = init_annotations["sampler"]
        self.assertEqual(get_origin(sampler_annotation).__name__, "Union")
        self.assertEqual(get_args(sampler_annotation), (Sampler[int], Iterable[int]))
        self.assertIs(init_annotations["batch_size"], int)
        self.assertIs(init_annotations["drop_last"], bool)
        self.assertIsNone(init_annotations["return"])
        self.assertEqual(
            BatchSampler.__iter__.__annotations__, {"return": Iterator[list[int]]}
        )
        self.assertEqual(BatchSampler.__len__.__annotations__, {"return": int})
        self.assertIsNone(BatchSampler.__init__.__doc__)
        self.assertIsNone(BatchSampler.__iter__.__doc__)
        self.assertIsNone(BatchSampler.__len__.__doc__)
        self.assertIn("Wraps another sampler", BatchSampler.__doc__)

    def test_imports_and_exports(self):
        data_module = importlib.import_module("torch_rs.utils.data")
        sampler_module = importlib.import_module("torch_rs.utils.data.sampler")

        self.assertIs(torch.utils.data.BatchSampler, BatchSampler)
        self.assertIs(data_module.BatchSampler, BatchSampler)
        self.assertIs(sampler_module.BatchSampler, BatchSampler)
        self.assertEqual(BatchSampler.__module__, "torch_rs.utils.data.sampler")
        self.assertEqual(BatchSampler.__name__, "BatchSampler")
        self.assertEqual(BatchSampler.__qualname__, "BatchSampler")
        self.assertIn("BatchSampler", data_module.__all__)
        self.assertIn("BatchSampler", sampler_module.__all__)


if __name__ == "__main__":
    unittest.main()

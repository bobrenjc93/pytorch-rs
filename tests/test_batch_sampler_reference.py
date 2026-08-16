import importlib
import inspect
import operator
import unittest
from typing import get_args, get_origin

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class BatchSamplerReferenceTests(unittest.TestCase):
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

    def test_construction_validation_and_call_forms_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_type = torch.utils.data.BatchSampler
        expected_type = reference_torch.utils.data.BatchSampler

        for batch_size in (0, -1, True, False, 1.0, "2", None):
            with self.subTest(batch_size=batch_size):
                self.assert_error_matches(
                    lambda batch_size=batch_size: actual_type([], batch_size, False),
                    lambda batch_size=batch_size: expected_type([], batch_size, False),
                )

        for drop_last in (0, 1, None, "False", []):
            with self.subTest(drop_last=drop_last):
                self.assert_error_matches(
                    lambda drop_last=drop_last: actual_type([], 1, drop_last),
                    lambda drop_last=drop_last: expected_type([], 1, drop_last),
                )

        error_pairs = (
            (lambda: actual_type(), lambda: expected_type()),
            (lambda: actual_type([], 1), lambda: expected_type([], 1)),
            (
                lambda: actual_type([], 1, False, None),
                lambda: expected_type([], 1, False, None),
            ),
            (
                lambda: actual_type(sampler=[], batch_size=1),
                lambda: expected_type(sampler=[], batch_size=1),
            ),
            (
                lambda: actual_type([], 1, sampler=[]),
                lambda: expected_type([], 1, sampler=[]),
            ),
            (
                lambda: actual_type([], 1, False, extra=True),
                lambda: expected_type([], 1, False, extra=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(error_pairs):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        actual_sampler = object()
        expected_sampler = object()
        actual = actual_type(actual_sampler, 3, True)
        expected = expected_type(expected_sampler, 3, True)
        self.assertIs(actual.sampler, actual_sampler)
        self.assertIs(expected.sampler, expected_sampler)
        self.assertEqual(
            actual.__dict__, expected.__dict__ | {"sampler": actual_sampler}
        )

    def test_arbitrary_finite_iterables_and_remainders_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_type = torch.utils.data.BatchSampler
        expected_type = reference_torch.utils.data.BatchSampler

        factories = (
            lambda: [],
            lambda: [8],
            lambda: range(7),
            lambda: iter([5, 4, 3, 2, 1]),
            lambda: (value * 2 for value in range(6)),
            lambda: "abcd",
        )
        for factory in factories:
            for batch_size in (1, 2, 3, 8):
                for drop_last in (False, True):
                    with self.subTest(
                        iterable=type(factory()).__name__,
                        batch_size=batch_size,
                        drop_last=drop_last,
                    ):
                        self.assertEqual(
                            list(actual_type(factory(), batch_size, drop_last)),
                            list(expected_type(factory(), batch_size, drop_last)),
                        )

    def test_length_and_unsized_failures_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_type = torch.utils.data.BatchSampler
        expected_type = reference_torch.utils.data.BatchSampler

        for size in range(9):
            for batch_size in (1, 2, 3, 10):
                for drop_last in (False, True):
                    actual = actual_type(range(size), batch_size, drop_last)
                    expected = expected_type(range(size), batch_size, drop_last)
                    with self.subTest(
                        size=size,
                        batch_size=batch_size,
                        drop_last=drop_last,
                    ):
                        self.assertEqual(len(actual), len(expected))
                        self.assertEqual(
                            operator.length_hint(actual), operator.length_hint(expected)
                        )

        for factory in (lambda: iter(range(4)), lambda: (x for x in range(4))):
            for drop_last in (False, True):
                actual = actual_type(factory(), 3, drop_last)
                expected = expected_type(factory(), 3, drop_last)
                with self.subTest(
                    sampler=type(factory()).__name__, drop_last=drop_last
                ):
                    self.assert_error_matches(
                        lambda: len(actual), lambda: len(expected)
                    )

    def test_signature_annotations_documentation_and_metadata_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.utils.data.BatchSampler
        expected = reference_torch.utils.data.BatchSampler

        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(inspect.isabstract(actual), inspect.isabstract(expected))
        actual_annotations = actual.__annotations__
        expected_annotations = expected.__annotations__
        self.assertEqual(tuple(actual.__dict__), tuple(expected.__dict__))
        self.assertEqual(
            str(inspect.signature(actual)).replace("torch_rs", "torch"),
            str(inspect.signature(expected)),
        )
        self.assertEqual(actual_annotations, expected_annotations)
        self.assertEqual(
            str(actual.__orig_bases__).replace("torch_rs", "torch"),
            str(expected.__orig_bases__),
        )
        self.assertEqual(actual.__parameters__, expected.__parameters__)
        self.assertIs(actual.__bases__[0], torch.utils.data.Sampler)
        self.assertIs(expected.__bases__[0], reference_torch.utils.data.Sampler)

        for name in ("__init__", "__iter__", "__len__"):
            with self.subTest(name=name):
                actual_method = getattr(actual, name)
                expected_method = getattr(expected, name)
                self.assertEqual(
                    str(inspect.signature(actual_method)).replace("torch_rs", "torch"),
                    str(inspect.signature(expected_method)),
                )
                self.assertEqual(actual_method.__doc__, expected_method.__doc__)
                self.assertEqual(
                    str(actual_method.__annotations__).replace("torch_rs", "torch"),
                    str(expected_method.__annotations__),
                )

        actual_origin = get_origin(actual.__orig_bases__[0])
        expected_origin = get_origin(expected.__orig_bases__[0])
        self.assertIs(actual_origin, torch.utils.data.Sampler)
        self.assertIs(expected_origin, reference_torch.utils.data.Sampler)
        self.assertEqual(get_args(actual.__orig_bases__[0]), (list[int],))
        self.assertEqual(get_args(expected.__orig_bases__[0]), (list[int],))

    def test_imports_exports_and_unsupported_neighbors_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_data = importlib.import_module("torch_rs.utils.data")
        expected_data = importlib.import_module("torch.utils.data")
        actual_module = importlib.import_module("torch_rs.utils.data.sampler")
        expected_module = importlib.import_module("torch.utils.data.sampler")

        self.assertIs(actual_data.BatchSampler, actual_module.BatchSampler)
        self.assertIs(expected_data.BatchSampler, expected_module.BatchSampler)
        self.assertEqual(
            actual_data.__all__,
            [
                name
                for name in expected_data.__all__
                if name
                in {
                    "BatchSampler",
                    "ChainDataset",
                    "ConcatDataset",
                    "DataChunk",
                    "Dataset",
                    "IterableDataset",
                    "Sampler",
                    "SequentialSampler",
                    "StackDataset",
                    "Subset",
                    "TensorDataset",
                }
            ],
        )
        self.assertEqual(
            actual_module.__all__,
            [
                name
                for name in expected_module.__all__
                if name in {"BatchSampler", "Sampler", "SequentialSampler"}
            ],
        )

        for name in (
            "DataLoader",
            "RandomSampler",
            "SubsetRandomSampler",
            "WeightedRandomSampler",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(actual_data, name))
                self.assertFalse(hasattr(actual_module, name))


if __name__ == "__main__":
    unittest.main()

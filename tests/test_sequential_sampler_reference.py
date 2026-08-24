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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SequentialSamplerReferenceTests(unittest.TestCase):
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

    def test_empty_mutable_and_custom_sized_iteration_matches(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_type = torch.utils.data.SequentialSampler
        expected_type = reference_torch.utils.data.SequentialSampler

        for size in (0, 1, 5):
            actual_source = MutableSizedSource(size)
            expected_source = MutableSizedSource(size)
            actual = actual_type(actual_source)
            expected = expected_type(expected_source)
            with self.subTest(size=size):
                self.assertEqual(type(iter(actual)), type(iter(expected)))
                self.assertEqual(list(actual), list(expected))

        actual_source = [object(), object()]
        expected_source = [object(), object()]
        actual = actual_type(actual_source)
        expected = expected_type(expected_source)
        self.assertEqual(list(actual), list(expected))
        actual_source.extend((object(), object(), object()))
        expected_source.extend((object(), object(), object()))
        self.assertEqual(list(actual), list(expected))
        actual_source.clear()
        expected_source.clear()
        self.assertEqual(list(actual), list(expected))

    def test_length_delegation_and_source_errors_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_source = MutableSizedSource(2)
        expected_source = MutableSizedSource(2)
        actual = torch.utils.data.SequentialSampler(actual_source)
        expected = reference_torch.utils.data.SequentialSampler(expected_source)

        self.assertEqual(len(actual), len(expected))
        self.assertEqual(actual_source.length_calls, expected_source.length_calls)
        actual_source.size = 7
        expected_source.size = 7
        self.assertEqual(operator.length_hint(actual), operator.length_hint(expected))
        self.assertEqual(actual_source.length_calls, expected_source.length_calls)

        actual_failing = torch.utils.data.SequentialSampler(FailingSizedSource())
        expected_failing = reference_torch.utils.data.SequentialSampler(
            FailingSizedSource()
        )
        for actual_call, expected_call in (
            (lambda: len(actual_failing), lambda: len(expected_failing)),
            (lambda: iter(actual_failing), lambda: iter(expected_failing)),
        ):
            self.assert_error_matches(actual_call, expected_call)

    def test_construction_call_forms_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_type = torch.utils.data.SequentialSampler
        expected_type = reference_torch.utils.data.SequentialSampler

        actual_source = object()
        expected_source = object()
        actual = actual_type(actual_source)
        expected = expected_type(expected_source)
        self.assertIs(actual.data_source, actual_source)
        self.assertIs(expected.data_source, expected_source)

        for actual_call, expected_call in (
            (lambda: actual_type(), lambda: expected_type()),
            (lambda: actual_type([], []), lambda: expected_type([], [])),
            (
                lambda: actual_type(data_source=[], extra=True),
                lambda: expected_type(data_source=[], extra=True),
            ),
            (
                lambda: actual_type([], data_source=[]),
                lambda: expected_type([], data_source=[]),
            ),
        ):
            self.assert_error_matches(actual_call, expected_call)

    def test_signature_annotations_documentation_and_metadata_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.utils.data.SequentialSampler
        expected = reference_torch.utils.data.SequentialSampler

        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(inspect.isabstract(actual), inspect.isabstract(expected))
        # Python 3.14 may materialize private annotation caches at different
        # times. Compare the stable class-defined method surface and the public
        # metadata below instead of the complete implementation namespace.
        declared_methods = ("__init__", "__iter__", "__len__")
        self.assertEqual(
            tuple(name for name in declared_methods if name in actual.__dict__),
            declared_methods,
        )
        self.assertEqual(
            tuple(name for name in declared_methods if name in expected.__dict__),
            declared_methods,
        )
        self.assertEqual(
            str(inspect.signature(actual)).replace("torch_rs", "torch"),
            str(inspect.signature(expected)),
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(
            str(actual.__orig_bases__).replace("torch_rs", "torch"),
            str(expected.__orig_bases__),
        )
        self.assertEqual(actual.__parameters__, expected.__parameters__)
        self.assertIs(actual.__bases__[0], torch.utils.data.Sampler)
        self.assertIs(expected.__bases__[0], reference_torch.utils.data.Sampler)

        for name in declared_methods:
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

        self.assertIs(get_origin(actual.__orig_bases__[0]), torch.utils.data.Sampler)
        self.assertIs(
            get_origin(expected.__orig_bases__[0]),
            reference_torch.utils.data.Sampler,
        )
        self.assertEqual(get_args(actual.__orig_bases__[0]), (int,))
        self.assertEqual(get_args(expected.__orig_bases__[0]), (int,))

    def test_imports_exports_and_unsupported_neighbors_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_data = importlib.import_module("torch_rs.utils.data")
        expected_data = importlib.import_module("torch.utils.data")
        actual_module = importlib.import_module("torch_rs.utils.data.sampler")
        expected_module = importlib.import_module("torch.utils.data.sampler")
        supported_data = {
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
        }
        supported_sampler = {"BatchSampler", "Sampler", "SequentialSampler"}

        self.assertIs(actual_data.SequentialSampler, actual_module.SequentialSampler)
        self.assertIs(
            expected_data.SequentialSampler, expected_module.SequentialSampler
        )
        self.assertEqual(
            actual_data.__all__,
            [name for name in expected_data.__all__ if name in supported_data],
        )
        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name in supported_sampler],
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

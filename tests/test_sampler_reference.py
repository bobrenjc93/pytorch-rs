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
class SamplerReferenceTests(unittest.TestCase):
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

    @staticmethod
    def make_iter_only_sampler(base):
        class IterOnlySampler(base[int]):
            def __init__(self, values):
                self.values = values

            def __iter__(self):
                return iter(self.values)

        return IterOnlySampler

    def test_imports_exports_and_unsupported_surface_match_pytorch_2_13(self):
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
            "get_worker_info",
        }

        self.assertIs(actual_data.Sampler, actual_module.Sampler)
        self.assertIs(expected_data.Sampler, expected_module.Sampler)
        self.assertEqual(
            actual_data.__all__,
            [name for name in expected_data.__all__ if name in supported_data],
        )
        self.assertEqual(
            actual_module.__all__,
            [
                name
                for name in expected_module.__all__
                if name in {"BatchSampler", "Sampler", "SequentialSampler"}
            ],
        )

        unsupported = (
            "DataLoader",
            "RandomSampler",
            "SubsetRandomSampler",
            "WeightedRandomSampler",
        )
        for name in unsupported:
            with self.subTest(name=name):
                self.assertFalse(hasattr(actual_data, name))
                self.assertFalse(hasattr(actual_module, name))

    def test_signatures_annotations_documentation_and_metadata_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.utils.data.Sampler
        expected = reference_torch.utils.data.Sampler

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

        actual_parameter = actual.__parameters__[0]
        expected_parameter = expected.__parameters__[0]
        for attribute in (
            "__name__",
            "__covariant__",
            "__contravariant__",
            "__bound__",
            "__constraints__",
        ):
            self.assertEqual(
                getattr(actual_parameter, attribute),
                getattr(expected_parameter, attribute),
            )
        self.assertIs(get_origin(actual[int]), actual)
        self.assertIs(get_origin(expected[int]), expected)
        self.assertEqual(get_args(actual[int]), get_args(expected[int]))

        self.assertEqual(
            str(inspect.signature(actual.__iter__)),
            str(inspect.signature(expected.__iter__)),
        )
        self.assertEqual(actual.__iter__.__doc__, expected.__iter__.__doc__)
        self.assertEqual(
            str(actual.__iter__.__annotations__),
            str(expected.__iter__.__annotations__),
        )
        self.assertIs(actual.__init__, object.__init__)
        self.assertIs(expected.__init__, object.__init__)
        self.assertNotIn("__len__", actual.__dict__)
        self.assertNotIn("__len__", expected.__dict__)

    def test_base_errors_and_subclass_iteration_fallbacks_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_base = torch.utils.data.Sampler
        expected_base = reference_torch.utils.data.Sampler

        error_pairs = (
            (lambda: iter(actual_base()), lambda: iter(expected_base())),
            (lambda: actual_base().__iter__(), lambda: expected_base().__iter__()),
            (lambda: len(actual_base()), lambda: len(expected_base())),
            (lambda: actual_base([0]), lambda: expected_base([0])),
            (
                lambda: actual_base(data_source=[0]),
                lambda: expected_base(data_source=[0]),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(error_pairs):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        actual_type = self.make_iter_only_sampler(actual_base)
        expected_type = self.make_iter_only_sampler(expected_base)
        actual = actual_type([2, 0, 1])
        expected = expected_type([2, 0, 1])

        self.assertEqual(operator.length_hint(actual), operator.length_hint(expected))
        self.assertEqual(list(actual), list(expected))
        self.assertEqual(tuple(actual), tuple(expected))
        self.assert_error_matches(lambda: len(actual), lambda: len(expected))
        self.assertEqual(
            str(actual_type.__orig_bases__).replace("torch_rs", "torch"),
            str(expected_type.__orig_bases__),
        )
        self.assertIsInstance(actual, actual_base)
        self.assertIsInstance(expected, expected_base)


if __name__ == "__main__":
    unittest.main()

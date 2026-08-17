import importlib
import inspect
import operator
import unittest
from collections.abc import Iterable as IterableABC
from typing import get_args, get_origin

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class IterableDatasetReferenceTests(unittest.TestCase):
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

    def make_stream(self, module, values, *, sized=False):
        base = module.utils.data.IterableDataset

        if sized:

            class FiniteStream(base):
                def __init__(self):
                    self.values = values
                    self.iteration_calls = 0
                    self.length_calls = 0

                def __iter__(self):
                    self.iteration_calls += 1
                    yield from self.values

                def __len__(self):
                    self.length_calls += 1
                    return len(self.values)

        else:

            class FiniteStream(base):
                def __init__(self):
                    self.values = values
                    self.iteration_calls = 0

                def __iter__(self):
                    self.iteration_calls += 1
                    yield from self.values

        return FiniteStream()

    def test_abstract_iteration_and_no_default_length_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_type = torch.utils.data.IterableDataset
        expected_type = reference_torch.utils.data.IterableDataset

        self.assertEqual(inspect.isabstract(actual_type), inspect.isabstract(expected_type))
        self.assertEqual(actual_type.__abstractmethods__, expected_type.__abstractmethods__)
        self.assertEqual("__iter__" in actual_type.__dict__, "__iter__" in expected_type.__dict__)
        self.assertEqual("__len__" in actual_type.__dict__, "__len__" in expected_type.__dict__)
        self.assertEqual(hasattr(actual_type, "__len__"), hasattr(expected_type, "__len__"))

        actual_missing = type("MissingIteration", (actual_type,), {})
        expected_missing = type("MissingIteration", (expected_type,), {})
        for actual_call, expected_call in (
            (lambda: actual_type(), lambda: expected_type()),
            (lambda: actual_type(1), lambda: expected_type(1)),
            (lambda: actual_missing(), lambda: expected_missing()),
        ):
            self.assert_error_matches(actual_call, expected_call)

        actual = self.make_stream(torch, [1, 2, 3])
        expected = self.make_stream(reference_torch, [1, 2, 3])
        self.assertEqual(type(iter(actual)).__name__, type(iter(expected)).__name__)
        self.assertEqual(list(actual), list(expected))
        self.assertEqual(list(actual), list(expected))
        self.assertEqual(actual.iteration_calls, expected.iteration_calls)
        self.assertEqual(operator.length_hint(actual), operator.length_hint(expected))
        self.assert_error_matches(lambda: len(actual), lambda: len(expected))

        actual_sized = self.make_stream(torch, [10, 20], sized=True)
        expected_sized = self.make_stream(reference_torch, [10, 20], sized=True)
        self.assertEqual(len(actual_sized), len(expected_sized))
        actual_sized.values.append(30)
        expected_sized.values.append(30)
        self.assertEqual(
            operator.length_hint(actual_sized), operator.length_hint(expected_sized)
        )
        self.assertEqual(actual_sized.length_calls, expected_sized.length_calls)

    def test_addition_is_lazy_and_selects_chain_dataset(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_left = self.make_stream(torch, [])
        expected_left = self.make_stream(reference_torch, [])
        actual_right = self.make_stream(torch, [1, 2])
        expected_right = self.make_stream(reference_torch, [1, 2])

        actual = actual_left + actual_right
        expected = expected_left + expected_right
        self.assertEqual(type(actual).__name__, type(expected).__name__)
        self.assertIs(actual.datasets[0], actual_left)
        self.assertIs(expected.datasets[0], expected_left)
        self.assertIs(actual.datasets[1], actual_right)
        self.assertIs(expected.datasets[1], expected_right)
        self.assertEqual(actual_left.iteration_calls, expected_left.iteration_calls)
        self.assertEqual(actual_right.iteration_calls, expected_right.iteration_calls)
        self.assertEqual(list(actual), list(expected))
        self.assertEqual(actual_left.iteration_calls, expected_left.iteration_calls)
        self.assertEqual(actual_right.iteration_calls, expected_right.iteration_calls)

        for actual_call, expected_call in (
            (lambda: actual_left.__add__(), lambda: expected_left.__add__()),
            (
                lambda: actual_left.__add__(actual_right, actual_right),
                lambda: expected_left.__add__(expected_right, expected_right),
            ),
        ):
            self.assert_error_matches(actual_call, expected_call)

    def test_signature_generic_inheritance_docs_and_metadata_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.utils.data.IterableDataset
        expected = reference_torch.utils.data.IterableDataset

        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(inspect.isabstract(actual), inspect.isabstract(expected))
        self.assertEqual(
            str(inspect.signature(actual)).replace("torch_rs", "torch"),
            str(inspect.signature(expected)),
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(
            str(actual.__orig_bases__).replace("torch_rs", "torch"),
            str(expected.__orig_bases__),
        )
        self.assertEqual(str(actual.__parameters__), str(expected.__parameters__))

        self.assertIs(actual.__bases__[0], torch.utils.data.Dataset)
        self.assertIs(expected.__bases__[0], reference_torch.utils.data.Dataset)
        self.assertIn(IterableABC, actual.__bases__)
        self.assertIn(IterableABC, expected.__bases__)
        self.assertEqual(
            tuple(
                name
                for name in ("__add__", "__iter__", "__len__")
                if name in actual.__dict__
            ),
            tuple(
                name
                for name in ("__add__", "__iter__", "__len__")
                if name in expected.__dict__
            ),
        )

        actual_parameter = actual.__parameters__[0]
        expected_parameter = expected.__parameters__[0]
        self.assertEqual(actual_parameter.__name__, expected_parameter.__name__)
        self.assertEqual(actual_parameter.__covariant__, expected_parameter.__covariant__)
        for actual_base, expected_base in zip(
            actual.__orig_bases__, expected.__orig_bases__, strict=True
        ):
            self.assertEqual(
                getattr(get_origin(actual_base), "__name__", None),
                getattr(get_origin(expected_base), "__name__", None),
            )
            self.assertEqual(
                tuple(str(arg) for arg in get_args(actual_base)),
                tuple(str(arg) for arg in get_args(expected_base)),
            )

        actual_add = actual.__add__
        expected_add = expected.__add__
        self.assertEqual(
            str(inspect.signature(actual_add)).replace("torch_rs", "torch"),
            str(inspect.signature(expected_add)),
        )
        self.assertEqual(actual_add.__doc__, expected_add.__doc__)
        self.assertEqual(
            str(actual_add.__annotations__).replace("torch_rs", "torch"),
            str(expected_add.__annotations__),
        )

    def test_imports_exports_and_out_of_scope_neighbors_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_data = importlib.import_module("torch_rs.utils.data")
        expected_data = importlib.import_module("torch.utils.data")
        actual_module = importlib.import_module("torch_rs.utils.data.dataset")
        expected_module = importlib.import_module("torch.utils.data.dataset")
        supported = {
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
            "get_worker_info",
        }

        self.assertIs(actual_data.IterableDataset, actual_module.IterableDataset)
        self.assertIs(expected_data.IterableDataset, expected_module.IterableDataset)
        self.assertEqual(
            actual_data.__all__,
            [name for name in expected_data.__all__ if name in supported],
        )
        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name in supported],
        )

        actual_namespace = {}
        exec("from torch_rs.utils.data import *", actual_namespace)
        self.assertIs(
            actual_namespace["IterableDataset"], actual_data.IterableDataset
        )

        for unsupported in (
            "DataLoader",
            "RandomSampler",
            "default_collate",
        ):
            with self.subTest(unsupported=unsupported):
                self.assertFalse(hasattr(actual_data, unsupported))
                self.assertFalse(hasattr(actual_module, unsupported))
                self.assertNotIn(unsupported, actual_namespace)


if __name__ == "__main__":
    unittest.main()

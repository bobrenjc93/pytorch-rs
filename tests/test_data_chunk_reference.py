import importlib
import inspect
import pickle
import unittest
from typing import get_args, get_origin

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DataChunkReferenceTests(unittest.TestCase):
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

    def make_mutated_chunk(self, data_chunk):
        shared = ["shared"]
        chunk = data_chunk([shared, "common", 3])
        chunk.append("list-only")
        chunk[1] = "list-mutated"
        chunk.items.append("raw-only")
        chunk.items[2] = "raw-mutated"
        shared.append("nested-mutated")
        return chunk

    def make_cases(self, data_chunk):
        return {
            "empty": data_chunk([]),
            "generator": data_chunk(
                value for value in (["generated"], {"value": 2})
            ),
            "nested": data_chunk([data_chunk([1, 2]), ["tail"]]),
            "mutated": self.make_mutated_chunk(data_chunk),
        }

    def snapshot(self, chunk):
        return {
            "list": list(chunk),
            "raw": list(chunk.raw_iterator()),
            "items": chunk.items,
            "str": str(chunk),
            "repr": repr(chunk),
            "as_str": chunk.as_str(),
            "indented": chunk.as_str("  "),
            "dict": chunk.__dict__,
            "list_types": [type(item).__name__ for item in chunk],
            "raw_types": [type(item).__name__ for item in chunk.raw_iterator()],
            "items_is_self": chunk.items is chunk,
        }

    def test_exports_generic_inheritance_and_metadata_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_data = importlib.import_module("torch_rs.utils.data")
        expected_data = importlib.import_module("torch.utils.data")
        actual_package = importlib.import_module("torch_rs.utils.data.datapipes")
        expected_package = importlib.import_module("torch.utils.data.datapipes")
        actual_module = importlib.import_module(
            "torch_rs.utils.data.datapipes.datapipe"
        )
        expected_module = importlib.import_module("torch.utils.data.datapipes.datapipe")
        actual = actual_data.DataChunk
        expected = expected_data.DataChunk
        supported = {
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

        self.assertIs(actual, actual_module.DataChunk)
        self.assertIs(expected, expected_module.DataChunk)
        self.assertEqual(
            actual_data.__all__,
            [name for name in expected_data.__all__ if name in supported],
        )
        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name == "DataChunk"],
        )
        self.assertEqual(
            hasattr(actual_package, "DataChunk"),
            hasattr(expected_package, "DataChunk"),
        )
        for unsupported in ("DFIterDataPipe", "IterDataPipe", "MapDataPipe"):
            with self.subTest(unsupported=unsupported):
                self.assertFalse(hasattr(actual_data, unsupported))
                self.assertFalse(hasattr(actual_module, unsupported))

        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(
            tuple(base.__name__ for base in actual.__bases__),
            tuple(base.__name__ for base in expected.__bases__),
        )
        self.assertEqual(
            tuple(base.__name__ for base in actual.__mro__),
            tuple(base.__name__ for base in expected.__mro__),
        )

        (actual_generic_base,) = actual.__orig_bases__
        (expected_generic_base,) = expected.__orig_bases__
        self.assertIs(get_origin(actual_generic_base), list)
        self.assertIs(get_origin(expected_generic_base), list)
        (actual_type_parameter,) = get_args(actual_generic_base)
        (expected_type_parameter,) = get_args(expected_generic_base)
        for attribute in (
            "__name__",
            "__covariant__",
            "__contravariant__",
            "__constraints__",
            "__bound__",
        ):
            self.assertEqual(
                getattr(actual_type_parameter, attribute),
                getattr(expected_type_parameter, attribute),
            )
        self.assertEqual(
            hasattr(actual, "__parameters__"),
            hasattr(expected, "__parameters__"),
        )
        self.assertEqual(
            repr(actual[int]).replace("torch_rs", "torch"), repr(expected[int])
        )

        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        for name in ("__init__", "__iter__", "raw_iterator", "as_str"):
            actual_method = getattr(actual, name)
            expected_method = getattr(expected, name)
            with self.subTest(method=name):
                self.assertEqual(
                    str(inspect.signature(actual_method)),
                    str(inspect.signature(expected_method)),
                )
                self.assertEqual(
                    str(actual_method.__annotations__),
                    str(expected_method.__annotations__),
                )
                self.assertEqual(actual_method.__doc__, expected_method.__doc__)
                self.assertEqual(
                    actual_method.__module__.replace("torch_rs", "torch"),
                    expected_method.__module__,
                )
                self.assertEqual(actual_method.__name__, expected_method.__name__)
                self.assertEqual(
                    actual_method.__qualname__, expected_method.__qualname__
                )

    def test_empty_generator_nested_and_mutated_behavior_matches(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_cases = self.make_cases(torch.utils.data.DataChunk)
        expected_cases = self.make_cases(reference_torch.utils.data.DataChunk)

        for name in actual_cases:
            with self.subTest(name=name):
                actual = actual_cases[name]
                expected = expected_cases[name]
                self.assertEqual(self.snapshot(actual), self.snapshot(expected))
                self.assertIsNot(actual.items, actual)
                self.assertIsNot(expected.items, expected)
                if actual:
                    self.assertEqual(
                        actual[0] is actual.items[0],
                        expected[0] is expected.items[0],
                    )

        actual = self.make_mutated_chunk(torch.utils.data.DataChunk)
        expected = self.make_mutated_chunk(reference_torch.utils.data.DataChunk)
        actual_iterator = iter(actual)
        expected_iterator = iter(expected)
        actual_raw_iterator = actual.raw_iterator()
        expected_raw_iterator = expected.raw_iterator()
        self.assertEqual(next(actual_iterator), next(expected_iterator))
        self.assertEqual(next(actual_raw_iterator), next(expected_raw_iterator))
        actual.append("late-list")
        expected.append("late-list")
        actual.items.append("late-raw")
        expected.items.append("late-raw")
        self.assertEqual(list(actual_iterator), list(expected_iterator))
        self.assertEqual(list(actual_raw_iterator), list(expected_raw_iterator))
        self.assertNotEqual(list(actual), list(actual.raw_iterator()))

    def test_pickle_round_trips_match_for_all_supported_inputs_and_protocols(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_cases = self.make_cases(torch.utils.data.DataChunk)
        expected_cases = self.make_cases(reference_torch.utils.data.DataChunk)

        for name in actual_cases:
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    actual = pickle.loads(
                        pickle.dumps(actual_cases[name], protocol=protocol)
                    )
                    expected = pickle.loads(
                        pickle.dumps(expected_cases[name], protocol=protocol)
                    )
                    self.assertEqual(type(actual).__name__, type(expected).__name__)
                    self.assertEqual(self.snapshot(actual), self.snapshot(expected))
                    if actual:
                        self.assertEqual(
                            actual[0] is actual.items[0],
                            expected[0] is expected.items[0],
                        )
                    if name == "nested":
                        self.assertEqual(
                            type(actual[0]).__name__, type(expected[0]).__name__
                        )
                        self.assertEqual(
                            actual[0].items is actual[0],
                            expected[0].items is expected[0],
                        )

    def test_constructor_and_method_errors_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.utils.data.DataChunk
        expected = reference_torch.utils.data.DataChunk
        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual([], []), lambda: expected([], [])),
            (lambda: actual(None), lambda: expected(None)),
            (
                lambda: actual([]).as_str(None),
                lambda: expected([]).as_str(None),
            ),
            (
                lambda: actual([]).as_str("x", "y"),
                lambda: expected([]).as_str("x", "y"),
            ),
            (
                lambda: actual([]).raw_iterator(1),
                lambda: expected([]).raw_iterator(1),
            ),
            (
                lambda: actual([]).__iter__(1),
                lambda: expected([]).__iter__(1),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()

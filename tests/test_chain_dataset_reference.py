import importlib
import inspect
import operator
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class StreamLengthError(Exception):
    pass


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ChainDatasetReferenceTests(unittest.TestCase):
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

    def make_stream(self, module, name, values, events=None, *, sized=True):
        base = module.utils.data.IterableDataset

        if sized:

            class RecordingStream(base):
                def __init__(self):
                    self.name = name
                    self.values = values
                    self.events = [] if events is None else events
                    self.iteration_calls = 0
                    self.length_calls = 0

                def __iter__(self):
                    self.iteration_calls += 1
                    self.events.append(f"iter:{self.name}")
                    yield from self.values

                def __len__(self):
                    self.length_calls += 1
                    self.events.append(f"len:{self.name}")
                    return len(self.values)

        else:

            class RecordingStream(base):
                def __init__(self):
                    self.name = name
                    self.values = values
                    self.events = [] if events is None else events
                    self.iteration_calls = 0

                def __iter__(self):
                    self.iteration_calls += 1
                    self.events.append(f"iter:{self.name}")
                    yield from self.values

        return RecordingStream()

    def make_map_dataset(self, module, values):
        base = module.utils.data.Dataset

        class ListDataset(base):
            def __getitem__(self, index):
                return values[index]

            def __len__(self):
                return len(values)

        return ListDataset()

    def test_lazy_empty_and_finite_stream_concatenation_matches(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_events = []
        expected_events = []
        actual_children = [
            self.make_stream(torch, "empty-1", [], actual_events),
            self.make_stream(torch, "first", [1, 2], actual_events),
            self.make_stream(torch, "empty-2", [], actual_events),
            self.make_stream(torch, "last", [3], actual_events),
        ]
        expected_children = [
            self.make_stream(reference_torch, "empty-1", [], expected_events),
            self.make_stream(reference_torch, "first", [1, 2], expected_events),
            self.make_stream(reference_torch, "empty-2", [], expected_events),
            self.make_stream(reference_torch, "last", [3], expected_events),
        ]

        def actual_source():
            actual_events.append("source:start")
            for child in actual_children:
                actual_events.append(f"source:{child.name}")
                yield child
            actual_events.append("source:end")

        def expected_source():
            expected_events.append("source:start")
            for child in expected_children:
                expected_events.append(f"source:{child.name}")
                yield child
            expected_events.append("source:end")

        actual_source_object = actual_source()
        expected_source_object = expected_source()
        actual = torch.utils.data.ChainDataset(actual_source_object)
        expected = reference_torch.utils.data.ChainDataset(expected_source_object)
        self.assertIs(actual.datasets, actual_source_object)
        self.assertIs(expected.datasets, expected_source_object)
        self.assertEqual(actual_events, expected_events)

        actual_iterator = iter(actual)
        expected_iterator = iter(expected)
        self.assertEqual(actual_events, expected_events)
        self.assertEqual(next(actual_iterator), next(expected_iterator))
        self.assertEqual(actual_events, expected_events)
        self.assertEqual(list(actual_iterator), list(expected_iterator))
        self.assertEqual(actual_events, expected_events)

        for source_factory in (list, tuple):
            actual = torch.utils.data.ChainDataset(source_factory(actual_children))
            expected = reference_torch.utils.data.ChainDataset(
                source_factory(expected_children)
            )
            with self.subTest(source=source_factory.__name__):
                self.assertEqual(list(actual), list(expected))
                self.assertEqual(list(actual), list(expected))

        for actual_source_object, expected_source_object in (
            ([], []),
            ((), ()),
            (iter(()), iter(())),
        ):
            actual = torch.utils.data.ChainDataset(actual_source_object)
            expected = reference_torch.utils.data.ChainDataset(expected_source_object)
            self.assertEqual(list(actual), list(expected))
            self.assertEqual(len(actual), len(expected))

    def test_addition_nested_chains_and_deferred_validation_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_left = self.make_stream(torch, "left", [])
        expected_left = self.make_stream(reference_torch, "left", [])
        actual_right = self.make_stream(torch, "right", [1, 2])
        expected_right = self.make_stream(reference_torch, "right", [1, 2])
        actual_tail = self.make_stream(torch, "tail", [3])
        expected_tail = self.make_stream(reference_torch, "tail", [3])

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

        actual_nested = actual + actual_tail
        expected_nested = expected + expected_tail
        self.assertEqual(type(actual_nested).__name__, type(expected_nested).__name__)
        self.assertIs(actual_nested.datasets[0], actual)
        self.assertIs(expected_nested.datasets[0], expected)
        self.assertEqual(list(actual_nested), list(expected_nested))
        self.assertEqual(
            type(actual_nested.datasets[0]).__name__,
            type(expected_nested.datasets[0]).__name__,
        )

        actual_invalid = actual_left + self.make_map_dataset(torch, [4])
        expected_invalid = expected_left + self.make_map_dataset(reference_torch, [4])
        actual_iterator = iter(actual_invalid)
        expected_iterator = iter(expected_invalid)
        self.assert_error_matches(
            lambda: next(actual_iterator), lambda: next(expected_iterator)
        )

    def test_length_delegation_and_child_errors_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_empty = self.make_stream(torch, "empty", [])
        expected_empty = self.make_stream(reference_torch, "empty", [])
        actual_finite = self.make_stream(torch, "finite", [1, 2])
        expected_finite = self.make_stream(reference_torch, "finite", [1, 2])
        actual = torch.utils.data.ChainDataset([actual_empty, actual_finite])
        expected = reference_torch.utils.data.ChainDataset(
            [expected_empty, expected_finite]
        )

        self.assertEqual(len(actual), len(expected))
        self.assertEqual(actual_empty.length_calls, expected_empty.length_calls)
        self.assertEqual(actual_finite.length_calls, expected_finite.length_calls)
        actual_finite.values.append(3)
        expected_finite.values.append(3)
        self.assertEqual(operator.length_hint(actual), operator.length_hint(expected))
        self.assertEqual(actual_empty.length_calls, expected_empty.length_calls)
        self.assertEqual(actual_finite.length_calls, expected_finite.length_calls)

        actual_unsized = torch.utils.data.ChainDataset(
            [self.make_stream(torch, "unsized", [1], sized=False)]
        )
        expected_unsized = reference_torch.utils.data.ChainDataset(
            [self.make_stream(reference_torch, "unsized", [1], sized=False)]
        )
        self.assert_error_matches(
            lambda: len(actual_unsized), lambda: len(expected_unsized)
        )

        def failing_stream(module):
            base = module.utils.data.IterableDataset

            class FailingLengthStream(base):
                def __iter__(self):
                    return iter(())

                def __len__(self):
                    raise StreamLengthError("stream length is unavailable")

            return FailingLengthStream()

        actual_failing = torch.utils.data.ChainDataset([failing_stream(torch)])
        expected_failing = reference_torch.utils.data.ChainDataset(
            [failing_stream(reference_torch)]
        )
        self.assert_error_matches(
            lambda: len(actual_failing), lambda: len(expected_failing)
        )

    def test_incremental_type_validation_and_concat_guard_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_valid = self.make_stream(torch, "valid", [1])
        expected_valid = self.make_stream(reference_torch, "valid", [1])
        actual = torch.utils.data.ChainDataset([actual_valid, [2]])
        expected = reference_torch.utils.data.ChainDataset([expected_valid, [2]])

        actual_iterator = iter(actual)
        expected_iterator = iter(expected)
        self.assertEqual(next(actual_iterator), next(expected_iterator))
        self.assert_error_matches(
            lambda: next(actual_iterator), lambda: next(expected_iterator)
        )
        self.assertEqual(actual_valid.iteration_calls, expected_valid.iteration_calls)

        self.assert_error_matches(lambda: len(actual), lambda: len(expected))
        self.assertEqual(actual_valid.length_calls, expected_valid.length_calls)

        actual_duck = torch.utils.data.ChainDataset([iter((1, 2))])
        expected_duck = reference_torch.utils.data.ChainDataset([iter((1, 2))])
        self.assert_error_matches(lambda: list(actual_duck), lambda: list(expected_duck))

        for actual_source, expected_source in ((None, None), (1, 1)):
            actual = torch.utils.data.ChainDataset(actual_source)
            expected = reference_torch.utils.data.ChainDataset(expected_source)
            self.assertIs(actual.datasets, actual_source)
            self.assertIs(expected.datasets, expected_source)
            self.assert_error_matches(lambda: list(actual), lambda: list(expected))
            self.assert_error_matches(lambda: len(actual), lambda: len(expected))

        actual_iterable = self.make_stream(torch, "iterable", [1])
        expected_iterable = self.make_stream(reference_torch, "iterable", [1])
        self.assert_error_matches(
            lambda: torch.utils.data.ConcatDataset([actual_iterable]),
            lambda: reference_torch.utils.data.ConcatDataset([expected_iterable]),
        )

    def test_construction_signatures_docs_and_metadata_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.utils.data.ChainDataset
        expected = reference_torch.utils.data.ChainDataset

        actual_source = []
        expected_source = []
        actual_instance = actual(datasets=actual_source)
        expected_instance = expected(datasets=expected_source)
        self.assertIs(actual_instance.datasets, actual_source)
        self.assertIs(expected_instance.datasets, expected_source)

        for actual_call, expected_call in (
            (lambda: actual(), lambda: expected()),
            (lambda: actual([], []), lambda: expected([], [])),
            (lambda: actual(foo=[]), lambda: expected(foo=[])),
            (
                lambda: actual_instance.__iter__(1),
                lambda: expected_instance.__iter__(1),
            ),
            (
                lambda: actual_instance.__len__(1),
                lambda: expected_instance.__len__(1),
            ),
        ):
            self.assert_error_matches(actual_call, expected_call)

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
        self.assertEqual(actual.__parameters__, expected.__parameters__)
        self.assertIs(actual.__bases__[0], torch.utils.data.IterableDataset)
        self.assertIs(
            expected.__bases__[0], reference_torch.utils.data.IterableDataset
        )

        declared_methods = ("__init__", "__iter__", "__len__")
        self.assertEqual(
            tuple(name for name in declared_methods if name in actual.__dict__),
            declared_methods,
        )
        self.assertEqual(
            tuple(name for name in declared_methods if name in expected.__dict__),
            declared_methods,
        )
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

    def test_imports_and_exports_match(self):
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

        self.assertIs(actual_data.ChainDataset, actual_module.ChainDataset)
        self.assertIs(expected_data.ChainDataset, expected_module.ChainDataset)
        self.assertEqual(
            actual_data.__all__,
            [name for name in expected_data.__all__ if name in supported],
        )
        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name in supported],
        )

        namespace = {}
        exec("from torch_rs.utils.data import *", namespace)
        self.assertIs(namespace["ChainDataset"], actual_data.ChainDataset)


if __name__ == "__main__":
    unittest.main()

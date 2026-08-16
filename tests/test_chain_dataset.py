import importlib
import inspect
import operator
import unittest
from collections.abc import Iterable as IterableABC
from typing import Iterable

from torch_rs.utils.data import (
    ChainDataset,
    ConcatDataset,
    Dataset,
    IterableDataset,
)


class RecordingStream(IterableDataset):
    def __init__(self, name, values, events=None):
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


class UnsizedStream(IterableDataset):
    def __iter__(self):
        yield from (1, 2)


class StreamLengthError(Exception):
    pass


class FailingLengthStream(IterableDataset):
    def __iter__(self):
        return iter(())

    def __len__(self):
        raise StreamLengthError("stream length is unavailable")


class ListDataset(Dataset):
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)


class ChainDatasetTests(unittest.TestCase):
    def test_explicit_construction_is_lazy_and_chains_empty_finite_streams(self):
        events = []
        children = [
            RecordingStream("empty-1", [], events),
            RecordingStream("first", [1, 2], events),
            RecordingStream("empty-2", [], events),
            RecordingStream("last", [3], events),
        ]

        def child_stream():
            events.append("source:start")
            for child in children:
                events.append(f"source:{child.name}")
                yield child
            events.append("source:end")

        source = child_stream()
        dataset = ChainDataset(source)
        self.assertIs(dataset.datasets, source)
        self.assertEqual(events, [])

        iterator = iter(dataset)
        self.assertEqual(events, [])
        self.assertEqual(next(iterator), 1)
        self.assertEqual(
            events,
            [
                "source:start",
                "source:empty-1",
                "iter:empty-1",
                "source:first",
                "iter:first",
            ],
        )
        self.assertEqual(list(iterator), [2, 3])
        self.assertEqual(
            events,
            [
                "source:start",
                "source:empty-1",
                "iter:empty-1",
                "source:first",
                "iter:first",
                "source:empty-2",
                "iter:empty-2",
                "source:last",
                "iter:last",
                "source:end",
            ],
        )

        replayable = ChainDataset(children)
        self.assertIs(replayable.datasets, children)
        self.assertEqual(list(replayable), [1, 2, 3])
        self.assertEqual(list(replayable), [1, 2, 3])
        self.assertEqual([child.iteration_calls for child in children], [3, 3, 3, 3])

        for empty_source in ([], (), iter(())):
            with self.subTest(empty_source=type(empty_source).__name__):
                empty = ChainDataset(empty_source)
                self.assertEqual(list(empty), [])
                self.assertEqual(len(empty), 0)

    def test_iterable_dataset_addition_is_lazy_and_preserves_nested_chains(self):
        left = RecordingStream("left", [])
        right = RecordingStream("right", [1, 2])
        tail = RecordingStream("tail", [3])

        direct = left + right
        self.assertIs(type(direct), ChainDataset)
        self.assertEqual(direct.datasets, [left, right])
        self.assertIs(direct.datasets[0], left)
        self.assertIs(direct.datasets[1], right)
        self.assertEqual(left.iteration_calls, 0)
        self.assertEqual(right.iteration_calls, 0)
        self.assertEqual(list(direct), [1, 2])

        nested = direct + tail
        self.assertIs(type(nested), ChainDataset)
        self.assertEqual(nested.datasets, [direct, tail])
        self.assertIs(nested.datasets[0], direct)
        self.assertIs(nested.datasets[1], tail)
        self.assertEqual(list(nested), [1, 2, 3])
        self.assertIs(type(nested.datasets[0]), ChainDataset)

        invalid = left + ListDataset([4])
        invalid_iterator = iter(invalid)
        with self.assertRaisesRegex(
            AssertionError, "^ChainDataset only supports IterableDataset$"
        ):
            next(invalid_iterator)

    def test_length_delegates_without_caching_and_preserves_child_errors(self):
        empty = RecordingStream("empty", [])
        finite = RecordingStream("finite", [1, 2])
        dataset = ChainDataset([empty, finite])

        self.assertEqual(empty.length_calls, 0)
        self.assertEqual(finite.length_calls, 0)
        self.assertEqual(len(dataset), 2)
        self.assertEqual((empty.length_calls, finite.length_calls), (1, 1))
        finite.values.append(3)
        self.assertEqual(operator.length_hint(dataset), 3)
        self.assertEqual((empty.length_calls, finite.length_calls), (2, 2))

        unsized = ChainDataset([UnsizedStream()])
        with self.assertRaisesRegex(
            TypeError, r"^object of type 'UnsizedStream' has no len\(\)$"
        ):
            len(unsized)

        failing = ChainDataset([FailingLengthStream()])
        with self.assertRaises(StreamLengthError) as raised:
            len(failing)
        self.assertEqual(raised.exception.args, ("stream length is unavailable",))

    def test_child_type_validation_is_incremental_for_iteration_and_length(self):
        valid = RecordingStream("valid", [1])
        invalid = [2]
        dataset = ChainDataset([valid, invalid])
        self.assertEqual(valid.iteration_calls, 0)

        iterator = iter(dataset)
        self.assertEqual(valid.iteration_calls, 0)
        self.assertEqual(next(iterator), 1)
        self.assertEqual(valid.iteration_calls, 1)
        with self.assertRaisesRegex(
            AssertionError, "^ChainDataset only supports IterableDataset$"
        ):
            next(iterator)

        self.assertEqual(valid.length_calls, 0)
        with self.assertRaisesRegex(
            AssertionError, "^ChainDataset only supports IterableDataset$"
        ):
            len(dataset)
        self.assertEqual(valid.length_calls, 1)

        duck_iterable = ChainDataset([iter((1, 2))])
        with self.assertRaisesRegex(
            AssertionError, "^ChainDataset only supports IterableDataset$"
        ):
            list(duck_iterable)

        for invalid_source in (None, 1):
            with self.subTest(invalid_source=invalid_source):
                constructed = ChainDataset(invalid_source)
                self.assertIs(constructed.datasets, invalid_source)
                with self.assertRaises(TypeError):
                    list(constructed)
                with self.assertRaises(TypeError):
                    len(constructed)

    def test_concat_dataset_rejects_iterable_dataset_children(self):
        iterable = RecordingStream("stream", [1])
        for children in ([iterable], [ListDataset([0]), iterable]):
            with self.subTest(position=len(children) - 1):
                with self.assertRaisesRegex(
                    AssertionError,
                    "^ConcatDataset does not support IterableDataset$",
                ):
                    ConcatDataset(children)

    def test_signature_inheritance_documentation_and_metadata(self):
        signature = inspect.signature(ChainDataset)
        self.assertEqual(tuple(signature.parameters), ("datasets",))
        parameter = signature.parameters["datasets"]
        self.assertIs(parameter.kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        self.assertIs(parameter.default, inspect.Parameter.empty)
        self.assertEqual(parameter.annotation, Iterable[Dataset])
        self.assertIs(signature.return_annotation, None)

        self.assertEqual(ChainDataset.__annotations__, {})
        self.assertEqual(ChainDataset.__parameters__, ())
        self.assertIs(ChainDataset.__bases__[0], IterableDataset)
        self.assertEqual(
            ChainDataset.__orig_bases__, IterableDataset.__orig_bases__
        )
        self.assertFalse(inspect.isabstract(ChainDataset))
        self.assertTrue(issubclass(ChainDataset, IterableDataset))
        self.assertTrue(issubclass(ChainDataset, IterableABC))
        self.assertIsInstance(ChainDataset([]), IterableDataset)
        self.assertIs(ChainDataset.__add__, IterableDataset.__add__)

        self.assertEqual(
            ChainDataset.__init__.__annotations__,
            {"datasets": Iterable[Dataset], "return": None},
        )
        self.assertEqual(ChainDataset.__iter__.__annotations__, {})
        self.assertEqual(ChainDataset.__len__.__annotations__, {"return": int})
        self.assertEqual(str(inspect.signature(ChainDataset.__iter__)), "(self)")
        self.assertEqual(str(inspect.signature(ChainDataset.__len__)), "(self) -> int")
        self.assertIsNone(ChainDataset.__init__.__doc__)
        self.assertIsNone(ChainDataset.__iter__.__doc__)
        self.assertIsNone(ChainDataset.__len__.__doc__)
        self.assertIn("Dataset for chaining", ChainDataset.__doc__)
        self.assertIn("on-the-fly", ChainDataset.__doc__)
        self.assertEqual(ChainDataset.__module__, "torch_rs.utils.data.dataset")
        self.assertEqual(ChainDataset.__name__, "ChainDataset")
        self.assertEqual(ChainDataset.__qualname__, "ChainDataset")

    def test_construction_call_forms_and_exports(self):
        source = []
        dataset = ChainDataset(datasets=source)
        self.assertIs(dataset.datasets, source)

        for call in (
            lambda: ChainDataset(),
            lambda: ChainDataset([], []),
            lambda: ChainDataset(foo=[]),
            lambda: dataset.__iter__(1),
            lambda: dataset.__len__(1),
        ):
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

        data_module = importlib.import_module("torch_rs.utils.data")
        dataset_module = importlib.import_module("torch_rs.utils.data.dataset")
        self.assertIs(data_module.ChainDataset, ChainDataset)
        self.assertIs(dataset_module.ChainDataset, ChainDataset)
        self.assertIn("ChainDataset", data_module.__all__)
        self.assertIn("ChainDataset", dataset_module.__all__)
        namespace = {}
        exec("from torch_rs.utils.data import *", namespace)
        self.assertIs(namespace["ChainDataset"], ChainDataset)


if __name__ == "__main__":
    unittest.main()

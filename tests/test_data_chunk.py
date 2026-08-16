import importlib
import inspect
import pickle
import unittest
from collections.abc import Iterable, Iterator
from typing import get_args, get_origin

import torch_rs as torch

from torch_rs.utils.data import DataChunk


class DataChunkTests(unittest.TestCase):
    def make_mutated_chunk(self):
        shared = ["shared"]
        chunk = DataChunk([shared, "common", 3])
        chunk.append("list-only")
        chunk[1] = "list-mutated"
        chunk.items.append("raw-only")
        chunk.items[2] = "raw-mutated"
        shared.append("nested-mutated")
        return chunk

    def test_construction_snapshots_finite_iterables(self):
        source = [["shared"], 2]
        chunk = DataChunk(source)

        self.assertIs(type(chunk), DataChunk)
        self.assertIsInstance(chunk, list)
        self.assertIs(type(chunk.items), list)
        self.assertIsNot(chunk.items, chunk)
        self.assertIsNot(chunk.items, source)
        self.assertIs(chunk[0], source[0])
        self.assertIs(chunk.items[0], source[0])

        source.append(3)
        source[0].append("nested")
        self.assertEqual(list(chunk), [["shared", "nested"], 2])
        self.assertEqual(list(chunk.raw_iterator()), [["shared", "nested"], 2])

        yielded = []

        def values():
            for value in ("first", "second", "third"):
                yielded.append(value)
                yield value

        generator = values()
        generated = DataChunk(generator)
        self.assertEqual(yielded, ["first", "second", "third"])
        self.assertEqual(list(generator), [])
        self.assertEqual(list(generated), yielded)
        self.assertEqual(list(generated.raw_iterator()), yielded)

        self.assertEqual(list(DataChunk(())), [])
        self.assertEqual(list(DataChunk(iter(()))), [])

    def test_list_and_raw_iteration_retain_distinct_mutation_state(self):
        chunk = self.make_mutated_chunk()

        self.assertEqual(
            list(chunk),
            [["shared", "nested-mutated"], "list-mutated", 3, "list-only"],
        )
        self.assertEqual(
            list(chunk.raw_iterator()),
            [["shared", "nested-mutated"], "common", "raw-mutated", "raw-only"],
        )

        list_iterator = iter(chunk)
        raw_iterator = chunk.raw_iterator()
        self.assertIs(next(list_iterator), chunk[0])
        self.assertIs(next(raw_iterator), chunk.items[0])

        chunk.append("late-list")
        chunk.items.append("late-raw")
        self.assertEqual(
            list(list_iterator),
            ["list-mutated", 3, "list-only", "late-list"],
        )
        self.assertEqual(
            list(raw_iterator),
            ["common", "raw-mutated", "raw-only", "late-raw"],
        )

    def test_as_str_uses_list_iteration_for_all_supported_inputs(self):
        empty = DataChunk([])
        generated = DataChunk(value for value in ("alpha", 2))
        nested = DataChunk([DataChunk(["inner", 3]), ["plain", 4]])
        mutated = self.make_mutated_chunk()

        self.assertEqual(empty.as_str(), "[]")
        self.assertEqual(empty.as_str("  "), "  []")
        self.assertEqual(generated.as_str(), "[alpha, 2]")
        self.assertEqual(generated.as_str("->"), "->[alpha, 2]")
        self.assertEqual(nested.as_str(), "[['inner', 3], ['plain', 4]]")
        self.assertEqual(
            mutated.as_str("  "),
            "  [['shared', 'nested-mutated'], list-mutated, 3, list-only]",
        )
        self.assertNotIn("raw-mutated", mutated.as_str())
        self.assertNotIn("raw-only", mutated.as_str())

    def test_pickle_preserves_all_list_raw_and_aliasing_state(self):
        cases = {
            "empty": DataChunk([]),
            "generator": DataChunk(value for value in (["generated"], {"value": 2})),
            "nested": DataChunk([DataChunk([1, 2]), ["tail"]]),
            "mutated": self.make_mutated_chunk(),
        }

        for name, chunk in cases.items():
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    restored = pickle.loads(pickle.dumps(chunk, protocol=protocol))
                    self.assertIs(type(restored), DataChunk)
                    self.assertEqual(list(restored), list(chunk))
                    self.assertEqual(
                        list(restored.raw_iterator()), list(chunk.raw_iterator())
                    )
                    self.assertEqual(restored.as_str(), chunk.as_str())
                    self.assertEqual(restored.__dict__, {"items": restored.items})
                    self.assertIsNot(restored.items, restored)
                    if restored:
                        self.assertIs(restored[0], restored.items[0])
                    if name == "nested":
                        self.assertIs(type(restored[0]), DataChunk)
                        self.assertIsNot(restored[0].items, restored[0])

        restored = pickle.loads(pickle.dumps(cases["mutated"]))
        restored.append("post-pickle-list")
        restored.items.append("post-pickle-raw")
        self.assertNotIn("post-pickle-list", restored.items)
        self.assertNotIn("post-pickle-raw", list(restored))

    def test_exports_generic_inheritance_signatures_annotations_and_metadata(self):
        data_module = importlib.import_module("torch_rs.utils.data")
        datapipes_package = importlib.import_module("torch_rs.utils.data.datapipes")
        datapipe_module = importlib.import_module(
            "torch_rs.utils.data.datapipes.datapipe"
        )

        self.assertIs(torch.utils.data, data_module)
        self.assertIs(data_module.DataChunk, DataChunk)
        self.assertIs(datapipe_module.DataChunk, DataChunk)
        self.assertFalse(hasattr(datapipes_package, "DataChunk"))
        self.assertEqual(datapipe_module.__all__, ["DataChunk"])
        self.assertEqual(
            data_module.__all__,
            [
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
            ],
        )

        wildcard_namespace = {}
        exec("from torch_rs.utils.data import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["DataChunk"], DataChunk)

        for unsupported in ("DFIterDataPipe", "IterDataPipe", "MapDataPipe"):
            with self.subTest(unsupported=unsupported):
                self.assertFalse(hasattr(data_module, unsupported))
                self.assertFalse(hasattr(datapipe_module, unsupported))
                self.assertNotIn(unsupported, wildcard_namespace)

        self.assertEqual(DataChunk.__module__, "torch_rs.utils.data.datapipes.datapipe")
        self.assertEqual(DataChunk.__name__, "DataChunk")
        self.assertEqual(DataChunk.__qualname__, "DataChunk")
        self.assertIsNone(DataChunk.__doc__)
        self.assertEqual(DataChunk.__annotations__, {})
        self.assertEqual(DataChunk.__bases__, (list,))
        self.assertEqual(DataChunk.__mro__, (DataChunk, list, object))

        (generic_base,) = DataChunk.__orig_bases__
        self.assertIs(get_origin(generic_base), list)
        (type_parameter,) = get_args(generic_base)
        self.assertEqual(type_parameter.__name__, "_T")
        self.assertFalse(type_parameter.__covariant__)
        self.assertFalse(type_parameter.__contravariant__)
        self.assertEqual(type_parameter.__constraints__, ())
        self.assertIsNone(type_parameter.__bound__)
        self.assertIs(get_origin(DataChunk[int]), DataChunk)
        self.assertEqual(get_args(DataChunk[int]), (int,))

        signature = inspect.signature(DataChunk)
        self.assertEqual(tuple(signature.parameters), ("items",))
        items_parameter = signature.parameters["items"]
        self.assertIs(items_parameter.kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        self.assertIs(items_parameter.default, inspect.Parameter.empty)
        self.assertEqual(items_parameter.annotation, Iterable[type_parameter])
        self.assertIs(signature.return_annotation, None)
        self.assertEqual(
            DataChunk.__init__.__annotations__,
            {"items": Iterable[type_parameter], "return": None},
        )
        self.assertEqual(
            DataChunk.__iter__.__annotations__,
            {"return": Iterator[type_parameter]},
        )
        self.assertEqual(
            DataChunk.raw_iterator.__annotations__,
            {"return": Iterator[type_parameter]},
        )
        self.assertEqual(
            DataChunk.as_str.__annotations__, {"indent": str, "return": str}
        )
        self.assertEqual(
            str(inspect.signature(DataChunk.__iter__)),
            "(self) -> collections.abc.Iterator[~_T]",
        )
        self.assertEqual(
            str(inspect.signature(DataChunk.raw_iterator)),
            "(self) -> collections.abc.Iterator[~_T]",
        )
        self.assertEqual(
            str(inspect.signature(DataChunk.as_str)),
            "(self, indent: str = '') -> str",
        )
        for method in (
            DataChunk.__init__,
            DataChunk.__iter__,
            DataChunk.raw_iterator,
            DataChunk.as_str,
        ):
            self.assertEqual(method.__module__, DataChunk.__module__)
            self.assertIsNone(method.__doc__)


if __name__ == "__main__":
    unittest.main()

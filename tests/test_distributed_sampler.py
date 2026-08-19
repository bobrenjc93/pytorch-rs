import importlib
import inspect
import math
import operator
import pickle
import unittest
from collections.abc import Iterator
from typing import get_args, get_origin

import torch_rs as torch

from torch_rs.utils.data import Dataset, DistributedSampler, Sampler


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


class PickleDistributedSampler(DistributedSampler[int]):
    pass


class DistributedSamplerTests(unittest.TestCase):
    @staticmethod
    def expected_indices(size, num_replicas, rank, drop_last):
        if drop_last:
            num_samples = size // num_replicas
        else:
            num_samples = math.ceil(size / num_replicas)
        total_size = num_samples * num_replicas
        indices = list(range(size))
        if drop_last:
            indices = indices[:total_size]
        else:
            padding_size = total_size - size
            if padding_size <= size:
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / size))[:padding_size]
        return indices[rank:total_size:num_replicas], num_samples, total_size

    def test_round_robin_partitioning_padding_drop_last_and_empty_datasets(self):
        for size in range(10):
            for num_replicas in (1, 2, 3, 5):
                for drop_last in (False, True):
                    for rank in range(num_replicas):
                        dataset = list(range(size))
                        sampler = DistributedSampler(
                            dataset,
                            num_replicas=num_replicas,
                            rank=rank,
                            shuffle=False,
                            seed=17,
                            drop_last=drop_last,
                        )
                        expected, num_samples, total_size = self.expected_indices(
                            size, num_replicas, rank, drop_last
                        )
                        with self.subTest(
                            size=size,
                            num_replicas=num_replicas,
                            rank=rank,
                            drop_last=drop_last,
                        ):
                            self.assertIs(sampler.dataset, dataset)
                            self.assertEqual(
                                sampler.__dict__,
                                {
                                    "dataset": dataset,
                                    "num_replicas": num_replicas,
                                    "rank": rank,
                                    "epoch": 0,
                                    "drop_last": drop_last,
                                    "num_samples": num_samples,
                                    "total_size": total_size,
                                    "shuffle": False,
                                    "seed": 17,
                                },
                            )
                            self.assertIs(type(iter(sampler)), type(iter([])))
                            self.assertEqual(list(sampler), expected)
                            self.assertEqual(len(sampler), num_samples)
                            self.assertEqual(
                                operator.length_hint(sampler), num_samples
                            )

    def test_short_datasets_repeat_from_the_front_for_padding(self):
        outputs = [
            list(
                DistributedSampler(
                    range(2),
                    num_replicas=5,
                    rank=rank,
                    shuffle=False,
                )
            )
            for rank in range(5)
        ]
        self.assertEqual(outputs, [[0], [1], [0], [1], [0]])

        dropped = [
            list(
                DistributedSampler(
                    range(2),
                    num_replicas=5,
                    rank=rank,
                    shuffle=False,
                    drop_last=True,
                )
            )
            for rank in range(5)
        ]
        self.assertEqual(dropped, [[], [], [], [], []])

    def test_size_is_cached_while_iteration_reads_the_current_dataset(self):
        source = MutableSizedSource(5)
        sampler = DistributedSampler(
            source, num_replicas=2, rank=1, shuffle=False
        )

        self.assertEqual(source.length_calls, 1)
        self.assertEqual(len(sampler), 3)
        self.assertEqual(source.length_calls, 1)
        self.assertEqual(list(sampler), [1, 3, 0])
        self.assertEqual(source.length_calls, 2)

        source.size = 4
        self.assertEqual(len(sampler), 3)
        self.assertEqual(list(sampler), [1, 3, 1])

        source.size = 7
        with self.assertRaisesRegex(
            AssertionError,
            r"^Number of indices \(13\) does not match total_size \(6\)$",
        ):
            iter(sampler)

        source.size = 0
        with self.assertRaisesRegex(ZeroDivisionError, r"^division by zero$"):
            iter(sampler)

        dropped_source = MutableSizedSource(5)
        dropped = DistributedSampler(
            dropped_source,
            num_replicas=2,
            rank=0,
            shuffle=False,
            drop_last=True,
        )
        dropped_source.size = 3
        with self.assertRaisesRegex(
            AssertionError,
            r"^Number of indices \(3\) does not match total_size \(4\)$",
        ):
            iter(dropped)

    def test_state_set_epoch_and_pickle_round_trips(self):
        dataset = list(range(8))
        sampler = DistributedSampler(
            dataset,
            num_replicas=3,
            rank=1,
            shuffle=False,
            seed=23,
            drop_last=False,
        )
        before = list(sampler)
        self.assertIsNone(sampler.set_epoch(9))
        self.assertEqual(sampler.epoch, 9)
        self.assertEqual(list(sampler), before)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(sampler, protocol=protocol))
                self.assertIs(type(restored), DistributedSampler)
                self.assertEqual(restored.__dict__, sampler.__dict__)
                self.assertIsNot(restored.dataset, sampler.dataset)
                self.assertEqual(list(restored), before)
                self.assertEqual(len(restored), len(sampler))
                self.assertIs(
                    pickle.loads(
                        pickle.dumps(DistributedSampler, protocol=protocol)
                    ),
                    DistributedSampler,
                )

        subclass = PickleDistributedSampler(
            dataset,
            num_replicas=3,
            rank=2,
            shuffle=False,
            drop_last=True,
        )
        subclass.tag = "preserved"
        restored_subclass = pickle.loads(pickle.dumps(subclass))
        self.assertIs(type(restored_subclass), PickleDistributedSampler)
        self.assertEqual(restored_subclass.__dict__, subclass.__dict__)
        self.assertEqual(list(restored_subclass), list(subclass))

    def test_supported_validation_and_length_errors(self):
        for num_replicas, rank, interval_end in (
            (3, -1, 2),
            (3, 3, 2),
            (3, 7, 2),
            (0, 0, -1),
            (-2, 0, -3),
        ):
            with self.subTest(num_replicas=num_replicas, rank=rank):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^Invalid rank {rank}, rank should be in the interval "
                    rf"\[0, {interval_end}\]$",
                ):
                    DistributedSampler(
                        [],
                        num_replicas=num_replicas,
                        rank=rank,
                        shuffle=False,
                    )

        with self.assertRaisesRegex(
            TypeError, r"^object of type 'object' has no len\(\)$"
        ):
            DistributedSampler(
                object(), num_replicas=2, rank=0, shuffle=False
            )

        with self.assertRaises(SourceLengthError) as raised:
            DistributedSampler(
                FailingSizedSource(), num_replicas=2, rank=0, shuffle=False
            )
        self.assertEqual(raised.exception.args, ("source length is unavailable",))

    def test_implicit_discovery_and_shuffling_are_explicitly_unsupported(self):
        cases = (
            (
                lambda: DistributedSampler([], shuffle=False),
                "DistributedSampler requires an explicit num_replicas",
            ),
            (
                lambda: DistributedSampler(
                    [], num_replicas=2, rank=None, shuffle=False
                ),
                "DistributedSampler requires an explicit rank",
            ),
            (
                lambda: DistributedSampler([], num_replicas=2, rank=0),
                "DistributedSampler only supports shuffle=False",
            ),
            (
                lambda: DistributedSampler(
                    [], num_replicas=2, rank=0, shuffle="yes"
                ),
                "DistributedSampler only supports shuffle=False",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                    NotImplementedError, rf"^{message}$"
                ):
                    call()

        sampler = DistributedSampler(
            range(4), num_replicas=2, rank=0, shuffle=False
        )
        sampler.shuffle = True
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^DistributedSampler only supports shuffle=False$",
        ):
            iter(sampler)

    def test_signature_annotations_inheritance_and_metadata(self):
        signature = inspect.signature(DistributedSampler)
        self.assertEqual(
            tuple(signature.parameters),
            (
                "dataset",
                "num_replicas",
                "rank",
                "shuffle",
                "seed",
                "drop_last",
            ),
        )
        for parameter in signature.parameters.values():
            self.assertIs(parameter.kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        self.assertIs(signature.parameters["dataset"].annotation, Dataset)
        self.assertEqual(
            signature.parameters["num_replicas"].annotation, int | None
        )
        self.assertEqual(signature.parameters["rank"].annotation, int | None)
        self.assertIs(signature.parameters["shuffle"].annotation, bool)
        self.assertIs(signature.parameters["seed"].annotation, int)
        self.assertIs(signature.parameters["drop_last"].annotation, bool)
        self.assertIs(signature.parameters["dataset"].default, inspect.Parameter.empty)
        self.assertIsNone(signature.parameters["num_replicas"].default)
        self.assertIsNone(signature.parameters["rank"].default)
        self.assertIs(signature.parameters["shuffle"].default, True)
        self.assertEqual(signature.parameters["seed"].default, 0)
        self.assertIs(signature.parameters["drop_last"].default, False)
        self.assertIsNone(signature.return_annotation)

        declared_methods = ("__init__", "__iter__", "__len__", "set_epoch")
        self.assertEqual(
            tuple(
                name
                for name in declared_methods
                if name in DistributedSampler.__dict__
            ),
            declared_methods,
        )
        self.assertEqual(DistributedSampler.__annotations__, {})
        self.assertIs(DistributedSampler.__bases__[0], Sampler)
        self.assertIsInstance(DistributedSampler([], 1, 0, False), Sampler)
        base = DistributedSampler.__orig_bases__[0]
        parameter = DistributedSampler.__parameters__[0]
        self.assertIs(get_origin(base), Sampler)
        self.assertIs(get_args(base)[0], parameter)
        self.assertEqual(parameter.__name__, "_T_co")
        self.assertTrue(parameter.__covariant__)

        self.assertEqual(
            DistributedSampler.__iter__.__annotations__,
            {"return": Iterator[parameter]},
        )
        self.assertEqual(
            DistributedSampler.__len__.__annotations__, {"return": int}
        )
        self.assertEqual(
            DistributedSampler.set_epoch.__annotations__,
            {"epoch": int, "return": None},
        )
        self.assertIsNone(DistributedSampler.__init__.__doc__)
        self.assertIsNone(DistributedSampler.__iter__.__doc__)
        self.assertIsNone(DistributedSampler.__len__.__doc__)
        self.assertIn("Sampler that restricts data loading", DistributedSampler.__doc__)
        self.assertIn("Set the epoch for this sampler", DistributedSampler.set_epoch.__doc__)

    def test_imports_exports_and_unsupported_neighbors(self):
        data_module = importlib.import_module("torch_rs.utils.data")
        distributed_module = importlib.import_module(
            "torch_rs.utils.data.distributed"
        )
        sampler_module = importlib.import_module("torch_rs.utils.data.sampler")

        self.assertIs(torch.utils.data.DistributedSampler, DistributedSampler)
        self.assertIs(data_module.DistributedSampler, DistributedSampler)
        self.assertIs(distributed_module.DistributedSampler, DistributedSampler)
        self.assertIs(data_module.distributed, distributed_module)
        self.assertEqual(
            DistributedSampler.__module__, "torch_rs.utils.data.distributed"
        )
        self.assertEqual(DistributedSampler.__name__, "DistributedSampler")
        self.assertEqual(DistributedSampler.__qualname__, "DistributedSampler")
        self.assertEqual(distributed_module.__all__, ["DistributedSampler"])
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
                "get_worker_info",
            ],
        )

        data_namespace = {}
        distributed_namespace = {}
        exec("from torch_rs.utils.data import *", data_namespace)
        exec(
            "from torch_rs.utils.data.distributed import *",
            distributed_namespace,
        )
        self.assertIs(data_namespace["DistributedSampler"], DistributedSampler)
        self.assertIs(
            distributed_namespace["DistributedSampler"], DistributedSampler
        )
        self.assertFalse(hasattr(sampler_module, "DistributedSampler"))

        for name in (
            "DataLoader",
            "RandomSampler",
            "SubsetRandomSampler",
            "WeightedRandomSampler",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(data_module, name))
                self.assertFalse(hasattr(distributed_module, name))


if __name__ == "__main__":
    unittest.main()

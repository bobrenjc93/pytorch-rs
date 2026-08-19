import importlib
import inspect
import operator
import pickle
import unittest
from collections.abc import Iterator
from typing import get_args, get_origin

import torch_rs as torch

from torch_rs.utils.data import Dataset, DistributedSampler, Sampler


class LengthError(Exception):
    pass


class FailingDataset:
    def __len__(self):
        raise LengthError("dataset length is unavailable")


class DistributedSamplerTests(unittest.TestCase):
    def test_partitioning_padding_drop_last_lengths_and_empty_datasets(self):
        cases = (
            (0, 4, False, ([], [], [], []), 0, 0),
            (1, 4, False, ([0], [0], [0], [0]), 1, 4),
            (2, 5, False, ([0], [1], [0], [1], [0]), 1, 5),
            (5, 2, False, ([0, 2, 4], [1, 3, 0]), 3, 6),
            (6, 3, False, ([0, 3], [1, 4], [2, 5]), 2, 6),
            (7, 3, False, ([0, 3, 6], [1, 4, 0], [2, 5, 1]), 3, 9),
            (0, 4, True, ([], [], [], []), 0, 0),
            (2, 5, True, ([], [], [], [], []), 0, 0),
            (6, 3, True, ([0, 3], [1, 4], [2, 5]), 2, 6),
            (7, 3, True, ([0, 3], [1, 4], [2, 5]), 2, 6),
        )
        for size, replicas, drop_last, expected, num_samples, total_size in cases:
            for rank, expected_indices in enumerate(expected):
                sampler = DistributedSampler(
                    range(size),
                    num_replicas=replicas,
                    rank=rank,
                    shuffle=False,
                    seed=17,
                    drop_last=drop_last,
                )
                with self.subTest(
                    size=size,
                    replicas=replicas,
                    rank=rank,
                    drop_last=drop_last,
                ):
                    self.assertEqual(list(sampler), expected_indices)
                    self.assertEqual(len(sampler), num_samples)
                    self.assertEqual(operator.length_hint(sampler), num_samples)
                    self.assertEqual(sampler.num_samples, num_samples)
                    self.assertEqual(sampler.total_size, total_size)

    def test_state_set_epoch_and_cached_length(self):
        dataset = list(range(7))
        sampler = DistributedSampler(
            dataset,
            num_replicas=3,
            rank=1,
            shuffle=False,
            seed=23,
            drop_last=False,
        )

        self.assertEqual(
            sampler.__dict__,
            {
                "dataset": dataset,
                "num_replicas": 3,
                "rank": 1,
                "epoch": 0,
                "drop_last": False,
                "num_samples": 3,
                "total_size": 9,
                "shuffle": False,
                "seed": 23,
            },
        )
        self.assertIs(sampler.dataset, dataset)
        self.assertEqual(list(sampler), [1, 4, 0])

        result = sampler.set_epoch(9)
        self.assertIsNone(result)
        self.assertEqual(sampler.epoch, 9)
        self.assertEqual(list(sampler), [1, 4, 0])

        marker = object()
        sampler.set_epoch(marker)
        self.assertIs(sampler.epoch, marker)
        self.assertEqual(list(sampler), [1, 4, 0])

        dataset.pop()
        self.assertEqual(len(sampler), 3)
        self.assertEqual(list(sampler), [1, 4, 1])

    def test_rank_and_dataset_length_validation(self):
        cases = (
            (3, -1, "Invalid rank -1, rank should be in the interval [0, 2]"),
            (3, 3, "Invalid rank 3, rank should be in the interval [0, 2]"),
            (0, 0, "Invalid rank 0, rank should be in the interval [0, -1]"),
            (-2, 0, "Invalid rank 0, rank should be in the interval [0, -3]"),
        )
        for replicas, rank, message in cases:
            with self.subTest(replicas=replicas, rank=rank):
                with self.assertRaises(ValueError) as raised:
                    DistributedSampler(
                        [],
                        num_replicas=replicas,
                        rank=rank,
                        shuffle=False,
                    )
                self.assertEqual(str(raised.exception), message)

        with self.assertRaises(LengthError) as raised:
            DistributedSampler(
                FailingDataset(),
                num_replicas=1,
                rank=0,
                shuffle=False,
            )
        self.assertEqual(raised.exception.args, ("dataset length is unavailable",))

        with self.assertRaisesRegex(
            TypeError, r"^object of type 'object' has no len\(\)$"
        ):
            DistributedSampler(
                object(),
                num_replicas=1,
                rank=0,
                shuffle=False,
            )

    def test_implicit_discovery_and_shuffling_are_explicitly_unsupported(self):
        discovery_message = (
            "torch_rs.utils.data.DistributedSampler requires explicit "
            "num_replicas and rank; process-group discovery is not supported"
        )
        for call in (
            lambda: DistributedSampler([], shuffle=False),
            lambda: DistributedSampler([], num_replicas=1, shuffle=False),
            lambda: DistributedSampler([], rank=0, shuffle=False),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    NotImplementedError, f"^{discovery_message}$"
                ):
                    call()

        shuffle_message = (
            "torch_rs.utils.data.DistributedSampler does not support shuffle=True"
        )
        for shuffle in (True, 1, "yes"):
            with self.subTest(shuffle=shuffle):
                with self.assertRaisesRegex(
                    NotImplementedError, f"^{shuffle_message}$"
                ):
                    DistributedSampler([], num_replicas=1, rank=0, shuffle=shuffle)

        sampler = DistributedSampler(range(3), num_replicas=1, rank=0, shuffle=False)
        sampler.shuffle = True
        with self.assertRaisesRegex(NotImplementedError, f"^{shuffle_message}$"):
            iter(sampler)

        module = importlib.import_module("torch_rs.utils.data.distributed")
        self.assertNotIn("torch", module.__dict__)
        self.assertNotIn("dist", module.__dict__)

    def test_pytorch_truthiness_and_rank_validation_order_are_preserved(self):
        for shuffle in (False, 0, None, "", []):
            sampler = DistributedSampler(
                range(3), num_replicas=2, rank=0, shuffle=shuffle
            )
            with self.subTest(shuffle=shuffle):
                self.assertIs(sampler.shuffle, shuffle)
                self.assertEqual(list(sampler), [0, 2])

        with self.assertRaisesRegex(
            ValueError,
            r"^Invalid rank 2, rank should be in the interval \[0, 1\]$",
        ):
            DistributedSampler([], num_replicas=2, rank=2, shuffle=True)

    def test_pickle_round_trips_state_and_behavior_for_every_protocol(self):
        sampler = DistributedSampler(
            range(7),
            num_replicas=3,
            rank=2,
            shuffle=False,
            seed=101,
            drop_last=False,
        )
        sampler.set_epoch(12)

        self.assertIs(
            pickle.loads(pickle.dumps(DistributedSampler)), DistributedSampler
        )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(sampler, protocol=protocol))
                self.assertIs(type(restored), DistributedSampler)
                self.assertEqual(restored.__dict__, sampler.__dict__)
                self.assertEqual(list(restored), [2, 5, 1])
                self.assertEqual(len(restored), 3)

    def test_imports_exports_signature_annotations_and_metadata(self):
        data_module = importlib.import_module("torch_rs.utils.data")
        distributed_module = importlib.import_module("torch_rs.utils.data.distributed")

        self.assertIs(torch.utils.data.DistributedSampler, DistributedSampler)
        self.assertIs(data_module.DistributedSampler, DistributedSampler)
        self.assertIs(distributed_module.DistributedSampler, DistributedSampler)
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
        self.assertIs(distributed_namespace["DistributedSampler"], DistributedSampler)

        signature = inspect.signature(DistributedSampler)
        self.assertEqual(
            tuple(signature.parameters),
            ("dataset", "num_replicas", "rank", "shuffle", "seed", "drop_last"),
        )
        defaults = (None, None, True, 0, False)
        for parameter, default in zip(
            tuple(signature.parameters.values())[1:], defaults, strict=True
        ):
            self.assertIs(parameter.kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            self.assertEqual(parameter.default, default)
        self.assertIs(signature.parameters["dataset"].annotation, Dataset)
        self.assertEqual(signature.parameters["num_replicas"].annotation, int | None)
        self.assertEqual(signature.parameters["rank"].annotation, int | None)
        self.assertIs(signature.parameters["shuffle"].annotation, bool)
        self.assertIs(signature.parameters["seed"].annotation, int)
        self.assertIs(signature.parameters["drop_last"].annotation, bool)
        self.assertIs(signature.return_annotation, None)

        self.assertEqual(DistributedSampler.__annotations__, {})
        self.assertIs(DistributedSampler.__bases__[0], Sampler)
        self.assertIsInstance(DistributedSampler([], 1, 0, False), Sampler)
        self.assertIs(get_origin(DistributedSampler.__orig_bases__[0]), Sampler)
        (parameter,) = get_args(DistributedSampler.__orig_bases__[0])
        self.assertEqual(parameter.__name__, "_T_co")
        self.assertTrue(parameter.__covariant__)
        self.assertIs(get_origin(DistributedSampler[int]), DistributedSampler)
        self.assertEqual(get_args(DistributedSampler[int]), (int,))

        self.assertEqual(
            DistributedSampler.__iter__.__annotations__,
            {"return": Iterator[parameter]},
        )
        self.assertEqual(DistributedSampler.__len__.__annotations__, {"return": int})
        self.assertEqual(
            DistributedSampler.set_epoch.__annotations__,
            {"epoch": int, "return": None},
        )
        self.assertIn("Sampler that restricts data loading", DistributedSampler.__doc__)
        self.assertIn(
            "Set the epoch for this sampler", DistributedSampler.set_epoch.__doc__
        )


if __name__ == "__main__":
    unittest.main()

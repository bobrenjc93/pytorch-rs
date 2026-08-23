import importlib
import inspect
import operator
import pickle
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
class DistributedSamplerReferenceTests(unittest.TestCase):
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

    def assert_state_matches(self, actual, expected):
        self.assertEqual(tuple(actual.__dict__), tuple(expected.__dict__))
        actual_state = actual.__dict__.copy()
        expected_state = expected.__dict__.copy()
        actual_dataset = actual_state.pop("dataset")
        expected_dataset = expected_state.pop("dataset")
        self.assertEqual(actual_dataset, expected_dataset)
        self.assertEqual(actual_state, expected_state)

    def test_partitioning_padding_drop_last_length_and_empty_datasets_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_type = torch.utils.data.DistributedSampler
        expected_type = reference_torch.utils.data.DistributedSampler

        for size in range(11):
            for num_replicas in (1, 2, 3, 5):
                for drop_last in (False, True):
                    for rank in range(num_replicas):
                        actual_dataset = list(range(size))
                        expected_dataset = list(range(size))
                        actual = actual_type(
                            actual_dataset,
                            num_replicas=num_replicas,
                            rank=rank,
                            shuffle=False,
                            seed=19,
                            drop_last=drop_last,
                        )
                        expected = expected_type(
                            expected_dataset,
                            num_replicas=num_replicas,
                            rank=rank,
                            shuffle=False,
                            seed=19,
                            drop_last=drop_last,
                        )
                        with self.subTest(
                            size=size,
                            num_replicas=num_replicas,
                            rank=rank,
                            drop_last=drop_last,
                        ):
                            self.assert_state_matches(actual, expected)
                            self.assertEqual(type(iter(actual)), type(iter(expected)))
                            self.assertEqual(list(actual), list(expected))
                            self.assertEqual(len(actual), len(expected))
                            self.assertEqual(
                                operator.length_hint(actual),
                                operator.length_hint(expected),
                            )

    def test_cached_size_and_mutated_dataset_behavior_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_source = MutableSizedSource(5)
        expected_source = MutableSizedSource(5)
        actual = torch.utils.data.DistributedSampler(
            actual_source, num_replicas=2, rank=1, shuffle=False
        )
        expected = reference_torch.utils.data.DistributedSampler(
            expected_source, num_replicas=2, rank=1, shuffle=False
        )

        self.assertEqual(actual_source.length_calls, expected_source.length_calls)
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(actual_source.length_calls, expected_source.length_calls)
        self.assertEqual(list(actual), list(expected))
        self.assertEqual(actual_source.length_calls, expected_source.length_calls)

        for size in (4, 6):
            actual_source.size = size
            expected_source.size = size
            with self.subTest(size=size):
                self.assertEqual(list(actual), list(expected))
                self.assertEqual(
                    actual_source.length_calls, expected_source.length_calls
                )

        for size in (7, 0):
            actual_source.size = size
            expected_source.size = size
            with self.subTest(size=size):
                self.assert_error_matches(lambda: iter(actual), lambda: iter(expected))

        actual_source = MutableSizedSource(5)
        expected_source = MutableSizedSource(5)
        actual = torch.utils.data.DistributedSampler(
            actual_source,
            num_replicas=2,
            rank=0,
            shuffle=False,
            drop_last=True,
        )
        expected = reference_torch.utils.data.DistributedSampler(
            expected_source,
            num_replicas=2,
            rank=0,
            shuffle=False,
            drop_last=True,
        )
        actual_source.size = 3
        expected_source.size = 3
        self.assert_error_matches(lambda: iter(actual), lambda: iter(expected))

    def test_validation_call_forms_and_falsey_shuffle_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_type = torch.utils.data.DistributedSampler
        expected_type = reference_torch.utils.data.DistributedSampler

        for num_replicas, rank in (
            (3, -1),
            (3, 3),
            (3, 8),
            (0, 0),
            (-2, 0),
            ("3", 0),
            (3, "0"),
        ):
            with self.subTest(num_replicas=num_replicas, rank=rank):
                self.assert_error_matches(
                    lambda num_replicas=num_replicas, rank=rank: actual_type(
                        [], num_replicas, rank, False
                    ),
                    lambda num_replicas=num_replicas, rank=rank: expected_type(
                        [], num_replicas, rank, False
                    ),
                )

        error_pairs = (
            (lambda: actual_type(), lambda: expected_type()),
            (
                lambda: actual_type([], 2, 0, False, 0, False, None),
                lambda: expected_type([], 2, 0, False, 0, False, None),
            ),
            (
                lambda: actual_type([], 2, 0, False, extra=True),
                lambda: expected_type([], 2, 0, False, extra=True),
            ),
            (
                lambda: actual_type([], 2, 0, False, dataset=[]),
                lambda: expected_type([], 2, 0, False, dataset=[]),
            ),
            (
                lambda: actual_type(
                    dataset=[], num_replicas=2, rank=0, shuffle=False, seed=0,
                    drop_last=False, extra=True
                ),
                lambda: expected_type(
                    dataset=[], num_replicas=2, rank=0, shuffle=False, seed=0,
                    drop_last=False, extra=True
                ),
            ),
            (
                lambda: actual_type(
                    object(), num_replicas=2, rank=0, shuffle=False
                ),
                lambda: expected_type(
                    object(), num_replicas=2, rank=0, shuffle=False
                ),
            ),
            (
                lambda: actual_type(
                    FailingSizedSource(),
                    num_replicas=2,
                    rank=0,
                    shuffle=False,
                ),
                lambda: expected_type(
                    FailingSizedSource(),
                    num_replicas=2,
                    rank=0,
                    shuffle=False,
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(error_pairs):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        for shuffle in (False, 0, None, "", []):
            for drop_last in (False, True, 0, 1, None, ""):
                with self.subTest(shuffle=shuffle, drop_last=drop_last):
                    actual = actual_type(
                        range(5), 3, 1, shuffle, seed="stored", drop_last=drop_last
                    )
                    expected = expected_type(
                        range(5), 3, 1, shuffle, seed="stored", drop_last=drop_last
                    )
                    self.assert_state_matches(actual, expected)
                    self.assertEqual(list(actual), list(expected))
                    self.assertEqual(len(actual), len(expected))

    def test_set_epoch_state_and_pickle_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.utils.data.DistributedSampler(
            list(range(8)), 3, 1, False, seed=29, drop_last=False
        )
        expected = reference_torch.utils.data.DistributedSampler(
            list(range(8)), 3, 1, False, seed=29, drop_last=False
        )

        for epoch in (4, -3, "epoch", None):
            with self.subTest(epoch=epoch):
                self.assertEqual(actual.set_epoch(epoch), expected.set_epoch(epoch))
                self.assert_state_matches(actual, expected)
                self.assertEqual(list(actual), list(expected))

        actual.set_epoch(11)
        expected.set_epoch(11)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored_actual = pickle.loads(pickle.dumps(actual, protocol=protocol))
                restored_expected = pickle.loads(
                    pickle.dumps(expected, protocol=protocol)
                )
                self.assertEqual(
                    restored_actual.__module__.replace("torch_rs", "torch"),
                    restored_expected.__module__,
                )
                self.assertEqual(
                    restored_actual.__class__.__name__,
                    restored_expected.__class__.__name__,
                )
                self.assert_state_matches(restored_actual, restored_expected)
                self.assertEqual(list(restored_actual), list(restored_expected))
                self.assertEqual(len(restored_actual), len(restored_expected))
                self.assertIs(
                    pickle.loads(
                        pickle.dumps(type(actual), protocol=protocol)
                    ),
                    type(actual),
                )
                self.assertIs(
                    pickle.loads(
                        pickle.dumps(type(expected), protocol=protocol)
                    ),
                    type(expected),
                )

    def test_signature_annotations_documentation_and_metadata_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.utils.data.DistributedSampler
        expected = reference_torch.utils.data.DistributedSampler

        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(inspect.isabstract(actual), inspect.isabstract(expected))
        declared_methods = ("__init__", "__iter__", "__len__", "set_epoch")
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
        self.assertEqual(len(actual.__parameters__), len(expected.__parameters__))
        self.assertIs(actual.__bases__[0], torch.utils.data.Sampler)
        self.assertIs(expected.__bases__[0], reference_torch.utils.data.Sampler)

        for name in ("__init__", "__iter__", "__len__", "set_epoch"):
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

        actual_base = actual.__orig_bases__[0]
        expected_base = expected.__orig_bases__[0]
        self.assertIs(get_origin(actual_base), torch.utils.data.Sampler)
        self.assertIs(get_origin(expected_base), reference_torch.utils.data.Sampler)
        actual_parameter = get_args(actual_base)[0]
        expected_parameter = get_args(expected_base)[0]
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

    def test_imports_exports_and_unsupported_neighbors_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_data = importlib.import_module("torch_rs.utils.data")
        expected_data = importlib.import_module("torch.utils.data")
        actual_module = importlib.import_module("torch_rs.utils.data.distributed")
        expected_module = importlib.import_module("torch.utils.data.distributed")
        actual_sampler = importlib.import_module("torch_rs.utils.data.sampler")
        expected_sampler = importlib.import_module("torch.utils.data.sampler")
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
            "default_convert",
            "get_worker_info",
        }

        self.assertIs(actual_data.DistributedSampler, actual_module.DistributedSampler)
        self.assertIs(
            expected_data.DistributedSampler, expected_module.DistributedSampler
        )
        self.assertEqual(
            actual_data.__all__,
            [name for name in expected_data.__all__ if name in supported],
        )
        self.assertEqual(actual_module.__all__, expected_module.__all__)
        self.assertEqual(
            hasattr(actual_sampler, "DistributedSampler"),
            hasattr(expected_sampler, "DistributedSampler"),
        )

        actual_data_namespace = {}
        expected_data_namespace = {}
        actual_module_namespace = {}
        expected_module_namespace = {}
        exec("from torch_rs.utils.data import *", actual_data_namespace)
        exec("from torch.utils.data import *", expected_data_namespace)
        exec(
            "from torch_rs.utils.data.distributed import *",
            actual_module_namespace,
        )
        exec(
            "from torch.utils.data.distributed import *",
            expected_module_namespace,
        )
        self.assertIs(
            actual_data_namespace["DistributedSampler"],
            actual_data.DistributedSampler,
        )
        self.assertIs(
            expected_data_namespace["DistributedSampler"],
            expected_data.DistributedSampler,
        )
        self.assertEqual(
            tuple(actual_module_namespace), tuple(expected_module_namespace)
        )

        for name in (
            "DataLoader",
            "RandomSampler",
            "SubsetRandomSampler",
            "WeightedRandomSampler",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(actual_module, name))


if __name__ == "__main__":
    unittest.main()

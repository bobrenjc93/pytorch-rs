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


class LengthError(Exception):
    pass


class FailingDataset:
    def __len__(self):
        raise LengthError("dataset length is unavailable")


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

    def assert_state_matches(self, actual, expected, actual_dataset, expected_dataset):
        self.assertEqual(tuple(actual.__dict__), tuple(expected.__dict__))
        self.assertIs(actual.dataset, actual_dataset)
        self.assertIs(expected.dataset, expected_dataset)
        for name in actual.__dict__.keys() - {"dataset"}:
            self.assertEqual(getattr(actual, name), getattr(expected, name))

    def test_partitioning_padding_drop_last_lengths_and_empty_datasets_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_type = torch.utils.data.DistributedSampler
        expected_type = reference_torch.utils.data.DistributedSampler

        for size in range(11):
            for replicas in range(1, 7):
                for drop_last in (False, True):
                    for rank in range(replicas):
                        actual_dataset = range(size)
                        expected_dataset = range(size)
                        actual = actual_type(
                            actual_dataset,
                            num_replicas=replicas,
                            rank=rank,
                            shuffle=False,
                            seed=37,
                            drop_last=drop_last,
                        )
                        expected = expected_type(
                            expected_dataset,
                            num_replicas=replicas,
                            rank=rank,
                            shuffle=False,
                            seed=37,
                            drop_last=drop_last,
                        )
                        with self.subTest(
                            size=size,
                            replicas=replicas,
                            rank=rank,
                            drop_last=drop_last,
                        ):
                            self.assert_state_matches(
                                actual,
                                expected,
                                actual_dataset,
                                expected_dataset,
                            )
                            self.assertEqual(type(iter(actual)), type(iter(expected)))
                            self.assertEqual(list(actual), list(expected))
                            self.assertEqual(len(actual), len(expected))
                            self.assertEqual(
                                operator.length_hint(actual),
                                operator.length_hint(expected),
                            )

    def test_set_epoch_cached_state_and_falsy_shuffle_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_type = torch.utils.data.DistributedSampler
        expected_type = reference_torch.utils.data.DistributedSampler

        for shuffle in (False, 0, None, "", []):
            actual_dataset = list(range(7))
            expected_dataset = list(range(7))
            actual = actual_type(actual_dataset, 3, 1, shuffle, 91, False)
            expected = expected_type(expected_dataset, 3, 1, shuffle, 91, False)
            with self.subTest(shuffle=shuffle):
                self.assert_state_matches(
                    actual, expected, actual_dataset, expected_dataset
                )
                self.assertEqual(list(actual), list(expected))
                for epoch in (0, 7, -3, "epoch"):
                    self.assertIsNone(actual.set_epoch(epoch))
                    self.assertIsNone(expected.set_epoch(epoch))
                    self.assertEqual(actual.epoch, expected.epoch)
                    self.assertEqual(list(actual), list(expected))

                actual_dataset.pop()
                expected_dataset.pop()
                self.assertEqual(len(actual), len(expected))
                self.assertEqual(list(actual), list(expected))

    def test_rank_dataset_length_and_call_form_validation_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_type = torch.utils.data.DistributedSampler
        expected_type = reference_torch.utils.data.DistributedSampler

        for replicas, rank in ((3, -1), (3, 3), (0, 0), (-2, 0)):
            with self.subTest(replicas=replicas, rank=rank):
                self.assert_error_matches(
                    lambda replicas=replicas, rank=rank: actual_type(
                        [], replicas, rank, False
                    ),
                    lambda replicas=replicas, rank=rank: expected_type(
                        [], replicas, rank, False
                    ),
                )

        self.assert_error_matches(
            lambda: actual_type(FailingDataset(), 1, 0, False),
            lambda: expected_type(FailingDataset(), 1, 0, False),
        )
        self.assert_error_matches(
            lambda: actual_type(object(), 1, 0, False),
            lambda: expected_type(object(), 1, 0, False),
        )

        for actual_call, expected_call in (
            (
                lambda: actual_type([], 1, 0, False, 0, False, None),
                lambda: expected_type([], 1, 0, False, 0, False, None),
            ),
            (
                lambda: actual_type([], 1, 0, False, num_replicas=1),
                lambda: expected_type([], 1, 0, False, num_replicas=1),
            ),
            (
                lambda: actual_type([], 1, 0, False, extra=True),
                lambda: expected_type([], 1, 0, False, extra=True),
            ),
        ):
            self.assert_error_matches(actual_call, expected_call)

    def test_pickle_round_trips_match_for_every_protocol(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.utils.data.DistributedSampler(range(7), 3, 2, False, 101, False)
        expected = reference_torch.utils.data.DistributedSampler(
            range(7), 3, 2, False, 101, False
        )
        actual.set_epoch(12)
        expected.set_epoch(12)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                actual_restored = pickle.loads(pickle.dumps(actual, protocol=protocol))
                expected_restored = pickle.loads(
                    pickle.dumps(expected, protocol=protocol)
                )
                self.assertEqual(
                    tuple(actual_restored.__dict__),
                    tuple(expected_restored.__dict__),
                )
                self.assertEqual(actual_restored.__dict__, expected_restored.__dict__)
                self.assertEqual(list(actual_restored), list(expected_restored))
                self.assertEqual(len(actual_restored), len(expected_restored))

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
        actual_annotations = actual.__annotations__
        expected_annotations = expected.__annotations__
        dynamic_cache_names = {"__annotations__", "__slotnames__"}
        self.assertEqual(
            tuple(name for name in actual.__dict__ if name not in dynamic_cache_names),
            tuple(
                name for name in expected.__dict__ if name not in dynamic_cache_names
            ),
        )
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
        self.assertIs(actual.__bases__[0], torch.utils.data.Sampler)
        self.assertIs(expected.__bases__[0], reference_torch.utils.data.Sampler)
        self.assertIs(get_origin(actual[int]), actual)
        self.assertIs(get_origin(expected[int]), expected)
        self.assertEqual(get_args(actual[int]), get_args(expected[int]))

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

        self.assertIs(get_origin(actual.__orig_bases__[0]), torch.utils.data.Sampler)
        self.assertIs(
            get_origin(expected.__orig_bases__[0]),
            reference_torch.utils.data.Sampler,
        )
        self.assertIs(get_args(actual.__orig_bases__[0])[0], actual_parameter)
        self.assertIs(get_args(expected.__orig_bases__[0])[0], expected_parameter)

    def test_imports_exports_and_unsupported_boundaries_match_scope(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_data = importlib.import_module("torch_rs.utils.data")
        expected_data = importlib.import_module("torch.utils.data")
        actual_module = importlib.import_module("torch_rs.utils.data.distributed")
        expected_module = importlib.import_module("torch.utils.data.distributed")
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

        discovery_message = (
            "torch_rs.utils.data.DistributedSampler requires explicit "
            "num_replicas and rank; process-group discovery is not supported"
        )
        for call in (
            lambda: actual_data.DistributedSampler([], shuffle=False),
            lambda: actual_data.DistributedSampler([], num_replicas=1, shuffle=False),
            lambda: actual_data.DistributedSampler([], rank=0, shuffle=False),
        ):
            with self.assertRaisesRegex(NotImplementedError, f"^{discovery_message}$"):
                call()

        with self.assertRaisesRegex(
            NotImplementedError,
            "^torch_rs.utils.data.DistributedSampler does not support shuffle=True$",
        ):
            actual_data.DistributedSampler([], 1, 0, True)
        shuffled_reference = expected_data.DistributedSampler([], 1, 0, True)
        self.assertEqual(list(shuffled_reference), [])


if __name__ == "__main__":
    unittest.main()

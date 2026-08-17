import importlib
import inspect
from typing import get_args, get_origin
import unittest

import numpy as np
import torch_rs as torch

from torch_rs.utils.data import Dataset, StackDataset, TensorDataset


class RecordingDataset(Dataset):
    def __init__(self, prefix):
        self.prefix = prefix
        self.calls = []

    def __getitem__(self, index):
        self.calls.append(index)
        return f"{self.prefix}-item-{index}"

    def __len__(self):
        return 3


class BatchedRecordingDataset(RecordingDataset):
    def __getitems__(self, indices):
        self.calls.append(("batch", indices))
        return [f"{self.prefix}-batch-{index}" for index in indices]


class NonCallableBatchedDataset(RecordingDataset):
    __getitems__ = None


class MismatchedBatchedDataset(RecordingDataset):
    def __getitems__(self, indices):
        return [f"{self.prefix}-batch-{index}" for index in indices[:-1]]


class StackDatasetTests(unittest.TestCase):
    def test_positional_and_keyword_datasets_return_matching_containers(self):
        features = TensorDataset(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        targets = TensorDataset(torch.tensor([10.0, 20.0, 30.0]))

        positional = StackDataset(features, targets)
        self.assertIs(type(positional.datasets), tuple)
        self.assertEqual(len(positional.datasets), 2)
        self.assertIs(positional.datasets[0], features)
        self.assertIs(positional.datasets[1], targets)
        self.assertEqual(len(positional), 3)

        positional_sample = positional[-1]
        self.assertIs(type(positional_sample), tuple)
        self.assertEqual(len(positional_sample), 2)
        self.assertIs(type(positional_sample[0]), tuple)
        self.assertIs(type(positional_sample[1]), tuple)
        np.testing.assert_array_equal(np.asarray(positional_sample[0][0]), [5.0, 6.0])
        self.assertEqual(positional_sample[1][0].item(), 30.0)

        keyword = StackDataset(image=features, label=targets)
        self.assertIs(type(keyword.datasets), dict)
        self.assertEqual(list(keyword.datasets), ["image", "label"])
        self.assertIs(keyword.datasets["image"], features)
        self.assertIs(keyword.datasets["label"], targets)
        self.assertEqual(len(keyword), 3)

        keyword_sample = keyword[0]
        self.assertIs(type(keyword_sample), dict)
        self.assertEqual(list(keyword_sample), ["image", "label"])
        np.testing.assert_array_equal(
            np.asarray(keyword_sample["image"][0]), [1.0, 2.0]
        )
        self.assertEqual(keyword_sample["label"][0].item(), 10.0)

    def test_length_and_integer_indexing_preserve_views_and_autograd(self):
        feature_source = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            requires_grad=True,
        )
        target_source = torch.tensor([10.0, 20.0, 30.0])
        dataset = StackDataset(
            feature=TensorDataset(feature_source),
            target=TensorDataset(target_source),
        )

        self.assertEqual(len(dataset), 3)
        for index, normalized in ((0, 0), (1, 1), (2, 2), (-1, 2), (-3, 0)):
            with self.subTest(index=index):
                sample = dataset[index]
                feature = sample["feature"][0]
                target = sample["target"][0]

                self.assertEqual(feature.shape, (3,))
                self.assertEqual(feature.stride(), (1,))
                self.assertEqual(feature.storage_offset(), normalized * 3)
                self.assertEqual(target.shape, ())
                self.assertEqual(target.storage_offset(), normalized)
                self.assertTrue(feature.requires_grad)
                self.assertFalse(feature.is_leaf)
                self.assertEqual(
                    feature.data_ptr() - feature_source.data_ptr(),
                    feature.storage_offset() * feature_source.element_size(),
                )
                self.assertEqual(
                    target.data_ptr() - target_source.data_ptr(),
                    target.storage_offset() * target_source.element_size(),
                )

        sample = dataset[-2]["feature"][0]
        (sample * torch.tensor([2.0, 3.0, 5.0])).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(feature_source.grad),
            [[0.0, 0.0, 0.0], [2.0, 3.0, 5.0], [0.0, 0.0, 0.0]],
        )

    def test_getitems_preserves_containers_views_and_autograd_lineage(self):
        indices = [2, 0, 2, -1]
        normalized_indices = [2, 0, 2, 2]

        for keyword in (False, True):
            with self.subTest(keyword=keyword):
                feature_source = torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                    requires_grad=True,
                )
                target_source = torch.tensor([10.0, 20.0, 30.0])
                features = TensorDataset(feature_source)
                targets = TensorDataset(target_source)
                if keyword:
                    dataset = StackDataset(feature=features, target=targets)
                else:
                    dataset = StackDataset(features, targets)

                self.assertEqual(dataset.__getitems__([]), [])
                batch = dataset.__getitems__(indices)
                self.assertIs(type(batch), list)
                self.assertEqual(len(batch), len(indices))

                selected_features = []
                for sample, normalized in zip(
                    batch, normalized_indices, strict=True
                ):
                    self.assertIs(type(sample), dict if keyword else tuple)
                    if keyword:
                        self.assertEqual(list(sample), ["feature", "target"])
                        feature = sample["feature"][0]
                        target = sample["target"][0]
                    else:
                        self.assertEqual(len(sample), 2)
                        feature = sample[0][0]
                        target = sample[1][0]

                    self.assertEqual(feature.shape, (2,))
                    self.assertEqual(feature.stride(), (1,))
                    self.assertEqual(feature.storage_offset(), normalized * 2)
                    self.assertEqual(target.shape, ())
                    self.assertEqual(target.storage_offset(), normalized)
                    self.assertTrue(feature.requires_grad)
                    self.assertFalse(feature.is_leaf)
                    self.assertEqual(
                        feature.data_ptr() - feature_source.data_ptr(),
                        feature.storage_offset() * feature_source.element_size(),
                    )
                    self.assertEqual(
                        target.data_ptr() - target_source.data_ptr(),
                        target.storage_offset() * target_source.element_size(),
                    )
                    selected_features.append(feature)

                (selected_features[-1] * torch.tensor([2.0, 3.0])).sum().backward()
                np.testing.assert_array_equal(
                    np.asarray(feature_source.grad),
                    [[0.0, 0.0], [0.0, 0.0], [2.0, 3.0]],
                )

    def test_getitems_delegates_batches_and_falls_back_to_indexing(self):
        indices = [2, 0, 2, -1]

        for keyword in (False, True):
            with self.subTest(keyword=keyword):
                delegated = BatchedRecordingDataset("delegated")
                fallback = RecordingDataset("fallback")
                noncallable = NonCallableBatchedDataset("noncallable")
                if keyword:
                    dataset = StackDataset(
                        delegated=delegated,
                        fallback=fallback,
                        noncallable=noncallable,
                    )
                    expected = [
                        {
                            "delegated": f"delegated-batch-{index}",
                            "fallback": f"fallback-item-{index}",
                            "noncallable": f"noncallable-item-{index}",
                        }
                        for index in indices
                    ]
                else:
                    dataset = StackDataset(delegated, fallback, noncallable)
                    expected = [
                        (
                            f"delegated-batch-{index}",
                            f"fallback-item-{index}",
                            f"noncallable-item-{index}",
                        )
                        for index in indices
                    ]

                self.assertEqual(dataset.__getitems__(indices), expected)
                self.assertEqual(delegated.calls, [("batch", indices)])
                self.assertEqual(fallback.calls, indices)
                self.assertEqual(noncallable.calls, indices)

                empty_indices = []
                self.assertEqual(dataset.__getitems__(empty_indices), [])
                self.assertEqual(
                    delegated.calls,
                    [("batch", indices), ("batch", empty_indices)],
                )
                self.assertEqual(fallback.calls, indices)
                self.assertEqual(noncallable.calls, indices)

    def test_getitems_rejects_mismatched_nested_batch_lengths(self):
        for dataset in (
            StackDataset(MismatchedBatchedDataset("child")),
            StackDataset(child=MismatchedBatchedDataset("child")),
        ):
            with self.subTest(container=type(dataset.datasets).__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    r"^Nested dataset's output size mismatch\. Expected 3, got 2$",
                ):
                    dataset.__getitems__([2, 0, -1])

    def test_empty_mixed_and_length_mismatch_validation(self):
        first_empty = TensorDataset(torch.zeros((0, 2)))
        second_empty = TensorDataset(torch.zeros((0,)))
        for dataset in (
            StackDataset(first_empty, second_empty),
            StackDataset(first=first_empty, second=second_empty),
        ):
            with self.subTest(container=type(dataset.datasets).__name__):
                self.assertEqual(len(dataset), 0)
                for index in (0, -1):
                    with self.assertRaisesRegex(
                        IndexError,
                        rf"^index {index} is out of bounds for dimension 0 with size 0$",
                    ):
                        dataset[index]

        with self.assertRaisesRegex(
            ValueError, "^At least one dataset should be passed$"
        ):
            StackDataset()
        with self.assertRaisesRegex(
            ValueError,
            r"^Supported either ``tuple``- \(via ``args``\) or"
            r"``dict``- \(via ``kwargs``\) like input/output, but both types are given\.$",
        ):
            StackDataset([1], named=[1])
        for call in (
            lambda: StackDataset([1], [2, 3]),
            lambda: StackDataset(first=[1], second=[2, 3]),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    ValueError, "^Size mismatch between datasets$"
                ):
                    call()

    def test_imports_exports_signatures_annotations_and_documentation(self):
        data_module = importlib.import_module("torch_rs.utils.data")
        dataset_module = importlib.import_module("torch_rs.utils.data.dataset")

        self.assertIs(torch.utils.data, data_module)
        self.assertIs(data_module.StackDataset, StackDataset)
        self.assertIs(dataset_module.StackDataset, StackDataset)
        self.assertTrue(issubclass(StackDataset, Dataset))
        self.assertIsInstance(StackDataset([1]), Dataset)
        self.assertEqual(StackDataset.__module__, "torch_rs.utils.data.dataset")
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
                "get_worker_info",
            ],
        )
        self.assertEqual(
            dataset_module.__all__,
            [
                "Dataset",
                "IterableDataset",
                "TensorDataset",
                "StackDataset",
                "ConcatDataset",
                "ChainDataset",
                "Subset",
            ],
        )

        wildcard_namespace = {}
        exec("from torch_rs.utils.data import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["StackDataset"], StackDataset)

        signature = inspect.signature(StackDataset)
        self.assertEqual(tuple(signature.parameters), ("args", "kwargs"))
        args_parameter = signature.parameters["args"]
        kwargs_parameter = signature.parameters["kwargs"]
        self.assertIs(args_parameter.kind, inspect.Parameter.VAR_POSITIONAL)
        self.assertIs(kwargs_parameter.kind, inspect.Parameter.VAR_KEYWORD)
        self.assertIs(get_origin(args_parameter.annotation), Dataset)
        self.assertEqual(args_parameter.annotation, kwargs_parameter.annotation)
        (sample_type,) = get_args(args_parameter.annotation)
        self.assertTrue(sample_type.__covariant__)
        self.assertIs(signature.return_annotation, None)
        self.assertEqual(StackDataset.__annotations__, {"datasets": tuple | dict})
        self.assertEqual(
            str(inspect.signature(StackDataset.__getitem__)), "(self, index)"
        )
        self.assertEqual(StackDataset.__getitems__.__annotations__, {"indices": list})
        self.assertEqual(
            str(inspect.signature(StackDataset.__getitems__)),
            "(self, indices: list)",
        )
        self.assertEqual(str(inspect.signature(StackDataset.__len__)), "(self) -> int")
        self.assertIn("__getitems__", StackDataset.__dict__)
        self.assertIsNone(StackDataset.__getitems__.__doc__)
        self.assertIn(
            "Dataset as a stacking of multiple datasets.", StackDataset.__doc__
        )
        self.assertIn(
            "*args (Dataset): Datasets for stacking returned as tuple.",
            StackDataset.__doc__,
        )
        self.assertIn(
            "**kwargs (Dataset): Datasets for stacking returned as dict.",
            StackDataset.__doc__,
        )

        dataset = StackDataset([10])
        with self.assertRaisesRegex(
            TypeError,
            r"^StackDataset.__getitem__\(\) missing 1 required positional argument: "
            "'index'$",
        ):
            dataset.__getitem__()
        with self.assertRaisesRegex(
            TypeError,
            r"^StackDataset.__getitem__\(\) takes 2 positional arguments but 3 were given$",
        ):
            dataset.__getitem__(0, 1)
        with self.assertRaisesRegex(
            TypeError,
            r"^StackDataset.__getitems__\(\) missing 1 required positional argument: "
            "'indices'$",
        ):
            dataset.__getitems__()
        with self.assertRaisesRegex(
            TypeError,
            r"^StackDataset.__getitems__\(\) takes 2 positional arguments but 3 were given$",
        ):
            dataset.__getitems__([], [])
        with self.assertRaisesRegex(
            TypeError,
            r"^StackDataset.__len__\(\) takes 1 positional argument but 2 were given$",
        ):
            dataset.__len__(1)


if __name__ == "__main__":
    unittest.main()

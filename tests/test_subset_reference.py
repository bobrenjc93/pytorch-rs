import importlib
import inspect
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SubsetReferenceTests(unittest.TestCase):
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

    def assert_tensor_matches(self, actual, expected, actual_source, expected_source):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        np.testing.assert_array_equal(
            np.asarray(actual.detach()), expected.detach().cpu().numpy()
        )
        self.assertEqual(
            actual.data_ptr() - actual_source.data_ptr(),
            expected.data_ptr() - expected_source.data_ptr(),
        )
        self.assertEqual(
            actual.storage_offset() - actual_source.storage_offset(),
            expected.storage_offset() - expected_source.storage_offset(),
        )
        self.assertEqual(
            expected.untyped_storage().data_ptr(),
            expected_source.untyped_storage().data_ptr(),
        )

    def recording_dataset(self, base, batched):
        class RecordingDataset(base):
            def __init__(self):
                self.calls = []

            def __getitem__(self, index):
                self.calls.append(index)
                if isinstance(index, list):
                    return [f"item-{item}" for item in index]
                return f"item-{index}"

        if batched is True:
            def getitems(self, indices):
                self.calls.append(("batch", indices))
                return [f"batch-{item}" for item in indices]

            RecordingDataset.__getitems__ = getitems
        elif batched is None:
            RecordingDataset.__getitems__ = None
        return RecordingDataset()

    def test_length_integer_indexing_views_and_autograd_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        actual_source = torch.tensor(values, requires_grad=True)
        expected_source = reference_torch.tensor(values, requires_grad=True)
        actual_dataset = torch.utils.data.TensorDataset(actual_source)
        expected_dataset = reference_torch.utils.data.TensorDataset(expected_source)
        actual_indices = [2, 0, 1]
        expected_indices = [2, 0, 1]
        actual = torch.utils.data.Subset(actual_dataset, actual_indices)
        expected = reference_torch.utils.data.Subset(
            expected_dataset, expected_indices
        )

        self.assertEqual(len(actual), len(expected))
        self.assertIs(actual.dataset, actual_dataset)
        self.assertIs(expected.dataset, expected_dataset)
        self.assertIs(actual.indices, actual_indices)
        self.assertIs(expected.indices, expected_indices)
        for index in (0, 1, 2, -1, -3, np.int64(1)):
            with self.subTest(index=index):
                self.assert_tensor_matches(
                    actual[index][0],
                    expected[index][0],
                    actual_source,
                    expected_source,
                )

        actual_selected = actual[1][0]
        expected_selected = expected[1][0]
        (actual_selected * torch.tensor([2.0, 3.0, 5.0])).sum().backward()
        (
            expected_selected * reference_torch.tensor([2.0, 3.0, 5.0])
        ).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_source.grad), expected_source.grad.detach().cpu().numpy()
        )

    def test_list_indexing_getitems_delegation_and_fallback_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        for batched in (False, True, None):
            with self.subTest(batched=batched):
                actual_dataset = self.recording_dataset(
                    torch.utils.data.Dataset, batched
                )
                expected_dataset = self.recording_dataset(
                    reference_torch.utils.data.Dataset, batched
                )
                actual = torch.utils.data.Subset(actual_dataset, (4, 1, 3))
                expected = reference_torch.utils.data.Subset(
                    expected_dataset, (4, 1, 3)
                )

                self.assertEqual(actual[[2, 0, -1]], expected[[2, 0, -1]])
                self.assertEqual(actual_dataset.calls, expected_dataset.calls)
                actual_dataset.calls.clear()
                expected_dataset.calls.clear()

                self.assertEqual(
                    actual.__getitems__([2, 0, -1]),
                    expected.__getitems__([2, 0, -1]),
                )
                self.assertEqual(actual_dataset.calls, expected_dataset.calls)

        self.assert_error_matches(
            lambda: torch.utils.data.Subset([10, 20], [1])[[0]],
            lambda: reference_torch.utils.data.Subset([10, 20], [1])[[0]],
        )

    def test_subclass_override_guard_matches(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        def incomplete(subset_type):
            class IncompleteSubset(subset_type):
                def __getitem__(self, idx):
                    return subset_type.__getitem__(self, idx)

            return IncompleteSubset

        self.assert_error_matches(
            lambda: incomplete(torch.utils.data.Subset)([10], [0]),
            lambda: incomplete(reference_torch.utils.data.Subset)([10], [0]),
        )

        def complete(subset_type):
            class CompleteSubset(subset_type):
                def __getitem__(self, idx):
                    return subset_type.__getitem__(self, idx)

                def __getitems__(self, indices):
                    return [self.__getitem__(idx) for idx in indices]

            return CompleteSubset

        actual = complete(torch.utils.data.Subset)([10, 20], [1, 0])
        expected = complete(reference_torch.utils.data.Subset)([10, 20], [1, 0])
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual.__getitems__([0, 1]), expected.__getitems__([0, 1]))

        def disabled(subset_type):
            class DisabledBatchSubset(subset_type):
                __getitems__ = None

                def __getitem__(self, idx):
                    return subset_type.__getitem__(self, idx)

            return DisabledBatchSubset

        self.assertEqual(
            disabled(torch.utils.data.Subset)([10], [0])[0],
            disabled(reference_torch.utils.data.Subset)([10], [0])[0],
        )

    def test_imports_exports_annotations_signatures_docs_and_errors_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_data = importlib.import_module("torch_rs.utils.data")
        expected_data = importlib.import_module("torch.utils.data")
        actual_module = importlib.import_module("torch_rs.utils.data.dataset")
        expected_module = importlib.import_module("torch.utils.data.dataset")
        actual = actual_data.Subset
        expected = expected_data.Subset
        supported = {
            "BatchSampler",
            "ConcatDataset",
            "Dataset",
            "Sampler",
            "StackDataset",
            "Subset",
            "TensorDataset",
        }

        self.assertIs(actual, actual_module.Subset)
        self.assertIs(expected, expected_module.Subset)
        self.assertEqual(
            actual_data.__all__,
            [name for name in expected_data.__all__ if name in supported],
        )
        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name in supported],
        )
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(
            str(inspect.signature(actual)).replace("torch_rs", "torch"),
            str(inspect.signature(expected)),
        )
        self.assertEqual(
            str(actual.__annotations__).replace("torch_rs", "torch"),
            str(expected.__annotations__),
        )
        self.assertTrue(issubclass(actual, actual_data.Dataset))
        self.assertTrue(issubclass(expected, expected_data.Dataset))

        for actual_method, expected_method in (
            (actual.__init__, expected.__init__),
            (actual.__getitem__, expected.__getitem__),
            (actual.__getitems__, expected.__getitems__),
            (actual.__len__, expected.__len__),
        ):
            self.assertEqual(
                str(inspect.signature(actual_method)).replace("torch_rs", "torch"),
                str(inspect.signature(expected_method)),
            )
            self.assertEqual(actual_method.__doc__, expected_method.__doc__)
            self.assertEqual(
                str(actual_method.__annotations__).replace("torch_rs", "torch"),
                str(expected_method.__annotations__),
            )

        error_pairs = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual([10, 20]), lambda: expected([10, 20])),
            (
                lambda: actual([10, 20], [1], None),
                lambda: expected([10, 20], [1], None),
            ),
            (
                lambda: actual(foo=[10, 20], indices=[1]),
                lambda: expected(foo=[10, 20], indices=[1]),
            ),
            (lambda: actual([10, 20], [1])[1], lambda: expected([10, 20], [1])[1]),
            (
                lambda: actual([10, 20], [1])[-2],
                lambda: expected([10, 20], [1])[-2],
            ),
            (
                lambda: actual([10, 20], [1])[0.0],
                lambda: expected([10, 20], [1])[0.0],
            ),
            (
                lambda: actual.__getitem__(actual([10], [0])),
                lambda: expected.__getitem__(expected([10], [0])),
            ),
            (
                lambda: actual.__getitems__(actual([10], [0]), [], []),
                lambda: expected.__getitems__(expected([10], [0]), [], []),
            ),
            (
                lambda: actual.__len__(actual([10], [0]), 1),
                lambda: expected.__len__(expected([10], [0]), 1),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(error_pairs):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()

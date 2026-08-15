import importlib
import inspect
import unittest
import warnings

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ConcatDatasetReferenceTests(unittest.TestCase):
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

    def make_dataset(self, module):
        direct_source = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        subset_source = module.tensor(
            [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]],
            requires_grad=True,
        )
        empty = module.utils.data.TensorDataset(module.zeros((0, 3)))
        direct = module.utils.data.TensorDataset(direct_source)
        subset = module.utils.data.Subset(
            module.utils.data.TensorDataset(subset_source), [2, 0]
        )
        children = [empty, direct, module.utils.data.Subset(direct, []), subset, empty]
        yielded = []

        def child_iterable():
            for index, child in enumerate(children):
                yielded.append(index)
                yield child

        dataset = module.utils.data.ConcatDataset(child_iterable())
        return dataset, children, yielded, direct_source, subset_source

    def make_list_dataset(self, module, values):
        class ListDataset(module.utils.data.Dataset):
            def __getitem__(self, index):
                return values[index]

            def __len__(self):
                return len(values)

        return ListDataset()

    def test_direct_and_chained_dataset_addition_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_left = torch.utils.data.TensorDataset(
            torch.tensor([[1.0], [2.0]])
        )
        expected_left = reference_torch.utils.data.TensorDataset(
            reference_torch.tensor([[1.0], [2.0]])
        )
        actual_right = torch.utils.data.Subset(
            torch.utils.data.TensorDataset(torch.tensor([[3.0], [4.0], [5.0]])),
            [2, 0],
        )
        expected_right = reference_torch.utils.data.Subset(
            reference_torch.utils.data.TensorDataset(
                reference_torch.tensor([[3.0], [4.0], [5.0]])
            ),
            [2, 0],
        )
        actual_tail = self.make_list_dataset(torch, ["tail"])
        expected_tail = self.make_list_dataset(reference_torch, ["tail"])

        actual_direct = actual_left + actual_right
        expected_direct = expected_left + expected_right
        self.assertEqual(
            type(actual_direct).__name__, type(expected_direct).__name__
        )
        self.assertIs(actual_direct.datasets[0], actual_left)
        self.assertIs(actual_direct.datasets[1], actual_right)
        self.assertIs(expected_direct.datasets[0], expected_left)
        self.assertIs(expected_direct.datasets[1], expected_right)
        self.assertEqual(
            actual_direct.cumulative_sizes, expected_direct.cumulative_sizes
        )
        for index in (0, 1, 2, 3, -1, -4):
            with self.subTest(stage="direct", index=index):
                np.testing.assert_array_equal(
                    np.asarray(actual_direct[index][0]),
                    expected_direct[index][0].cpu().numpy(),
                )

        actual_chained = actual_direct + actual_tail
        expected_chained = expected_direct + expected_tail
        self.assertEqual(
            type(actual_chained).__name__, type(expected_chained).__name__
        )
        self.assertIs(actual_chained.datasets[0], actual_direct)
        self.assertIs(actual_chained.datasets[1], actual_tail)
        self.assertIs(expected_chained.datasets[0], expected_direct)
        self.assertIs(expected_chained.datasets[1], expected_tail)
        self.assertEqual(
            actual_chained.cumulative_sizes, expected_chained.cumulative_sizes
        )
        for index in (0, 1, 2, 3, -2, -5):
            with self.subTest(stage="chained", index=index):
                np.testing.assert_array_equal(
                    np.asarray(actual_chained[index][0]),
                    expected_chained[index][0].cpu().numpy(),
                )
        self.assertEqual(actual_chained[-1], expected_chained[-1])
        self.assertEqual(
            type(actual_chained.datasets[0]).__name__,
            type(expected_chained.datasets[0]).__name__,
        )

        for actual_owner, expected_owner in (
            (actual_right, expected_right),
            (actual_tail, expected_tail),
        ):
            actual_combined = actual_owner + actual_left
            expected_combined = expected_owner + expected_left
            self.assertEqual(
                actual_combined.cumulative_sizes,
                expected_combined.cumulative_sizes,
            )
            self.assertIs(actual_combined.datasets[0], actual_owner)
            self.assertIs(expected_combined.datasets[0], expected_owner)

        error_pairs = (
            (lambda: actual_left + 3, lambda: expected_left + 3),
            (lambda: 3 + actual_left, lambda: 3 + expected_left),
            (lambda: actual_left.__add__(), lambda: expected_left.__add__()),
            (
                lambda: actual_left.__add__(actual_right, actual_tail),
                lambda: expected_left.__add__(expected_right, expected_tail),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(error_pairs):
            with self.subTest(stage="errors", case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_construction_indexing_views_and_autograd_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        (
            actual,
            actual_children,
            actual_yielded,
            actual_direct_source,
            actual_subset_source,
        ) = self.make_dataset(torch)
        (
            expected,
            expected_children,
            expected_yielded,
            expected_direct_source,
            expected_subset_source,
        ) = self.make_dataset(reference_torch)

        self.assertIs(type(actual.datasets), type(expected.datasets))
        self.assertEqual(actual_yielded, expected_yielded)
        self.assertEqual(len(actual.datasets), len(expected.datasets))
        for stored, child in zip(actual.datasets, actual_children):
            self.assertIs(stored, child)
        for stored, child in zip(expected.datasets, expected_children):
            self.assertIs(stored, child)
        self.assertEqual(actual.cumulative_sizes, expected.cumulative_sizes)
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(
            torch.utils.data.ConcatDataset.cumsum(actual_children),
            reference_torch.utils.data.ConcatDataset.cumsum(expected_children),
        )

        source_cases = (
            (0, actual_direct_source, expected_direct_source),
            (1, actual_direct_source, expected_direct_source),
            (2, actual_subset_source, expected_subset_source),
            (3, actual_subset_source, expected_subset_source),
            (-1, actual_subset_source, expected_subset_source),
            (-2, actual_subset_source, expected_subset_source),
            (-3, actual_direct_source, expected_direct_source),
            (-4, actual_direct_source, expected_direct_source),
            (np.int64(2), actual_subset_source, expected_subset_source),
        )
        for index, actual_source, expected_source in source_cases:
            with self.subTest(index=index):
                self.assert_tensor_matches(
                    actual[index][0],
                    expected[index][0],
                    actual_source,
                    expected_source,
                )

        (actual[1][0] * torch.tensor([2.0, 3.0, 5.0])).sum().backward()
        (
            expected[1][0] * reference_torch.tensor([2.0, 3.0, 5.0])
        ).sum().backward()
        (actual[-2][0] * torch.tensor([7.0, 11.0, 13.0])).sum().backward()
        (
            expected[-2][0] * reference_torch.tensor([7.0, 11.0, 13.0])
        ).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_direct_source.grad),
            expected_direct_source.grad.detach().cpu().numpy(),
        )
        np.testing.assert_array_equal(
            np.asarray(actual_subset_source.grad),
            expected_subset_source.grad.detach().cpu().numpy(),
        )

    def test_validation_and_index_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.utils.data.ConcatDataset([[], [10], []])
        expected = reference_torch.utils.data.ConcatDataset([[], [10], []])
        actual_empty = torch.utils.data.ConcatDataset([[], []])
        expected_empty = reference_torch.utils.data.ConcatDataset([[], []])

        for index in (0, -1, False, np.int64(0)):
            with self.subTest(index=index):
                self.assertEqual(actual[index], expected[index])

        error_pairs = (
            (
                lambda: torch.utils.data.ConcatDataset([]),
                lambda: reference_torch.utils.data.ConcatDataset([]),
            ),
            (
                lambda: torch.utils.data.ConcatDataset(iter(())),
                lambda: reference_torch.utils.data.ConcatDataset(iter(())),
            ),
            (
                lambda: torch.utils.data.ConcatDataset(None),
                lambda: reference_torch.utils.data.ConcatDataset(None),
            ),
            (
                lambda: torch.utils.data.ConcatDataset([object()]),
                lambda: reference_torch.utils.data.ConcatDataset([object()]),
            ),
            (lambda: actual[1], lambda: expected[1]),
            (lambda: actual[-2], lambda: expected[-2]),
            (lambda: actual_empty[0], lambda: expected_empty[0]),
            (lambda: actual_empty[-1], lambda: expected_empty[-1]),
            (lambda: actual[0.5], lambda: expected[0.5]),
            (lambda: actual[:], lambda: expected[:]),
            (
                lambda: torch.utils.data.ConcatDataset(),
                lambda: reference_torch.utils.data.ConcatDataset(),
            ),
            (
                lambda: torch.utils.data.ConcatDataset([], []),
                lambda: reference_torch.utils.data.ConcatDataset([], []),
            ),
            (
                lambda: torch.utils.data.ConcatDataset(foo=[]),
                lambda: reference_torch.utils.data.ConcatDataset(foo=[]),
            ),
            (
                lambda: actual.__getitem__(),
                lambda: expected.__getitem__(),
            ),
            (
                lambda: actual.__getitem__(0, 1),
                lambda: expected.__getitem__(0, 1),
            ),
            (lambda: actual.__len__(1), lambda: expected.__len__(1)),
            (
                lambda: torch.utils.data.ConcatDataset.cumsum(None),
                lambda: reference_torch.utils.data.ConcatDataset.cumsum(None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(error_pairs):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_imports_metadata_and_deprecated_alias_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_data = importlib.import_module("torch_rs.utils.data")
        expected_data = importlib.import_module("torch.utils.data")
        actual_module = importlib.import_module("torch_rs.utils.data.dataset")
        expected_module = importlib.import_module("torch.utils.data.dataset")
        actual = actual_data.ConcatDataset
        expected = expected_data.ConcatDataset
        supported = {"ConcatDataset", "Dataset", "Subset", "TensorDataset"}

        self.assertIs(actual, actual_module.ConcatDataset)
        self.assertIs(expected, expected_module.ConcatDataset)
        self.assertEqual(
            actual_data.__all__,
            [name for name in expected_data.__all__ if name in supported],
        )
        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name in supported],
        )
        for unsupported in ("DataLoader", "IterableDataset"):
            self.assertFalse(hasattr(actual_data, unsupported))
            self.assertFalse(hasattr(actual_module, unsupported))

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

        self.assertEqual(
            str(inspect.signature(actual_data.Dataset.__add__)).replace(
                "torch_rs", "torch"
            ),
            str(inspect.signature(expected_data.Dataset.__add__)),
        )
        self.assertEqual(
            actual_data.Dataset.__add__.__annotations__,
            expected_data.Dataset.__add__.__annotations__,
        )
        self.assertEqual(
            actual_data.Dataset.__add__.__doc__, expected_data.Dataset.__add__.__doc__
        )

        for actual_method, expected_method in (
            (actual.__init__, expected.__init__),
            (actual.cumsum, expected.cumsum),
            (actual.__getitem__, expected.__getitem__),
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

        actual_property = inspect.getattr_static(actual, "cummulative_sizes")
        expected_property = inspect.getattr_static(expected, "cummulative_sizes")
        self.assertIs(type(actual_property), type(expected_property))
        self.assertEqual(
            str(inspect.signature(actual_property.fget)),
            str(inspect.signature(expected_property.fget)),
        )
        self.assertEqual(actual_property.fget.__doc__, expected_property.fget.__doc__)
        self.assertEqual(
            actual_property.fget.__deprecated__,
            expected_property.fget.__deprecated__,
        )

        actual_dataset = actual([[1], [2, 3]])
        expected_dataset = expected([[1], [2, 3]])
        with warnings.catch_warnings(record=True) as actual_warnings:
            warnings.simplefilter("always")
            actual_sizes = actual_dataset.cummulative_sizes
        with warnings.catch_warnings(record=True) as expected_warnings:
            warnings.simplefilter("always")
            expected_sizes = expected_dataset.cummulative_sizes
        self.assertIs(actual_sizes, actual_dataset.cumulative_sizes)
        self.assertIs(expected_sizes, expected_dataset.cumulative_sizes)
        self.assertEqual(actual_sizes, expected_sizes)
        self.assertEqual(len(actual_warnings), len(expected_warnings))
        self.assertEqual(
            [warning.category for warning in actual_warnings],
            [warning.category for warning in expected_warnings],
        )
        self.assertEqual(
            [str(warning.message) for warning in actual_warnings],
            [str(warning.message) for warning in expected_warnings],
        )


if __name__ == "__main__":
    unittest.main()

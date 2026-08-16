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
class StackDatasetReferenceTests(unittest.TestCase):
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
            np.asarray(actual), expected.detach().cpu().numpy()
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

    def make_stacks(self, keyword):
        actual_sources = (
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            torch.tensor([10.0, 20.0, 30.0]),
        )
        expected_sources = (
            reference_torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            reference_torch.tensor([10.0, 20.0, 30.0]),
        )
        actual_children = tuple(
            torch.utils.data.TensorDataset(source) for source in actual_sources
        )
        expected_children = tuple(
            reference_torch.utils.data.TensorDataset(source)
            for source in expected_sources
        )
        if keyword:
            actual = torch.utils.data.StackDataset(
                feature=actual_children[0], target=actual_children[1]
            )
            expected = reference_torch.utils.data.StackDataset(
                feature=expected_children[0], target=expected_children[1]
            )
        else:
            actual = torch.utils.data.StackDataset(*actual_children)
            expected = reference_torch.utils.data.StackDataset(*expected_children)
        return actual, expected, actual_sources, expected_sources

    def test_positional_and_keyword_integer_indexing_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        for keyword in (False, True):
            actual, expected, actual_sources, expected_sources = self.make_stacks(
                keyword
            )
            with self.subTest(keyword=keyword):
                self.assertEqual(len(actual), len(expected))
                self.assertIs(type(actual.datasets), type(expected.datasets))

            for index in (0, 1, 2, -1, -3, np.int64(1)):
                with self.subTest(keyword=keyword, index=index):
                    actual_sample = actual[index]
                    expected_sample = expected[index]
                    self.assertIs(type(actual_sample), type(expected_sample))
                    if keyword:
                        self.assertEqual(list(actual_sample), list(expected_sample))
                        pairs = zip(actual_sample.values(), expected_sample.values())
                    else:
                        self.assertEqual(len(actual_sample), len(expected_sample))
                        pairs = zip(actual_sample, expected_sample)

                    for child_index, (actual_child, expected_child) in enumerate(pairs):
                        self.assertIs(type(actual_child), type(expected_child))
                        self.assertEqual(len(actual_child), len(expected_child))
                        self.assert_tensor_matches(
                            actual_child[0],
                            expected_child[0],
                            actual_sources[child_index],
                            expected_sources[child_index],
                        )

    def test_view_aliasing_and_autograd_lineage_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)
        actual = torch.utils.data.StackDataset(
            torch.utils.data.TensorDataset(actual_leaf)
        )
        expected = reference_torch.utils.data.StackDataset(
            reference_torch.utils.data.TensorDataset(expected_leaf)
        )

        actual_sample = actual[-2][0][0]
        expected_sample = expected[-2][0][0]
        self.assert_tensor_matches(
            actual_sample, expected_sample, actual_leaf, expected_leaf
        )

        actual_weights = torch.tensor([2.0, 3.0, 5.0])
        expected_weights = reference_torch.tensor([2.0, 3.0, 5.0])
        (actual_sample * actual_weights).sum().backward()
        (expected_sample * expected_weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), expected_leaf.grad.detach().cpu().numpy()
        )

    def test_empty_mixed_and_length_validation_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_empty_children = (
            torch.utils.data.TensorDataset(torch.zeros((0, 2))),
            torch.utils.data.TensorDataset(torch.zeros((0,))),
        )
        expected_empty_children = (
            reference_torch.utils.data.TensorDataset(reference_torch.zeros((0, 2))),
            reference_torch.utils.data.TensorDataset(reference_torch.zeros((0,))),
        )
        actual_empty = torch.utils.data.StackDataset(*actual_empty_children)
        expected_empty = reference_torch.utils.data.StackDataset(
            *expected_empty_children
        )
        self.assertEqual(len(actual_empty), len(expected_empty))
        for index in (0, -1):
            self.assert_error_matches(
                lambda index=index: actual_empty[index],
                lambda index=index: expected_empty[index],
            )

        cases = (
            (
                lambda: torch.utils.data.StackDataset(),
                lambda: reference_torch.utils.data.StackDataset(),
            ),
            (
                lambda: torch.utils.data.StackDataset([1], named=[1]),
                lambda: reference_torch.utils.data.StackDataset([1], named=[1]),
            ),
            (
                lambda: torch.utils.data.StackDataset([1], [2, 3]),
                lambda: reference_torch.utils.data.StackDataset([1], [2, 3]),
            ),
            (
                lambda: torch.utils.data.StackDataset(first=[1], second=[2, 3]),
                lambda: reference_torch.utils.data.StackDataset(
                    first=[1], second=[2, 3]
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_imports_exports_signatures_annotations_and_docs_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_data = importlib.import_module("torch_rs.utils.data")
        expected_data = importlib.import_module("torch.utils.data")
        actual_module = importlib.import_module("torch_rs.utils.data.dataset")
        expected_module = importlib.import_module("torch.utils.data.dataset")
        actual = actual_data.StackDataset
        expected = expected_data.StackDataset
        supported = {
            "ConcatDataset",
            "Dataset",
            "StackDataset",
            "Subset",
            "TensorDataset",
        }

        self.assertIs(actual, actual_module.StackDataset)
        self.assertIs(expected, expected_module.StackDataset)
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
        self.assertEqual(
            str(actual.__init__.__annotations__).replace("torch_rs", "torch"),
            str(expected.__init__.__annotations__),
        )
        self.assertTrue(issubclass(actual, torch.utils.data.Dataset))
        self.assertTrue(issubclass(expected, reference_torch.utils.data.Dataset))
        self.assertNotIn("__getitems__", actual.__dict__)
        self.assertIn("__getitems__", expected.__dict__)

        method_pairs = (
            (actual.__init__, expected.__init__),
            (actual.__getitem__, expected.__getitem__),
            (actual.__len__, expected.__len__),
        )
        for actual_method, expected_method in method_pairs:
            self.assertEqual(
                str(inspect.signature(actual_method)).replace("torch_rs", "torch"),
                str(inspect.signature(expected_method)),
            )
            self.assertEqual(actual_method.__doc__, expected_method.__doc__)

        actual_dataset = actual([10])
        expected_dataset = expected([10])
        error_pairs = (
            (
                lambda: actual_dataset.__getitem__(),
                lambda: expected_dataset.__getitem__(),
            ),
            (
                lambda: actual_dataset.__getitem__(0, 1),
                lambda: expected_dataset.__getitem__(0, 1),
            ),
            (
                lambda: actual_dataset.__len__(1),
                lambda: expected_dataset.__len__(1),
            ),
        )
        for actual_call, expected_call in error_pairs:
            self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()

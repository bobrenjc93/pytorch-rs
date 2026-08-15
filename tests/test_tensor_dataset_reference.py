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
class TensorDatasetReferenceTests(unittest.TestCase):
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
        self.assertIs(actual.dtype, torch.float32)
        self.assertEqual(actual.device, torch.device("cpu"))
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

    def test_values_layout_integer_indexing_and_aliasing_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
        actual_sources = (
            torch.tensor(values.tolist()).transpose(1, 2),
            torch.tensor([10.0, 20.0, 30.0]),
        )
        expected_sources = (
            reference_torch.tensor(values).transpose(1, 2),
            reference_torch.tensor([10.0, 20.0, 30.0]),
        )
        actual_dataset = torch.utils.data.TensorDataset(*actual_sources)
        expected_dataset = reference_torch.utils.data.TensorDataset(*expected_sources)

        self.assertEqual(len(actual_dataset), len(expected_dataset))
        for index in (0, 1, 2, -1, -3, np.int64(1)):
            with self.subTest(index=index):
                actual_sample = actual_dataset[index]
                expected_sample = expected_dataset[index]
                self.assertIs(type(actual_sample), type(expected_sample))
                self.assertEqual(len(actual_sample), len(expected_sample))
                for actual, expected, actual_source, expected_source in zip(
                    actual_sample,
                    expected_sample,
                    actual_sources,
                    expected_sources,
                ):
                    self.assert_tensor_matches(
                        actual, expected, actual_source, expected_source
                    )

    def test_autograd_lineage_matches_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)
        actual_sample = torch.utils.data.TensorDataset(actual_leaf)[-2][0]
        expected_sample = reference_torch.utils.data.TensorDataset(expected_leaf)[-2][0]

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

    def test_validation_empty_scalar_and_no_argument_cases_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_empty = torch.utils.data.TensorDataset(
            torch.zeros((0, 2)), torch.zeros((0,))
        )
        expected_empty = reference_torch.utils.data.TensorDataset(
            reference_torch.zeros((0, 2)), reference_torch.zeros((0,))
        )
        self.assertEqual(len(actual_empty), len(expected_empty))
        for index in (0, -1):
            self.assert_error_matches(
                lambda index=index: actual_empty[index],
                lambda index=index: expected_empty[index],
            )

        actual_no_arguments = torch.utils.data.TensorDataset()
        expected_no_arguments = reference_torch.utils.data.TensorDataset()
        self.assertEqual(actual_no_arguments.tensors, expected_no_arguments.tensors)
        for index in (0, -1, 100):
            self.assertEqual(actual_no_arguments[index], expected_no_arguments[index])
        self.assert_error_matches(
            lambda: len(actual_no_arguments), lambda: len(expected_no_arguments)
        )

        cases = (
            (
                lambda: torch.utils.data.TensorDataset(torch.tensor(1.0)),
                lambda: reference_torch.utils.data.TensorDataset(
                    reference_torch.tensor(1.0)
                ),
            ),
            (
                lambda: torch.utils.data.TensorDataset(
                    torch.zeros((2,)), torch.tensor(1.0)
                ),
                lambda: reference_torch.utils.data.TensorDataset(
                    reference_torch.zeros((2,)), reference_torch.tensor(1.0)
                ),
            ),
            (
                lambda: torch.utils.data.TensorDataset(
                    torch.zeros((2, 3)), torch.zeros((3,))
                ),
                lambda: reference_torch.utils.data.TensorDataset(
                    reference_torch.zeros((2, 3)), reference_torch.zeros((3,))
                ),
            ),
            (
                lambda: torch.utils.data.TensorDataset([1.0], [2.0]),
                lambda: reference_torch.utils.data.TensorDataset([1.0], [2.0]),
            ),
            (
                lambda: torch.utils.data.TensorDataset(
                    tensors=(torch.zeros((1,)),)
                ),
                lambda: reference_torch.utils.data.TensorDataset(
                    tensors=(reference_torch.zeros((1,)),)
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_imports_inheritance_signatures_docs_and_diagnostics_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_data = importlib.import_module("torch_rs.utils.data")
        expected_data = importlib.import_module("torch.utils.data")
        actual_module = importlib.import_module("torch_rs.utils.data.dataset")
        expected_module = importlib.import_module("torch.utils.data.dataset")

        for name in ("Dataset", "TensorDataset"):
            actual = getattr(actual_data, name)
            expected = getattr(expected_data, name)
            self.assertIs(actual, getattr(actual_module, name))
            self.assertIs(expected, getattr(expected_module, name))
            self.assertIn(name, actual_data.__all__)
            self.assertIn(name, expected_data.__all__)
            self.assertIn(name, actual_module.__all__)
            self.assertIn(name, expected_module.__all__)
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
            torch.utils.data.TensorDataset.__annotations__,
            {"tensors": tuple[torch.Tensor, ...]},
        )
        self.assertEqual(
            str(torch.utils.data.TensorDataset.__annotations__).replace(
                "torch_rs", "torch"
            ),
            str(reference_torch.utils.data.TensorDataset.__annotations__),
        )
        self.assertTrue(
            issubclass(torch.utils.data.TensorDataset, torch.utils.data.Dataset)
        )
        self.assertTrue(
            issubclass(
                reference_torch.utils.data.TensorDataset,
                reference_torch.utils.data.Dataset,
            )
        )

        actual_dataset = torch.utils.data.TensorDataset(torch.zeros((3, 2)))
        expected_dataset = reference_torch.utils.data.TensorDataset(
            reference_torch.zeros((3, 2))
        )
        for index in (3, -4, 1.5):
            with self.subTest(index=index):
                self.assert_error_matches(
                    lambda index=index: actual_dataset[index],
                    lambda index=index: expected_dataset[index],
                )
        self.assert_error_matches(
            lambda: torch.utils.data.Dataset()[0],
            lambda: reference_torch.utils.data.Dataset()[0],
        )

        method_pairs = (
            (
                torch.utils.data.TensorDataset.__init__,
                reference_torch.utils.data.TensorDataset.__init__,
            ),
            (
                torch.utils.data.TensorDataset.__getitem__,
                reference_torch.utils.data.TensorDataset.__getitem__,
            ),
            (
                torch.utils.data.TensorDataset.__len__,
                reference_torch.utils.data.TensorDataset.__len__,
            ),
        )
        for actual, expected in method_pairs:
            self.assertEqual(
                str(inspect.signature(actual)).replace("torch_rs", "torch"),
                str(inspect.signature(expected)),
            )
            self.assertEqual(actual.__doc__, expected.__doc__)


if __name__ == "__main__":
    unittest.main()

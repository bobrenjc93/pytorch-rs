import importlib
import inspect
import unittest

import numpy as np
import torch_rs as torch

from torch_rs.utils.data import Dataset, TensorDataset


class TensorDatasetTests(unittest.TestCase):
    def test_length_integer_indexing_layout_and_shared_storage(self):
        values = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
        features = torch.tensor(values.tolist()).transpose(1, 2)
        targets = torch.tensor([10.0, 20.0, 30.0])
        dataset = TensorDataset(features, targets)

        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.tensors, (features, targets))
        self.assertIs(dataset.tensors[0], features)
        self.assertIs(dataset.tensors[1], targets)

        for index, normalized in ((0, 0), (1, 1), (2, 2), (-1, 2), (-3, 0)):
            with self.subTest(index=index):
                sample = dataset[index]
                self.assertIs(type(sample), tuple)
                self.assertEqual(len(sample), 2)
                self.assertEqual(sample[0].shape, (4, 2))
                self.assertEqual(sample[0].stride(), (1, 4))
                self.assertEqual(sample[0].storage_offset(), normalized * 8)
                self.assertEqual(sample[1].shape, ())
                self.assertEqual(sample[1].storage_offset(), normalized)
                np.testing.assert_array_equal(
                    np.asarray(sample[0]), values.transpose(0, 2, 1)[normalized]
                )
                self.assertEqual(sample[1].item(), float((normalized + 1) * 10))

                for output, source in zip(sample, dataset.tensors):
                    pointer_delta = output.data_ptr() - source.data_ptr()
                    offset_delta = output.storage_offset() - source.storage_offset()
                    self.assertEqual(
                        pointer_delta, offset_delta * source.element_size()
                    )

    def test_integer_indexing_preserves_autograd_lineage(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            requires_grad=True,
        )
        sample = TensorDataset(leaf)[-2][0]

        self.assertTrue(sample.requires_grad)
        self.assertFalse(sample.is_leaf)
        self.assertEqual(sample.storage_offset(), 3)
        self.assertEqual(
            sample.data_ptr() - leaf.data_ptr(),
            sample.storage_offset() * leaf.element_size(),
        )

        weights = torch.tensor([2.0, 3.0, 5.0])
        (sample * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad),
            [[0.0, 0.0, 0.0], [2.0, 3.0, 5.0], [0.0, 0.0, 0.0]],
        )

    def test_empty_batches_scalars_no_arguments_and_validation(self):
        empty = TensorDataset(torch.zeros((0, 2)), torch.zeros((0,)))
        self.assertEqual(len(empty), 0)
        for index in (0, -1):
            with self.subTest(index=index):
                with self.assertRaisesRegex(
                    IndexError,
                    rf"^index {index} is out of bounds for dimension 0 with size 0$",
                ):
                    empty[index]

        no_arguments = TensorDataset()
        self.assertEqual(no_arguments.tensors, ())
        self.assertEqual(no_arguments[0], ())
        self.assertEqual(no_arguments[-1], ())
        with self.assertRaisesRegex(IndexError, "^tuple index out of range$"):
            len(no_arguments)

        with self.assertRaisesRegex(
            IndexError,
            "^Dimension specified as 0 but tensor has no dimensions$",
        ):
            TensorDataset(torch.tensor(1.0))
        with self.assertRaisesRegex(
            IndexError,
            "^Dimension specified as 0 but tensor has no dimensions$",
        ):
            TensorDataset(torch.zeros((2,)), torch.tensor(1.0))
        with self.assertRaisesRegex(
            AssertionError, "^Size mismatch between tensors$"
        ):
            TensorDataset(torch.zeros((2, 3)), torch.zeros((3,)))
        with self.assertRaisesRegex(
            AttributeError, "^'list' object has no attribute 'size'$"
        ):
            TensorDataset([1.0], [2.0])
        with self.assertRaisesRegex(
            TypeError,
            "^TensorDataset.__init__\\(\\) got an unexpected keyword argument 'tensors'$",
        ):
            TensorDataset(tensors=(torch.zeros((1,)),))

    def test_index_and_method_diagnostics(self):
        dataset = TensorDataset(torch.zeros((3, 2)))
        self.assertEqual(dataset.__getitem__(index=1)[0].shape, (2,))

        for index in (3, -4):
            with self.subTest(index=index):
                with self.assertRaisesRegex(
                    IndexError,
                    rf"^index {index} is out of bounds for dimension 0 with size 3$",
                ):
                    dataset[index]

        with self.assertRaisesRegex(
            IndexError,
            "^only integers, slices \\(`:`\\), ellipsis \\(`\\.\\.\\.`\\), "
            "None and long or byte Variables are valid indices \\(got float\\)$",
        ):
            dataset[1.5]
        with self.assertRaises(IndexError):
            dataset[1:]

        with self.assertRaisesRegex(
            TypeError,
            "^TensorDataset.__getitem__\\(\\) missing 1 required positional argument: "
            "'index'$",
        ):
            dataset.__getitem__()
        with self.assertRaisesRegex(
            TypeError,
            "^TensorDataset.__getitem__\\(\\) takes 2 positional arguments but 3 "
            "were given$",
        ):
            dataset.__getitem__(0, 1)
        with self.assertRaisesRegex(
            TypeError,
            "^TensorDataset.__len__\\(\\) takes 1 positional argument but 2 were given$",
        ):
            dataset.__len__(1)

    def test_dataset_inheritance_imports_exports_and_signatures(self):
        data_module = importlib.import_module("torch_rs.utils.data")
        dataset_module = importlib.import_module("torch_rs.utils.data.dataset")

        self.assertIs(torch.utils, importlib.import_module("torch_rs.utils"))
        self.assertIs(torch.utils.data, data_module)
        self.assertIs(data_module.Dataset, Dataset)
        self.assertIs(data_module.TensorDataset, TensorDataset)
        self.assertIs(dataset_module.Dataset, Dataset)
        self.assertIs(dataset_module.TensorDataset, TensorDataset)
        self.assertTrue(issubclass(TensorDataset, Dataset))
        self.assertIsInstance(TensorDataset(torch.zeros((1,))), Dataset)
        self.assertEqual(Dataset.__module__, "torch_rs.utils.data.dataset")
        self.assertEqual(TensorDataset.__module__, "torch_rs.utils.data.dataset")

        self.assertEqual(
            data_module.__all__,
            ["ConcatDataset", "Dataset", "StackDataset", "Subset", "TensorDataset"],
        )
        self.assertEqual(
            dataset_module.__all__,
            ["Dataset", "TensorDataset", "StackDataset", "ConcatDataset", "Subset"],
        )
        self.assertFalse(hasattr(data_module, "DataLoader"))
        self.assertNotIn("utils", torch.__all__)
        wildcard_namespace = {}
        exec("from torch_rs.utils.data import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["Dataset"], Dataset)
        self.assertIs(wildcard_namespace["TensorDataset"], TensorDataset)
        self.assertNotIn("DataLoader", wildcard_namespace)

        signature = inspect.signature(TensorDataset)
        self.assertEqual(tuple(signature.parameters), ("tensors",))
        tensors_parameter = signature.parameters["tensors"]
        self.assertIs(tensors_parameter.kind, inspect.Parameter.VAR_POSITIONAL)
        self.assertIs(tensors_parameter.annotation, torch.Tensor)
        self.assertIs(signature.return_annotation, None)
        self.assertEqual(
            TensorDataset.__annotations__,
            {"tensors": tuple[torch.Tensor, ...]},
        )
        self.assertEqual(str(inspect.signature(TensorDataset.__getitem__)), "(self, index)")
        self.assertEqual(str(inspect.signature(TensorDataset.__len__)), "(self) -> int")

        with self.assertRaisesRegex(
            NotImplementedError,
            "^Subclasses of Dataset should implement __getitem__\\.$",
        ):
            Dataset()[0]


if __name__ == "__main__":
    unittest.main()

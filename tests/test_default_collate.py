import importlib
import inspect
import unittest
from collections import OrderedDict, namedtuple

import numpy as np
import torch_rs as torch

from torch_rs.utils.data import default_collate


Point = namedtuple("Point", ["x", "y"])


class DefaultCollateTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected_values, *, shape, case):
        expected = np.asarray(expected_values, dtype=np.float32)
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(shape))
            self.assertEqual(actual.storage_offset(), 0)
            self.assertTrue(actual.is_contiguous())
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                expected.reshape(-1).view(np.uint32),
            )

    def test_tensor_batches_stack_along_outer_dimension(self):
        base = torch.tensor([[1.0, -0.0, 3.0], [4.0, 5.0, 6.0]])
        view = base.transpose(0, 1)
        result = default_collate([view[0], view[2]])
        self.assert_tensor_matches(
            result,
            [[1.0, 4.0], [3.0, 6.0]],
            shape=(2, 2),
            case="noncontiguous views",
        )
        self.assertNotEqual(result.data_ptr(), view.data_ptr())

        scalar_result = default_collate((torch.tensor(2.0), torch.tensor(-0.0)))
        self.assert_tensor_matches(
            scalar_result,
            [2.0, -0.0],
            shape=(2,),
            case="scalars",
        )

        empty_result = default_collate(
            [torch.zeros((2, 0, 3)), torch.zeros((2, 0, 3))]
        )
        self.assert_tensor_matches(
            empty_result,
            np.empty((2, 2, 0, 3), dtype=np.float32),
            shape=(2, 2, 0, 3),
            case="empty tensors",
        )

    def test_nested_containers_preserve_type_and_key_order(self):
        batch = [
            OrderedDict(
                [
                    (
                        "z",
                        [
                            torch.tensor([1.0, 2.0]),
                            (
                                torch.tensor(3.0),
                                Point(torch.tensor([4.0]), torch.tensor([5.0])),
                            ),
                        ],
                    ),
                    ("a", (torch.tensor([6.0]),)),
                ]
            ),
            OrderedDict(
                [
                    (
                        "z",
                        [
                            torch.tensor([10.0, 20.0]),
                            (
                                torch.tensor(30.0),
                                Point(torch.tensor([40.0]), torch.tensor([50.0])),
                            ),
                        ],
                    ),
                    ("a", (torch.tensor([60.0]),)),
                ]
            ),
        ]

        result = default_collate(batch)

        self.assertIs(type(result), OrderedDict)
        self.assertEqual(list(result), ["z", "a"])
        self.assertIs(type(result["z"]), list)
        self.assertIs(type(result["z"][1]), list)
        self.assertIs(type(result["z"][1][1]), Point)
        self.assertIs(type(result["a"]), list)
        self.assert_tensor_matches(
            result["z"][0],
            [[1.0, 2.0], [10.0, 20.0]],
            shape=(2, 2),
            case="nested list leaf",
        )
        self.assert_tensor_matches(
            result["z"][1][0],
            [3.0, 30.0],
            shape=(2,),
            case="nested plain tuple scalar leaf",
        )
        self.assert_tensor_matches(
            result["z"][1][1].x,
            [[4.0], [40.0]],
            shape=(2, 1),
            case="namedtuple x",
        )
        self.assert_tensor_matches(
            result["z"][1][1].y,
            [[5.0], [50.0]],
            shape=(2, 1),
            case="namedtuple y",
        )
        self.assert_tensor_matches(
            result["a"][0],
            [[6.0], [60.0]],
            shape=(2, 1),
            case="dict plain tuple leaf",
        )

        tuple_batch = [
            (torch.tensor([1.0]), torch.tensor([2.0])),
            (torch.tensor([3.0]), torch.tensor([4.0])),
        ]
        tuple_result = default_collate(tuple_batch)
        self.assertIs(type(tuple_result), list)
        self.assertEqual(len(tuple_result), 2)
        self.assert_tensor_matches(
            tuple_result[0],
            [[1.0], [3.0]],
            shape=(2, 1),
            case="plain tuple first field",
        )
        self.assert_tensor_matches(
            tuple_result[1],
            [[2.0], [4.0]],
            shape=(2, 1),
            case="plain tuple second field",
        )

    def test_unsupported_leaves_and_mismatched_structures_fail_closed(self):
        unsupported_batches = (
            [np.asarray([1.0], dtype=np.float32)],
            [np.float32(1.0)],
            [1.0],
            [1],
            ["sample"],
            [b"sample"],
            [object()],
            [None],
        )
        for batch in unsupported_batches:
            with self.subTest(batch_type=type(batch[0]).__name__):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^default_collate\(\): only batches of exact native CPU "
                    r"float32 tensors",
                ):
                    default_collate(batch)

        tensor = torch.tensor([1.0])
        structure_errors = (
            (
                lambda: default_collate([[tensor], [tensor, tensor]]),
                RuntimeError,
                "^each element in list of batch should be of equal size$",
            ),
            (
                lambda: default_collate([(tensor,), [tensor]]),
                TypeError,
                "^default_collate\\(\\): each sequence batch element must have "
                "the same container type$",
            ),
            (
                lambda: default_collate([{"a": tensor}, {"a": tensor, "b": tensor}]),
                RuntimeError,
                "^each element in dict batch should have the same keys$",
            ),
        )
        for call, error_type, regex in structure_errors:
            with self.subTest(call=call):
                with self.assertRaisesRegex(error_type, regex):
                    call()

        with self.assertRaises(IndexError):
            default_collate([])

    def test_imports_exports_signature_and_unsupported_neighbors(self):
        data_module = importlib.import_module("torch_rs.utils.data")
        collate_module = importlib.import_module("torch_rs.utils.data._utils.collate")
        utils_package = importlib.import_module("torch_rs.utils.data._utils")

        self.assertIs(torch.utils.data.default_collate, default_collate)
        self.assertIs(data_module.default_collate, default_collate)
        self.assertIs(collate_module.default_collate, default_collate)
        self.assertIs(utils_package.collate, collate_module)
        self.assertEqual(default_collate.__module__, "torch_rs.utils.data._utils.collate")
        self.assertEqual(default_collate.__name__, "default_collate")
        self.assertEqual(default_collate.__qualname__, "default_collate")
        self.assertEqual(str(inspect.signature(default_collate)), "(batch)")
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
                "default_collate",
                "get_worker_info",
            ],
        )

        wildcard_namespace = {}
        exec("from torch_rs.utils.data import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["default_collate"], default_collate)

        for unsupported in (
            "DataLoader",
            "RandomSampler",
            "SubsetRandomSampler",
            "WeightedRandomSampler",
        ):
            with self.subTest(unsupported=unsupported):
                self.assertFalse(hasattr(data_module, unsupported))
                self.assertNotIn(unsupported, wildcard_namespace)


if __name__ == "__main__":
    unittest.main()

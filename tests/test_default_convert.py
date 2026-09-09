import importlib
import inspect
import unittest
from collections import OrderedDict, namedtuple

import numpy as np
import torch_rs as torch

from torch_rs.utils.data import default_collate, default_convert


Point = namedtuple("Point", ["x", "y"])


class DefaultConvertTests(unittest.TestCase):
    def test_exact_python_scalar_leaves_are_preserved(self):
        for value in (True, False, 1, -(2**200), -0.0, float("nan"), 1 + 2j, None):
            with self.subTest(value=value):
                self.assertIs(default_convert(value), value)
                result = default_convert({"value": [Point(value, (value,))]})
                self.assertIs(result["value"][0].x, value)
                self.assertIs(result["value"][0].y[0], value)

    def test_tensor_leaves_are_preserved_without_batching(self):
        base = torch.tensor([[1.0, -0.0, 3.0], [4.0, 5.0, 6.0]])
        first = base[0]
        second = base[1]

        result = default_convert([first, second])

        self.assertIs(type(result), list)
        self.assertIs(result[0], first)
        self.assertIs(result[1], second)
        self.assertNotIsInstance(result, torch.Tensor)

        tuple_result = default_convert((first, second))
        self.assertIs(type(tuple_result), list)
        self.assertIs(tuple_result[0], first)
        self.assertIs(tuple_result[1], second)

    def test_nested_containers_preserve_supported_structure_and_leaves(self):
        label = "sample"
        payload = b"payload"
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor(3.0)
        z = torch.tensor([4.0])
        data = OrderedDict(
            [
                ("z", [x, (y, Point(z, label))]),
                ("a", {"payload": payload}),
            ]
        )

        result = default_convert(data)

        self.assertIs(type(result), OrderedDict)
        self.assertIsNot(result, data)
        self.assertEqual(list(result), ["z", "a"])
        self.assertIs(type(result["z"]), list)
        self.assertIs(type(result["z"][1]), list)
        self.assertIs(type(result["z"][1][1]), Point)
        self.assertIs(type(result["a"]), dict)
        self.assertIs(result["z"][0], x)
        self.assertIs(result["z"][1][0], y)
        self.assertIs(result["z"][1][1].x, z)
        self.assertIs(result["z"][1][1].y, label)
        self.assertIs(result["a"]["payload"], payload)

    def test_empty_containers_are_supported(self):
        self.assertEqual(default_convert([]), [])
        self.assertEqual(default_convert(()), [])
        self.assertEqual(default_convert({}), {})
        self.assertEqual(default_convert(Point([], ())), Point([], []))

    def test_unsupported_leaves_fail_closed(self):
        class IntSubclass(int):
            pass

        class FloatSubclass(float):
            pass

        class ComplexSubclass(complex):
            pass

        unsupported_values = (
            np.asarray([1.0], dtype=np.float32),
            np.float32(1.0),
            np.int64(1),
            np.bool_(True),
            np.float64(1.0),
            np.complex128(1 + 2j),
            IntSubclass(1),
            FloatSubclass(1.0),
            ComplexSubclass(1 + 2j),
            object(),
            {"ok": torch.tensor([1.0]), "bad": object()},
            [torch.tensor([1.0]), object()],
        )
        for value in unsupported_values:
            for data in (value, {"nested": [Point(None, (value,))]}):
                with self.subTest(
                    value_type=type(value).__name__, nested=data is not value
                ):
                    with self.assertRaisesRegex(
                        TypeError,
                        r"^default_convert\(\): only exact native tensors, "
                        r"exact Python bool, int, float, complex, None, "
                        r"strings, bytes, and list, tuple, namedtuple, or dict "
                        r"containers are supported; found ",
                    ):
                        default_convert(data)

    def test_default_collate_remains_separate_and_fail_closed(self):
        left = torch.tensor([1.0])
        right = torch.tensor([2.0])
        converted = default_convert([left, right])
        collated = default_collate([left, right])

        self.assertIs(type(converted), list)
        self.assertIs(converted[0], left)
        self.assertIs(converted[1], right)
        self.assertEqual(collated.shape, (2, 1))

        for value in (True, 1, 1.0, 1 + 2j, None):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^default_collate\(\): only batches of exact native CPU float32 "
                    r"tensors",
                ):
                    default_collate([value, value])

    def test_imports_exports_and_signature(self):
        data_module = importlib.import_module("torch_rs.utils.data")
        collate_module = importlib.import_module("torch_rs.utils.data._utils.collate")
        utils_package = importlib.import_module("torch_rs.utils.data._utils")

        self.assertIs(torch.utils.data.default_convert, default_convert)
        self.assertIs(data_module.default_convert, default_convert)
        self.assertIs(collate_module.default_convert, default_convert)
        self.assertIs(utils_package.collate, collate_module)
        self.assertEqual(default_convert.__module__, "torch_rs.utils.data._utils.collate")
        self.assertEqual(default_convert.__name__, "default_convert")
        self.assertEqual(default_convert.__qualname__, "default_convert")
        self.assertEqual(str(inspect.signature(default_convert)), "(data)")
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
                "default_convert",
                "get_worker_info",
            ],
        )

        wildcard_namespace = {}
        exec("from torch_rs.utils.data import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["default_convert"], default_convert)

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

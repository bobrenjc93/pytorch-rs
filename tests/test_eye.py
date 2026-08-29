import re
import sys
import unittest

import numpy as np
import torch_rs as torch


class EyeTests(unittest.TestCase):
    def tensor_observation(self, tensor):
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "numel": tensor.numel(),
            "values": tensor.tolist(),
            "dtype": str(tensor.dtype),
            "dtype_identity": tensor.dtype is torch.float32,
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "grad_is_none": tensor.grad is None,
        }

    def test_square_rectangular_and_zero_size_results(self):
        cases = (
            (
                lambda: torch.eye(3),
                {
                    "shape": (3, 3),
                    "stride": (3, 1),
                    "storage_offset": 0,
                    "numel": 9,
                    "values": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "dtype": "torch.float32",
                    "dtype_identity": True,
                    "device": "cpu",
                    "layout": "torch.strided",
                    "requires_grad": False,
                    "is_leaf": True,
                    "grad_is_none": True,
                },
            ),
            (
                lambda: torch.eye(2, 4),
                {
                    "shape": (2, 4),
                    "stride": (4, 1),
                    "storage_offset": 0,
                    "numel": 8,
                    "values": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ],
                    "dtype": "torch.float32",
                    "dtype_identity": True,
                    "device": "cpu",
                    "layout": "torch.strided",
                    "requires_grad": False,
                    "is_leaf": True,
                    "grad_is_none": True,
                },
            ),
            (
                lambda: torch.eye(n=4, m=2),
                {
                    "shape": (4, 2),
                    "stride": (2, 1),
                    "storage_offset": 0,
                    "numel": 8,
                    "values": [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ],
                    "dtype": "torch.float32",
                    "dtype_identity": True,
                    "device": "cpu",
                    "layout": "torch.strided",
                    "requires_grad": False,
                    "is_leaf": True,
                    "grad_is_none": True,
                },
            ),
            (
                lambda: torch.eye(n=0),
                {
                    "shape": (0, 0),
                    "stride": (1, 1),
                    "storage_offset": 0,
                    "numel": 0,
                    "values": [],
                    "dtype": "torch.float32",
                    "dtype_identity": True,
                    "device": "cpu",
                    "layout": "torch.strided",
                    "requires_grad": False,
                    "is_leaf": True,
                    "grad_is_none": True,
                },
            ),
            (
                lambda: torch.eye(3, 0),
                {
                    "shape": (3, 0),
                    "stride": (1, 1),
                    "storage_offset": 0,
                    "numel": 0,
                    "values": [[], [], []],
                    "dtype": "torch.float32",
                    "dtype_identity": True,
                    "device": "cpu",
                    "layout": "torch.strided",
                    "requires_grad": False,
                    "is_leaf": True,
                    "grad_is_none": True,
                },
            ),
            (
                lambda: torch.eye(0, 3),
                {
                    "shape": (0, 3),
                    "stride": (3, 1),
                    "storage_offset": 0,
                    "numel": 0,
                    "values": [],
                    "dtype": "torch.float32",
                    "dtype_identity": True,
                    "device": "cpu",
                    "layout": "torch.strided",
                    "requires_grad": False,
                    "is_leaf": True,
                    "grad_is_none": True,
                },
            ),
        )
        for create, expected in cases:
            with self.subTest(shape=expected["shape"]):
                self.assertEqual(self.tensor_observation(create()), expected)

    def test_supported_metadata_creates_cpu_float32_leaves(self):
        cases = (
            {"dtype": torch.float32},
            {"dtype": torch.float},
            {"device": "cpu"},
            {"device": "cpu:0"},
            {"device": torch.device("cpu", 2)},
            {"dtype": torch.float32, "device": torch.device("cpu")},
            {"requires_grad": None},
            {"requires_grad": False},
            {"requires_grad": True},
            {
                "dtype": torch.float,
                "device": torch.device("cpu"),
                "requires_grad": True,
            },
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with torch.no_grad():
                    tensor = torch.eye(2, 3, **kwargs)
                self.assertEqual(tensor.shape, (2, 3))
                self.assertEqual(tensor.stride(), (3, 1))
                self.assertEqual(
                    tensor.tolist(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                )
                self.assertIs(tensor.dtype, torch.float32)
                self.assertEqual(tensor.device, torch.device("cpu"))
                self.assertEqual(
                    tensor.requires_grad, kwargs.get("requires_grad") is True
                )
                self.assertTrue(tensor.is_leaf)
                self.assertIsNone(tensor.grad)

    def test_integer_protocol_inputs(self):
        class IntSubclass(int):
            pass

        class IndexDimension:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        class IntOnly:
            def __int__(self):
                return 2

        rows = IndexDimension(2)
        columns = IndexDimension(3)
        tensor = torch.eye(rows, columns)
        self.assertEqual(tensor.shape, (2, 3))
        self.assertEqual(tensor.tolist(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertEqual((rows.calls, columns.calls), (1, 1))

        self.assertEqual(torch.eye(IntSubclass(2)).shape, (2, 2))
        self.assertEqual(torch.eye(np.int64(2), np.uint32(3)).shape, (2, 3))

        for dimensions in (
            (True,),
            (2, False),
            (np.bool_(True),),
            (IntOnly(),),
            (2.0,),
        ):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(TypeError):
                    torch.eye(*dimensions)

    def test_negative_and_overflow_errors(self):
        with self.assertRaisesRegex(
            RuntimeError, re.escape("n must be greater or equal to 0, got -1")
        ):
            torch.eye(-1)
        with self.assertRaisesRegex(
            RuntimeError, re.escape("m must be greater or equal to 0, got -2")
        ):
            torch.eye(1, -2)

        for dimensions in (
            (2**63,),
            (-(2**63) - 1,),
            (1, 2**63),
            (1, -(2**63) - 1),
            (np.uint64(2**63),),
        ):
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(
                    ValueError, "^Overflow when unpacking long long$"
                ):
                    torch.eye(*dimensions)

        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("numel: integer multiplication overflow"),
        ):
            torch.eye(sys.maxsize, 3)

        oversized = sys.maxsize // 4 + 1
        with self.assertRaisesRegex(RuntimeError, "exceeds the platform capacity"):
            torch.eye(oversized, 1)

        no_rows = torch.eye(0, sys.maxsize)
        self.assertEqual(no_rows.shape, (0, sys.maxsize))
        self.assertEqual(no_rows.stride(), (sys.maxsize, 1))
        self.assertEqual(no_rows.numel(), 0)

        no_columns = torch.eye(sys.maxsize, 0)
        self.assertEqual(no_columns.shape, (sys.maxsize, 0))
        self.assertEqual(no_columns.stride(), (1, 1))
        self.assertEqual(no_columns.numel(), 0)


if __name__ == "__main__":
    unittest.main()

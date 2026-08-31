import copy
import pickle
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
            "layout_identity": tensor.layout is torch.strided,
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
                    "layout_identity": True,
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
                    "layout_identity": True,
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
                    "layout_identity": True,
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
                    "layout_identity": True,
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
                    "layout_identity": True,
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
                    "layout_identity": True,
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
            {"out": None},
            {"dtype": torch.float32},
            {"dtype": torch.float},
            {"layout": None},
            {"layout": torch.strided},
            {"device": "cpu"},
            {"device": "cpu:0"},
            {"device": torch.device("cpu", 2)},
            {"pin_memory": None},
            {"pin_memory": False},
            {"dtype": torch.float32, "device": torch.device("cpu")},
            {"requires_grad": None},
            {"requires_grad": False},
            {"requires_grad": True},
            {
                "out": None,
                "dtype": torch.float,
                "layout": torch.strided,
                "device": torch.device("cpu"),
                "pin_memory": False,
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
                self.assertIs(tensor.layout, torch.strided)
                self.assertEqual(
                    tensor.requires_grad, kwargs.get("requires_grad") is True
                )
                self.assertTrue(tensor.is_leaf)
                self.assertIsNone(tensor.grad)

    def test_out_none_and_strided_layout_keep_fresh_storage(self):
        cases = (
            lambda: torch.eye(2, out=None),
            lambda: torch.eye(2, 3, layout=None),
            lambda: torch.eye(2, 3, layout=torch.strided),
            lambda: torch.eye(0, out=None, layout=torch.strided),
            lambda: torch.eye(3, 0, out=None, layout=torch.strided),
            lambda: torch.eye(0, 3, out=None, layout=torch.strided),
        )
        for create in cases:
            first = create()
            second = create()
            with self.subTest(shape=first.shape):
                self.assertFalse(first.is_set_to(second))
                if first.numel() != 0:
                    self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_unsupported_output_layout_and_pinned_memory_options(self):
        destination = torch.zeros((1, 1))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^eye\(\): the 'out' argument is not supported$",
        ):
            torch.eye(1, out=destination)
        self.assertEqual(destination.tolist(), [[0.0]])

        with self.assertRaisesRegex(
            TypeError,
            r"^eye\(\): argument 'out' must be Tensor, not list$",
        ):
            torch.eye(1, out=[])

        with self.assertRaisesRegex(
            TypeError,
            r"^eye\(\): argument 'layout' must be torch\.layout, not object$",
        ):
            torch.eye(1, layout=object())

        with self.assertRaisesRegex(
            RuntimeError,
            r"^eye\(\): pin_memory=True is not supported; only unpinned CPU storage is implemented$",
        ):
            torch.eye(1, pin_memory=True)

        with self.assertRaisesRegex(
            TypeError,
            r"^eye\(\): argument 'pin_memory' must be bool, not int$",
        ):
            torch.eye(1, pin_memory=1)

    def test_callable_import_wildcard_copy_and_pickle_contract(self):
        function = torch.eye
        import_namespace = {}
        wildcard_namespace = {}
        exec("from torch_rs import eye as imported_eye", import_namespace)
        exec("from torch_rs import *", wildcard_namespace)

        self.assertTrue(callable(function))
        self.assertEqual(type(function).__name__, "builtin_function_or_method")
        self.assertEqual(function.__name__, "eye")
        self.assertEqual(torch.__all__.count("eye"), 1)
        self.assertIs(import_namespace["imported_eye"], function)
        self.assertIs(wildcard_namespace["eye"], function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(function, protocol)), function)

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

        for dimensions in ((sys.maxsize, 2), (sys.maxsize, 3)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(
                    RuntimeError,
                    re.escape("numel: integer multiplication overflow"),
                ):
                    torch.eye(*dimensions)

        for dimensions in ((sys.maxsize, 1), (sys.maxsize // 4 + 1, 1)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(
                    RuntimeError,
                    re.escape(
                        "Storage size calculation overflowed with "
                        f"sizes={list(dimensions)}"
                    ),
                ):
                    torch.eye(*dimensions)

        no_rows = torch.eye(0, sys.maxsize)
        self.assertEqual(no_rows.shape, (0, sys.maxsize))
        self.assertEqual(no_rows.stride(), (sys.maxsize, 1))
        self.assertEqual(no_rows.numel(), 0)

        no_columns = torch.eye(sys.maxsize, 0)
        self.assertEqual(no_columns.shape, (sys.maxsize, 0))
        self.assertEqual(no_columns.stride(), (1, 1))
        self.assertEqual(no_columns.numel(), 0)

    def test_new_keyword_error_ordering(self):
        invalid_type_cases = (
            lambda: torch.eye(2**63, out=[]),
            lambda: torch.eye(2**63, layout=object()),
            lambda: torch.eye(2**63, pin_memory=1),
        )
        for call in invalid_type_cases:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

        destination = torch.zeros((1, 1))
        for call in (
            lambda: torch.eye(2**63, out=None),
            lambda: torch.eye(2**63, layout=torch.strided),
            lambda: torch.eye(2**63, pin_memory=True),
            lambda: torch.eye(2**63, out=destination),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    ValueError, "^Overflow when unpacking long long$"
                ):
                    call()

        for call in (
            lambda: torch.eye(-1, out=None),
            lambda: torch.eye(-1, layout=torch.strided),
            lambda: torch.eye(-1, pin_memory=True),
            lambda: torch.eye(-1, out=destination),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    RuntimeError,
                    re.escape("n must be greater or equal to 0, got -1"),
                ):
                    call()

        overflow_cases = (
            (
                lambda: torch.eye(sys.maxsize, 3, out=destination),
                "numel: integer multiplication overflow",
            ),
            (
                lambda: torch.eye(sys.maxsize, 3, pin_memory=True),
                "numel: integer multiplication overflow",
            ),
            (
                lambda: torch.eye(sys.maxsize, 1, out=destination),
                "Storage size calculation overflowed with "
                f"sizes={[sys.maxsize, 1]}",
            ),
            (
                lambda: torch.eye(sys.maxsize // 4 + 1, 1, pin_memory=True),
                "Storage size calculation overflowed with "
                f"sizes={[sys.maxsize // 4 + 1, 1]}",
            ),
        )
        for call, message in overflow_cases:
            with self.subTest(call=call):
                with self.assertRaisesRegex(RuntimeError, re.escape(message)):
                    call()


if __name__ == "__main__":
    unittest.main()

import copy
import inspect
import pickle
import re
import sys
import types
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
            {"out": None},
            {"dtype": torch.float32},
            {"dtype": torch.float},
            {"layout": None},
            {"layout": torch.strided},
            {"device": "cpu"},
            {"device": "cpu:0"},
            {"device": torch.device("cpu", 2)},
            {"dtype": torch.float32, "device": torch.device("cpu")},
            {"pin_memory": None},
            {"pin_memory": False},
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
                self.assertEqual(
                    tensor.requires_grad, kwargs.get("requires_grad") is True
                )
                self.assertTrue(tensor.is_leaf)
                self.assertIsNone(tensor.grad)

    def test_default_out_and_layout_keep_fresh_allocation_behavior(self):
        cases = (
            ("square", lambda keywords: torch.eye(3, **keywords)),
            ("wide", lambda keywords: torch.eye(2, 4, **keywords)),
            ("tall", lambda keywords: torch.eye(n=4, m=2, **keywords)),
            ("empty square", lambda keywords: torch.eye(0, **keywords)),
            ("empty rows", lambda keywords: torch.eye(0, 3, **keywords)),
            ("empty columns", lambda keywords: torch.eye(3, 0, **keywords)),
            (
                "requires grad",
                lambda keywords: torch.eye(2, requires_grad=True, **keywords),
            ),
        )
        option_sets = (
            {"out": None},
            {"layout": None},
            {"layout": torch.strided},
            {"out": None, "layout": None},
            {"out": None, "layout": torch.strided},
        )

        for case, factory in cases:
            for options in option_sets:
                with self.subTest(case=case, options=options):
                    baseline = factory({})
                    explicit = factory(options)
                    self.assertEqual(
                        self.tensor_observation(explicit),
                        self.tensor_observation(baseline),
                    )
                    self.assertFalse(explicit.is_set_to(baseline))

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

    def test_concrete_out_and_nondefault_options_remain_unsupported(self):
        destination = torch.full((2, 2), 9.0)
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("eye(): the 'out' argument is not supported"),
        ):
            torch.eye(2, out=destination)
        self.assertEqual(destination.tolist(), [[9.0, 9.0], [9.0, 9.0]])

        for call, error_type, message in (
            (
                lambda: torch.eye(1, out=object()),
                TypeError,
                "eye(): argument 'out' must be Tensor, not object",
            ),
            (
                lambda: torch.eye(1, layout=object()),
                TypeError,
                "eye(): argument 'layout' must be torch.layout, not object",
            ),
            (
                lambda: torch.eye(1, device="meta"),
                RuntimeError,
                "eye(): device 'meta' is not supported; only 'cpu' is implemented",
            ),
            (
                lambda: torch.eye(1, device="cuda"),
                RuntimeError,
                "eye(): device 'cuda' is not supported; only 'cpu' is implemented",
            ),
            (
                lambda: torch.eye(1, pin_memory=True),
                RuntimeError,
                "eye(): pin_memory=True is not supported; only unpinned CPU storage is implemented",
            ),
            (
                lambda: torch.eye(1, pin_memory=0),
                TypeError,
                "eye(): argument 'pin_memory' must be bool, not int",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_callable_metadata_and_exports_use_variable_function_owner(self):
        function = torch.eye
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)

        with self.assertRaises(ValueError):
            inspect.signature(function)

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "eye")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.eye")
        self.assertEqual(function.__module__, "torch")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.eye, function)
        self.assertIs(wildcard_namespace["eye"], function)
        self.assertEqual(torch.__all__.count("eye"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )


if __name__ == "__main__":
    unittest.main()

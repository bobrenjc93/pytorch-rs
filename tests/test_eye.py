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

    def test_default_out_and_layout_keywords_preserve_allocation_behavior(self):
        cases = (
            ((3,), {"out": None}),
            ((2, 4), {"layout": None}),
            ((4, 2), {"layout": torch.strided}),
            ((0,), {"out": None, "layout": torch.strided}),
            ((3, 0), {"out": None, "layout": None}),
            ((0, 3), {"out": None, "layout": torch.strided}),
        )
        for arguments, kwargs in cases:
            with self.subTest(arguments=arguments, kwargs=kwargs):
                actual = torch.eye(*arguments, **kwargs)
                expected = torch.eye(*arguments)
                self.assertEqual(
                    self.tensor_observation(actual),
                    self.tensor_observation(expected),
                )

        first = torch.eye(2, out=None, layout=torch.strided)
        second = torch.eye(2, out=None, layout=torch.strided)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_callable_exports_and_pickling_match_generated_builtin_shape(self):
        function = torch.eye
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)

        self.assertTrue(callable(function))
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "eye")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.eye")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method eye of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.eye, function)
        self.assertEqual(torch.__all__.count("eye"), 1)
        self.assertIs(wildcard_namespace["eye"], function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(function, protocol)), function)

    def test_unsupported_output_layout_and_pin_memory_options(self):
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("eye(): the 'out' argument is not supported"),
        ):
            torch.eye(1, out=torch.zeros((1, 1)))
        with self.assertRaisesRegex(
            TypeError,
            re.escape("eye(): argument 'out' must be Tensor, not object"),
        ):
            torch.eye(1, out=object())

        for layout, type_name in ((object(), "object"), (torch.float32, "torch.dtype")):
            with self.subTest(argument="layout", value=layout):
                with self.assertRaisesRegex(
                    TypeError,
                    re.escape(
                        "eye(): argument 'layout' must be torch.layout, "
                        f"not {type_name}"
                    ),
                ):
                    torch.eye(1, layout=layout)

        for value in (None, False, True):
            with self.subTest(argument="pin_memory", value=value):
                with self.assertRaisesRegex(
                    TypeError,
                    re.escape("eye() got an unexpected keyword argument 'pin_memory'"),
                ):
                    torch.eye(1, pin_memory=value)

    def test_supported_option_type_errors_precede_dimension_conversion(self):
        invalid_metadata = (
            lambda: torch.eye(2**63, out=object(), requires_grad=1),
            lambda: torch.eye(2**63, layout=object(), requires_grad=1),
            lambda: torch.eye(2**63, dtype=object(), requires_grad=1),
            lambda: torch.eye(2**63, device=object(), requires_grad=1),
        )
        for call in invalid_metadata:
            with self.subTest(call=call):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertNotIn("argument 'requires_grad'", str(raised.exception))

        out = torch.eye(1)
        for call, message in (
            (
                lambda: torch.eye(2**63, out=out),
                "Overflow when unpacking long long",
            ),
            (
                lambda: torch.eye(-1, out=out),
                "n must be greater or equal to 0, got -1",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex((RuntimeError, ValueError), re.escape(message)):
                    call()

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


if __name__ == "__main__":
    unittest.main()

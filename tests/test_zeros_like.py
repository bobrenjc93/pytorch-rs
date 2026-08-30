import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC_PREFIX = (
    "\nzeros_like(input, *, dtype=None, layout=None, device=None, "
    "requires_grad=False, memory_format=None) -> Tensor\n\n"
)

UNSUPPORTED_INPUT = (
    "zeros_like(): only exact native CPU float32 row-major contiguous Tensor "
    "inputs are supported"
)


class ZerosLikeTests(unittest.TestCase):
    def tensor_contract(self, tensor):
        return (
            np.asarray(tensor).reshape(-1).view(np.uint32).tolist(),
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.dtype,
            tensor.device,
            tensor.layout,
            tensor.requires_grad,
            tensor.is_leaf,
        )

    def assert_error(self, call, error_type, message):
        with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
            call()

    def contiguous_inputs(self):
        base = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32
        )
        return (
            ("scalar", torch.tensor(-3.5, dtype=torch.float32)),
            ("empty vector", torch.zeros((0,), dtype=torch.float32)),
            ("empty multidimensional", torch.zeros((2, 0, 3), dtype=torch.float32)),
            ("matrix", base),
            ("offset row", base[1]),
            (
                "requires grad input",
                torch.ones((2, 3), dtype=torch.float32, requires_grad=True) * 2.0,
            ),
        )

    def test_supported_default_equivalent_metadata_allocates_fresh_zeros(self):
        option_cases = (
            {},
            {"dtype": None},
            {"dtype": torch.float32},
            {"dtype": torch.float},
            {"layout": None},
            {"layout": torch.strided},
            {"device": None},
            {"device": "cpu"},
            {"device": "cpu:0"},
            {"device": torch.device("cpu")},
            {"device": torch.device("cpu", 2)},
            {"requires_grad": None},
            {"requires_grad": False},
            {"requires_grad": True},
            {"memory_format": None},
            {"memory_format": torch.preserve_format},
            {"memory_format": torch.contiguous_format},
        )

        for case, source in self.contiguous_inputs():
            for options in option_cases:
                with self.subTest(case=case, options=options):
                    output = torch.zeros_like(source, **options)
                    expected = torch.zeros(
                        source.shape,
                        dtype=torch.float32,
                        device=torch.device("cpu"),
                        requires_grad=options.get("requires_grad") is True,
                    )
                    self.assertEqual(
                        self.tensor_contract(output), self.tensor_contract(expected)
                    )
                    self.assertFalse(output.is_set_to(source))
                    if source.numel() != 0:
                        self.assertNotEqual(output.data_ptr(), source.data_ptr())

    def test_fresh_storage_and_no_input_autograd_edge(self):
        leaf = torch.ones((2, 3), dtype=torch.float32, requires_grad=True)
        source = leaf * 5.0
        first = torch.zeros_like(source, requires_grad=True)
        second = torch.zeros_like(source, requires_grad=True)

        self.assertTrue(first.requires_grad)
        self.assertTrue(first.is_leaf)
        self.assertFalse(first.is_set_to(second))
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertFalse(first.is_set_to(source))

        first.sum().backward()
        self.assertIsNone(leaf.grad)
        self.assertEqual(first.grad.tolist(), torch.ones(first.shape).tolist())

    def test_no_grad_does_not_override_explicit_requires_grad(self):
        source = torch.ones((2,), dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            default = torch.zeros_like(source)
            requested = torch.zeros_like(source, requires_grad=True)

        self.assertFalse(default.requires_grad)
        self.assertTrue(default.is_leaf)
        self.assertTrue(requested.requires_grad)
        self.assertTrue(requested.is_leaf)

    def test_rejects_out_nondefault_formats_modes_and_noncontiguous_inputs(self):
        source = torch.ones((2, 3), dtype=torch.float32)
        cases = (
            (
                lambda: torch.zeros_like(),
                TypeError,
                'zeros_like() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.zeros_like(source, source),
                TypeError,
                "zeros_like() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.zeros_like(source, input=source),
                TypeError,
                "zeros_like() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.zeros_like(source, out=None),
                TypeError,
                "zeros_like() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: torch.zeros_like(source, unexpected=True),
                TypeError,
                "zeros_like() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: torch.zeros_like(source, dtype=object()),
                TypeError,
                "zeros_like(): argument 'dtype' must be torch.dtype, not object",
            ),
            (
                lambda: torch.zeros_like(source, layout=object()),
                TypeError,
                "zeros_like(): argument 'layout' must be torch.layout, not object",
            ),
            (
                lambda: torch.zeros_like(source, device=object()),
                TypeError,
                "zeros_like(): argument 'device' must be torch.device or str, not object",
            ),
            (
                lambda: torch.zeros_like(source, requires_grad=1),
                TypeError,
                "zeros_like(): argument 'requires_grad' must be bool, not int",
            ),
            (
                lambda: torch.zeros_like(source, memory_format=object()),
                TypeError,
                "zeros_like(): argument 'memory_format' must be torch.memory_format, not object",
            ),
            (
                lambda: torch.zeros_like(source, device="cuda"),
                RuntimeError,
                "zeros_like(): device 'cuda' is not supported; only 'cpu' is implemented",
            ),
            (
                lambda: torch.zeros_like(source, memory_format=torch.channels_last),
                NotImplementedError,
                "zeros_like(): only preserve_format and contiguous_format memory formats are supported",
            ),
            (
                lambda: torch.zeros_like(source, memory_format=torch.channels_last_3d),
                NotImplementedError,
                "zeros_like(): only preserve_format and contiguous_format memory formats are supported",
            ),
            (
                lambda: torch.zeros_like(source.transpose(0, 1)),
                NotImplementedError,
                UNSUPPORTED_INPUT,
            ),
            (
                lambda: torch.zeros_like(
                    torch.ones((2, 3, 4, 5)).contiguous(
                        memory_format=torch.channels_last
                    )
                ),
                NotImplementedError,
                UNSUPPORTED_INPUT,
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                self.assert_error(call, error_type, message)

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return object()

        self.assert_error(
            lambda: torch.zeros_like(Override()), NotImplementedError, UNSUPPORTED_INPUT
        )
        self.assertEqual(Override.calls, [])

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return object()

        mode = RecordingMode()
        with mode:
            self.assert_error(
                lambda: torch.zeros_like(source),
                NotImplementedError,
                "zeros_like(): TorchFunctionMode dispatch is not supported",
            )
        self.assertEqual(mode.calls, [])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_callable_metadata_exports_copy_pickle_and_reload(self):
        package = importlib.import_module("torch_rs")
        native = package._C
        function = package.zeros_like

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "zeros_like")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.zeros_like")
        self.assertEqual(function.__module__, "torch")
        self.assertTrue(function.__doc__.startswith(FUNCTION_DOC_PREFIX))
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method zeros_like of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, package._C._VariableFunctionsClass)
        self.assertIs(owner.zeros_like, function)
        self.assertIs(native.zeros_like, function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(package.__all__.count("zeros_like"), 1)
        self.assertNotIn("_VariableFunctionsClass", package.__all__)
        self.assertFalse(hasattr(package, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["zeros_like"], function)

        self.assertFalse(hasattr(package, "ones_like"))
        self.assertFalse(hasattr(package, "empty_like"))
        self.assertFalse(hasattr(package, "full_like"))

        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.zeros_like, function)
        self.assertIs(importlib.reload(package), package)
        self.assertIs(package.zeros_like, function)
        self.assertEqual(package.__all__.count("zeros_like"), 1)


if __name__ == "__main__":
    unittest.main()

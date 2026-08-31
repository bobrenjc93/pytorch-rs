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
    "Returns a tensor filled with the scalar value 0"
)


class ZerosLikeTests(unittest.TestCase):
    def assert_zero_like_result(self, source, result, *, requires_grad=False):
        expected = torch.zeros(source.shape, dtype=torch.float32)
        self.assertIsNot(result, source)
        self.assertFalse(result.is_set_to(source))
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.stride(), expected.stride())
        self.assertEqual(result.storage_offset(), 0)
        self.assertIs(result.dtype, torch.float32)
        self.assertEqual(result.device, torch.device("cpu"))
        self.assertIs(result.layout, torch.strided)
        self.assertEqual(result.requires_grad, requires_grad)
        self.assertTrue(result.is_leaf)
        value_tensor = result.detach() if result.requires_grad else result
        np.testing.assert_array_equal(
            np.asarray(value_tensor).reshape(-1).view(np.uint32),
            np.zeros(result.numel(), dtype=np.uint32),
        )
        if source.numel() != 0:
            self.assertNotEqual(result.data_ptr(), source.data_ptr())

    def test_supported_default_equivalent_metadata(self):
        cases = (
            ("scalar", torch.tensor(-7.0, dtype=torch.float32)),
            ("empty", torch.zeros((2, 0, 3), dtype=torch.float32)),
            (
                "multidimensional",
                torch.tensor(
                    [[1.0, -2.0, 3.5], [4.0, 5.0, -6.0]],
                    dtype=torch.float32,
                ),
            ),
            ("requires grad input", torch.ones((2, 3), requires_grad=True) * 2.0),
        )
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
            {"memory_format": None},
            {"memory_format": torch.preserve_format},
            {"memory_format": torch.contiguous_format},
            {
                "dtype": torch.float32,
                "layout": torch.strided,
                "device": torch.device("cpu"),
                "requires_grad": True,
                "memory_format": torch.preserve_format,
            },
        )

        for case, source in cases:
            for options in option_cases:
                with self.subTest(case=case, options=options):
                    result = torch.zeros_like(source, **options)
                    self.assert_zero_like_result(
                        source,
                        result,
                        requires_grad=options.get("requires_grad") is True,
                    )

    def test_requires_grad_is_explicit_and_no_grad_does_not_disable_it(self):
        leaf = torch.ones((2, 3), requires_grad=True)
        source = leaf * 3.0

        default = torch.zeros_like(source)
        self.assertFalse(default.requires_grad)
        self.assertTrue(default.is_leaf)

        requested = torch.zeros_like(source, requires_grad=True)
        self.assertTrue(requested.requires_grad)
        self.assertTrue(requested.is_leaf)

        with torch.no_grad():
            no_grad_default = torch.zeros_like(source)
            no_grad_requested = torch.zeros_like(source, requires_grad=True)
        self.assertFalse(no_grad_default.requires_grad)
        self.assertTrue(no_grad_default.is_leaf)
        self.assertTrue(no_grad_requested.requires_grad)
        self.assertTrue(no_grad_requested.is_leaf)

    def test_rejects_unsupported_inputs_and_options(self):
        source = torch.ones((2, 3), dtype=torch.float32)
        noncontiguous = source.transpose(0, 1)
        channels_last = torch.zeros((2, 3, 4, 5)).contiguous(
            memory_format=torch.channels_last
        )
        self.assertFalse(noncontiguous.is_contiguous())
        self.assertTrue(channels_last.is_contiguous(memory_format=torch.channels_last))
        self.assertFalse(channels_last.is_contiguous())

        cases = (
            (
                lambda: torch.zeros_like([1.0]),
                TypeError,
                "zeros_like(): argument 'input' (position 1) must be Tensor, not list",
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
                "zeros_like(): argument 'device' must be torch.device, not object",
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
                lambda: torch.zeros_like(
                    source, memory_format=torch.channels_last
                ),
                NotImplementedError,
                "zeros_like(): only preserve_format and contiguous_format are supported",
            ),
            (
                lambda: torch.zeros_like(
                    source, memory_format=torch.channels_last_3d
                ),
                NotImplementedError,
                "zeros_like(): only preserve_format and contiguous_format are supported",
            ),
            (
                lambda: torch.zeros_like(noncontiguous),
                NotImplementedError,
                "zeros_like(): only row-major contiguous inputs are supported",
            ),
            (
                lambda: torch.zeros_like(channels_last),
                NotImplementedError,
                "zeros_like(): only row-major contiguous inputs are supported",
            ),
            (
                lambda: torch.zeros_like(source, device="cuda"),
                RuntimeError,
                "zeros_like(): device 'cuda' is not supported; only 'cpu' is implemented",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                    call()

    def test_rejects_torch_function_modes_without_dispatching(self):
        source = torch.ones((2, 3), dtype=torch.float32)

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return object()

        mode = RecordingMode()
        with mode:
            with self.assertRaisesRegex(
                NotImplementedError,
                r"^zeros_like\(\) does not support an active TorchFunctionMode$",
            ):
                torch.zeros_like(source)
        self.assertEqual(mode.calls, [])

    def test_rejects_torch_function_override_objects_without_dispatching(self):
        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return object()

        with self.assertRaisesRegex(
            TypeError,
            r"^zeros_like\(\): argument 'input' \(position 1\) must be Tensor, not Override$",
        ):
            torch.zeros_like(Override())
        self.assertEqual(Override.calls, [])

    def test_callable_metadata_exports_copy_pickle_and_reload(self):
        package = importlib.import_module("torch_rs")
        native = package._C
        function = package.zeros_like

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "zeros_like")
        self.assertEqual(
            function.__qualname__, "_VariableFunctionsClass.zeros_like"
        )
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
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(package.__all__.count("zeros_like"), 1)
        self.assertNotIn("_VariableFunctionsClass", package.__all__)
        self.assertFalse(hasattr(package, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["zeros_like"], function)

        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.zeros_like, function)
        self.assertIs(importlib.reload(package), package)
        self.assertIs(package.zeros_like, function)
        self.assertEqual(package.__all__.count("zeros_like"), 1)

    def test_other_like_factories_remain_absent(self):
        for name in ("empty_like", "ones_like", "full_like"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)


if __name__ == "__main__":
    unittest.main()

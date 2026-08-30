import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


class ZerosLikeTests(unittest.TestCase):
    def expected_stride(self, source, kwargs):
        if kwargs.get("memory_format") is torch.contiguous_format:
            return torch.zeros(tuple(source.shape)).stride()
        return source.stride()

    def assert_tensor_matches(self, actual, source, *, requires_grad, expected_stride):
        self.assertEqual(actual.shape, source.shape)
        self.assertEqual(actual.stride(), expected_stride)
        self.assertEqual(actual.storage_offset(), 0)
        np.testing.assert_array_equal(
            np.asarray(actual),
            np.zeros(tuple(source.shape), dtype=np.float32),
        )
        self.assertIs(actual.dtype, torch.float32)
        self.assertEqual(actual.device, torch.device("cpu"))
        self.assertIs(actual.layout, torch.strided)
        self.assertEqual(actual.requires_grad, requires_grad)
        self.assertTrue(actual.is_leaf)
        self.assertIsNone(actual.grad)
        self.assertFalse(actual.is_set_to(source))
        if source.numel() != 0:
            self.assertNotEqual(actual.data_ptr(), source.data_ptr())

    def test_scalar_empty_and_multidimensional_results_are_fresh(self):
        cases = (
            ("scalar", torch.tensor(-3.5, requires_grad=True)),
            ("empty", torch.zeros((0,), requires_grad=True)),
            ("empty leading dimension", torch.zeros((0, 3), requires_grad=True)),
            (
                "multidimensional",
                torch.tensor([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]]),
            ),
            ("offset contiguous view", torch.ones((3, 2))[1]),
            ("singleton transpose", torch.ones((2, 1)).transpose(0, 1)),
            ("empty transpose", torch.zeros((2, 0, 3)).transpose(0, 2)),
        )
        option_cases = (
            ({}, False),
            ({"dtype": None}, False),
            ({"dtype": torch.float32}, False),
            ({"dtype": torch.float}, False),
            ({"layout": None}, False),
            ({"layout": torch.strided}, False),
            ({"device": None}, False),
            ({"device": "cpu"}, False),
            ({"device": torch.device("cpu")}, False),
            ({"requires_grad": None}, False),
            ({"requires_grad": False}, False),
            ({"requires_grad": True}, True),
            ({"memory_format": None}, False),
            ({"memory_format": torch.preserve_format}, False),
            ({"memory_format": torch.contiguous_format}, False),
            (
                {
                    "dtype": torch.float32,
                    "layout": torch.strided,
                    "device": torch.device("cpu"),
                    "requires_grad": True,
                    "memory_format": torch.preserve_format,
                },
                True,
            ),
        )

        for case, source in cases:
            for kwargs, expected_requires_grad in option_cases:
                with self.subTest(case=case, kwargs=kwargs):
                    self.assert_tensor_matches(
                        torch.zeros_like(source, **kwargs),
                        source,
                        requires_grad=expected_requires_grad,
                        expected_stride=self.expected_stride(source, kwargs),
                    )

    def test_input_keyword_and_no_grad_metadata(self):
        source = torch.ones((2, 3), requires_grad=True)
        keyword = torch.zeros_like(input=source)
        self.assert_tensor_matches(
            keyword,
            source,
            requires_grad=False,
            expected_stride=source.stride(),
        )

        with torch.no_grad():
            default = torch.zeros_like(source)
            tracked = torch.zeros_like(source, requires_grad=True)
        self.assertFalse(default.requires_grad)
        self.assertTrue(default.is_leaf)
        self.assertTrue(tracked.requires_grad)
        self.assertTrue(tracked.is_leaf)

    def test_requires_grad_result_is_independent_leaf(self):
        source = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = torch.zeros_like(source, requires_grad=True)
        result.sum().backward()

        self.assertIsNone(source.grad)
        self.assertIsNotNone(result.grad)
        np.testing.assert_array_equal(np.asarray(result.grad), np.ones(3, dtype=np.float32))

    def test_rejects_noncontiguous_channels_last_and_unsupported_metadata(self):
        source = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        noncontiguous = source.transpose(0, 1)
        channels_last = torch.clone(
            torch.ones((2, 3, 4, 5)),
            memory_format=torch.channels_last,
        )
        channels_last_3d = torch.clone(
            torch.ones((2, 3, 4, 5, 6)),
            memory_format=torch.channels_last_3d,
        )
        ambiguous_channels_last = torch.clone(
            torch.ones((1, 1, 2, 2)),
            memory_format=torch.channels_last,
        )

        for tensor in (
            noncontiguous,
            channels_last,
            channels_last_3d,
            ambiguous_channels_last,
        ):
            with self.subTest(stride=tensor.stride()):
                with self.assertRaisesRegex(
                    RuntimeError,
                    re.escape(
                        "zeros_like(): only row-major contiguous input tensors are supported"
                    ),
                ):
                    torch.zeros_like(tensor)

        for kwargs, error_type, message in (
            (
                {"dtype": object()},
                TypeError,
                "zeros_like(): argument 'dtype' must be torch.dtype, not object",
            ),
            (
                {"layout": object()},
                TypeError,
                "zeros_like(): argument 'layout' must be torch.layout, not object",
            ),
            (
                {"device": object()},
                TypeError,
                "zeros_like(): argument 'device' must be torch.device, not object",
            ),
            (
                {"device": "cuda"},
                RuntimeError,
                "zeros_like(): device 'cuda' is not supported; only 'cpu' is implemented",
            ),
            (
                {"requires_grad": 1},
                TypeError,
                "zeros_like(): argument 'requires_grad' must be bool, not int",
            ),
            (
                {"memory_format": object()},
                TypeError,
                "zeros_like(): argument 'memory_format' must be torch.memory_format, not object",
            ),
            (
                {"memory_format": torch.channels_last},
                NotImplementedError,
                "zeros_like(): only default-equivalent memory_format values are supported",
            ),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                    torch.zeros_like(source, **kwargs)

    def test_binding_errors_out_and_override_boundaries(self):
        source = torch.ones((2,))
        for call, message in (
            (
                lambda: torch.zeros_like(),
                'zeros_like() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.zeros_like(source, source),
                "zeros_like() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.zeros_like(source, input=source),
                "zeros_like() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.zeros_like(source, out=None),
                "zeros_like() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: torch.zeros_like(source, out=source),
                "zeros_like() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: torch.zeros_like(source, unexpected=True),
                "zeros_like() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: torch.zeros_like(x=source),
                'zeros_like() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.zeros_like(1),
                "zeros_like(): argument 'input' (position 1) must be Tensor, not int",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return object()

        with self.assertRaisesRegex(
            TypeError,
            r"^zeros_like\(\): argument 'input' \(position 1\) must be Tensor, "
            r"not (?:.*\.)?Override$",
        ):
            torch.zeros_like(Override())
        self.assertEqual(Override.calls, [])

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return object()

        mode = RecordingMode()
        with mode:
            with self.assertRaisesRegex(
                TypeError,
                r"^zeros_like\(\) does not support TorchFunctionMode$",
            ):
                torch.zeros_like(source)
        self.assertEqual(mode.calls, [])

    def test_top_level_builtin_metadata_exports_copying_and_pickling(self):
        function = torch.zeros_like
        self.assertIs(function, torch._C.zeros_like)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "zeros_like")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.zeros_like")
        self.assertEqual(function.__module__, "torch")
        self.assertIn(
            "zeros_like(input, *, dtype=None, layout=None, device=None, "
            "requires_grad=False, memory_format=torch.preserve_format) -> Tensor",
            function.__doc__,
        )
        self.assertIsNone(function.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.zeros_like, function)
        for action in (
            lambda: setattr(owner, "zeros_like", None),
            lambda: delattr(owner, "zeros_like"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.zeros_like, function)

        imported = importlib.import_module("torch_rs").zeros_like
        native_imported = importlib.import_module("torch_rs._C").zeros_like
        self.assertIs(imported, function)
        self.assertIs(native_imported, function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("zeros_like"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["zeros_like"], function)

    def test_other_like_factories_remain_unexported(self):
        for name in ("ones_like", "empty_like", "full_like"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertFalse(hasattr(torch._C, name))
                self.assertNotIn(name, torch.__all__)


if __name__ == "__main__":
    unittest.main()

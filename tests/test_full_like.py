import copy
import importlib
import inspect
import math
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


SUPPORTED_INPUT_ERROR = (
    "full_like(): only exact native CPU float32 row-major contiguous Tensor "
    "inputs are supported"
)


def tensor_bits(tensor):
    return np.asarray(tensor.detach()).reshape(-1).view(np.uint32).tolist()


class FullLikeTests(unittest.TestCase):
    def assert_full_like_result(
        self, source, result, fill_value, *, requires_grad=False
    ):
        self.assertIs(type(result), torch.Tensor)
        self.assertIsNot(result, source)
        self.assertFalse(result.is_set_to(source))
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.stride(), source.stride())
        self.assertEqual(result.storage_offset(), 0)
        self.assertIs(result.dtype, torch.float32)
        self.assertEqual(result.device, torch.device("cpu"))
        self.assertEqual(result.requires_grad, requires_grad)
        self.assertTrue(result.is_leaf)
        self.assertEqual(tensor_bits(result), tensor_bits(torch.full(source.shape, fill_value)))

    def supported_sources(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        return (
            ("scalar", torch.tensor(-3.0)),
            ("empty", torch.ones((0,))),
            ("empty multidimensional", torch.ones((2, 0, 3))),
            ("multidimensional", torch.ones((2, 3, 4))),
            ("offset contiguous", base[1]),
        )

    def test_supported_default_metadata_matches_input_shape_stride_and_values(self):
        option_cases = (
            {},
            {"dtype": None},
            {"dtype": torch.float32},
            {"dtype": torch.float},
            {"layout": None},
            {"layout": torch.strided},
            {"device": None},
            {"device": "cpu"},
            {"device": torch.device("cpu")},
            {"memory_format": None},
            {"memory_format": torch.preserve_format},
            {"memory_format": torch.contiguous_format},
        )
        for case, source in self.supported_sources():
            before = (
                source.shape,
                source.stride(),
                source.storage_offset(),
                source.data_ptr(),
                source.tolist(),
            )
            for options in option_cases:
                with self.subTest(case=case, options=options):
                    result = torch.full_like(source, -1.25, **options)
                    self.assert_full_like_result(source, result, -1.25)
                    self.assertEqual(
                        (
                            source.shape,
                            source.stride(),
                            source.storage_offset(),
                            source.data_ptr(),
                            source.tolist(),
                        ),
                        before,
                    )

    def test_fill_value_scalars_preserve_float32_bits(self):
        source = torch.ones((2,))
        fill_values = (
            -0.0,
            math.inf,
            -math.inf,
            math.nan,
            3,
            -2,
            True,
            False,
            np.bool_(True),
            np.int64(-5),
            np.uint32(7),
            np.float64(1.25),
            torch.tensor(2.5),
        )
        for fill_value in fill_values:
            with self.subTest(fill_value=repr(fill_value)):
                self.assert_full_like_result(
                    source, torch.full_like(source, fill_value), fill_value
                )

    def test_requires_grad_and_no_grad_match_factory_semantics(self):
        leaf = torch.ones((2, 3), requires_grad=True)
        source = leaf * 2.0

        default = torch.full_like(source, 4.0)
        self.assert_full_like_result(source, default, 4.0)

        tracked = torch.full_like(source, 4.0, requires_grad=True)
        self.assert_full_like_result(source, tracked, 4.0, requires_grad=True)

        with torch.no_grad():
            no_grad_default = torch.full_like(source, 4.0)
            no_grad_tracked = torch.full_like(source, 4.0, requires_grad=True)
        self.assert_full_like_result(source, no_grad_default, 4.0)
        self.assert_full_like_result(
            source, no_grad_tracked, 4.0, requires_grad=True
        )

        tracked.sum().backward()
        self.assertEqual(tracked.grad.tolist(), [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        self.assertIsNone(leaf.grad)

    def test_rejects_noncontiguous_channels_last_and_nondefault_metadata(self):
        source = torch.ones((2, 3))
        noncontiguous = source.transpose(0, 1)
        relaxed_singleton_contiguous = torch.ones((3, 1)).transpose(0, 1)
        relaxed_empty_contiguous = torch.ones((2, 0, 3)).transpose(0, 2)
        channels_last = torch.ones((2, 3, 4, 5)).contiguous(
            memory_format=torch.channels_last
        )

        self.assertTrue(relaxed_singleton_contiguous.is_contiguous())
        self.assertEqual(relaxed_singleton_contiguous.shape, (1, 3))
        self.assertEqual(relaxed_singleton_contiguous.stride(), (1, 1))
        self.assertTrue(relaxed_empty_contiguous.is_contiguous())
        self.assertEqual(relaxed_empty_contiguous.shape, (3, 0, 2))
        self.assertEqual(relaxed_empty_contiguous.stride(), (1, 3, 3))

        for case, call in (
            ("noncontiguous", lambda: torch.full_like(noncontiguous, 2.0)),
            (
                "relaxed singleton contiguous",
                lambda: torch.full_like(relaxed_singleton_contiguous, 2.0),
            ),
            (
                "relaxed empty contiguous",
                lambda: torch.full_like(relaxed_empty_contiguous, 2.0),
            ),
            ("channels last", lambda: torch.full_like(channels_last, 2.0)),
        ):
            with self.subTest(case=case), self.assertRaisesRegex(
                NotImplementedError, f"^{re.escape(SUPPORTED_INPUT_ERROR)}$"
            ):
                call()

        error_cases = (
            (
                lambda: torch.full_like(source, 2.0, memory_format=torch.channels_last),
                NotImplementedError,
                "full_like(): only default-equivalent memory_format is supported",
            ),
            (
                lambda: torch.full_like(source, 2.0, device="cuda"),
                RuntimeError,
                "full_like(): device 'cuda' is not supported; only 'cpu' is implemented",
            ),
            (
                lambda: torch.full_like(source, 2.0, device=torch.device("cpu", 0)),
                NotImplementedError,
                "full_like(): indexed CPU devices require a copy and are not supported",
            ),
            (
                lambda: torch.full_like(source, 2.0, out=None),
                TypeError,
                "full_like() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: torch.full_like(source, 2.0, out=torch.zeros((2, 3))),
                TypeError,
                "full_like() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: torch.full_like(source, 2.0, dtype=object()),
                TypeError,
                "full_like(): argument 'dtype' must be torch.dtype, not object",
            ),
            (
                lambda: torch.full_like(source, 2.0, layout=object()),
                TypeError,
                "full_like(): argument 'layout' must be torch.layout, not object",
            ),
            (
                lambda: torch.full_like(source, 2.0, memory_format=True),
                TypeError,
                "full_like(): argument 'memory_format' must be torch.memory_format, not bool",
            ),
            (
                lambda: torch.full_like(source, 2.0, requires_grad=1),
                TypeError,
                "full_like(): argument 'requires_grad' must be bool, not int",
            ),
        )
        for call, error_type, message in error_cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                error_type, f"^{re.escape(message)}$"
            ):
                call()

    def test_rejects_non_scalar_fills_subclasses_modes_and_bad_call_forms(self):
        source = torch.ones((2, 3))

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return object()

        bad_calls = (
            (
                lambda: torch.full_like(),
                TypeError,
                'full_like() missing 2 required positional argument: "input", "fill_value"',
            ),
            (
                lambda: torch.full_like(source),
                TypeError,
                'full_like() missing 1 required positional arguments: "fill_value"',
            ),
            (
                lambda: torch.full_like(source, 2.0, source),
                TypeError,
                "full_like() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.full_like(source, 2.0, input=source),
                TypeError,
                "full_like() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.full_like(source, 2.0, fill_value=3.0),
                TypeError,
                "full_like() got multiple values for argument 'fill_value'",
            ),
            (
                lambda: torch.full_like([1.0], 2.0),
                TypeError,
                "full_like(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.full_like(source, [2.0]),
                TypeError,
                "full_like(): argument 'fill_value' (position 2) must be Number, not list",
            ),
            (
                lambda: torch.full_like(source, fill_value=[2.0]),
                TypeError,
                "full_like(): argument 'fill_value' must be Number, not list",
            ),
            (
                lambda: torch.full_like(source, torch.ones((1,))),
                TypeError,
                "full_like(): argument 'fill_value' (position 2) must be Number, not Tensor",
            ),
            (
                lambda: torch.full_like(source, fill_value=torch.ones((1,))),
                TypeError,
                "full_like(): argument 'fill_value' must be Number, not Tensor",
            ),
            (
                lambda: torch.full_like(Override(), 2.0),
                TypeError,
                "full_like(): argument 'input' (position 1) must be Tensor, not Override",
            ),
        )
        for call, error_type, message in bad_calls:
            with self.subTest(message=message), self.assertRaisesRegex(
                error_type, f"^{re.escape(message)}$"
            ):
                call()
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
                NotImplementedError,
                r"^full_like\(\): __torch_function__ modes are not supported$",
            ):
                torch.full_like(source, 2.0)
        self.assertEqual(mode.calls, [])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_input_keyword_aliases_match_generated_binding_forms(self):
        source = torch.ones((2,))
        for keyword in ("input", "x", "a", "x1"):
            with self.subTest(keyword=keyword):
                result = torch.full_like(**{keyword: source}, fill_value=2.5)
                self.assert_full_like_result(source, result, 2.5)

    def test_callable_metadata_exports_copy_pickle_and_reload(self):
        package = importlib.import_module("torch_rs")
        native = package._C
        function = package.full_like
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "full_like")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.full_like")
        self.assertEqual(function.__module__, "torch")
        self.assertIn(
            "full_like(input, fill_value, *, dtype=None, layout=None, device=None, "
            "requires_grad=False, memory_format=None) -> Tensor",
            function.__doc__,
        )
        self.assertIsNone(function.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertIs(owner, package._C._VariableFunctionsClass)
        self.assertIs(owner.full_like, function)
        self.assertIs(native.full_like, function)
        self.assertEqual(package.__all__.count("full_like"), 1)
        self.assertIs(wildcard_namespace["full_like"], function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.full_like, function)
        self.assertIs(importlib.reload(package), package)
        self.assertIs(package.full_like, function)
        self.assertEqual(package.__all__.count("full_like"), 1)

    def test_empty_like_remains_unsupported(self):
        package = importlib.import_module("torch_rs")
        native = package._C
        self.assertFalse(hasattr(package, "empty_like"))
        self.assertFalse(hasattr(native, "empty_like"))
        self.assertNotIn("empty_like", package.__all__)
        self.assertNotIn("empty_like", native.__all__)


if __name__ == "__main__":
    unittest.main()

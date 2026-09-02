import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


SUPPORTED_INPUT_ERROR = (
    "ones_like(): only exact native CPU float32 row-major contiguous or rank-4 "
    "channels-last-contiguous non-overlapping dense Tensor inputs are supported"
)


class OnesLikeTests(unittest.TestCase):
    def assert_ones_like_result(self, source, result, *, requires_grad=False):
        self.assertIs(type(result), torch.Tensor)
        self.assertIsNot(result, source)
        self.assertFalse(result.is_set_to(source))
        if result.numel() != 0:
            self.assertNotEqual(result.data_ptr(), source.data_ptr())
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.stride(), source.stride())
        self.assertEqual(result.storage_offset(), 0)
        self.assertIs(result.dtype, torch.float32)
        self.assertEqual(result.device, torch.device("cpu"))
        self.assertEqual(result.requires_grad, requires_grad)
        self.assertTrue(result.is_leaf)
        self.assertEqual(result.tolist(), torch.ones(source.shape).tolist())

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

    def channels_last_sources(self):
        base = torch.ones((3, 3, 4, 5), dtype=torch.float32).contiguous(
            memory_format=torch.channels_last
        )
        return (
            (
                "standard",
                torch.ones((2, 3, 4, 5), dtype=torch.float32).contiguous(
                    memory_format=torch.channels_last
                ),
            ),
            (
                "singleton channel",
                torch.ones((2, 1, 4, 5), dtype=torch.float32).contiguous(
                    memory_format=torch.channels_last
                ),
            ),
            (
                "singleton height",
                torch.ones((2, 3, 1, 5), dtype=torch.float32).contiguous(
                    memory_format=torch.channels_last
                ),
            ),
            (
                "singleton width",
                torch.ones((2, 3, 4, 1), dtype=torch.float32).contiguous(
                    memory_format=torch.channels_last
                ),
            ),
            ("offset", base[1][None]),
        )

    def observed_source_state(self, source):
        return (
            source.shape,
            source.stride(),
            source.storage_offset(),
            source.data_ptr(),
            source.tolist(),
            source.is_contiguous(),
            source.is_contiguous(memory_format=torch.channels_last),
        )

    def test_supported_default_metadata_matches_input_shape_and_stride(self):
        option_cases = (
            {},
            {"dtype": None},
            {"dtype": torch.float32},
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
                    result = torch.ones_like(source, **options)
                    self.assert_ones_like_result(source, result)
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

    def test_channels_last_preserve_format_matches_input_shape_stride_and_storage(self):
        option_cases = (
            {},
            {"memory_format": None},
            {"memory_format": torch.preserve_format},
            {"dtype": torch.float32},
            {"layout": torch.strided},
            {"device": torch.device("cpu")},
        )
        for case, source in self.channels_last_sources():
            self.assertTrue(source.is_contiguous(memory_format=torch.channels_last))
            before = self.observed_source_state(source)
            for options in option_cases:
                with self.subTest(case=case, options=options):
                    result = torch.ones_like(source, **options)
                    self.assert_ones_like_result(source, result)
                    self.assertTrue(
                        result.is_contiguous(memory_format=torch.channels_last)
                    )
                    self.assertEqual(self.observed_source_state(source), before)

    def test_requires_grad_and_no_grad_match_factory_semantics(self):
        leaf = torch.ones((2, 3), requires_grad=True)
        source = leaf * 2.0

        default = torch.ones_like(source)
        self.assert_ones_like_result(source, default)

        tracked = torch.ones_like(source, requires_grad=True)
        self.assert_ones_like_result(source, tracked, requires_grad=True)

        with torch.no_grad():
            no_grad_default = torch.ones_like(source)
            no_grad_tracked = torch.ones_like(source, requires_grad=True)
        self.assert_ones_like_result(source, no_grad_default)
        self.assert_ones_like_result(source, no_grad_tracked, requires_grad=True)

        tracked.sum().backward()
        self.assertEqual(tracked.grad.tolist(), [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        self.assertIsNone(leaf.grad)

    def test_channels_last_respects_requires_grad_and_no_grad(self):
        source = (
            torch.ones((2, 3, 4, 5), dtype=torch.float32, requires_grad=True) * 2.0
        ).contiguous(memory_format=torch.channels_last)

        default = torch.ones_like(source)
        self.assert_ones_like_result(source, default)
        self.assertTrue(default.is_contiguous(memory_format=torch.channels_last))

        tracked = torch.ones_like(source, requires_grad=True)
        self.assert_ones_like_result(source, tracked, requires_grad=True)
        self.assertTrue(tracked.is_contiguous(memory_format=torch.channels_last))

        with torch.no_grad():
            no_grad_default = torch.ones_like(source)
            no_grad_tracked = torch.ones_like(source, requires_grad=True)
        self.assert_ones_like_result(source, no_grad_default)
        self.assert_ones_like_result(source, no_grad_tracked, requires_grad=True)

        tracked.sum().backward()
        self.assertEqual(tracked.grad.tolist(), torch.ones(source.shape).tolist())

    def test_rejects_noncontiguous_and_nondefault_metadata(self):
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
            ("noncontiguous", lambda: torch.ones_like(noncontiguous)),
            (
                "relaxed singleton contiguous",
                lambda: torch.ones_like(relaxed_singleton_contiguous),
            ),
            (
                "relaxed empty contiguous",
                lambda: torch.ones_like(relaxed_empty_contiguous),
            ),
        ):
            with self.subTest(case=case), self.assertRaisesRegex(
                NotImplementedError, f"^{re.escape(SUPPORTED_INPUT_ERROR)}$"
            ):
                call()

        error_cases = (
            (
                lambda: torch.ones_like(source, memory_format=torch.channels_last),
                NotImplementedError,
                "ones_like(): only default-equivalent memory_format is supported",
            ),
            (
                lambda: torch.ones_like(
                    channels_last, memory_format=torch.channels_last
                ),
                NotImplementedError,
                "ones_like(): only default-equivalent memory_format is supported",
            ),
            (
                lambda: torch.ones_like(source, device="cuda"),
                RuntimeError,
                "ones_like(): device 'cuda' is not supported; only 'cpu' is implemented",
            ),
            (
                lambda: torch.ones_like(source, device=torch.device("cpu", 0)),
                NotImplementedError,
                "ones_like(): indexed CPU devices require a copy and are not supported",
            ),
            (
                lambda: torch.ones_like(source, out=None),
                TypeError,
                "ones_like() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: torch.ones_like(source, dtype=object()),
                TypeError,
                "ones_like(): argument 'dtype' must be torch.dtype, not object",
            ),
            (
                lambda: torch.ones_like(source, layout=object()),
                TypeError,
                "ones_like(): argument 'layout' must be torch.layout, not object",
            ),
            (
                lambda: torch.ones_like(source, memory_format=True),
                TypeError,
                "ones_like(): argument 'memory_format' must be torch.memory_format, not bool",
            ),
            (
                lambda: torch.ones_like(source, requires_grad=1),
                TypeError,
                "ones_like(): argument 'requires_grad' must be bool, not int",
            ),
        )
        for call, error_type, message in error_cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                error_type, f"^{re.escape(message)}$"
            ):
                call()

    def test_rejects_bad_input_subclasses_modes_and_bad_call_forms(self):
        source = torch.ones((2, 3))

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return object()

        bad_calls = (
            (
                lambda: torch.ones_like(),
                TypeError,
                'ones_like() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.ones_like(source, source),
                TypeError,
                "ones_like() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.ones_like(source, input=source),
                TypeError,
                "ones_like() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.ones_like([1.0]),
                TypeError,
                "ones_like(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.ones_like(Override()),
                TypeError,
                "ones_like(): argument 'input' (position 1) must be Tensor, not Override",
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
                r"^ones_like\(\): __torch_function__ modes are not supported$",
            ):
                torch.ones_like(source)
        self.assertEqual(mode.calls, [])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_input_keyword_aliases_match_generated_binding_forms(self):
        source = torch.ones((2,))
        for keyword in ("input", "x", "a", "x1"):
            with self.subTest(keyword=keyword):
                result = torch.ones_like(**{keyword: source})
                self.assert_ones_like_result(source, result)

    def test_callable_metadata_exports_copy_pickle_and_reload(self):
        package = importlib.import_module("torch_rs")
        native = package._C
        function = package.ones_like
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "ones_like")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.ones_like")
        self.assertEqual(function.__module__, "torch")
        self.assertIn(
            "ones_like(input, *, dtype=None, layout=None, device=None, "
            "requires_grad=False, memory_format=None) -> Tensor",
            function.__doc__,
        )
        self.assertIsNone(function.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertIs(owner, package._C._VariableFunctionsClass)
        self.assertIs(owner.ones_like, function)
        self.assertIs(native.ones_like, function)
        self.assertEqual(package.__all__.count("ones_like"), 1)
        self.assertIs(wildcard_namespace["ones_like"], function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.ones_like, function)
        self.assertIs(importlib.reload(package), package)
        self.assertIs(package.ones_like, function)
        self.assertEqual(package.__all__.count("ones_like"), 1)


if __name__ == "__main__":
    unittest.main()

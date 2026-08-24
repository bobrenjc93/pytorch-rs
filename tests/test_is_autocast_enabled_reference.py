import copy
import importlib
import inspect
import os
import pickle
import subprocess
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class IsAutocastEnabledReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "autocast enabled-state differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def signature_outcome(self, function):
        try:
            return "return", str(inspect.signature(function))
        except BaseException as error:
            return "error", type(error).__name__, str(error)

    def test_default_cpu_cuda_and_indexed_forms_match_pytorch_2_13(self):
        forms = (
            (),
            ("cpu",),
            ("cpu:0",),
            ("cpu:2147483647",),
            ("cuda",),
            ("cuda:0",),
            ("cuda:7",),
            (b"cpu",),
            (b"cuda:0",),
            (np.str_("cpu"),),
        )
        for arguments in forms:
            with self.subTest(arguments=arguments):
                actual = torch.is_autocast_enabled(*arguments)
                expected = reference_torch.is_autocast_enabled(*arguments)
                self.assertIs(actual, expected)
                self.assertIs(actual, False)

        for device_type in ("cpu", "cpu:0", "cuda", "cuda:0", b"cuda"):
            with self.subTest(device_type=device_type, binding="keyword"):
                actual = torch.is_autocast_enabled(device_type=device_type)
                expected = reference_torch.is_autocast_enabled(
                    device_type=device_type
                )
                self.assertIs(actual, expected)
                self.assertIs(actual, False)

    def test_queries_do_not_initialize_cuda_in_a_fresh_process(self):
        script = r"""
import torch as reference_torch
import torch_rs as actual_torch

assert reference_torch.__version__.split("+")[0] == "2.13.0"
assert reference_torch.cuda.is_initialized() is False
for arguments in ((), ("cpu",), ("cpu:0",), ("cuda",), ("cuda:0",)):
    assert actual_torch.is_autocast_enabled(*arguments) is False
    assert reference_torch.is_autocast_enabled(*arguments) is False
    assert reference_torch.cuda.is_initialized() is False
"""
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES="0",
            PYTHONDONTWRITEBYTECODE="1",
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )

    def test_cpu_cuda_default_state_matches_after_h100_initialization(self):
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch build")

        device_name = reference_torch.cuda.get_device_name(0)
        if "H100" not in device_name:
            self.skipTest(f"requires an NVIDIA H100, found {device_name}")

        probe = reference_torch.ones(1, device="cuda:0")
        reference_torch.cuda.synchronize(0)
        self.assertEqual(probe.item(), 1.0)
        self.assertIs(reference_torch.cuda.is_initialized(), True)
        for arguments in ((), ("cpu",), ("cpu:0",), ("cuda",), ("cuda:0",)):
            with self.subTest(arguments=arguments):
                self.assertIs(torch.is_autocast_enabled(*arguments), False)
                self.assertIs(
                    reference_torch.is_autocast_enabled(*arguments), False
                )

    def test_builtin_metadata_exports_copying_and_pickling_match(self):
        actual = torch.is_autocast_enabled
        expected = reference_torch.is_autocast_enabled

        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs.torch_rs", "torch"),
            expected.__module__,
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        self.assertEqual(
            hasattr(actual, "__annotations__"),
            hasattr(expected, "__annotations__"),
        )
        self.assertEqual(repr(actual), repr(expected))
        self.assertIs(actual.__self__, torch._C)
        self.assertIs(expected.__self__, reference_torch._C)
        self.assertIs(torch._C.is_autocast_enabled, actual)
        self.assertIs(reference_torch._C.is_autocast_enabled, expected)
        self.assertEqual(actual.__reduce__(), expected.__reduce__())
        self.assertEqual(
            self.signature_outcome(actual), self.signature_outcome(expected)
        )
        self.assertEqual(
            torch.__all__.count("is_autocast_enabled"),
            reference_torch.__all__.count("is_autocast_enabled"),
        )

        for function in (actual, expected):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(
                    module=function.__module__, protocol=protocol
                ):
                    restored = pickle.loads(
                        pickle.dumps(function, protocol=protocol)
                    )
                    self.assertIs(restored, function)

        for module, function in (
            (torch, actual),
            (reference_torch, expected),
        ):
            wildcard_namespace = {}
            exec(f"from {module.__name__} import *", wildcard_namespace)
            self.assertIs(wildcard_namespace["is_autocast_enabled"], function)

        actual_native = importlib.import_module("torch_rs._C")
        expected_native = importlib.import_module("torch._C")
        self.assertIs(actual_native.is_autocast_enabled, actual)
        self.assertIs(expected_native.is_autocast_enabled, expected)

    def test_supported_binding_and_device_errors_match_pytorch_2_13(self):
        actual = torch.is_autocast_enabled
        expected = reference_torch.is_autocast_enabled

        class HostileKeyword(str):
            __hash__ = str.__hash__

            def __eq__(self, other):
                raise RuntimeError("keyword equality trap")

        paired_values = (
            (None, None),
            (1, 1),
            (True, True),
            (bytearray(b"cpu"), bytearray(b"cpu")),
            (torch.device("cpu"), reference_torch.device("cpu")),
        )
        for actual_value, expected_value in paired_values:
            with self.subTest(value=actual_value, binding="positional"):
                self.assert_error_matches(
                    lambda: actual(actual_value),
                    lambda: expected(expected_value),
                )
            with self.subTest(value=actual_value, binding="keyword"):
                self.assert_error_matches(
                    lambda: actual(device_type=actual_value),
                    lambda: expected(device_type=expected_value),
                )

        cases = (
            (
                lambda function: function(enabled=True),
                lambda function: function(enabled=True),
            ),
            (
                lambda function: function("cpu", "cuda"),
                lambda function: function("cpu", "cuda"),
            ),
            (
                lambda function: function("cpu", device_type="cuda"),
                lambda function: function("cpu", device_type="cuda"),
            ),
            (
                lambda function: function("cpu", enabled=True),
                lambda function: function("cpu", enabled=True),
            ),
            (
                lambda function: function(
                    **{HostileKeyword("device_type"): "cpu"}
                ),
                lambda function: function(
                    **{HostileKeyword("device_type"): "cpu"}
                ),
            ),
            (
                lambda function: function("cpu", **{"bad\0tail": 1}),
                lambda function: function("cpu", **{"bad\0tail": 1}),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(
                    lambda: actual_call(actual),
                    lambda: expected_call(expected),
                )

        invalid_strings = (
            "",
            b"",
            "CPU",
            "banana",
            "cpu:-1",
            "cpu:00",
            "cuda:",
            "cuda:abc",
            "cuda:2147483648",
            "\ud800",
        )
        for value in invalid_strings:
            with self.subTest(value=value, binding="positional"):
                self.assert_error_matches(
                    lambda: actual(value),
                    lambda: expected(value),
                )
            with self.subTest(value=value, binding="keyword"):
                self.assert_error_matches(
                    lambda: actual(device_type=value),
                    lambda: expected(device_type=value),
                )

    def test_only_the_query_and_existing_cache_controls_are_supported(self):
        self.assertIs(torch.is_autocast_enabled, torch._C.is_autocast_enabled)
        self.assertIs(
            reference_torch.is_autocast_enabled,
            reference_torch._C.is_autocast_enabled,
        )
        self.assertIs(reference_torch.is_autocast_enabled("xpu"), False)
        with self.assertRaisesRegex(RuntimeError, "only 'cpu' and 'cuda'"):
            torch.is_autocast_enabled("xpu")

        unsupported = (
            "autocast",
            "autocast_decrement_nesting",
            "autocast_increment_nesting",
            "get_autocast_cpu_dtype",
            "get_autocast_dtype",
            "get_autocast_gpu_dtype",
            "set_autocast_cpu_dtype",
            "set_autocast_cpu_enabled",
            "set_autocast_dtype",
            "set_autocast_enabled",
            "set_autocast_gpu_dtype",
        )
        for name in unsupported:
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertTrue(hasattr(reference_torch, name))
        self.assertFalse(hasattr(torch, "amp"))
        self.assertTrue(hasattr(reference_torch, "amp"))
        self.assertFalse(hasattr(torch.cpu, "amp"))
        self.assertTrue(hasattr(reference_torch.cpu, "amp"))


if __name__ == "__main__":
    unittest.main()

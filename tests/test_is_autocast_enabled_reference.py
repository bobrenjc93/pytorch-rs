import copy
import inspect
import json
import os
import pickle
import subprocess
import sys
import types
import unittest

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
                "autocast state differentials require pinned PyTorch 2.13.0"
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

    def test_default_cpu_and_cuda_forms_match_pytorch_2_13(self):
        calls = (
            lambda function: function(),
            lambda function: function("cpu"),
            lambda function: function("cuda"),
            lambda function: function("cpu:0"),
            lambda function: function("cuda:0"),
            lambda function: function(device_type="cpu"),
            lambda function: function(device_type="cuda"),
            lambda function: function(**{}),
        )
        for case, call in enumerate(calls):
            with self.subTest(case=case):
                actual = call(torch.is_autocast_enabled)
                expected = call(reference_torch.is_autocast_enabled)
                self.assertIs(actual, False)
                self.assertIs(expected, False)
                self.assertIs(actual, expected)

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
            self.signature_outcome(actual),
            self.signature_outcome(expected),
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
                    module=function.__module__,
                    protocol=protocol,
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

    def test_binding_and_supported_form_errors_match_pytorch_2_13(self):
        class HasString:
            def __str__(self):
                return "cpu"

        paired_value_factories = (
            lambda module: None,
            lambda module: 1,
            lambda module: True,
            lambda module: module.device("cpu"),
            lambda module: module.float32,
            lambda module: HasString(),
        )
        for case, factory in enumerate(paired_value_factories):
            with self.subTest(form="positional_type", case=case):
                self.assert_error_matches(
                    lambda: torch.is_autocast_enabled(factory(torch)),
                    lambda: reference_torch.is_autocast_enabled(
                        factory(reference_torch)
                    ),
                )
            with self.subTest(form="keyword_type", case=case):
                self.assert_error_matches(
                    lambda: torch.is_autocast_enabled(
                        device_type=factory(torch)
                    ),
                    lambda: reference_torch.is_autocast_enabled(
                        device_type=factory(reference_torch)
                    ),
                )

        calls = (
            lambda function: function(foo="cpu"),
            lambda function: function("cpu", "cuda"),
            lambda function: function("cpu", device_type="cuda"),
            lambda function: function(""),
            lambda function: function("cuda:"),
            lambda function: function("cpu:-1"),
            lambda function: function("cpu:01"),
            lambda function: function("cpu:2147483648"),
            lambda function: function("CUDA"),
        )
        for case, call in enumerate(calls):
            with self.subTest(form="binding_or_device", case=case):
                self.assert_error_matches(
                    lambda: call(torch.is_autocast_enabled),
                    lambda: call(reference_torch.is_autocast_enabled),
                )

    def test_cuda_visible_parity_does_not_initialize_cuda(self):
        script = r"""
import json
import torch as reference_torch
import torch_rs as torch

initialized_before = reference_torch.cuda.is_initialized()
actual = [
    torch.is_autocast_enabled(),
    torch.is_autocast_enabled("cpu"),
    torch.is_autocast_enabled("cuda"),
    torch.is_autocast_enabled(device_type="cpu"),
    torch.is_autocast_enabled(device_type="cuda"),
]
expected = [
    reference_torch.is_autocast_enabled(),
    reference_torch.is_autocast_enabled("cpu"),
    reference_torch.is_autocast_enabled("cuda"),
    reference_torch.is_autocast_enabled(device_type="cpu"),
    reference_torch.is_autocast_enabled(device_type="cuda"),
]
initialized_after_queries = reference_torch.cuda.is_initialized()
available = reference_torch.cuda.is_available()
payload = {
    "actual": actual,
    "expected": expected,
    "initialized_before": initialized_before,
    "initialized_after_queries": initialized_after_queries,
    "available": available,
    "torch_version": reference_torch.__version__,
    "cuda_runtime": reference_torch.version.cuda,
}
if available:
    payload["device_count"] = reference_torch.cuda.device_count()
    payload["device_name"] = reference_torch.cuda.get_device_name(0)
print(json.dumps(payload))
"""
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["actual"], [False] * 5)
        self.assertEqual(payload["expected"], [False] * 5)
        self.assertFalse(payload["initialized_before"])
        self.assertFalse(payload["initialized_after_queries"])
        if not payload["available"]:
            self.skipTest("reference PyTorch has no visible CUDA accelerator")
        self.assertGreaterEqual(payload["device_count"], 1)
        self.assertTrue(payload["device_name"])

    def test_contexts_setters_and_dtype_controls_remain_unsupported(self):
        self.assertFalse(hasattr(torch, "autocast"))
        self.assertTrue(hasattr(reference_torch, "autocast"))
        self.assertFalse(hasattr(torch, "amp"))
        self.assertTrue(hasattr(reference_torch, "amp"))
        self.assertFalse(hasattr(torch.cpu, "amp"))
        self.assertTrue(hasattr(reference_torch.cpu, "amp"))
        self.assertFalse(hasattr(torch, "cuda"))
        self.assertTrue(hasattr(reference_torch, "cuda"))

        for name in (
            "get_autocast_dtype",
            "get_autocast_cpu_dtype",
            "get_autocast_gpu_dtype",
            "is_autocast_cpu_enabled",
            "set_autocast_enabled",
            "set_autocast_cpu_enabled",
            "set_autocast_cpu_dtype",
            "set_autocast_dtype",
            "set_autocast_gpu_dtype",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertTrue(hasattr(reference_torch, name))


if __name__ == "__main__":
    unittest.main()

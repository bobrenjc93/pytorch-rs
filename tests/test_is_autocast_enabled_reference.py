import copy
import importlib
import inspect
import pickle
import threading
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


AUTOCAST_DEVICE_TYPES = (
    "cpu",
    "cuda",
    "ipu",
    "xpu",
    "maia",
    "xla",
    "mps",
    "hpu",
    "mtia",
    "privateuseone",
)

NON_AUTOCAST_DEVICE_TYPES = (
    "mkldnn",
    "opengl",
    "opencl",
    "ideep",
    "hip",
    "ve",
    "fpga",
    "lazy",
    "vulkan",
    "meta",
)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class IsAutocastEnabledReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "is_autocast_enabled differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(Exception) as actual_raised:
                actual_call()
            with self.assertRaises(Exception) as expected_raised:
                expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def test_default_and_recognized_device_results_match_pytorch_2_13(self):
        self.assertIs(torch.is_autocast_enabled(), False)
        self.assertIs(reference_torch.is_autocast_enabled(), False)
        values = []
        for device_type in AUTOCAST_DEVICE_TYPES:
            values.extend(
                (
                    device_type,
                    f"{device_type}:0",
                    device_type.encode(),
                )
            )
        values.extend((np.str_("cpu"), np.bytes_(b"cuda")))
        for value in values:
            with self.subTest(value=value):
                actual = torch.is_autocast_enabled(value)
                expected = reference_torch.is_autocast_enabled(value)
                self.assertIs(type(actual), type(expected))
                self.assertIs(actual, expected)
                self.assertIs(actual, False)
                self.assertIs(
                    torch.is_autocast_enabled(device_type=value),
                    reference_torch.is_autocast_enabled(device_type=value),
                )

    def thread_contract(self, module):
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=10)
                results[index] = (
                    module.is_autocast_enabled(),
                    module.is_autocast_enabled("cpu"),
                    module.is_autocast_enabled(device_type="cuda"),
                )
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        return results

    def test_default_thread_state_matches_pytorch_2_13(self):
        self.assertEqual(
            self.thread_contract(torch),
            self.thread_contract(reference_torch),
        )

    def test_builtin_contract_reload_copying_and_pickling_match(self):
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
        self.assertIsNone(actual.__doc__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        self.assertEqual(
            hasattr(actual, "__annotations__"),
            hasattr(expected, "__annotations__"),
        )
        self.assertEqual(repr(actual), repr(expected))
        self.assertIs(actual.__self__, torch._C)
        self.assertIs(expected.__self__, reference_torch._C)
        self.assertEqual(actual.__reduce__(), expected.__reduce__())
        self.assertIs(torch._C.is_autocast_enabled, actual)
        self.assertIs(reference_torch._C.is_autocast_enabled, expected)

        for function in (actual, expected):
            with self.assertRaises(ValueError):
                inspect.signature(function)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=function, protocol=protocol):
                    restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                    self.assertIs(restored, function)

        self.assertEqual(
            torch.__all__.count("is_autocast_enabled"),
            reference_torch.__all__.count("is_autocast_enabled"),
        )
        for module, function in ((torch, actual), (reference_torch, expected)):
            wildcard_namespace = {}
            exec(f"from {module.__name__} import *", wildcard_namespace)
            self.assertIs(wildcard_namespace["is_autocast_enabled"], function)

        for module, function in ((torch, actual), (reference_torch, expected)):
            native_module = module._C
            self.assertIs(importlib.reload(native_module), native_module)
            self.assertIs(native_module.is_autocast_enabled, function)
            self.assertIs(module.is_autocast_enabled, function)

    def test_invalid_devices_types_and_arities_match_pytorch_2_13(self):
        calls = (
            lambda module: module.is_autocast_enabled(None),
            lambda module: module.is_autocast_enabled(device_type=module.device("cpu")),
            lambda module: module.is_autocast_enabled(""),
            lambda module: module.is_autocast_enabled("banana"),
            lambda module: module.is_autocast_enabled("CPU"),
            lambda module: module.is_autocast_enabled("cpu:01"),
            lambda module: module.is_autocast_enabled("cpu:2147483648"),
            lambda module: module.is_autocast_enabled("cpu\x00tail"),
            lambda module: module.is_autocast_enabled("\udcff"),
            lambda module: module.is_autocast_enabled(b"\xff"),
            lambda module: module.is_autocast_enabled("cpu", "cuda"),
            lambda module: module.is_autocast_enabled(
                "cpu", device_type="cuda"
            ),
            lambda module: module.is_autocast_enabled(foo="cpu"),
            lambda module: module.is_autocast_enabled(
                device_type="cpu", foo=1
            ),
            lambda module: module.is_autocast_enabled(foo=1, bar="cpu"),
        )
        for call in calls:
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda: call(torch),
                    lambda: call(reference_torch),
                )

        for device_type in NON_AUTOCAST_DEVICE_TYPES:
            for value in (device_type, f"{device_type}:0"):
                with self.subTest(value=value):
                    self.assert_error_matches(
                        lambda value=value: torch.is_autocast_enabled(value),
                        lambda value=value: reference_torch.is_autocast_enabled(
                            value
                        ),
                    )

        self.assertIs(torch.is_autocast_enabled(**{}), False)
        self.assertIs(reference_torch.is_autocast_enabled(**{}), False)

    def test_reference_cpu_context_bounds_unsupported_enabled_transition(self):
        self.assertIs(torch.is_autocast_enabled("cpu"), False)
        self.assertIs(reference_torch.is_autocast_enabled("cpu"), False)
        with reference_torch.autocast(device_type="cpu"):
            self.assertIs(torch.is_autocast_enabled("cpu"), False)
            self.assertIs(reference_torch.is_autocast_enabled("cpu"), True)
            self.assertIs(reference_torch.is_autocast_enabled(), False)
        self.assertIs(reference_torch.is_autocast_enabled("cpu"), False)

    @unittest.skipUnless(
        reference_torch is not None and reference_torch.cuda.is_available(),
        "PyTorch CUDA is required for the CUDA autocast reference check",
    )
    def test_real_cuda_reference_bounds_unsupported_enabled_transition(self):
        self.assertGreaterEqual(reference_torch.cuda.device_count(), 1)
        self.assertTrue(reference_torch.cuda.get_device_name(0))
        self.assertIs(torch.is_autocast_enabled(), False)
        self.assertIs(torch.is_autocast_enabled("cuda"), False)
        self.assertIs(reference_torch.is_autocast_enabled(), False)
        self.assertIs(reference_torch.is_autocast_enabled("cuda"), False)

        values = reference_torch.ones(16, device="cuda")
        with reference_torch.autocast(device_type="cuda"):
            self.assertIs(torch.is_autocast_enabled(), False)
            self.assertIs(torch.is_autocast_enabled("cuda"), False)
            self.assertIs(reference_torch.is_autocast_enabled(), True)
            self.assertIs(reference_torch.is_autocast_enabled("cuda"), True)
            result = (values * 2).sum()
        reference_torch.cuda.synchronize(0)
        self.assertEqual(result.item(), 32.0)
        self.assertIs(reference_torch.is_autocast_enabled(), False)
        self.assertIs(reference_torch.is_autocast_enabled("cuda"), False)

    def test_contexts_dtype_controls_and_setters_remain_deliberately_absent(self):
        top_level_names = (
            "set_autocast_enabled",
            "get_autocast_dtype",
            "set_autocast_dtype",
            "get_autocast_gpu_dtype",
            "set_autocast_gpu_dtype",
            "get_autocast_cpu_dtype",
            "set_autocast_cpu_dtype",
            "is_autocast_cpu_enabled",
            "autocast",
            "amp",
        )
        native_names = (
            "set_autocast_enabled",
            "get_autocast_dtype",
            "set_autocast_dtype",
            "set_autocast_cache_enabled",
        )
        for name in top_level_names:
            with self.subTest(owner="torch", name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertTrue(hasattr(reference_torch, name))
        for name in native_names:
            with self.subTest(owner="torch._C", name=name):
                self.assertFalse(hasattr(torch._C, name))
                self.assertTrue(hasattr(reference_torch._C, name))
        self.assertFalse(hasattr(torch.cpu, "amp"))
        self.assertTrue(hasattr(reference_torch.cpu, "amp"))


if __name__ == "__main__":
    unittest.main()

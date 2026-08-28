import copy
import importlib
import inspect
import pickle
import pickletools
import sys
import threading
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class AutogradKinetoAvailableReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "kineto_available differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def threaded_contract(self, root):
        function = root.autograd.kineto_available
        native = root._C._autograd
        worker_count = 16
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=10)
                first = function()
                second = native.kineto_available()
                results[index] = (
                    type(first) is bool,
                    type(second) is bool,
                    first is second,
                    function is root.autograd.kineto_available,
                    function is native.kineto_available,
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
            thread.join(timeout=15)

        return [thread.is_alive() for thread in threads], errors, results

    def test_native_alias_and_callable_metadata_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs._C._autograd")
        expected_module = importlib.import_module("torch._C._autograd")
        actual = torch.autograd.kineto_available
        expected = reference_torch.autograd.kineto_available

        self.assertIs(actual, actual_module.kineto_available)
        self.assertIs(expected, expected_module.kineto_available)
        self.assertIs(type(actual_module), type(expected_module))
        self.assertEqual(
            actual_module.__name__.replace("torch_rs", "torch"),
            expected_module.__name__,
        )
        self.assertEqual(actual_module.__doc__, expected_module.__doc__)
        self.assertEqual(actual_module.__package__, expected_module.__package__)
        self.assertEqual(actual_module.__loader__, expected_module.__loader__)
        self.assertEqual(actual_module.__spec__, expected_module.__spec__)
        self.assertEqual(actual_module.__annotations__, expected_module.__annotations__)
        self.assertEqual(
            hasattr(actual_module, "__all__"),
            hasattr(expected_module, "__all__"),
        )

        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        for name in ("__annotations__", "__defaults__", "__kwdefaults__", "__dict__"):
            self.assertEqual(hasattr(actual, name), hasattr(expected, name))
        self.assertEqual(
            type(actual.__self__).__module__,
            type(expected.__self__).__module__,
        )
        self.assertEqual(
            type(actual.__self__).__name__,
            type(expected.__self__).__name__,
        )
        self.assertEqual(repr(type(actual.__self__)), repr(type(expected.__self__)))
        self.assertIs(inspect.getmodule(actual), actual_module)
        self.assertIs(inspect.getmodule(expected), expected_module)

        for function in (actual, expected):
            with self.assertRaises(ValueError):
                inspect.signature(function)

    def test_no_argument_errors_match_pytorch_2_13(self):
        actual = torch.autograd.kineto_available
        expected = reference_torch.autograd.kineto_available
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(enabled=True), lambda: expected(enabled=True)),
            (
                lambda: actual(None, enabled=True),
                lambda: expected(None, enabled=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        self.assertIs(actual(**{}), False)
        self.assertIs(type(expected(**{})), bool)

    def test_import_wildcard_copy_and_pickle_rules_match_pytorch_2_13(self):
        actual = torch.autograd.kineto_available
        expected = reference_torch.autograd.kineto_available

        for package_name, public, native, function in (
            ("torch_rs", torch.autograd, torch._C._autograd, actual),
            ("torch", reference_torch.autograd, reference_torch._C._autograd, expected),
        ):
            public_import = {}
            native_import = {}
            public_wildcard = {}
            native_wildcard = {}
            exec(
                f"from {package_name}.autograd import kineto_available",
                public_import,
            )
            exec(
                f"from {package_name}._C._autograd import kineto_available",
                native_import,
            )
            exec(f"from {package_name}.autograd import *", public_wildcard)
            exec(f"from {package_name}._C._autograd import *", native_wildcard)

            self.assertIs(public_import["kineto_available"], function)
            self.assertIs(native_import["kineto_available"], function)
            self.assertNotIn("kineto_available", public.__all__)
            self.assertNotIn("kineto_available", public_wildcard)
            self.assertIs(native_wildcard["kineto_available"], function)
            self.assertIs(function, native.kineto_available)

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_thread_and_capability_shapes_match_pytorch_2_13(self):
        actual = self.threaded_contract(torch)
        expected = self.threaded_contract(reference_torch)

        self.assertEqual(actual, expected)
        self.assertIs(torch.autograd.kineto_available(), False)
        self.assertIs(type(reference_torch.autograd.kineto_available()), bool)

    def test_torch_rs_reloads_preserve_the_native_alias(self):
        public = torch.autograd
        native = torch._C._autograd
        function = public.kineto_available

        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.autograd, public)
        self.assertIs(torch._C._autograd, native)
        self.assertIs(public.kineto_available, function)

        self.assertIs(importlib.reload(public), public)
        self.assertIs(public.kineto_available, function)
        self.assertIs(native.kineto_available, function)

    def test_profiler_surface_stays_outside_the_supported_subset(self):
        self.assertFalse(hasattr(torch.autograd, "profiler"))
        self.assertFalse(hasattr(torch.autograd, "ProfilerActivity"))
        self.assertFalse(hasattr(torch.autograd, "ProfilerEvent"))
        self.assertFalse(hasattr(torch._C._autograd, "_supported_activities"))

        self.assertTrue(hasattr(reference_torch.autograd, "profiler"))
        self.assertTrue(hasattr(reference_torch.autograd, "ProfilerActivity"))
        self.assertTrue(hasattr(reference_torch.autograd, "ProfilerEvent"))
        self.assertTrue(
            hasattr(reference_torch._C._autograd, "_supported_activities")
        )


if __name__ == "__main__":
    unittest.main()

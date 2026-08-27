import copy
import importlib
import inspect
import pickle
import pickletools
import re
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
class KinetoAvailableReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "autograd.kineto_available differentials require pinned PyTorch 2.13.0"
            )

    def normalize(self, value):
        return (
            value.replace("torch_rs.torch_rs", "torch._C")
            .replace("torch_rs", "torch")
        )

    def normalized_repr(self, value):
        return re.sub(r"0x[0-9a-fA-F]+", "0x...", self.normalize(repr(value)))

    def signature_outcome(self, function):
        try:
            return "return", str(inspect.signature(function))
        except BaseException as error:
            return "error", type(error).__name__, self.normalize(
                re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error))
            )

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = self.normalize(argument)
            shape.append((opcode.name, argument))
        return shape

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def test_native_callable_metadata_matches_pytorch_2_13(self):
        actual_module = torch._C._autograd
        expected_module = reference_torch._C._autograd
        actual = torch.autograd.kineto_available
        expected = reference_torch.autograd.kineto_available

        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(self.normalize(actual.__module__), expected.__module__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        self.assertEqual(
            hasattr(actual, "__annotations__"),
            hasattr(expected, "__annotations__"),
        )
        self.assertEqual(hasattr(actual, "__dict__"), hasattr(expected, "__dict__"))
        self.assertEqual(self.normalized_repr(actual), self.normalized_repr(expected))
        self.assertEqual(
            type(actual.__self__).__module__, type(expected.__self__).__module__
        )
        self.assertEqual(type(actual.__self__).__name__, type(expected.__self__).__name__)
        self.assertEqual(
            type(actual.__self__).__qualname__,
            type(expected.__self__).__qualname__,
        )
        self.assertIsNot(actual.__self__, actual_module)
        self.assertIsNot(expected.__self__, expected_module)
        self.assertIs(inspect.getmodule(actual), actual_module)
        self.assertIs(inspect.getmodule(expected), expected_module)
        self.assertEqual(self.signature_outcome(actual), self.signature_outcome(expected))
        self.assertEqual(inspect.get_annotations(actual), inspect.get_annotations(expected))

        for function in (actual, expected):
            reduction = function.__reduce__()
            self.assertIs(reduction[0], getattr)
            self.assertIs(reduction[1][0], function.__self__)
            self.assertEqual(reduction[1][1], "kineto_available")
            record_reduction = function.__self__.__reduce_ex__(
                pickle.HIGHEST_PROTOCOL
            )
            self.assertIs(record_reduction[0], eval)
            self.assertEqual(
                self.normalize(record_reduction[1][0]),
                "__import__('importlib').import_module('torch._C._autograd')",
            )

        self.assertEqual(actual_module.__doc__, expected_module.__doc__)
        self.assertEqual(actual_module.__package__, expected_module.__package__)
        self.assertEqual(actual_module.__loader__, expected_module.__loader__)
        self.assertEqual(actual_module.__spec__, expected_module.__spec__)
        self.assertFalse(hasattr(actual_module, "__all__"))
        self.assertFalse(hasattr(expected_module, "__all__"))

    def test_supported_false_result_bounds_reference_kineto_build(self):
        actual = torch.autograd.kineto_available()
        expected = reference_torch.autograd.kineto_available()

        self.assertIs(type(actual), bool)
        self.assertIs(type(expected), bool)
        self.assertIs(actual, False)
        self.assertIs(actual, torch._C._autograd.kineto_available())

    def test_import_and_non_wildcard_behavior_matches_pytorch_2_13(self):
        actual = torch.autograd.kineto_available
        expected = reference_torch.autograd.kineto_available

        for module, native, function in (
            (torch.autograd, torch._C._autograd, actual),
            (reference_torch.autograd, reference_torch._C._autograd, expected),
        ):
            self.assertIs(module.kineto_available, native.kineto_available)
            self.assertNotIn("kineto_available", module.__all__)
            public_import = {}
            public_wildcard = {}
            native_import = {}
            native_wildcard = {}
            exec(
                f"from {module.__name__} import kineto_available",
                public_import,
            )
            exec(f"from {module.__name__} import *", public_wildcard)
            exec(
                f"from {native.__name__} import kineto_available",
                native_import,
            )
            exec(f"from {native.__name__} import *", native_wildcard)
            self.assertIs(public_import["kineto_available"], function)
            self.assertNotIn("kineto_available", public_wildcard)
            self.assertIs(native_import["kineto_available"], function)
            self.assertIs(native_wildcard["kineto_available"], function)

        self.assertFalse(hasattr(torch, "kineto_available"))
        self.assertFalse(hasattr(reference_torch, "kineto_available"))

    def test_argument_errors_match_pytorch_2_13(self):
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

    def test_copying_and_pickling_match_pytorch_2_13(self):
        actual = torch.autograd.kineto_available
        expected = reference_torch.autograd.kineto_available

        for function in (actual, expected):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def threaded_contract(self, root):
        function = root.autograd.kineto_available
        barrier = threading.Barrier(17)
        results = [None] * 16
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=10)
                values = tuple(function() for _ in range(1000))
                results[index] = (
                    function is root.autograd.kineto_available,
                    function is root._C._autograd.kineto_available,
                    all(type(value) is bool for value in values),
                    len(set(values)) == 1,
                )
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(16)]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=10)
        for thread in threads:
            thread.join(timeout=10)
        return [thread.is_alive() for thread in threads], errors, results

    def test_thread_behavior_matches_pytorch_2_13(self):
        actual = self.threaded_contract(torch)
        expected = self.threaded_contract(reference_torch)
        self.assertEqual(actual, expected)
        self.assertEqual(actual, ([False] * 16, [], [(True, True, True, True)] * 16))

    def native_reload_contract(self, root):
        native = root._C
        native_autograd = native._autograd
        function = root.autograd.kineto_available
        reloaded = importlib.reload(native)
        return (
            reloaded is native,
            native._autograd is native_autograd,
            root.autograd.kineto_available is function,
            native._autograd.kineto_available is function,
            pickle.loads(pickle.dumps(function)) is function,
        )

    def native_submodule_reload_error(self, root):
        try:
            importlib.reload(root._C._autograd)
        except BaseException as error:
            return type(error).__name__, self.normalize(str(error))
        self.fail("native autograd submodule unexpectedly reloaded")

    def test_native_reload_identity_rules_match_pytorch_2_13(self):
        self.assertEqual(
            self.native_reload_contract(torch),
            self.native_reload_contract(reference_torch),
        )
        self.assertEqual(
            self.native_submodule_reload_error(torch),
            self.native_submodule_reload_error(reference_torch),
        )

    def test_profiler_surface_remains_deliberately_unsupported(self):
        unsupported = (
            "ProfilerActivity",
            "ProfilerConfig",
            "ProfilerEvent",
            "ProfilerState",
            "_KinetoEvent",
            "_ProfilerResult",
            "_disable_profiler",
            "_enable_profiler",
            "_kineto_step",
            "_supported_activities",
            "profiler",
            "profiler_legacy",
        )
        for name in unsupported:
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.autograd, name))
                self.assertFalse(hasattr(torch._C._autograd, name))
                self.assertTrue(
                    hasattr(reference_torch.autograd, name)
                    or hasattr(reference_torch._C._autograd, name)
                )

        self.assertFalse(hasattr(torch, "profiler"))
        self.assertTrue(hasattr(reference_torch, "profiler"))
        self.assertEqual(
            {name for name in vars(torch._C._autograd) if not name.startswith("_")},
            {"kineto_available"},
        )


if __name__ == "__main__":
    unittest.main()

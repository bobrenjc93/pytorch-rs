import copy
import importlib
import inspect
import os
import pickle
import pickletools
import re
import sys
import threading
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CpuCapabilityReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "CPU capability differentials require pinned PyTorch 2.13.0"
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

    def thread_contract(self, root):
        function = root.backends.cpu.get_cpu_capability
        native_function = root._C._get_cpu_capability
        baseline = function()
        native_baseline = native_function()
        worker_count = 16
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=5)
                results[index] = (
                    function() == baseline,
                    native_function() == native_baseline,
                    function() == native_function(),
                    type(function()) is str,
                    type(native_function()) is str,
                    function is root.backends.cpu.get_cpu_capability,
                    native_function is root._C._get_cpu_capability,
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

        return (
            [thread.is_alive() for thread in threads],
            errors,
            results,
        )

    def environment_contract(self, root):
        function = root.backends.cpu.get_cpu_capability
        native_function = root._C._get_cpu_capability
        baseline = function()
        observations = []
        environments = (
            {},
            {"ATEN_CPU_CAPABILITY": "default"},
            {"ATEN_CPU_CAPABILITY": "avx2"},
            {
                "ATEN_CPU_CAPABILITY": "avx512",
                "MKL_DEBUG_CPU_TYPE": "5",
                "OMP_NUM_THREADS": "64",
                "OPENBLAS_CORETYPE": "SAPPHIRERAPIDS",
            },
        )
        for environment in environments:
            with mock.patch.dict(os.environ, environment, clear=True):
                observations.append(
                    (
                        function() == baseline,
                        native_function() == baseline,
                        type(function()) is str,
                        type(native_function()) is str,
                    )
                )
        return observations

    def test_value_and_invariance_use_each_engines_native_dispatch_level(self):
        self.assertEqual(torch.backends.cpu.get_cpu_capability(), "DEFAULT")
        self.assertEqual(torch._C._get_cpu_capability(), "DEFAULT")
        self.assertIn(
            reference_torch.backends.cpu.get_cpu_capability(),
            {"DEFAULT", "VSX", "Z VECTOR", "NO AVX", "AVX2", "AVX512", "SVE256"},
        )
        self.assertEqual(
            reference_torch.backends.cpu.get_cpu_capability(),
            reference_torch._C._get_cpu_capability(),
        )
        self.assertEqual(
            self.environment_contract(torch),
            self.environment_contract(reference_torch),
        )
        self.assertEqual(
            self.environment_contract(torch),
            [(True, True, True, True)] * 4,
        )

    def test_thread_behavior_matches_pytorch_2_13(self):
        actual = self.thread_contract(torch)
        expected = self.thread_contract(reference_torch)

        self.assertEqual(actual, expected)
        self.assertEqual(
            actual,
            (
                [False] * 16,
                [],
                [(True, True, True, True, True, True, True)] * 16,
            ),
        )

    def test_signature_documentation_and_identity_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.backends.cpu")
        expected_module = importlib.import_module("torch.backends.cpu")
        actual = actual_module.get_cpu_capability
        expected = expected_module.get_cpu_capability

        self.assertIs(torch.backends.cpu, actual_module)
        self.assertIs(reference_torch.backends.cpu, expected_module)
        self.assertIs(sys.modules[actual_module.__name__], actual_module)
        self.assertIs(sys.modules[expected_module.__name__], expected_module)
        self.assertIs(type(actual_module), types.ModuleType)
        self.assertIs(type(expected_module), types.ModuleType)
        self.assertEqual(actual_module.__doc__, expected_module.__doc__)
        self.assertEqual(actual_module.__all__, expected_module.__all__)
        self.assertEqual(actual_module.__annotations__, expected_module.__annotations__)
        self.assertEqual(
            {name for name in vars(actual_module) if not name.startswith("_")},
            {name for name in vars(expected_module) if not name.startswith("_")},
        )
        self.assertIs(actual_module.torch, torch)
        self.assertIs(expected_module.torch, reference_torch)

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(inspect.get_annotations(actual), inspect.get_annotations(expected))
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_module)
        self.assertIs(inspect.getmodule(expected), expected_module)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(actual.__code__.co_names, expected.__code__.co_names)
        self.assertEqual(actual.__code__.co_freevars, expected.__code__.co_freevars)
        self.assertEqual(actual.__code__.co_cellvars, expected.__code__.co_cellvars)

    def test_imports_copying_and_pickling_match_pytorch_2_13(self):
        actual_backends = importlib.import_module("torch_rs.backends")
        expected_backends = importlib.import_module("torch.backends")
        actual_module = importlib.import_module("torch_rs.backends.cpu")
        expected_module = importlib.import_module("torch.backends.cpu")
        actual = actual_module.get_cpu_capability
        expected = expected_module.get_cpu_capability
        supported_backends = {
            "cpu",
            "cuda",
            "cudnn",
            "mha",
            "mkl",
            "nnpack",
            "openmp",
        }

        for package_name, module, function in (
            ("torch_rs", actual_module, actual),
            ("torch", expected_module, expected),
        ):
            backend_import = {}
            function_import = {}
            child_wildcard = {}
            exec(f"from {package_name}.backends import cpu", backend_import)
            exec(
                f"from {package_name}.backends.cpu import get_cpu_capability",
                function_import,
            )
            exec(f"from {package_name}.backends.cpu import *", child_wildcard)
            self.assertIs(backend_import["cpu"], module)
            self.assertIs(function_import["get_cpu_capability"], function)
            self.assertIs(child_wildcard["get_cpu_capability"], function)
            self.assertEqual(
                {name for name in child_wildcard if not name.startswith("__")},
                {"get_cpu_capability"},
            )

        actual_parent_wildcard = {}
        expected_parent_wildcard = {}
        exec("from torch_rs.backends import *", actual_parent_wildcard)
        exec("from torch.backends import *", expected_parent_wildcard)
        self.assertEqual(
            {
                name
                for name in actual_parent_wildcard
                if not name.startswith("__")
            },
            {
                name
                for name in expected_parent_wildcard
                if name in supported_backends
            },
        )
        self.assertIs(actual_backends.cpu, actual_module)
        self.assertIs(expected_backends.cpu, expected_module)

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

    def reload_contract(self, root):
        parent = root.backends
        module = parent.cpu
        old_function = module.get_cpu_capability
        namespace = module.__dict__
        reloaded = importlib.reload(module)
        new_function = module.get_cpu_capability

        try:
            pickle.dumps(old_function)
        except Exception as error:
            stale_pickle_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                    "torch_rs", "torch"
                ),
            )
        else:
            self.fail("a stale CPU capability query remained pickleable")

        return (
            reloaded is module,
            module.__dict__ is namespace,
            parent.cpu is module,
            sys.modules[module.__name__] is module,
            old_function is not new_function,
            new_function() == root._C._get_cpu_capability(),
            type(new_function()) is str,
            copy.copy(new_function) is new_function,
            copy.deepcopy(new_function) is new_function,
            pickle.loads(pickle.dumps(new_function)) is new_function,
            stale_pickle_error,
        )

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )
        actual = torch.backends.cpu.get_cpu_capability
        expected = reference_torch.backends.cpu.get_cpu_capability
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.backends.cpu.get_cpu_capability
        expected = reference_torch.backends.cpu.get_cpu_capability
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (
                lambda: actual(device="cpu"),
                lambda: expected(device="cpu"),
            ),
            (
                lambda: actual(None, device="cpu"),
                lambda: expected(None, device="cpu"),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()

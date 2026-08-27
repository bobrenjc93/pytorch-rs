import copy
import importlib
import inspect
import pickle
import pickletools
import re
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


DOCUMENTED_CAPABILITIES = {
    "DEFAULT",
    "VSX",
    "Z VECTOR",
    "NO AVX",
    "AVX2",
    "AVX512",
    "SVE256",
}


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class BackendsCpuCapabilityReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.cpu.get_cpu_capability differentials require pinned "
                "PyTorch 2.13.0"
            )

    def normalize(self, value):
        return str(value).replace("torch_rs.torch_rs", "torch._C").replace(
            "torch_rs", "torch"
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
                argument = self.normalize(argument)
            shape.append((opcode.name, argument))
        return shape

    def threaded_contract(self, root):
        function = root.backends.cpu.get_cpu_capability
        native_function = root._C._get_cpu_capability
        worker_count = 16
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=5)
                results[index] = (
                    function(),
                    native_function(),
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

        return [thread.is_alive() for thread in threads], errors, results

    def test_signature_documentation_and_identity_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.backends.cpu")
        expected_module = importlib.import_module("torch.backends.cpu")
        actual = actual_module.get_cpu_capability
        expected = expected_module.get_cpu_capability

        self.assertEqual(actual_module.__doc__, expected_module.__doc__)
        self.assertEqual(actual_module.__all__, expected_module.__all__)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(inspect.get_annotations(actual), inspect.get_annotations(expected))
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
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

    def test_unspecialized_value_and_thread_behavior_are_truthful(self):
        actual = self.threaded_contract(torch)
        expected = self.threaded_contract(reference_torch)

        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        self.assertEqual(actual[0], [False] * 16)
        self.assertEqual(actual[1], [])
        self.assertEqual(
            actual[2],
            [("DEFAULT", "DEFAULT", True, True)] * 16,
        )
        expected_capability = reference_torch.backends.cpu.get_cpu_capability()
        self.assertIn(expected_capability, DOCUMENTED_CAPABILITIES)
        self.assertEqual(
            expected[2],
            [(expected_capability, expected_capability, True, True)] * 16,
        )

    def test_native_query_contract_matches_pytorch_2_13(self):
        actual = torch._C._get_cpu_capability
        expected = reference_torch._C._get_cpu_capability

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
        self.assertEqual(repr(actual), repr(expected))
        self.assertIs(actual.__self__, torch._C)
        self.assertIs(expected.__self__, reference_torch._C)
        self.assertEqual(actual.__reduce__(), expected.__reduce__())
        for function in (actual, expected):
            assert_no_argument_signature(self, function, "()")
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

    def test_imports_copying_and_pickling_match_pytorch_2_13(self):
        actual_backends = importlib.import_module("torch_rs.backends")
        expected_backends = importlib.import_module("torch.backends")
        actual_module = importlib.import_module("torch_rs.backends.cpu")
        expected_module = importlib.import_module("torch.backends.cpu")
        actual = actual_module.get_cpu_capability
        expected = expected_module.get_cpu_capability

        self.assertIs(torch.backends, actual_backends)
        self.assertIs(reference_torch.backends, expected_backends)
        self.assertIs(actual_backends.cpu, actual_module)
        self.assertIs(expected_backends.cpu, expected_module)
        self.assertEqual(
            {name for name in vars(actual_module) if not name.startswith("_")},
            {name for name in vars(expected_module) if not name.startswith("_")},
        )

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
            self.assertEqual(
                {name for name in child_wildcard if not name.startswith("__")},
                {"get_cpu_capability"},
            )

        actual_parent_wildcard = {}
        expected_parent_wildcard = {}
        exec("from torch_rs.backends import *", actual_parent_wildcard)
        exec("from torch.backends import *", expected_parent_wildcard)
        supported_backends = {
            "cpu",
            "cuda",
            "cudnn",
            "mha",
            "mkl",
            "nnpack",
            "openmp",
        }
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

        for root in (torch, reference_torch):
            namespace = {}
            exec(f"from {root.__name__} import *", namespace)
            self.assertNotIn("backends", namespace)
            self.assertNotIn("get_cpu_capability", namespace)
            self.assertFalse(hasattr(root, "get_cpu_capability"))

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
                lambda: actual(capability="avx2"),
                lambda: expected(capability="avx2"),
            ),
            (
                lambda: actual(None, capability="avx2"),
                lambda: expected(None, capability="avx2"),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        actual_native = torch._C._get_cpu_capability
        expected_native = reference_torch._C._get_cpu_capability
        native_cases = (
            (lambda: actual_native(None), lambda: expected_native(None)),
            (
                lambda: actual_native(None, None),
                lambda: expected_native(None, None),
            ),
            (
                lambda: actual_native(capability="avx2"),
                lambda: expected_native(capability="avx2"),
            ),
            (
                lambda: actual_native(None, capability="avx2"),
                lambda: expected_native(None, capability="avx2"),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(native_cases):
            with self.subTest(native_case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_only_cpu_capability_introspection_is_added(self):
        actual = torch.backends.cpu
        expected = reference_torch.backends.cpu
        actual_public = {
            name for name in vars(actual) if not name.startswith("_")
        }
        expected_public = {
            name for name in vars(expected) if not name.startswith("_")
        }
        self.assertEqual(actual_public, {"get_cpu_capability", "torch"})
        self.assertEqual(actual_public, expected_public)
        self.assertFalse(hasattr(torch.backends.cpu, "set_cpu_capability"))
        self.assertFalse(hasattr(reference_torch.backends.cpu, "set_cpu_capability"))


if __name__ == "__main__":
    unittest.main()

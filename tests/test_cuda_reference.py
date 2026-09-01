import copy
import importlib
import inspect
import pickle
import pickletools
import subprocess
import sys
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudaReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "cuda probe differentials require pinned PyTorch 2.13.0"
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

    def test_is_initialized_signature_and_errors_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.cuda")
        expected_module = importlib.import_module("torch.cuda")
        actual = actual_module.is_initialized
        expected = expected_module.is_initialized

        self.assertIs(torch.cuda, actual_module)
        self.assertIs(reference_torch.cuda, expected_module)
        self.assertIs(sys.modules["torch_rs.cuda"], actual_module)
        self.assertIs(sys.modules["torch.cuda"], expected_module)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
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

    def test_imports_wildcards_copy_pickle_and_reload_match_supported_scope(self):
        actual_cuda = torch.cuda
        expected_cuda = reference_torch.cuda
        supported = {"device_count", "is_available", "is_initialized"}

        self.assertEqual(
            actual_cuda.__all__,
            [name for name in expected_cuda.__all__ if name in supported],
        )
        for name in ("cuda", *sorted(supported)):
            with self.subTest(top_level_export=name):
                self.assertEqual(
                    torch.__all__.count(name), reference_torch.__all__.count(name)
                )

        actual_package_import = {}
        expected_package_import = {}
        exec("from torch_rs import cuda", actual_package_import)
        exec("from torch import cuda", expected_package_import)
        self.assertIs(actual_package_import["cuda"], actual_cuda)
        self.assertIs(expected_package_import["cuda"], expected_cuda)

        actual_direct_import = {}
        expected_direct_import = {}
        exec(
            "from torch_rs.cuda import device_count, is_available, is_initialized",
            actual_direct_import,
        )
        exec(
            "from torch.cuda import device_count, is_available, is_initialized",
            expected_direct_import,
        )
        for name in supported:
            with self.subTest(direct_import=name):
                self.assertIs(actual_direct_import[name], getattr(actual_cuda, name))
                self.assertIs(expected_direct_import[name], getattr(expected_cuda, name))

        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.cuda import *", actual_wildcard)
        exec("from torch.cuda import *", expected_wildcard)
        self.assertEqual(
            {name for name in actual_wildcard if not name.startswith("__")},
            supported,
        )
        for name in supported:
            with self.subTest(wildcard=name):
                self.assertIs(actual_wildcard[name], getattr(actual_cuda, name))
                self.assertIs(expected_wildcard[name], getattr(expected_cuda, name))

        for module, function in (
            (torch, actual_cuda.is_initialized),
            (reference_torch, expected_cuda.is_initialized),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("cuda", namespace)
            self.assertNotIn("is_initialized", namespace)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(actual_cuda.is_initialized, protocol)),
                    actual_cuda.is_initialized,
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(expected_cuda.is_initialized, protocol)),
                    expected_cuda.is_initialized,
                )
                self.assertEqual(
                    self.pickle_shape(actual_cuda.is_initialized, protocol),
                    self.pickle_shape(expected_cuda.is_initialized, protocol),
                )

        actual_old = actual_cuda.is_initialized
        expected_old = expected_cuda.is_initialized
        self.assertIs(importlib.reload(actual_cuda), actual_cuda)
        self.assertIs(importlib.reload(expected_cuda), expected_cuda)
        self.assertIsNot(actual_cuda.is_initialized, actual_old)
        self.assertIsNot(expected_cuda.is_initialized, expected_old)
        self.assertIs(torch.cuda, actual_cuda)
        self.assertIs(reference_torch.cuda, expected_cuda)
        self.assertIs(sys.modules["torch_rs.cuda"], actual_cuda)
        self.assertIs(sys.modules["torch.cuda"], expected_cuda)
        self.assertIs(actual_cuda.is_initialized(), False)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(actual_old)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(expected_old)

    def test_cpu_build_probe_values_are_static_without_changing_cuda_runtime_state(self):
        cuda = torch.cuda
        self.assertIs(cuda.is_initialized(), False)
        self.assertIs(cuda.is_available(), False)
        self.assertEqual(cuda.device_count(), 0)
        self.assertIs(cuda.is_initialized(), False)
        self.assertFalse(hasattr(cuda, "_initialized"))
        self.assertFalse(hasattr(cuda, "_cached_device_count"))

        script = r"""
import torch

assert torch.cuda.is_initialized() is False
assert type(torch.cuda.is_available()) is bool
assert type(torch.cuda.device_count()) is int
assert torch.cuda.is_initialized() is False
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()

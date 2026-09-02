import copy
import importlib
import inspect
import os
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

    def test_max_memory_allocated_signature_and_errors_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.cuda")
        expected_module = importlib.import_module("torch.cuda")
        actual = actual_module.max_memory_allocated
        expected = expected_module.max_memory_allocated

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
        for function in (actual, expected):
            with self.subTest(type_hints=function.__module__):
                with self.assertRaises(NameError):
                    typing.get_type_hints(function)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(
            inspect.getmodule(actual), sys.modules["torch_rs.cuda.memory"]
        )
        self.assertIs(inspect.getmodule(expected), reference_torch.cuda.memory)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

        cases = (
            (lambda: actual(None, None), lambda: expected(None, None)),
            (
                lambda: actual(unexpected=True),
                lambda: expected(unexpected=True),
            ),
            (
                lambda: actual(None, device=None),
                lambda: expected(None, device=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_imports_wildcards_copy_pickle_and_reload_match_supported_scope(self):
        actual_cuda = torch.cuda
        expected_cuda = reference_torch.cuda
        supported = {
            "device_count",
            "is_available",
            "is_initialized",
            "max_memory_allocated",
        }

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
            "from torch_rs.cuda import device_count, is_available, "
            "is_initialized, max_memory_allocated",
            actual_direct_import,
        )
        exec(
            "from torch.cuda import device_count, is_available, "
            "is_initialized, max_memory_allocated",
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

        for module, functions in (
            (
                torch,
                (actual_cuda.is_initialized, actual_cuda.max_memory_allocated),
            ),
            (
                reference_torch,
                (
                    expected_cuda.is_initialized,
                    expected_cuda.max_memory_allocated,
                ),
            ),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("cuda", namespace)
            for name, function in (
                (function.__name__, function) for function in functions
            ):
                with self.subTest(package=module.__name__, name=name):
                    self.assertNotIn(name, namespace)
                    self.assertIs(copy.copy(function), function)
                    self.assertIs(copy.deepcopy(function), function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                for actual, expected in (
                    (
                        actual_cuda.is_initialized,
                        expected_cuda.is_initialized,
                    ),
                    (
                        actual_cuda.max_memory_allocated,
                        expected_cuda.max_memory_allocated,
                    ),
                ):
                    with self.subTest(name=actual.__name__):
                        self.assertIs(
                            pickle.loads(pickle.dumps(actual, protocol)),
                            actual,
                        )
                        self.assertIs(
                            pickle.loads(pickle.dumps(expected, protocol)),
                            expected,
                        )
                        self.assertEqual(
                            self.pickle_shape(actual, protocol),
                            self.pickle_shape(expected, protocol),
                        )

        actual_old = actual_cuda.is_initialized
        expected_old = expected_cuda.is_initialized
        actual_old_max_memory_allocated = actual_cuda.max_memory_allocated
        expected_old_max_memory_allocated = expected_cuda.max_memory_allocated
        self.assertIs(importlib.reload(actual_cuda), actual_cuda)
        self.assertIs(importlib.reload(expected_cuda), expected_cuda)
        self.assertIsNot(actual_cuda.is_initialized, actual_old)
        self.assertIsNot(expected_cuda.is_initialized, expected_old)
        self.assertIs(
            actual_cuda.max_memory_allocated,
            actual_old_max_memory_allocated,
        )
        self.assertIs(
            expected_cuda.max_memory_allocated,
            expected_old_max_memory_allocated,
        )
        self.assertIs(torch.cuda, actual_cuda)
        self.assertIs(reference_torch.cuda, expected_cuda)
        self.assertIs(sys.modules["torch_rs.cuda"], actual_cuda)
        self.assertIs(sys.modules["torch.cuda"], expected_cuda)
        self.assertIs(actual_cuda.is_initialized(), False)
        self.assertEqual(actual_cuda.max_memory_allocated(), 0)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(actual_old)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(expected_old)
        self.assertIs(
            pickle.loads(pickle.dumps(actual_old_max_memory_allocated)),
            actual_cuda.max_memory_allocated,
        )
        self.assertIs(
            pickle.loads(pickle.dumps(expected_old_max_memory_allocated)),
            expected_cuda.max_memory_allocated,
        )

    def test_cpu_build_probe_values_are_static_without_changing_cuda_runtime_state(self):
        cuda = torch.cuda
        self.assertIs(cuda.is_initialized(), False)
        self.assertIs(cuda.is_available(), False)
        self.assertEqual(cuda.device_count(), 0)
        self.assertEqual(cuda.max_memory_allocated(), 0)
        self.assertIs(cuda.is_initialized(), False)
        self.assertFalse(hasattr(cuda, "_initialized"))
        self.assertFalse(hasattr(cuda, "_cached_device_count"))

        script = r"""
import torch

assert torch.cuda.is_initialized() is False
assert torch.cuda.is_available() is False
assert torch.cuda.device_count() == 0
assert torch.cuda.is_initialized() is False
class ExplodingDeviceToken:
    def __bool__(self):
        raise AssertionError("device token truth value was inspected")
    def __index__(self):
        raise AssertionError("device token index was inspected")
    def __int__(self):
        raise AssertionError("device token integer value was inspected")
    def __str__(self):
        raise AssertionError("device token string value was inspected")
tokens = (
    None,
    0,
    -1,
    True,
    False,
    1.5,
    "",
    "cpu",
    "cpu:0",
    "cuda",
    "cuda:0",
    "banana",
    torch.device("cpu"),
    object(),
    [],
    {},
    ExplodingDeviceToken(),
)
for token in tokens:
    result = torch.cuda.max_memory_allocated(token)
    assert type(result) is int and result == 0
    result = torch.cuda.max_memory_allocated(device=token)
    assert type(result) is int and result == 0
    assert torch.cuda.is_initialized() is False
assert torch.cuda.is_available() is False
assert torch.cuda.device_count() == 0
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
            },
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()

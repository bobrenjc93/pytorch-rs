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


SUPPORTED = {"device_count", "is_available", "is_initialized", "memory_reserved"}


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

    def type_hints_outcome(self, function):
        try:
            return ("return", typing.get_type_hints(function))
        except Exception as error:
            return ("raise", type(error), str(error), error.args)

    def test_supported_signature_and_errors_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.cuda")
        expected_module = importlib.import_module("torch.cuda")
        actual_memory_module = sys.modules["torch_rs.cuda.memory"]
        expected_memory_module = importlib.import_module("torch.cuda.memory")

        self.assertIs(torch.cuda, actual_module)
        self.assertIs(reference_torch.cuda, expected_module)
        self.assertIs(sys.modules["torch_rs.cuda"], actual_module)
        self.assertIs(sys.modules["torch.cuda"], expected_module)
        self.assertIs(
            actual_module.memory_reserved,
            actual_memory_module.memory_reserved,
        )
        self.assertIs(
            expected_module.memory_reserved,
            expected_memory_module.memory_reserved,
        )

        for name in ("is_initialized", "memory_reserved"):
            with self.subTest(name=name):
                actual = getattr(actual_module, name)
                expected = getattr(expected_module, name)
                actual_owner = (
                    actual_memory_module if name == "memory_reserved" else actual_module
                )
                expected_owner = (
                    expected_memory_module if name == "memory_reserved" else expected_module
                )

                self.assertIs(type(actual), types.FunctionType)
                self.assertIs(type(expected), types.FunctionType)
                self.assertEqual(
                    str(inspect.signature(actual)), str(inspect.signature(expected))
                )
                self.assertEqual(actual.__annotations__, expected.__annotations__)
                self.assertEqual(
                    self.type_hints_outcome(actual),
                    self.type_hints_outcome(expected),
                )
                self.assertEqual(actual.__name__, expected.__name__)
                self.assertEqual(actual.__qualname__, expected.__qualname__)
                self.assertEqual(
                    actual.__module__.replace("torch_rs", "torch"),
                    expected.__module__,
                )
                self.assertIs(inspect.getmodule(actual), actual_owner)
                self.assertIs(inspect.getmodule(expected), expected_owner)
                self.assertEqual(actual.__doc__, expected.__doc__)
                self.assertEqual(actual.__defaults__, expected.__defaults__)
                self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
                self.assertEqual(actual.__dict__, expected.__dict__)
                self.assertEqual(
                    hasattr(actual, "__text_signature__"),
                    hasattr(expected, "__text_signature__"),
                )

        is_initialized_cases = (
            (
                lambda: actual_module.is_initialized(None),
                lambda: expected_module.is_initialized(None),
            ),
            (
                lambda: actual_module.is_initialized(None, None),
                lambda: expected_module.is_initialized(None, None),
            ),
            (
                lambda: actual_module.is_initialized(enabled=True),
                lambda: expected_module.is_initialized(enabled=True),
            ),
            (
                lambda: actual_module.is_initialized(None, enabled=True),
                lambda: expected_module.is_initialized(None, enabled=True),
            ),
        )
        memory_reserved_cases = (
            (
                lambda: actual_module.memory_reserved(None, None),
                lambda: expected_module.memory_reserved(None, None),
            ),
            (
                lambda: actual_module.memory_reserved(device_index=0),
                lambda: expected_module.memory_reserved(device_index=0),
            ),
            (
                lambda: actual_module.memory_reserved(None, device=0),
                lambda: expected_module.memory_reserved(None, device=0),
            ),
            (
                lambda: actual_module.memory_reserved(unexpected=True),
                lambda: expected_module.memory_reserved(unexpected=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(
            is_initialized_cases + memory_reserved_cases
        ):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_imports_wildcards_copy_pickle_and_reload_match_supported_scope(self):
        actual_cuda = torch.cuda
        expected_cuda = reference_torch.cuda

        self.assertEqual(
            actual_cuda.__all__,
            [name for name in expected_cuda.__all__ if name in SUPPORTED],
        )
        for name in ("cuda", *sorted(SUPPORTED)):
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
            "from torch_rs.cuda import "
            "device_count, is_available, is_initialized, memory_reserved",
            actual_direct_import,
        )
        exec(
            "from torch.cuda import "
            "device_count, is_available, is_initialized, memory_reserved",
            expected_direct_import,
        )
        for name in SUPPORTED:
            with self.subTest(direct_import=name):
                self.assertIs(actual_direct_import[name], getattr(actual_cuda, name))
                self.assertIs(expected_direct_import[name], getattr(expected_cuda, name))

        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.cuda import *", actual_wildcard)
        exec("from torch.cuda import *", expected_wildcard)
        self.assertEqual(
            {name for name in actual_wildcard if not name.startswith("__")},
            SUPPORTED,
        )
        for name in SUPPORTED:
            with self.subTest(wildcard=name):
                self.assertIs(actual_wildcard[name], getattr(actual_cuda, name))
                self.assertIs(expected_wildcard[name], getattr(expected_cuda, name))

        for module, functions in (
            (torch, (actual_cuda.is_initialized, actual_cuda.memory_reserved)),
            (
                reference_torch,
                (expected_cuda.is_initialized, expected_cuda.memory_reserved),
            ),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("cuda", namespace)
            self.assertNotIn("is_initialized", namespace)
            self.assertNotIn("memory_reserved", namespace)
            for function in functions:
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                for name in ("is_initialized", "memory_reserved"):
                    actual = getattr(actual_cuda, name)
                    expected = getattr(expected_cuda, name)
                    self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
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
        actual_old_memory_reserved = actual_cuda.memory_reserved
        expected_old_memory_reserved = expected_cuda.memory_reserved
        self.assertIs(importlib.reload(actual_cuda), actual_cuda)
        self.assertIs(importlib.reload(expected_cuda), expected_cuda)
        self.assertIsNot(actual_cuda.is_initialized, actual_old)
        self.assertIsNot(expected_cuda.is_initialized, expected_old)
        self.assertIs(actual_cuda.memory_reserved, actual_old_memory_reserved)
        self.assertIs(expected_cuda.memory_reserved, expected_old_memory_reserved)
        self.assertIs(torch.cuda, actual_cuda)
        self.assertIs(reference_torch.cuda, expected_cuda)
        self.assertIs(sys.modules["torch_rs.cuda"], actual_cuda)
        self.assertIs(sys.modules["torch.cuda"], expected_cuda)
        self.assertIs(actual_cuda.is_initialized(), False)
        self.assertEqual(actual_cuda.memory_reserved(), 0)
        self.assertIs(type(expected_cuda.memory_reserved()), int)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(actual_old)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(expected_old)

    def test_memory_reserved_device_forms_match_pytorch_2_13_before_initialization(self):
        script = r'''
import torch as reference_torch
import torch_rs as torch

assert reference_torch.__version__.split("+")[0] == "2.13.0"
assert torch.cuda.is_initialized() is False
assert reference_torch.cuda.is_initialized() is False

class ExplodingDeviceToken:
    def __bool__(self):
        raise AssertionError("device token truth value was inspected")

    def __index__(self):
        raise AssertionError("device token index was inspected")

    def __int__(self):
        raise AssertionError("device token integer value was inspected")

    def __str__(self):
        raise AssertionError("device token string value was inspected")

expressions = (
    "module.cuda.memory_reserved()",
    "module.cuda.memory_reserved(None)",
    "module.cuda.memory_reserved(device=None)",
    "module.cuda.memory_reserved(0)",
    "module.cuda.memory_reserved(-1)",
    "module.cuda.memory_reserved(True)",
    "module.cuda.memory_reserved(1.5)",
    "module.cuda.memory_reserved('cpu')",
    "module.cuda.memory_reserved('cpu:0')",
    "module.cuda.memory_reserved('cuda')",
    "module.cuda.memory_reserved('cuda:0')",
    "module.cuda.memory_reserved('cuda:999999')",
    "module.cuda.memory_reserved('cuda:banana')",
    "module.cuda.memory_reserved('banana')",
    "module.cuda.memory_reserved('')",
    "module.cuda.memory_reserved(module.device('cpu'))",
    "module.cuda.memory_reserved(module.device('cpu', 0))",
    "module.cuda.memory_reserved(object())",
    "module.cuda.memory_reserved([])",
    "module.cuda.memory_reserved({})",
    "module.cuda.memory_reserved(ExplodingDeviceToken())",
)
for expression in expressions:
    actual = eval(
        expression,
        {"module": torch, "ExplodingDeviceToken": ExplodingDeviceToken},
    )
    expected = eval(
        expression,
        {"module": reference_torch, "ExplodingDeviceToken": ExplodingDeviceToken},
    )
    assert type(actual) is int, expression
    assert type(expected) is int, expression
    assert actual == expected == 0, expression
    assert torch.cuda.is_initialized() is False, expression
    assert reference_torch.cuda.is_initialized() is False, expression
foreign_cuda_device = reference_torch.device("cuda", 0)
assert torch.cuda.memory_reserved(foreign_cuda_device) == 0
assert reference_torch.cuda.memory_reserved(foreign_cuda_device) == 0
assert torch.cuda.is_available() is False
assert torch.cuda.device_count() == 0
assert torch.cuda.is_initialized() is False
assert reference_torch.cuda.is_initialized() is False
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )

    def test_cpu_build_probe_values_are_static_without_changing_cuda_runtime_state(self):
        cuda = torch.cuda
        self.assertIs(cuda.is_initialized(), False)
        self.assertIs(cuda.is_available(), False)
        self.assertEqual(cuda.device_count(), 0)
        self.assertEqual(cuda.memory_reserved(), 0)
        self.assertEqual(cuda.memory_reserved("cuda:0"), 0)
        self.assertIs(cuda.is_initialized(), False)
        self.assertFalse(hasattr(cuda, "_initialized"))
        self.assertFalse(hasattr(cuda, "_cached_device_count"))

        script = r"""
import torch

assert torch.cuda.is_initialized() is False
assert type(torch.cuda.is_available()) is bool
assert type(torch.cuda.device_count()) is int
assert type(torch.cuda.memory_reserved()) is int
assert torch.cuda.memory_reserved() == 0
assert torch.cuda.memory_reserved("cuda:0") == 0
assert torch.cuda.memory_reserved(object()) == 0
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

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

    def type_hints_outcome(self, function):
        try:
            return ("return", typing.get_type_hints(function))
        except Exception as error:
            return ("raise", type(error), str(error), error.args)

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

    def test_max_memory_reserved_signature_and_metadata_match_pytorch_2_13(self):
        actual_cuda = importlib.import_module("torch_rs.cuda")
        expected_cuda = importlib.import_module("torch.cuda")
        actual_memory = importlib.import_module("torch_rs.cuda.memory")
        expected_memory = importlib.import_module("torch.cuda.memory")
        actual = actual_cuda.max_memory_reserved
        expected = expected_cuda.max_memory_reserved

        self.assertIs(torch.cuda, actual_cuda)
        self.assertIs(reference_torch.cuda, expected_cuda)
        self.assertIs(sys.modules["torch_rs.cuda"], actual_cuda)
        self.assertIs(sys.modules["torch.cuda"], expected_cuda)
        self.assertIs(actual_cuda.memory, actual_memory)
        self.assertIs(expected_cuda.memory, expected_memory)
        self.assertIs(actual, actual_memory.max_memory_reserved)
        self.assertIs(expected, expected_memory.max_memory_reserved)
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
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_memory)
        self.assertIs(inspect.getmodule(expected), expected_memory)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

        cases = (
            (
                lambda: actual(None, None),
                lambda: expected(None, None),
            ),
            (
                lambda: actual(None, device=0),
                lambda: expected(None, device=0),
            ),
            (
                lambda: actual(device_index=0),
                lambda: expected(device_index=0),
            ),
            (
                lambda: actual(None, device_index=0),
                lambda: expected(None, device_index=0),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_imports_wildcards_copy_pickle_and_reload_match_supported_scope(self):
        actual_cuda = torch.cuda
        expected_cuda = reference_torch.cuda
        actual_memory = importlib.import_module("torch_rs.cuda.memory")
        expected_memory = importlib.import_module("torch.cuda.memory")
        supported = {
            "device_count",
            "is_available",
            "is_initialized",
            "max_memory_reserved",
        }

        self.assertEqual(
            actual_cuda.__all__,
            [name for name in expected_cuda.__all__ if name in supported],
        )
        self.assertEqual(
            actual_memory.__all__,
            [
                name
                for name in expected_memory.__all__
                if name in {"max_memory_reserved"}
            ],
        )
        self.assertIs(actual_cuda.memory, actual_memory)
        self.assertIs(expected_cuda.memory, expected_memory)
        self.assertIs(
            actual_cuda.max_memory_reserved,
            actual_memory.max_memory_reserved,
        )
        self.assertIs(
            expected_cuda.max_memory_reserved,
            expected_memory.max_memory_reserved,
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
            "is_initialized, max_memory_reserved",
            actual_direct_import,
        )
        exec(
            "from torch.cuda import device_count, is_available, "
            "is_initialized, max_memory_reserved",
            expected_direct_import,
        )
        for name in supported:
            with self.subTest(direct_import=name):
                self.assertIs(actual_direct_import[name], getattr(actual_cuda, name))
                self.assertIs(expected_direct_import[name], getattr(expected_cuda, name))

        actual_memory_import = {}
        expected_memory_import = {}
        exec(
            "from torch_rs.cuda.memory import max_memory_reserved",
            actual_memory_import,
        )
        exec(
            "from torch.cuda.memory import max_memory_reserved",
            expected_memory_import,
        )
        self.assertIs(
            actual_memory_import["max_memory_reserved"],
            actual_cuda.max_memory_reserved,
        )
        self.assertIs(
            expected_memory_import["max_memory_reserved"],
            expected_cuda.max_memory_reserved,
        )

        actual_wildcard = {}
        expected_wildcard = {}
        actual_memory_wildcard = {}
        expected_memory_wildcard = {}
        exec("from torch_rs.cuda import *", actual_wildcard)
        exec("from torch.cuda import *", expected_wildcard)
        exec("from torch_rs.cuda.memory import *", actual_memory_wildcard)
        exec("from torch.cuda.memory import *", expected_memory_wildcard)
        self.assertEqual(
            {name for name in actual_wildcard if not name.startswith("__")},
            supported,
        )
        self.assertEqual(
            {
                name
                for name in actual_memory_wildcard
                if not name.startswith("__")
            },
            {"max_memory_reserved"},
        )
        for name in supported:
            with self.subTest(wildcard=name):
                self.assertIs(actual_wildcard[name], getattr(actual_cuda, name))
                self.assertIs(expected_wildcard[name], getattr(expected_cuda, name))
        self.assertIs(
            actual_memory_wildcard["max_memory_reserved"],
            actual_cuda.max_memory_reserved,
        )
        self.assertIs(
            expected_memory_wildcard["max_memory_reserved"],
            expected_cuda.max_memory_reserved,
        )

        for module, functions in (
            (
                torch,
                (actual_cuda.is_initialized, actual_cuda.max_memory_reserved),
            ),
            (
                reference_torch,
                (expected_cuda.is_initialized, expected_cuda.max_memory_reserved),
            ),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("cuda", namespace)
            self.assertNotIn("is_initialized", namespace)
            self.assertNotIn("max_memory_reserved", namespace)
            for function in functions:
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
                self.assertIs(
                    pickle.loads(
                        pickle.dumps(actual_cuda.max_memory_reserved, protocol)
                    ),
                    actual_cuda.max_memory_reserved,
                )
                self.assertIs(
                    pickle.loads(
                        pickle.dumps(expected_cuda.max_memory_reserved, protocol)
                    ),
                    expected_cuda.max_memory_reserved,
                )
                self.assertEqual(
                    self.pickle_shape(actual_cuda.max_memory_reserved, protocol),
                    self.pickle_shape(expected_cuda.max_memory_reserved, protocol),
                )

        actual_old = actual_cuda.is_initialized
        expected_old = expected_cuda.is_initialized
        actual_old_max = actual_cuda.max_memory_reserved
        expected_old_max = expected_cuda.max_memory_reserved
        self.assertIs(importlib.reload(actual_cuda), actual_cuda)
        self.assertIs(importlib.reload(expected_cuda), expected_cuda)
        self.assertIsNot(actual_cuda.is_initialized, actual_old)
        self.assertIsNot(expected_cuda.is_initialized, expected_old)
        self.assertIs(actual_cuda.max_memory_reserved, actual_old_max)
        self.assertIs(expected_cuda.max_memory_reserved, expected_old_max)
        self.assertIs(torch.cuda, actual_cuda)
        self.assertIs(reference_torch.cuda, expected_cuda)
        self.assertIs(sys.modules["torch_rs.cuda"], actual_cuda)
        self.assertIs(sys.modules["torch.cuda"], expected_cuda)
        self.assertIs(actual_cuda.is_initialized(), False)
        self.assertEqual(actual_cuda.max_memory_reserved(), 0)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(actual_old)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(expected_old)

        self.assertIs(importlib.reload(actual_memory), actual_memory)
        self.assertIs(importlib.reload(expected_memory), expected_memory)
        self.assertIsNot(actual_memory.max_memory_reserved, actual_old_max)
        self.assertIsNot(expected_memory.max_memory_reserved, expected_old_max)
        self.assertIs(actual_cuda.max_memory_reserved, actual_old_max)
        self.assertIs(expected_cuda.max_memory_reserved, expected_old_max)
        self.assertIs(importlib.reload(actual_cuda), actual_cuda)
        self.assertIs(importlib.reload(expected_cuda), expected_cuda)
        self.assertIs(
            actual_cuda.max_memory_reserved,
            actual_memory.max_memory_reserved,
        )
        self.assertIs(
            expected_cuda.max_memory_reserved,
            expected_memory.max_memory_reserved,
        )
        self.assertIsNot(actual_cuda.max_memory_reserved, actual_old_max)
        self.assertIsNot(expected_cuda.max_memory_reserved, expected_old_max)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(actual_old_max)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(expected_old_max)

    def test_max_memory_reserved_uninitialized_device_behavior_matches_pytorch_2_13(self):
        script = r'''
import torch as reference_torch
import torch_rs as torch

assert reference_torch.__version__.split("+")[0] == "2.13.0"
assert torch.cuda.max_memory_reserved is torch.cuda.memory.max_memory_reserved
assert (
    reference_torch.cuda.max_memory_reserved
    is reference_torch.cuda.memory.max_memory_reserved
)
assert torch.cuda.is_available() is False
assert torch.cuda.device_count() == 0
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

actual_devices = [
    None,
    0,
    -1,
    True,
    False,
    "cuda",
    "cuda:0",
    "cuda:-1",
    "cuda:01",
    "cpu",
    "cpu:0",
    "",
    "banana",
    1.5,
    object(),
    [],
    {},
    torch.device("cuda"),
    torch.device("cuda", 0),
    torch.device("cpu"),
    torch.device("cpu", 0),
    ExplodingDeviceToken(),
]
expected_devices = [
    None,
    0,
    -1,
    True,
    False,
    "cuda",
    "cuda:0",
    "cuda:-1",
    "cuda:01",
    "cpu",
    "cpu:0",
    "",
    "banana",
    1.5,
    object(),
    [],
    {},
    reference_torch.device("cuda"),
    reference_torch.device("cuda", 0),
    reference_torch.device("cpu"),
    reference_torch.device("cpu", 0),
    ExplodingDeviceToken(),
]
actual = [torch.cuda.max_memory_reserved(device) for device in actual_devices]
expected = [
    reference_torch.cuda.max_memory_reserved(device)
    for device in expected_devices
]
assert actual == expected == [0] * len(actual_devices)
assert all(type(result) is int for result in actual)
assert all(type(result) is int for result in expected)
assert torch.cuda.max_memory_reserved(device=None) == 0
assert reference_torch.cuda.max_memory_reserved(device=None) == 0
assert torch.cuda.max_memory_reserved(device=0) == 0
assert reference_torch.cuda.max_memory_reserved(device=0) == 0
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
        self.assertEqual(cuda.max_memory_reserved(), 0)
        self.assertIs(cuda.is_initialized(), False)
        self.assertFalse(hasattr(cuda, "_initialized"))
        self.assertFalse(hasattr(cuda, "_cached_device_count"))

        script = r"""
import torch

assert torch.cuda.is_initialized() is False
assert type(torch.cuda.is_available()) is bool
assert type(torch.cuda.device_count()) is int
assert type(torch.cuda.max_memory_reserved()) is int
assert torch.cuda.max_memory_reserved() == 0
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

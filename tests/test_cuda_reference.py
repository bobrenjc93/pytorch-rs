import copy
import importlib
import inspect
import pickle
import pickletools
import re
import subprocess
import sys
import types
import typing
import unittest
from collections import OrderedDict

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SUPPORTED = {
    "device_count",
    "empty_cache",
    "is_available",
    "is_initialized",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
}
MEMORY_LOCAL = {
    "empty_cache",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
}


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
            return ("return", self.normalize(typing.get_type_hints(function)))
        except Exception as error:
            return ("raise", type(error), str(error), error.args)

    def pickle_outcome(self, function, protocol):
        try:
            payload = pickle.dumps(function, protocol=protocol)
        except Exception as error:
            return (
                "raise",
                type(error).__name__,
                re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                    "torch_rs",
                    "torch",
                ),
            )
        return (
            "return",
            pickle.loads(payload) is function,
            self.pickle_shape(function, protocol),
        )

    def normalize(self, value):
        return str(value).replace("torch_rs", "torch")

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

    def test_cuda_memory_signature_documentation_and_identity_match_pytorch_2_13(
        self,
    ):
        actual_module = importlib.import_module("torch_rs.cuda")
        expected_module = importlib.import_module("torch.cuda")
        actual_memory = importlib.import_module("torch_rs.cuda.memory")
        expected_memory = importlib.import_module("torch.cuda.memory")

        self.assertIs(torch.cuda, actual_module)
        self.assertIs(reference_torch.cuda, expected_module)
        self.assertIs(actual_module.memory, actual_memory)
        self.assertIs(expected_module.memory, expected_memory)
        self.assertIs(sys.modules["torch_rs.cuda.memory"], actual_memory)
        self.assertIs(sys.modules["torch.cuda.memory"], expected_memory)
        self.assertEqual(actual_memory.__doc__, expected_memory.__doc__)

        for name in sorted(MEMORY_LOCAL):
            with self.subTest(name=name):
                actual = getattr(actual_module, name)
                expected = getattr(expected_module, name)
                actual_memory_function = getattr(actual_memory, name)
                expected_memory_function = getattr(expected_memory, name)
                self.assertIs(actual, actual_memory_function)
                self.assertIs(expected, expected_memory_function)
                self.assertIs(type(actual), types.FunctionType)
                self.assertIs(type(expected), types.FunctionType)
                self.assertEqual(
                    self.normalize(inspect.signature(actual)),
                    str(inspect.signature(expected)),
                )
                self.assertEqual(
                    self.normalize(inspect.get_annotations(actual)),
                    str(inspect.get_annotations(expected)),
                )
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

    def test_imports_wildcards_copy_pickle_and_reload_match_supported_scope(self):
        actual_cuda = torch.cuda
        expected_cuda = reference_torch.cuda
        actual_memory = actual_cuda.memory
        expected_memory = expected_cuda.memory

        self.assertEqual(
            actual_cuda.__all__,
            [name for name in expected_cuda.__all__ if name in SUPPORTED],
        )
        self.assertEqual(
            actual_memory.__all__,
            [name for name in expected_memory.__all__ if name in MEMORY_LOCAL],
        )
        for name in MEMORY_LOCAL:
            with self.subTest(memory_alias=name):
                self.assertIs(getattr(actual_cuda, name), getattr(actual_memory, name))
                self.assertIs(
                    getattr(expected_cuda, name),
                    getattr(expected_memory, name),
                )
        self.assertIs(
            actual_memory,
            importlib.import_module("torch_rs.cuda.memory"),
        )
        self.assertIs(
            expected_memory,
            importlib.import_module("torch.cuda.memory"),
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
            "from torch_rs.cuda import device_count, empty_cache, is_available, is_initialized, max_memory_allocated, max_memory_reserved, memory_allocated, memory_reserved, memory_stats, reset_accumulated_memory_stats, reset_peak_memory_stats",
            actual_direct_import,
        )
        exec(
            "from torch.cuda import device_count, empty_cache, is_available, is_initialized, max_memory_allocated, max_memory_reserved, memory_allocated, memory_reserved, memory_stats, reset_accumulated_memory_stats, reset_peak_memory_stats",
            expected_direct_import,
        )
        for name in SUPPORTED:
            with self.subTest(direct_import=name):
                self.assertIs(actual_direct_import[name], getattr(actual_cuda, name))
                self.assertIs(expected_direct_import[name], getattr(expected_cuda, name))

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
            SUPPORTED,
        )
        self.assertEqual(
            {name for name in actual_memory_wildcard if not name.startswith("__")},
            MEMORY_LOCAL,
        )
        for name in SUPPORTED:
            with self.subTest(wildcard=name):
                self.assertIs(actual_wildcard[name], getattr(actual_cuda, name))
                self.assertIs(expected_wildcard[name], getattr(expected_cuda, name))
                if name in MEMORY_LOCAL:
                    self.assertIs(
                        actual_memory_wildcard[name],
                        getattr(actual_cuda, name),
                    )
                    self.assertIs(
                        expected_memory_wildcard[name],
                        getattr(expected_cuda, name),
                    )

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

        for name in SUPPORTED:
            actual = getattr(actual_cuda, name)
            expected = getattr(expected_cuda, name)
            with self.subTest(copy=name):
                self.assertIs(copy.copy(actual), actual)
                self.assertIs(copy.copy(expected), expected)
                self.assertIs(copy.deepcopy(actual), actual)
                self.assertIs(copy.deepcopy(expected), expected)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    self.assertEqual(
                        self.pickle_outcome(actual, protocol),
                        self.pickle_outcome(expected, protocol),
                    )

        actual_old = actual_cuda.is_initialized
        expected_old = expected_cuda.is_initialized
        actual_old_memory = {name: getattr(actual_cuda, name) for name in MEMORY_LOCAL}
        expected_old_memory = {
            name: getattr(expected_cuda, name) for name in MEMORY_LOCAL
        }
        self.assertIs(importlib.reload(actual_cuda), actual_cuda)
        self.assertIs(importlib.reload(expected_cuda), expected_cuda)
        self.assertIsNot(actual_cuda.is_initialized, actual_old)
        self.assertIsNot(expected_cuda.is_initialized, expected_old)
        for name in MEMORY_LOCAL:
            with self.subTest(reload_memory_alias=name):
                self.assertIs(getattr(actual_cuda, name), actual_old_memory[name])
                self.assertIs(getattr(expected_cuda, name), expected_old_memory[name])
        self.assertIs(torch.cuda, actual_cuda)
        self.assertIs(reference_torch.cuda, expected_cuda)
        self.assertIs(sys.modules["torch_rs.cuda"], actual_cuda)
        self.assertIs(sys.modules["torch.cuda"], expected_cuda)
        self.assertIs(actual_cuda.is_initialized(), False)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(actual_old)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(expected_old)

    def test_cuda_memory_reload_behavior_matches_pytorch_2_13(self):
        actual_cuda = torch.cuda
        expected_cuda = reference_torch.cuda
        actual_memory = actual_cuda.memory
        expected_memory = expected_cuda.memory
        actual_old = {name: getattr(actual_memory, name) for name in MEMORY_LOCAL}
        expected_old = {name: getattr(expected_memory, name) for name in MEMORY_LOCAL}

        self.assertIs(importlib.reload(actual_memory), actual_memory)
        self.assertIs(importlib.reload(expected_memory), expected_memory)
        actual_new = {name: getattr(actual_memory, name) for name in MEMORY_LOCAL}
        expected_new = {name: getattr(expected_memory, name) for name in MEMORY_LOCAL}

        for name in sorted(MEMORY_LOCAL):
            with self.subTest(name=name):
                self.assertIsNot(actual_new[name], actual_old[name])
                self.assertIsNot(expected_new[name], expected_old[name])
                self.assertIs(getattr(actual_cuda, name), actual_old[name])
                self.assertIs(getattr(expected_cuda, name), expected_old[name])
                self.assertEqual(
                    self.pickle_outcome(actual_old[name], pickle.HIGHEST_PROTOCOL),
                    self.pickle_outcome(expected_old[name], pickle.HIGHEST_PROTOCOL),
                )
                self.assertEqual(
                    self.pickle_outcome(actual_new[name], pickle.HIGHEST_PROTOCOL),
                    self.pickle_outcome(expected_new[name], pickle.HIGHEST_PROTOCOL),
                )

        self.assertIs(importlib.reload(actual_cuda), actual_cuda)
        self.assertIs(importlib.reload(expected_cuda), expected_cuda)
        for name in sorted(MEMORY_LOCAL):
            with self.subTest(cuda_reload_name=name):
                self.assertIs(getattr(actual_cuda, name), actual_new[name])
                self.assertIs(getattr(expected_cuda, name), expected_new[name])

    def test_cuda_memory_cpu_build_noops_match_preinit_query_contract(self):
        script = r'''
from collections import OrderedDict

import torch as reference_torch
import torch_rs as torch

assert reference_torch.__version__.split("+")[0] == "2.13.0"
assert torch.cuda.is_available() is False
assert torch.cuda.device_count() == 0
assert torch.cuda.is_initialized() is False
assert not hasattr(torch.cuda, "_initialized")
assert not hasattr(torch.cuda, "_cached_device_count")
assert not reference_torch.cuda.is_initialized()

class ExplodingDeviceToken:
    def __bool__(self):
        raise AssertionError("device token truth value was inspected")

    def __index__(self):
        raise AssertionError("device token index was inspected")

    def __int__(self):
        raise AssertionError("device token integer value was inspected")

    def __str__(self):
        raise AssertionError("device token string value was inspected")

actual_tokens = (
    None,
    0,
    -1,
    True,
    1.5,
    "cpu",
    "cuda:0",
    torch.device("cpu"),
    object(),
    [],
    {},
    ExplodingDeviceToken(),
)
expected_tokens = (
    None,
    0,
    -1,
    True,
    1.5,
    "cpu",
    "cuda:0",
    reference_torch.device("cpu"),
    reference_torch.device("cuda:0"),
    object(),
    [],
    {},
    ExplodingDeviceToken(),
)
actual_stats = [
    torch.cuda.memory_stats(token) for token in actual_tokens
] + [
    torch.cuda.memory_stats(device=token) for token in actual_tokens
]
expected_stats = [
    reference_torch.cuda.memory_stats(token) for token in expected_tokens
] + [
    reference_torch.cuda.memory_stats(device=token) for token in expected_tokens
]
for stats in actual_stats + expected_stats:
    assert type(stats) is OrderedDict and not stats
assert len({id(stats) for stats in actual_stats}) == len(actual_stats)
assert len({id(stats) for stats in expected_stats}) == len(expected_stats)
for actual_name, expected_name in (
    ("memory_allocated", "memory_allocated"),
    ("max_memory_allocated", "max_memory_allocated"),
    ("memory_reserved", "memory_reserved"),
    ("max_memory_reserved", "max_memory_reserved"),
):
    actual_function = getattr(torch.cuda, actual_name)
    expected_function = getattr(reference_torch.cuda, expected_name)
    actual_values = [actual_function(token) for token in actual_tokens]
    actual_values += [actual_function(device=token) for token in actual_tokens]
    expected_values = [expected_function(token) for token in expected_tokens]
    expected_values += [expected_function(device=token) for token in expected_tokens]
    assert actual_values == [0] * len(actual_values)
    assert expected_values == [0] * len(expected_values)
    assert all(type(value) is int for value in actual_values)
    assert all(type(value) is int for value in expected_values)
assert torch.cuda.empty_cache() is None
assert not reference_torch.cuda.is_initialized()
state = (
    torch.cuda.is_available(),
    torch.cuda.device_count(),
    torch.cuda.is_initialized(),
    torch._C._has_cuda,
    torch.version.cuda,
)
for token in actual_tokens:
    assert torch.cuda.reset_accumulated_memory_stats(token) is None
    assert torch.cuda.reset_accumulated_memory_stats(device=token) is None
    assert torch.cuda.reset_peak_memory_stats(token) is None
    assert torch.cuda.reset_peak_memory_stats(device=token) is None
assert (
    torch.cuda.is_available(),
    torch.cuda.device_count(),
    torch.cuda.is_initialized(),
    torch._C._has_cuda,
    torch.version.cuda,
) == state == (False, 0, False, False, None)
assert torch.cuda.memory_stats() == OrderedDict()
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

    def test_cuda_memory_argument_binding_errors_match_pytorch_2_13(self):
        actual = torch.cuda
        expected = reference_torch.cuda
        cases = (
            (
                lambda: actual.empty_cache(None),
                lambda: expected.empty_cache(None),
            ),
            (
                lambda: actual.empty_cache(device=True),
                lambda: expected.empty_cache(device=True),
            ),
            (
                lambda: actual.memory_allocated(device_index=None),
                lambda: expected.memory_allocated(device_index=None),
            ),
            (
                lambda: actual.memory_allocated(None, None),
                lambda: expected.memory_allocated(None, None),
            ),
            (
                lambda: actual.memory_allocated(unexpected=True),
                lambda: expected.memory_allocated(unexpected=True),
            ),
            (
                lambda: actual.max_memory_allocated(device_index=None),
                lambda: expected.max_memory_allocated(device_index=None),
            ),
            (
                lambda: actual.max_memory_allocated(None, None),
                lambda: expected.max_memory_allocated(None, None),
            ),
            (
                lambda: actual.max_memory_allocated(unexpected=True),
                lambda: expected.max_memory_allocated(unexpected=True),
            ),
            (
                lambda: actual.memory_reserved(device_index=None),
                lambda: expected.memory_reserved(device_index=None),
            ),
            (
                lambda: actual.memory_reserved(None, None),
                lambda: expected.memory_reserved(None, None),
            ),
            (
                lambda: actual.memory_reserved(unexpected=True),
                lambda: expected.memory_reserved(unexpected=True),
            ),
            (
                lambda: actual.max_memory_reserved(device_index=None),
                lambda: expected.max_memory_reserved(device_index=None),
            ),
            (
                lambda: actual.max_memory_reserved(None, None),
                lambda: expected.max_memory_reserved(None, None),
            ),
            (
                lambda: actual.max_memory_reserved(unexpected=True),
                lambda: expected.max_memory_reserved(unexpected=True),
            ),
            (
                lambda: actual.memory_stats(device_index=None),
                lambda: expected.memory_stats(device_index=None),
            ),
            (
                lambda: actual.memory_stats(None, None),
                lambda: expected.memory_stats(None, None),
            ),
            (
                lambda: actual.memory_stats(unexpected=True),
                lambda: expected.memory_stats(unexpected=True),
            ),
            (
                lambda: actual.reset_accumulated_memory_stats(device_index=None),
                lambda: expected.reset_accumulated_memory_stats(device_index=None),
            ),
            (
                lambda: actual.reset_accumulated_memory_stats(None, None),
                lambda: expected.reset_accumulated_memory_stats(None, None),
            ),
            (
                lambda: actual.reset_accumulated_memory_stats(unexpected=True),
                lambda: expected.reset_accumulated_memory_stats(unexpected=True),
            ),
            (
                lambda: actual.reset_peak_memory_stats(device_index=None),
                lambda: expected.reset_peak_memory_stats(device_index=None),
            ),
            (
                lambda: actual.reset_peak_memory_stats(None, None),
                lambda: expected.reset_peak_memory_stats(None, None),
            ),
            (
                lambda: actual.reset_peak_memory_stats(unexpected=True),
                lambda: expected.reset_peak_memory_stats(unexpected=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

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

import contextlib
import copy
import importlib
import inspect
import pickle
import pickletools
import re
import subprocess
import sys
import threading
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
    "current_accelerator",
    "current_device_index",
    "device_count",
    "empty_cache",
    "is_available",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "reset_peak_memory_stats",
}

MEMORY_LOCAL = {
    "empty_cache",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "reset_peak_memory_stats",
}
ACCELERATOR_LOCAL = SUPPORTED - MEMORY_LOCAL

NO_ACCELERATOR_ERROR = "Cannot access accelerator device when none is available."


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class AcceleratorReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "accelerator discovery differentials require pinned PyTorch 2.13.0"
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

    def normalize(self, value):
        return str(value).replace("torch_rs", "torch")

    def call_outcome(self, function):
        try:
            return ("return", function())
        except Exception as error:
            return ("raise", type(error), str(error), error.args)

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_module = importlib.import_module("torch_rs.accelerator")
        expected_module = importlib.import_module("torch.accelerator")

        self.assertIs(torch.accelerator, actual_module)
        self.assertIs(reference_torch.accelerator, expected_module)
        self.assertIs(sys.modules["torch_rs.accelerator"], actual_module)
        self.assertIs(sys.modules["torch.accelerator"], expected_module)
        self.assertEqual(actual_module.__doc__, expected_module.__doc__)
        self.assertIs(
            actual_module.memory,
            importlib.import_module("torch_rs.accelerator.memory"),
        )
        self.assertIs(
            expected_module.memory,
            importlib.import_module("torch.accelerator.memory"),
        )

        for name in (
            "current_accelerator",
            "current_device_index",
            "device_count",
            "empty_cache",
            "is_available",
            "max_memory_allocated",
            "max_memory_reserved",
            "memory_allocated",
            "memory_reserved",
            "memory_stats",
            "reset_peak_memory_stats",
        ):
            with self.subTest(name=name):
                actual = getattr(actual_module, name)
                expected = getattr(expected_module, name)
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
                    self.normalize(typing.get_type_hints(actual)),
                    str(typing.get_type_hints(expected)),
                )
                self.assertEqual(actual.__name__, expected.__name__)
                self.assertEqual(actual.__qualname__, expected.__qualname__)
                self.assertEqual(
                    actual.__module__.replace("torch_rs", "torch"),
                    expected.__module__,
                )
                actual_owner = (
                    actual_module.memory if name in MEMORY_LOCAL else actual_module
                )
                expected_owner = (
                    expected_module.memory
                    if name in MEMORY_LOCAL
                    else expected_module
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

    def test_imports_exports_copy_and_pickle_match_the_supported_scope(self):
        actual_module = torch.accelerator
        expected_module = reference_torch.accelerator
        actual_memory = actual_module.memory
        expected_memory = expected_module.memory

        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name in SUPPORTED],
        )
        self.assertEqual(
            actual_memory.__all__,
            [name for name in expected_memory.__all__ if name in MEMORY_LOCAL],
        )
        self.assertIs(actual_module.empty_cache, actual_memory.empty_cache)
        self.assertIs(expected_module.empty_cache, expected_memory.empty_cache)
        self.assertIs(
            actual_module.max_memory_allocated,
            actual_memory.max_memory_allocated,
        )
        self.assertIs(
            expected_module.max_memory_allocated,
            expected_memory.max_memory_allocated,
        )
        self.assertIs(
            actual_module.max_memory_reserved,
            actual_memory.max_memory_reserved,
        )
        self.assertIs(
            expected_module.max_memory_reserved,
            expected_memory.max_memory_reserved,
        )
        self.assertIs(
            actual_module.memory_allocated,
            actual_memory.memory_allocated,
        )
        self.assertIs(
            expected_module.memory_allocated,
            expected_memory.memory_allocated,
        )
        self.assertIs(actual_module.memory_reserved, actual_memory.memory_reserved)
        self.assertIs(
            expected_module.memory_reserved,
            expected_memory.memory_reserved,
        )
        self.assertIs(actual_module.memory_stats, actual_memory.memory_stats)
        self.assertIs(expected_module.memory_stats, expected_memory.memory_stats)
        self.assertIs(
            actual_module.reset_peak_memory_stats,
            actual_memory.reset_peak_memory_stats,
        )
        self.assertIs(
            expected_module.reset_peak_memory_stats,
            expected_memory.reset_peak_memory_stats,
        )
        for name in ("accelerator", *SUPPORTED):
            with self.subTest(top_level_export=name):
                self.assertEqual(
                    torch.__all__.count(name),
                    reference_torch.__all__.count(name),
                )

        actual_package_import = {}
        expected_package_import = {}
        exec("from torch_rs import accelerator", actual_package_import)
        exec("from torch import accelerator", expected_package_import)
        self.assertIs(actual_package_import["accelerator"], actual_module)
        self.assertIs(expected_package_import["accelerator"], expected_module)

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.accelerator import *", actual_namespace)
        exec("from torch.accelerator import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            SUPPORTED,
        )
        actual_memory_namespace = {}
        expected_memory_namespace = {}
        exec("from torch_rs.accelerator.memory import *", actual_memory_namespace)
        exec("from torch.accelerator.memory import *", expected_memory_namespace)
        self.assertEqual(
            {name for name in actual_memory_namespace if not name.startswith("__")},
            MEMORY_LOCAL,
        )
        for name in SUPPORTED:
            with self.subTest(name=name):
                actual = getattr(actual_module, name)
                expected = getattr(expected_module, name)
                self.assertIs(actual_namespace[name], actual)
                self.assertIs(expected_namespace[name], expected)
                if name in MEMORY_LOCAL:
                    self.assertIs(actual_memory_namespace[name], actual)
                    self.assertIs(expected_memory_namespace[name], expected)
                self.assertIs(copy.copy(actual), actual)
                self.assertIs(copy.copy(expected), expected)
                self.assertIs(copy.deepcopy(actual), actual)
                self.assertIs(copy.deepcopy(expected), expected)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected, protocol)), expected
                    )
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            for name in ("accelerator", *SUPPORTED):
                self.assertNotIn(name, namespace)

    def test_memory_queries_before_allocator_initialization_match(self):
        script = r'''
from collections import OrderedDict

import torch as reference_torch
import torch_rs as torch

assert reference_torch.__version__.split("+")[0] == "2.13.0"
assert not reference_torch._C._accelerator_isAllocatorInitialized()
assert (
    torch.accelerator.memory_allocated
    is torch.accelerator.memory.memory_allocated
)
assert (
    reference_torch.accelerator.memory_allocated
    is reference_torch.accelerator.memory.memory_allocated
)
assert (
    torch.accelerator.max_memory_allocated
    is torch.accelerator.memory.max_memory_allocated
)
assert (
    reference_torch.accelerator.max_memory_allocated
    is reference_torch.accelerator.memory.max_memory_allocated
)
assert (
    torch.accelerator.max_memory_reserved
    is torch.accelerator.memory.max_memory_reserved
)
assert (
    reference_torch.accelerator.max_memory_reserved
    is reference_torch.accelerator.memory.max_memory_reserved
)
assert (
    torch.accelerator.memory_reserved
    is torch.accelerator.memory.memory_reserved
)
assert (
    reference_torch.accelerator.memory_reserved
    is reference_torch.accelerator.memory.memory_reserved
)
assert torch.accelerator.memory_stats is torch.accelerator.memory.memory_stats
assert (
    reference_torch.accelerator.memory_stats
    is reference_torch.accelerator.memory.memory_stats
)
assert (
    torch.accelerator.reset_peak_memory_stats
    is torch.accelerator.memory.reset_peak_memory_stats
)
assert (
    reference_torch.accelerator.reset_peak_memory_stats
    is reference_torch.accelerator.memory.reset_peak_memory_stats
)

class ExplodingDeviceToken:
    def __bool__(self):
        raise AssertionError("device token truth value was inspected")

    def __index__(self):
        raise AssertionError("device token index was inspected")

    def __int__(self):
        raise AssertionError("device token integer value was inspected")

    def __str__(self):
        raise AssertionError("device token string value was inspected")

actual = [
    torch.accelerator.memory_stats(),
    torch.accelerator.memory_stats(None),
    torch.accelerator.memory_stats("cuda:0"),
    torch.accelerator.memory_stats(torch.device("cpu")),
    torch.accelerator.memory_stats(ExplodingDeviceToken()),
]
expected = [
    reference_torch.accelerator.memory_stats(),
    reference_torch.accelerator.memory_stats(None),
    reference_torch.accelerator.memory_stats("cuda:0"),
    reference_torch.accelerator.memory_stats(reference_torch.device("cpu")),
    reference_torch.accelerator.memory_stats(ExplodingDeviceToken()),
]
for results in (actual, expected):
    assert all(type(result) is OrderedDict and not result for result in results)
    assert len({id(result) for result in results}) == len(results)
actual_allocated = [
    torch.accelerator.memory_allocated(),
    torch.accelerator.memory_allocated(None),
    torch.accelerator.memory_allocated("cuda:0"),
    torch.accelerator.memory_allocated(torch.device("cpu")),
    torch.accelerator.memory_allocated(ExplodingDeviceToken()),
]
expected_allocated = [
    reference_torch.accelerator.memory_allocated(),
    reference_torch.accelerator.memory_allocated(None),
    reference_torch.accelerator.memory_allocated("cuda:0"),
    reference_torch.accelerator.memory_allocated(reference_torch.device("cpu")),
    reference_torch.accelerator.memory_allocated(ExplodingDeviceToken()),
]
assert actual_allocated == expected_allocated == [0, 0, 0, 0, 0]
assert all(type(result) is int for result in actual_allocated)
assert all(type(result) is int for result in expected_allocated)
actual_max_allocated = [
    torch.accelerator.max_memory_allocated(),
    torch.accelerator.max_memory_allocated(None),
    torch.accelerator.max_memory_allocated("cuda:0"),
    torch.accelerator.max_memory_allocated(torch.device("cpu")),
    torch.accelerator.max_memory_allocated(ExplodingDeviceToken()),
]
expected_max_allocated = [
    reference_torch.accelerator.max_memory_allocated(),
    reference_torch.accelerator.max_memory_allocated(None),
    reference_torch.accelerator.max_memory_allocated("cuda:0"),
    reference_torch.accelerator.max_memory_allocated(
        reference_torch.device("cpu")
    ),
    reference_torch.accelerator.max_memory_allocated(ExplodingDeviceToken()),
]
assert actual_max_allocated == expected_max_allocated == [0, 0, 0, 0, 0]
assert all(type(result) is int for result in actual_max_allocated)
assert all(type(result) is int for result in expected_max_allocated)
actual_max_reserved = [
    torch.accelerator.max_memory_reserved(),
    torch.accelerator.max_memory_reserved(None),
    torch.accelerator.max_memory_reserved("cuda:0"),
    torch.accelerator.max_memory_reserved(torch.device("cpu")),
    torch.accelerator.max_memory_reserved(ExplodingDeviceToken()),
]
expected_max_reserved = [
    reference_torch.accelerator.max_memory_reserved(),
    reference_torch.accelerator.max_memory_reserved(None),
    reference_torch.accelerator.max_memory_reserved("cuda:0"),
    reference_torch.accelerator.max_memory_reserved(
        reference_torch.device("cpu")
    ),
    reference_torch.accelerator.max_memory_reserved(ExplodingDeviceToken()),
]
assert actual_max_reserved == expected_max_reserved == [0, 0, 0, 0, 0]
assert all(type(result) is int for result in actual_max_reserved)
assert all(type(result) is int for result in expected_max_reserved)
actual_reserved = [
    torch.accelerator.memory_reserved(),
    torch.accelerator.memory_reserved(None),
    torch.accelerator.memory_reserved("cuda:0"),
    torch.accelerator.memory_reserved(torch.device("cpu")),
    torch.accelerator.memory_reserved(ExplodingDeviceToken()),
]
expected_reserved = [
    reference_torch.accelerator.memory_reserved(),
    reference_torch.accelerator.memory_reserved(None),
    reference_torch.accelerator.memory_reserved("cuda:0"),
    reference_torch.accelerator.memory_reserved(reference_torch.device("cpu")),
    reference_torch.accelerator.memory_reserved(ExplodingDeviceToken()),
]
assert actual_reserved == expected_reserved == [0, 0, 0, 0, 0]
assert all(type(result) is int for result in actual_reserved)
assert all(type(result) is int for result in expected_reserved)
actual_reset = [
    torch.accelerator.reset_peak_memory_stats(),
    torch.accelerator.reset_peak_memory_stats(None),
    torch.accelerator.reset_peak_memory_stats("cuda:0"),
    torch.accelerator.reset_peak_memory_stats(torch.device("cpu")),
    torch.accelerator.reset_peak_memory_stats(ExplodingDeviceToken()),
]
expected_reset = [
    reference_torch.accelerator.reset_peak_memory_stats(),
    reference_torch.accelerator.reset_peak_memory_stats(None),
    reference_torch.accelerator.reset_peak_memory_stats("cuda:0"),
    reference_torch.accelerator.reset_peak_memory_stats(
        reference_torch.device("cpu")
    ),
    reference_torch.accelerator.reset_peak_memory_stats(ExplodingDeviceToken()),
]
assert actual_reset == expected_reset == [None, None, None, None, None]
assert torch.accelerator.memory_allocated() == 0
assert torch.accelerator.max_memory_allocated() == 0
assert torch.accelerator.memory_reserved() == 0
assert torch.accelerator.max_memory_reserved() == 0
assert not reference_torch._C._accelerator_isAllocatorInitialized()
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

    def test_cpu_only_values_bound_the_cuda_enabled_reference(self):
        torch_rs_build_metadata = (
            torch.accelerator.current_accelerator(),
            torch.accelerator.current_accelerator(check_available=True),
            self.call_outcome(torch.accelerator.current_device_index),
            self.call_outcome(torch.accelerator.current_device_index),
            torch.accelerator.is_available(),
            torch.accelerator.device_count(),
            torch._C._has_cuda,
            torch.version.cuda,
        )
        self.assertEqual(
            torch_rs_build_metadata,
            (
                None,
                None,
                (
                    "raise",
                    RuntimeError,
                    NO_ACCELERATOR_ERROR,
                    (NO_ACCELERATOR_ERROR,),
                ),
                (
                    "raise",
                    RuntimeError,
                    NO_ACCELERATOR_ERROR,
                    (NO_ACCELERATOR_ERROR,),
                ),
                False,
                0,
                False,
                None,
            ),
        )
        self.assertEqual(torch_rs_build_metadata[2], torch_rs_build_metadata[3])
        self.assertIs(torch_rs_build_metadata[4], False)
        self.assertIs(type(torch_rs_build_metadata[5]), int)
        self.assertIs(torch_rs_build_metadata[6], False)
        self.assertIs(torch_rs_build_metadata[7], None)

        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch build")

        device_name = reference_torch.cuda.get_device_name(0)
        if "H100" not in device_name:
            self.skipTest(f"requires an NVIDIA H100, found {device_name}")

        self.assertGreaterEqual(reference_torch.cuda.device_count(), 1)
        reference_accelerator = reference_torch.accelerator.current_accelerator()
        self.assertEqual(reference_accelerator, reference_torch.device("cuda"))
        self.assertIsNone(reference_accelerator.index)
        reference_index = reference_torch.accelerator.current_device_index()
        self.assertIs(type(reference_index), int)
        self.assertEqual(reference_index, reference_torch.cuda.current_device())
        self.assertEqual(
            tuple(
                reference_torch.accelerator.current_device_index()
                for _ in range(4)
            ),
            (reference_index,) * 4,
        )
        self.assertEqual(
            self.call_outcome(torch.accelerator.current_device_index),
            torch_rs_build_metadata[2],
        )
        self.assertIs(reference_torch.accelerator.is_available(), True)
        self.assertGreaterEqual(reference_torch.accelerator.device_count(), 1)

        probe = reference_torch.ones(
            1, device=reference_torch.device("cuda", reference_index)
        )
        self.assertEqual(probe.item(), 1.0)
        reference_torch.cuda.synchronize(reference_index)

        self.assertEqual(
            (
                torch.accelerator.current_accelerator(),
                torch.accelerator.current_accelerator(check_available=True),
                self.call_outcome(torch.accelerator.current_device_index),
                self.call_outcome(torch.accelerator.current_device_index),
                torch.accelerator.is_available(),
                torch.accelerator.device_count(),
                torch._C._has_cuda,
                torch.version.cuda,
            ),
            torch_rs_build_metadata,
        )
        self.assertFalse(hasattr(torch, "cuda"))
        self.assertNotIn("torch_rs.cuda", sys.modules)

    def test_empty_cache_cuda_differential_preserves_cpu_build_behavior(self):
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch build")

        script = r'''
import gc

import torch as reference_torch
import torch_rs as torch

assert reference_torch.__version__.split("+")[0] == "2.13.0"
assert reference_torch.cuda.is_available()
assert torch.accelerator.empty_cache is torch.accelerator.memory.empty_cache
assert torch.accelerator.empty_cache.__code__.co_names == ()

torch_rs_state = (
    torch.accelerator.current_accelerator(),
    torch.accelerator.is_available(),
    torch.accelerator.device_count(),
    torch._C._has_cuda,
    torch.version.cuda,
)
assert torch_rs_state == (None, False, 0, False, None)
assert tuple(torch.accelerator.empty_cache() for _ in range(8)) == (None,) * 8

device_index = reference_torch.cuda.current_device()
device = reference_torch.device("cuda", device_index)
assert reference_torch.accelerator.empty_cache() is None
reference_torch.cuda.synchronize(device_index)
baseline = (
    reference_torch.cuda.memory_allocated(device_index),
    reference_torch.cuda.memory_reserved(device_index),
)

allocation_bytes = 16 * 1024 * 1024
probe = reference_torch.empty(
    allocation_bytes,
    dtype=reference_torch.uint8,
    device=device,
)
probe.fill_(7)
reference_torch.cuda.synchronize(device_index)
assert probe[0].item() == 7
assert probe[-1].item() == 7
live = (
    reference_torch.cuda.memory_allocated(device_index),
    reference_torch.cuda.memory_reserved(device_index),
)
assert live[0] >= baseline[0] + allocation_bytes
assert live[1] >= live[0]

del probe
gc.collect()
reference_torch.cuda.synchronize(device_index)
cached = (
    reference_torch.cuda.memory_allocated(device_index),
    reference_torch.cuda.memory_reserved(device_index),
)
assert cached[0] == baseline[0]
assert cached[1] > baseline[1]

assert tuple(torch.accelerator.empty_cache() for _ in range(8)) == (None,) * 8
assert (
    reference_torch.cuda.memory_allocated(device_index),
    reference_torch.cuda.memory_reserved(device_index),
) == cached

assert reference_torch.accelerator.empty_cache() is None
reference_torch.cuda.synchronize(device_index)
emptied = (
    reference_torch.cuda.memory_allocated(device_index),
    reference_torch.cuda.memory_reserved(device_index),
)
assert emptied[0] == baseline[0]
assert emptied[1] <= baseline[1]
assert emptied[1] < cached[1]
assert (
    torch.accelerator.current_accelerator(),
    torch.accelerator.is_available(),
    torch.accelerator.device_count(),
    torch._C._has_cuda,
    torch.version.cuda,
) == torch_rs_state
assert not hasattr(torch, "cuda")
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

    def test_memory_queries_and_peak_reset_h100_allocation_differential(self):
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch build")

        device_name = reference_torch.cuda.get_device_name(0)
        if "H100" not in device_name:
            self.skipTest(f"requires an NVIDIA H100, found {device_name}")

        script = r'''
import gc
from collections import OrderedDict

import torch as reference_torch
import torch_rs as torch

assert reference_torch.__version__.split("+")[0] == "2.13.0"
assert reference_torch.cuda.is_available()
assert not reference_torch._C._accelerator_isAllocatorInitialized()
assert torch.accelerator.memory_allocated is torch.accelerator.memory.memory_allocated
assert torch.accelerator.memory_allocated.__code__.co_names == ("memory_stats", "get")
assert torch.accelerator.max_memory_allocated is torch.accelerator.memory.max_memory_allocated
assert torch.accelerator.max_memory_allocated.__code__.co_names == ("memory_stats", "get")
assert torch.accelerator.max_memory_reserved is torch.accelerator.memory.max_memory_reserved
assert torch.accelerator.max_memory_reserved.__code__.co_names == ("memory_stats", "get")
assert torch.accelerator.memory_reserved is torch.accelerator.memory.memory_reserved
assert torch.accelerator.memory_reserved.__code__.co_names == ("memory_stats", "get")
assert torch.accelerator.memory_stats is torch.accelerator.memory.memory_stats
assert torch.accelerator.memory_stats.__code__.co_names == ("_OrderedDict",)
assert torch.accelerator.reset_peak_memory_stats is torch.accelerator.memory.reset_peak_memory_stats
assert torch.accelerator.reset_peak_memory_stats.__code__.co_names == ()
assert reference_torch.accelerator.reset_peak_memory_stats is reference_torch.accelerator.memory.reset_peak_memory_stats

class ExplodingDeviceToken:
    def __bool__(self):
        raise AssertionError("device token truth value was inspected")

    def __index__(self):
        raise AssertionError("device token index was inspected")

    def __int__(self):
        raise AssertionError("device token integer value was inspected")

    def __str__(self):
        raise AssertionError("device token string value was inspected")

def reject_discovery():
    raise AssertionError("torch-rs accelerator discovery was attempted")

torch.accelerator._discover_accelerator = reject_discovery
torch_rs_reset_before = [
    torch.accelerator.reset_peak_memory_stats(),
    torch.accelerator.reset_peak_memory_stats("cuda:0"),
    torch.accelerator.reset_peak_memory_stats(ExplodingDeviceToken()),
]
assert torch_rs_reset_before == [None, None, None]
torch_rs_before = [
    torch.accelerator.memory_stats(),
    torch.accelerator.memory_stats("cuda:0"),
    torch.accelerator.memory_stats(ExplodingDeviceToken()),
]
assert all(type(stats) is OrderedDict and not stats for stats in torch_rs_before)
assert len({id(stats) for stats in torch_rs_before}) == len(torch_rs_before)
torch_rs_allocated_before = [
    torch.accelerator.memory_allocated(),
    torch.accelerator.memory_allocated("cuda:0"),
    torch.accelerator.memory_allocated(ExplodingDeviceToken()),
]
assert torch_rs_allocated_before == [0, 0, 0]
assert all(type(value) is int for value in torch_rs_allocated_before)
torch_rs_max_allocated_before = [
    torch.accelerator.max_memory_allocated(),
    torch.accelerator.max_memory_allocated("cuda:0"),
    torch.accelerator.max_memory_allocated(ExplodingDeviceToken()),
]
assert torch_rs_max_allocated_before == [0, 0, 0]
assert all(type(value) is int for value in torch_rs_max_allocated_before)
torch_rs_max_reserved_before = [
    torch.accelerator.max_memory_reserved(),
    torch.accelerator.max_memory_reserved("cuda:0"),
    torch.accelerator.max_memory_reserved(ExplodingDeviceToken()),
]
assert torch_rs_max_reserved_before == [0, 0, 0]
assert all(type(value) is int for value in torch_rs_max_reserved_before)
torch_rs_reserved_before = [
    torch.accelerator.memory_reserved(),
    torch.accelerator.memory_reserved("cuda:0"),
    torch.accelerator.memory_reserved(ExplodingDeviceToken()),
]
assert torch_rs_reserved_before == [0, 0, 0]
assert all(type(value) is int for value in torch_rs_reserved_before)
assert not reference_torch._C._accelerator_isAllocatorInitialized()

reference_before = reference_torch.accelerator.memory_stats(
    ExplodingDeviceToken()
)
assert type(reference_before) is OrderedDict and not reference_before
reference_allocated_before = reference_torch.accelerator.memory_allocated(
    ExplodingDeviceToken()
)
assert type(reference_allocated_before) is int
assert reference_allocated_before == 0
reference_max_allocated_before = reference_torch.accelerator.max_memory_allocated(
    ExplodingDeviceToken()
)
assert type(reference_max_allocated_before) is int
assert reference_max_allocated_before == 0
reference_max_reserved_before = reference_torch.accelerator.max_memory_reserved(
    ExplodingDeviceToken()
)
assert type(reference_max_reserved_before) is int
assert reference_max_reserved_before == 0
reference_reserved_before = reference_torch.accelerator.memory_reserved(
    ExplodingDeviceToken()
)
assert type(reference_reserved_before) is int
assert reference_reserved_before == 0
assert (
    reference_torch.accelerator.reset_peak_memory_stats(ExplodingDeviceToken())
    is None
)
assert not reference_torch._C._accelerator_isAllocatorInitialized()

device_index = 0
device = reference_torch.device("cuda", device_index)
allocation_bytes = 16 * 1024 * 1024
probe = reference_torch.empty(
    allocation_bytes,
    dtype=reference_torch.uint8,
    device=device,
)
probe.fill_(7)
reference_torch.cuda.synchronize(device_index)
assert "H100" in reference_torch.cuda.get_device_name(device_index)
assert probe[0].item() == 7
assert probe[-1].item() == 7
assert reference_torch._C._accelerator_isAllocatorInitialized()

reference_after = reference_torch.accelerator.memory_stats(device_index)
assert type(reference_after) is OrderedDict
assert reference_after
assert list(reference_after) == sorted(reference_after)
assert type(reference_after["allocated_bytes.all.current"]) is int
assert reference_after["allocated_bytes.all.current"] >= allocation_bytes
assert type(reference_after["allocated_bytes.all.peak"]) is int
assert (
    reference_after["allocated_bytes.all.peak"]
    >= reference_after["allocated_bytes.all.current"]
)
assert type(reference_after["reserved_bytes.all.current"]) is int
assert (
    reference_after["reserved_bytes.all.current"]
    >= reference_after["allocated_bytes.all.current"]
)
assert type(reference_after["reserved_bytes.all.peak"]) is int
assert (
    reference_after["reserved_bytes.all.peak"]
    >= reference_after["reserved_bytes.all.current"]
)
assert type(reference_after["allocation.all.current"]) is int
assert reference_after["allocation.all.current"] >= 1
reference_allocated_after = reference_torch.accelerator.memory_allocated(device_index)
assert type(reference_allocated_after) is int
assert reference_allocated_after == reference_after["allocated_bytes.all.current"]
assert reference_allocated_after >= reference_allocated_before + allocation_bytes
reference_max_allocated_after = (
    reference_torch.accelerator.max_memory_allocated(device_index)
)
assert type(reference_max_allocated_after) is int
assert reference_max_allocated_after == reference_after["allocated_bytes.all.peak"]
assert (
    reference_max_allocated_after
    >= reference_max_allocated_before + allocation_bytes
)
reference_max_reserved_after = (
    reference_torch.accelerator.max_memory_reserved(device_index)
)
assert type(reference_max_reserved_after) is int
assert reference_max_reserved_after == reference_after["reserved_bytes.all.peak"]
assert (
    reference_max_reserved_after
    >= reference_max_reserved_before + allocation_bytes
)
reference_reserved_after = reference_torch.accelerator.memory_reserved(device_index)
assert type(reference_reserved_after) is int
assert reference_reserved_after == reference_after["reserved_bytes.all.current"]
assert reference_reserved_after > reference_reserved_before

torch_rs_after = [
    torch.accelerator.memory_stats(),
    torch.accelerator.memory_stats(device_index),
    torch.accelerator.memory_stats(reference_torch.device("cuda:0")),
]
assert all(type(stats) is OrderedDict and not stats for stats in torch_rs_after)
assert len({id(stats) for stats in torch_rs_after}) == len(torch_rs_after)
torch_rs_allocated_after = [
    torch.accelerator.memory_allocated(),
    torch.accelerator.memory_allocated(device_index),
    torch.accelerator.memory_allocated(reference_torch.device("cuda:0")),
]
assert torch_rs_allocated_after == [0, 0, 0]
assert all(type(value) is int for value in torch_rs_allocated_after)
torch_rs_max_allocated_after = [
    torch.accelerator.max_memory_allocated(),
    torch.accelerator.max_memory_allocated(device_index),
    torch.accelerator.max_memory_allocated(
        reference_torch.device("cuda:0")
    ),
]
assert torch_rs_max_allocated_after == [0, 0, 0]
assert all(type(value) is int for value in torch_rs_max_allocated_after)
torch_rs_max_reserved_after = [
    torch.accelerator.max_memory_reserved(),
    torch.accelerator.max_memory_reserved(device_index),
    torch.accelerator.max_memory_reserved(
        reference_torch.device("cuda:0")
    ),
]
assert torch_rs_max_reserved_after == [0, 0, 0]
assert all(type(value) is int for value in torch_rs_max_reserved_after)
torch_rs_reserved_after = [
    torch.accelerator.memory_reserved(),
    torch.accelerator.memory_reserved(device_index),
    torch.accelerator.memory_reserved(reference_torch.device("cuda:0")),
]
assert torch_rs_reserved_after == [0, 0, 0]
assert all(type(value) is int for value in torch_rs_reserved_after)
assert torch.accelerator.reset_peak_memory_stats(ExplodingDeviceToken()) is None
assert torch.accelerator.memory_allocated() == 0
assert torch.accelerator.max_memory_allocated() == 0
assert torch.accelerator.memory_reserved() == 0
assert torch.accelerator.max_memory_reserved() == 0

del probe
gc.collect()
reference_torch.cuda.synchronize(device_index)
reference_released = reference_torch.accelerator.memory_stats(device_index)
assert type(reference_released) is OrderedDict
assert (
    reference_released["allocated_bytes.all.current"]
    == reference_allocated_before
)
assert (
    reference_released["allocated_bytes.all.peak"]
    == reference_max_allocated_after
)
assert (
    reference_released["reserved_bytes.all.peak"]
    == reference_max_reserved_after
)
reference_max_allocated_released = (
    reference_torch.accelerator.max_memory_allocated(device_index)
)
assert type(reference_max_allocated_released) is int
assert reference_max_allocated_released == reference_max_allocated_after
assert (
    reference_max_allocated_released
    > reference_torch.accelerator.memory_allocated(device_index)
)
reference_max_reserved_released = (
    reference_torch.accelerator.max_memory_reserved(device_index)
)
assert type(reference_max_reserved_released) is int
assert reference_max_reserved_released == reference_max_reserved_after
assert reference_torch.accelerator.reset_peak_memory_stats(device_index) is None
reference_reset = reference_torch.accelerator.memory_stats(device_index)
assert (
    reference_reset["allocated_bytes.all.peak"]
    == reference_reset["allocated_bytes.all.current"]
)
assert (
    reference_reset["reserved_bytes.all.peak"]
    == reference_reset["reserved_bytes.all.current"]
)
assert reference_torch.accelerator.max_memory_allocated(device_index) == (
    reference_torch.accelerator.memory_allocated(device_index)
)
assert reference_torch.accelerator.max_memory_reserved(device_index) == (
    reference_torch.accelerator.memory_reserved(device_index)
)
torch_rs_reset_released = [
    torch.accelerator.reset_peak_memory_stats(),
    torch.accelerator.reset_peak_memory_stats(device_index),
    torch.accelerator.reset_peak_memory_stats(ExplodingDeviceToken()),
]
assert torch_rs_reset_released == [None, None, None]
torch_rs_max_allocated_released = [
    torch.accelerator.max_memory_allocated(),
    torch.accelerator.max_memory_allocated(device_index),
    torch.accelerator.max_memory_allocated(ExplodingDeviceToken()),
]
assert torch_rs_max_allocated_released == [0, 0, 0]
assert all(type(value) is int for value in torch_rs_max_allocated_released)
torch_rs_max_reserved_released = [
    torch.accelerator.max_memory_reserved(),
    torch.accelerator.max_memory_reserved(device_index),
    torch.accelerator.max_memory_reserved(ExplodingDeviceToken()),
]
assert torch_rs_max_reserved_released == [0, 0, 0]
assert all(type(value) is int for value in torch_rs_max_reserved_released)
torch_rs_released = (
    torch.accelerator.memory_allocated(ExplodingDeviceToken()),
    torch.accelerator.max_memory_allocated(ExplodingDeviceToken()),
    torch.accelerator.memory_reserved(ExplodingDeviceToken()),
    torch.accelerator.max_memory_reserved(ExplodingDeviceToken()),
)
assert torch_rs_released == (0, 0, 0, 0)
torch_rs_released_stats = torch.accelerator.memory_stats(ExplodingDeviceToken())
assert type(torch_rs_released_stats) is OrderedDict
assert not torch_rs_released_stats
assert not hasattr(torch, "cuda")
assert torch._C._has_cuda is False
assert torch.version.cuda is None
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

    def threaded_outcome(self, module):
        accelerator = module.accelerator
        baseline = (
            accelerator.current_accelerator(),
            accelerator.current_accelerator(True),
            self.call_outcome(accelerator.current_device_index),
            self.call_outcome(accelerator.current_device_index),
            accelerator.is_available(),
            accelerator.device_count(),
            accelerator.memory_allocated(),
            accelerator.memory_allocated(),
            accelerator.max_memory_allocated(),
            accelerator.max_memory_allocated(),
            accelerator.max_memory_reserved(),
            accelerator.max_memory_reserved(),
            accelerator.memory_reserved(),
            accelerator.memory_reserved(),
            accelerator.memory_stats(),
            accelerator.memory_stats(),
        )
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = (
                        module.is_grad_enabled(),
                        accelerator.current_accelerator(),
                        accelerator.current_accelerator(True),
                        self.call_outcome(accelerator.current_device_index),
                        self.call_outcome(accelerator.current_device_index),
                        accelerator.is_available(),
                        accelerator.device_count(),
                        accelerator.memory_allocated(),
                        accelerator.memory_allocated(),
                        accelerator.max_memory_allocated(),
                        accelerator.max_memory_allocated(),
                        accelerator.max_memory_reserved(),
                        accelerator.max_memory_reserved(),
                        accelerator.memory_reserved(),
                        accelerator.memory_reserved(),
                        accelerator.memory_stats(),
                        accelerator.memory_stats(),
                        module.is_grad_enabled(),
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
        return baseline, results

    def test_queries_are_stable_across_threads_and_grad_modes(self):
        for module in (torch, reference_torch):
            with self.subTest(module=module.__name__):
                baseline, results = self.threaded_outcome(module)
                for index, result in enumerate(results):
                    expected_grad_state = index % 2 == 0
                    self.assertEqual(result[0], expected_grad_state)
                    self.assertEqual(result[1:17], baseline)
                    self.assertEqual(result[17], expected_grad_state)
                    self.assertIs(type(result[7]), int)
                    self.assertIs(type(result[8]), int)
                    self.assertIs(type(result[9]), int)
                    self.assertIs(type(result[10]), int)
                    self.assertIs(type(result[11]), int)
                    self.assertIs(type(result[12]), int)
                    self.assertIs(type(result[13]), int)
                    self.assertIs(type(result[14]), int)
                    self.assertIs(type(result[15]), OrderedDict)
                    self.assertIs(type(result[16]), OrderedDict)
                    self.assertIsNot(result[15], result[16])
                self.assertEqual(baseline[2], baseline[3])
                self.assertIs(type(baseline[6]), int)
                self.assertIs(type(baseline[7]), int)
                self.assertIs(type(baseline[8]), int)
                self.assertIs(type(baseline[9]), int)
                self.assertIs(type(baseline[10]), int)
                self.assertIs(type(baseline[11]), int)
                self.assertIs(type(baseline[12]), int)
                self.assertIs(type(baseline[13]), int)
                self.assertIs(type(baseline[14]), OrderedDict)
                self.assertIs(type(baseline[15]), OrderedDict)
                self.assertIsNot(baseline[14], baseline[15])
                if module is torch:
                    self.assertEqual(
                        baseline[2],
                        (
                            "raise",
                            RuntimeError,
                            NO_ACCELERATOR_ERROR,
                            (NO_ACCELERATOR_ERROR,),
                        ),
                    )
                self.assertIs(type(baseline[4]), bool)
                self.assertIs(type(baseline[5]), int)
                all_stats = [baseline[14], baseline[15]]
                all_stats.extend(result[15] for result in results)
                all_stats.extend(result[16] for result in results)
                self.assertEqual(
                    len({id(stats) for stats in all_stats}), len(all_stats)
                )

    def reload_contract(self, module):
        accelerator = module.accelerator
        memory = accelerator.memory
        old_all = accelerator.__all__
        old_empty_cache = accelerator.empty_cache
        old_max_memory_allocated = accelerator.max_memory_allocated
        max_memory_allocated_before = old_max_memory_allocated()
        old_max_memory_reserved = accelerator.max_memory_reserved
        max_memory_reserved_before = old_max_memory_reserved()
        old_memory_allocated = accelerator.memory_allocated
        old_memory_reserved = accelerator.memory_reserved
        old_memory_stats = accelerator.memory_stats
        old_reset_peak_memory_stats = accelerator.reset_peak_memory_stats
        old_functions = {
            name: getattr(accelerator, name) for name in ACCELERATOR_LOCAL
        }
        reloaded = importlib.reload(accelerator)
        new_functions = {
            name: getattr(accelerator, name) for name in ACCELERATOR_LOCAL
        }
        max_memory_allocated_after = accelerator.max_memory_allocated()
        max_memory_reserved_after = accelerator.max_memory_reserved()

        stale_pickle_errors = []
        for old_function in old_functions.values():
            try:
                pickle.dumps(old_function)
            except Exception as error:
                message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error))
                stale_pickle_errors.append(
                    (
                        type(error).__name__,
                        message.replace("torch_rs", "torch"),
                    )
                )
            else:
                self.fail("a stale function unexpectedly remained pickleable")

        return (
            reloaded is accelerator,
            module.accelerator is accelerator,
            sys.modules[accelerator.__name__] is accelerator,
            accelerator.__all__ is not old_all,
            accelerator.memory is memory,
            accelerator.empty_cache is old_empty_cache,
            accelerator.empty_cache is memory.empty_cache,
            accelerator.empty_cache() is None,
            accelerator.max_memory_allocated is old_max_memory_allocated,
            accelerator.max_memory_allocated is memory.max_memory_allocated,
            type(max_memory_allocated_after) is int,
            max_memory_allocated_after == max_memory_allocated_before,
            accelerator.max_memory_reserved is old_max_memory_reserved,
            accelerator.max_memory_reserved is memory.max_memory_reserved,
            type(max_memory_reserved_after) is int,
            max_memory_reserved_after == max_memory_reserved_before,
            accelerator.memory_allocated is old_memory_allocated,
            accelerator.memory_allocated is memory.memory_allocated,
            accelerator.memory_allocated(),
            accelerator.memory_reserved is old_memory_reserved,
            accelerator.memory_reserved is memory.memory_reserved,
            accelerator.memory_reserved(),
            accelerator.memory_stats is old_memory_stats,
            accelerator.memory_stats is memory.memory_stats,
            type(accelerator.memory_stats()) is OrderedDict,
            accelerator.reset_peak_memory_stats is old_reset_peak_memory_stats,
            accelerator.reset_peak_memory_stats
            is memory.reset_peak_memory_stats,
            accelerator.reset_peak_memory_stats() is None,
            tuple(
                old_functions[name] is not new_functions[name]
                for name in sorted(ACCELERATOR_LOCAL)
            ),
            tuple(
                copy.copy(new_functions[name]) is new_functions[name]
                for name in sorted(ACCELERATOR_LOCAL)
            ),
            tuple(
                copy.deepcopy(new_functions[name]) is new_functions[name]
                for name in sorted(ACCELERATOR_LOCAL)
            ),
            tuple(
                pickle.loads(pickle.dumps(new_functions[name]))
                is new_functions[name]
                for name in sorted(ACCELERATOR_LOCAL)
            ),
            tuple(stale_pickle_errors),
        )

    def memory_reload_contract(self, module):
        accelerator = module.accelerator
        memory = accelerator.memory
        old_all = memory.__all__
        old_functions = {name: getattr(memory, name) for name in MEMORY_LOCAL}

        reloaded = importlib.reload(memory)
        new_functions = {name: getattr(memory, name) for name in MEMORY_LOCAL}
        stale_pickle_errors = []
        for name in sorted(MEMORY_LOCAL):
            try:
                pickle.dumps(old_functions[name])
            except Exception as error:
                stale_pickle_errors.append(
                    (
                        type(error).__name__,
                        re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                            "torch_rs", "torch"
                        ),
                    )
                )
            else:
                self.fail("a stale memory function unexpectedly remained pickleable")

        old_reset = old_functions["reset_peak_memory_stats"]()
        new_reset = new_functions["reset_peak_memory_stats"]()
        old_stats = old_functions["memory_stats"]()
        new_stats = new_functions["memory_stats"]()
        old_max_allocated = old_functions["max_memory_allocated"]()
        new_max_allocated = new_functions["max_memory_allocated"]()
        old_max_reserved = old_functions["max_memory_reserved"]()
        new_max_reserved = new_functions["max_memory_reserved"]()
        old_allocated = old_functions["memory_allocated"]()
        new_allocated = new_functions["memory_allocated"]()
        old_reserved = old_functions["memory_reserved"]()
        new_reserved = new_functions["memory_reserved"]()
        result = (
            reloaded is memory,
            accelerator.memory is memory,
            sys.modules[memory.__name__] is memory,
            memory.__all__ is not old_all,
            tuple(
                new_functions[name] is not old_functions[name]
                for name in sorted(MEMORY_LOCAL)
            ),
            tuple(
                getattr(accelerator, name) is old_functions[name]
                for name in sorted(MEMORY_LOCAL)
            ),
            tuple(
                getattr(accelerator, name) is not new_functions[name]
                for name in sorted(MEMORY_LOCAL)
            ),
            old_functions["empty_cache"]() is None,
            new_functions["empty_cache"]() is None,
            type(old_stats) is OrderedDict,
            type(new_stats) is OrderedDict,
            old_stats is not new_stats,
            type(old_max_allocated) is int,
            type(new_max_allocated) is int,
            old_max_allocated == new_max_allocated,
            type(old_max_reserved) is int,
            type(new_max_reserved) is int,
            old_max_reserved == new_max_reserved,
            type(old_allocated) is int,
            type(new_allocated) is int,
            old_allocated,
            new_allocated,
            type(old_reserved) is int,
            type(new_reserved) is int,
            old_reserved,
            new_reserved,
            old_reset is None,
            new_reset is None,
            tuple(
                copy.copy(new_functions[name]) is new_functions[name]
                for name in sorted(MEMORY_LOCAL)
            ),
            tuple(
                copy.deepcopy(new_functions[name]) is new_functions[name]
                for name in sorted(MEMORY_LOCAL)
            ),
            tuple(
                pickle.loads(pickle.dumps(new_functions[name]))
                is new_functions[name]
                for name in sorted(MEMORY_LOCAL)
            ),
            tuple(stale_pickle_errors),
        )

        importlib.reload(accelerator)
        return result + (
            accelerator.empty_cache is new_functions["empty_cache"],
            accelerator.empty_cache is memory.empty_cache,
            accelerator.empty_cache() is None,
            accelerator.max_memory_allocated
            is new_functions["max_memory_allocated"],
            accelerator.max_memory_allocated is memory.max_memory_allocated,
            type(accelerator.max_memory_allocated()) is int,
            accelerator.max_memory_allocated() == new_max_allocated,
            accelerator.max_memory_reserved
            is new_functions["max_memory_reserved"],
            accelerator.max_memory_reserved is memory.max_memory_reserved,
            type(accelerator.max_memory_reserved()) is int,
            accelerator.max_memory_reserved() == new_max_reserved,
            accelerator.memory_allocated is new_functions["memory_allocated"],
            accelerator.memory_allocated is memory.memory_allocated,
            accelerator.memory_allocated(),
            accelerator.memory_reserved is new_functions["memory_reserved"],
            accelerator.memory_reserved is memory.memory_reserved,
            accelerator.memory_reserved(),
            accelerator.memory_stats is new_functions["memory_stats"],
            accelerator.memory_stats is memory.memory_stats,
            type(accelerator.memory_stats()) is OrderedDict,
            accelerator.reset_peak_memory_stats
            is new_functions["reset_peak_memory_stats"],
            accelerator.reset_peak_memory_stats
            is memory.reset_peak_memory_stats,
            accelerator.reset_peak_memory_stats() is None,
        )

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )
        self.assertEqual(
            self.memory_reload_contract(torch),
            self.memory_reload_contract(reference_torch),
        )
        self.assertEqual(
            torch.accelerator.__all__,
            [
                name
                for name in reference_torch.accelerator.__all__
                if name in SUPPORTED
            ],
        )
        for name in sorted(SUPPORTED):
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                self.assertEqual(
                    self.pickle_shape(
                        getattr(torch.accelerator, name), protocol
                    ),
                    self.pickle_shape(
                        getattr(reference_torch.accelerator, name), protocol
                    ),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.accelerator
        expected = reference_torch.accelerator
        cases = (
            (
                lambda: actual.current_accelerator(False, False),
                lambda: expected.current_accelerator(False, False),
            ),
            (
                lambda: actual.current_accelerator(
                    False, check_available=True
                ),
                lambda: expected.current_accelerator(
                    False, check_available=True
                ),
            ),
            (
                lambda: actual.current_accelerator(unexpected=True),
                lambda: expected.current_accelerator(unexpected=True),
            ),
            (
                lambda: actual.current_device_index(None),
                lambda: expected.current_device_index(None),
            ),
            (
                lambda: actual.current_device_index(None, None),
                lambda: expected.current_device_index(None, None),
            ),
            (
                lambda: actual.current_device_index(device=True),
                lambda: expected.current_device_index(device=True),
            ),
            (
                lambda: actual.empty_cache(None),
                lambda: expected.empty_cache(None),
            ),
            (
                lambda: actual.empty_cache(None, None),
                lambda: expected.empty_cache(None, None),
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
            (
                lambda: actual.is_available(None),
                lambda: expected.is_available(None),
            ),
            (
                lambda: actual.is_available(None, None),
                lambda: expected.is_available(None, None),
            ),
            (
                lambda: actual.is_available(device=True),
                lambda: expected.is_available(device=True),
            ),
            (
                lambda: actual.device_count(None),
                lambda: expected.device_count(None),
            ),
            (
                lambda: actual.device_count(None, None),
                lambda: expected.device_count(None, None),
            ),
            (
                lambda: actual.device_count(device=True),
                lambda: expected.device_count(device=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_selection_stream_memory_graph_and_execution_remain_unsupported(self):
        actual = torch.accelerator
        expected = reference_torch.accelerator
        unsupported = set(expected.__all__) - SUPPORTED
        self.assertTrue(
            {
                "Graph",
                "current_stream",
                "device_index",
                "get_memory_info",
                "set_device_index",
                "set_stream",
                "synchronize",
            }.issubset(unsupported)
        )
        for name in unsupported | {"graphs"}:
            with self.subTest(name=name):
                self.assertFalse(hasattr(actual, name))

        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.accelerator.graphs")

        actual_memory = importlib.import_module("torch_rs.accelerator.memory")
        expected_memory = importlib.import_module("torch.accelerator.memory")
        self.assertIs(actual.memory, actual_memory)
        self.assertIs(expected.memory, expected_memory)
        self.assertEqual(
            actual_memory.__all__,
            [name for name in expected_memory.__all__ if name in MEMORY_LOCAL],
        )
        for name in set(expected_memory.__all__) - MEMORY_LOCAL:
            with self.subTest(memory_name=name):
                self.assertFalse(hasattr(actual_memory, name))

        self.assertFalse(hasattr(torch, "cuda"))
        self.assertTrue(hasattr(reference_torch, "cuda"))
        for specification in ("cuda", "cuda:0"):
            with self.subTest(specification=specification):
                with self.assertRaises(RuntimeError):
                    torch.device(specification)
                with self.assertRaises(RuntimeError):
                    torch.tensor([1.0], device=specification)


if __name__ == "__main__":
    unittest.main()

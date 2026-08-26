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
}

ACCELERATOR_LOCAL = SUPPORTED - {"empty_cache"}

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
                    actual_module.memory if name == "empty_cache" else actual_module
                )
                expected_owner = (
                    expected_module.memory
                    if name == "empty_cache"
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
            [name for name in expected_memory.__all__ if name == "empty_cache"],
        )
        self.assertIs(actual_module.empty_cache, actual_memory.empty_cache)
        self.assertIs(expected_module.empty_cache, expected_memory.empty_cache)
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
            {"empty_cache"},
        )
        for name in SUPPORTED:
            with self.subTest(name=name):
                actual = getattr(actual_module, name)
                expected = getattr(expected_module, name)
                self.assertIs(actual_namespace[name], actual)
                self.assertIs(expected_namespace[name], expected)
                if name == "empty_cache":
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

    def threaded_outcome(self, module):
        accelerator = module.accelerator
        baseline = (
            accelerator.current_accelerator(),
            accelerator.current_accelerator(True),
            self.call_outcome(accelerator.current_device_index),
            self.call_outcome(accelerator.current_device_index),
            accelerator.is_available(),
            accelerator.device_count(),
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
                    self.assertEqual(result[1:7], baseline)
                    self.assertEqual(result[7], expected_grad_state)
                self.assertEqual(baseline[2], baseline[3])
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

    def reload_contract(self, module):
        accelerator = module.accelerator
        memory = accelerator.memory
        old_all = accelerator.__all__
        old_empty_cache = accelerator.empty_cache
        old_functions = {
            name: getattr(accelerator, name) for name in ACCELERATOR_LOCAL
        }
        reloaded = importlib.reload(accelerator)
        new_functions = {
            name: getattr(accelerator, name) for name in ACCELERATOR_LOCAL
        }

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
        old_empty_cache = memory.empty_cache

        reloaded = importlib.reload(memory)
        new_empty_cache = memory.empty_cache
        try:
            pickle.dumps(old_empty_cache)
        except Exception as error:
            stale_pickle_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                    "torch_rs", "torch"
                ),
            )
        else:
            self.fail("a stale memory function unexpectedly remained pickleable")

        result = (
            reloaded is memory,
            accelerator.memory is memory,
            sys.modules[memory.__name__] is memory,
            memory.__all__ is not old_all,
            new_empty_cache is not old_empty_cache,
            accelerator.empty_cache is old_empty_cache,
            accelerator.empty_cache is not new_empty_cache,
            old_empty_cache() is None,
            new_empty_cache() is None,
            copy.copy(new_empty_cache) is new_empty_cache,
            copy.deepcopy(new_empty_cache) is new_empty_cache,
            pickle.loads(pickle.dumps(new_empty_cache)) is new_empty_cache,
            stale_pickle_error,
        )

        importlib.reload(accelerator)
        return result + (
            accelerator.empty_cache is new_empty_cache,
            accelerator.empty_cache is memory.empty_cache,
            accelerator.empty_cache() is None,
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
                "memory_stats",
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
            [name for name in expected_memory.__all__ if name == "empty_cache"],
        )
        for name in set(expected_memory.__all__) - {"empty_cache"}:
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

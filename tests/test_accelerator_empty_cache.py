import contextlib
import copy
import importlib
import inspect
import os
import pickle
import re
import subprocess
import sys
import threading
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch


FUNCTION_DOC = """Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other application.

    .. note:: This function is a no-op if the memory allocator for the current
        :ref:`accelerator <accelerators>` has not been initialized.
    """


class AcceleratorEmptyCacheTests(unittest.TestCase):
    def test_returns_exact_none_repeatedly_without_runtime_probes(self):
        accelerator = torch.accelerator
        function = accelerator.empty_cache

        self.assertEqual(function.__code__.co_names, ())
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        environments = (
            {},
            {"CUDA_VISIBLE_DEVICES": ""},
            {"CUDA_VISIBLE_DEVICES": "0"},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "NVIDIA_VISIBLE_DEVICES": "all",
                "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    with mock.patch.object(
                        accelerator,
                        "_discover_accelerator",
                        side_effect=AssertionError("accelerator metadata was queried"),
                    ):
                        for _ in range(8):
                            self.assertIs(function(), None)
                            self.assertIs(accelerator.memory.empty_cache(), None)

        self.assertEqual(
            accelerator._discover_accelerator(), (None, False, 0, None)
        )

    def test_none_is_thread_safe_and_preserves_grad_mode(self):
        function = torch.accelerator.empty_cache
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = (
                        torch.is_grad_enabled(),
                        function(),
                        torch.is_grad_enabled(),
                        torch.accelerator.memory.empty_cache(),
                        torch.is_grad_enabled(),
                    )
            except BaseException as error:
                errors.append(error)

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
        for index, result in enumerate(results):
            expected_grad_state = index % 2 == 0
            self.assertEqual(
                result,
                (
                    expected_grad_state,
                    None,
                    expected_grad_state,
                    None,
                    expected_grad_state,
                ),
            )
            self.assertIs(result[1], None)
            self.assertIs(result[3], None)

    def test_signature_documentation_and_canonical_module_identity(self):
        accelerator = importlib.import_module("torch_rs.accelerator")
        memory = importlib.import_module("torch_rs.accelerator.memory")
        function = memory.empty_cache

        self.assertIs(torch.accelerator, accelerator)
        self.assertIs(accelerator.memory, memory)
        self.assertIs(accelerator.empty_cache, function)
        self.assertIs(sys.modules["torch_rs.accelerator.memory"], memory)
        self.assertIsNone(memory.__doc__)
        self.assertEqual(memory.__all__, ["empty_cache"])
        self.assertEqual(
            {name for name in vars(memory) if not name.startswith("_")},
            {"empty_cache"},
        )
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            inspect.signature(function),
            inspect.Signature(return_annotation=None),
        )
        self.assertEqual(inspect.get_annotations(function), {"return": None})
        self.assertEqual(typing.get_type_hints(function), {"return": type(None)})
        self.assertEqual(function.__name__, "empty_cache")
        self.assertEqual(function.__qualname__, "empty_cache")
        self.assertEqual(function.__module__, "torch_rs.accelerator.memory")
        self.assertIs(inspect.getmodule(function), memory)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_wildcards_copying_and_pickling_are_canonical(self):
        accelerator = torch.accelerator
        memory = accelerator.memory
        function = memory.empty_cache

        accelerator_import = {}
        memory_import = {}
        accelerator_wildcard = {}
        memory_wildcard = {}
        exec("from torch_rs.accelerator import empty_cache", accelerator_import)
        exec("from torch_rs.accelerator.memory import empty_cache", memory_import)
        exec("from torch_rs.accelerator import *", accelerator_wildcard)
        exec("from torch_rs.accelerator.memory import *", memory_wildcard)
        self.assertIs(accelerator_import["empty_cache"], function)
        self.assertIs(memory_import["empty_cache"], function)
        self.assertIs(accelerator_wildcard["empty_cache"], function)
        self.assertEqual(
            {name for name in memory_wildcard if not name.startswith("__")},
            {"empty_cache"},
        )
        self.assertIs(memory_wildcard["empty_cache"], function)

        self.assertNotIn("accelerator", torch.__all__)
        self.assertNotIn("empty_cache", torch.__all__)
        self.assertFalse(hasattr(torch, "empty_cache"))
        top_level_wildcard = {}
        exec("from torch_rs import *", top_level_wildcard)
        self.assertNotIn("accelerator", top_level_wildcard)
        self.assertNotIn("empty_cache", top_level_wildcard)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.accelerator.memory", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_reload_matches_the_parent_and_canonical_submodule_contract(self):
        accelerator = torch.accelerator
        memory = accelerator.memory
        accelerator_namespace = accelerator.__dict__
        memory_namespace = memory.__dict__
        old_accelerator_all = accelerator.__all__
        old_memory_all = memory.__all__
        old_function = memory.empty_cache

        self.assertIs(importlib.reload(accelerator), accelerator)
        self.assertIs(accelerator.__dict__, accelerator_namespace)
        self.assertIs(accelerator.memory, memory)
        self.assertIs(memory.empty_cache, old_function)
        self.assertIs(accelerator.empty_cache, old_function)
        self.assertIsNot(accelerator.__all__, old_accelerator_all)

        self.assertIs(importlib.reload(memory), memory)
        new_function = memory.empty_cache
        self.assertIs(memory.__dict__, memory_namespace)
        self.assertIs(accelerator.memory, memory)
        self.assertIsNot(memory.__all__, old_memory_all)
        self.assertIsNot(new_function, old_function)
        self.assertIs(accelerator.empty_cache, old_function)
        self.assertIs(old_function(), None)
        self.assertIs(new_function(), None)
        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_function)
        self.assertEqual(
            re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception)),
            "Can't pickle <function empty_cache at 0x...>: "
            "it's not the same object as torch_rs.accelerator.memory.empty_cache",
        )

        self.assertIs(importlib.reload(accelerator), accelerator)
        self.assertIs(accelerator.memory, memory)
        self.assertIs(accelerator.empty_cache, new_function)
        self.assertIs(copy.copy(new_function), new_function)
        self.assertIs(copy.deepcopy(new_function), new_function)
        self.assertIs(pickle.loads(pickle.dumps(new_function)), new_function)
        self.assertIs(new_function(), None)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.accelerator.empty_cache
        cases = (
            (
                lambda: function(None),
                "empty_cache() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: function(None, None),
                "empty_cache() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: function(device=True),
                "empty_cache() got an unexpected keyword argument 'device'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_importing_and_calling_does_not_import_or_probe_external_runtimes(self):
        script = r'''
import os
import sys

class RejectExternalRuntimeImport:
    blocked = {
        "amdsmi",
        "cupy",
        "intel_extension_for_pytorch",
        "nvidia",
        "numpy",
        "pyamdgpuinfo",
        "pynvml",
        "torch",
    }

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    CUDA_VISIBLE_DEVICES="0",
    NVIDIA_VISIBLE_DEVICES="all",
    PYTORCH_NVML_BASED_CUDA_CHECK="1",
)
import torch_rs as torch
from torch_rs.accelerator import empty_cache
from torch_rs.accelerator.memory import empty_cache as memory_empty_cache

assert empty_cache is memory_empty_cache is torch.accelerator.empty_cache
assert empty_cache.__code__.co_names == ()
modules_before_calls = set(sys.modules)
for _ in range(16):
    assert empty_cache() is None
    assert torch.accelerator.memory.empty_cache() is None
assert set(sys.modules) == modules_before_calls
assert torch.accelerator._discover_accelerator() == (None, False, 0, None)
assert not hasattr(torch, "cuda")
assert not any(
    name.split(".", 1)[0] in RejectExternalRuntimeImport.blocked
    for name in sys.modules
)
'''
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

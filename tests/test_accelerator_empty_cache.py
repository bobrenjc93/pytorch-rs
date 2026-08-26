import contextlib
import copy
import importlib
import inspect
import os
import pickle
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
    def test_is_a_repeatable_probe_free_cpu_build_noop(self):
        accelerator = torch.accelerator
        memory = importlib.import_module("torch_rs.accelerator.memory")
        function = memory.empty_cache

        self.assertEqual(function.__code__.co_names, ())
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())
        self.assertIs(accelerator.empty_cache, function)

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
                        with mock.patch(
                            "os.cpu_count",
                            side_effect=AssertionError("hardware was probed"),
                        ):
                            modules_before = set(sys.modules)
                            results = tuple(
                                call()
                                for _ in range(8)
                                for call in (function, accelerator.empty_cache)
                            )
                            self.assertEqual(results, (None,) * 16)
                            self.assertEqual(set(sys.modules), modules_before)

    def test_none_is_thread_safe_and_preserves_grad_mode(self):
        accelerator = torch.accelerator
        function = accelerator.memory.empty_cache
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    calls = tuple(
                        call()
                        for _ in range(16)
                        for call in (function, accelerator.empty_cache)
                    )
                    results[index] = (
                        torch.is_grad_enabled(),
                        calls,
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
                (expected_grad_state, (None,) * 32, expected_grad_state),
            )

    def test_signature_documentation_identity_and_exports(self):
        accelerator = importlib.import_module("torch_rs.accelerator")
        memory = importlib.import_module("torch_rs.accelerator.memory")
        function = memory.empty_cache

        self.assertIs(torch.accelerator, accelerator)
        self.assertIs(accelerator.memory, memory)
        self.assertIs(sys.modules["torch_rs.accelerator.memory"], memory)
        self.assertIs(accelerator.empty_cache, function)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(inspect.signature(function), inspect.Signature(return_annotation=None))
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
        self.assertIsNone(memory.__doc__)

        self.assertEqual(memory.__all__, ["empty_cache"])
        self.assertEqual(
            {name for name in vars(memory) if not name.startswith("_")},
            {"empty_cache"},
        )
        self.assertEqual(accelerator.__all__.count("empty_cache"), 1)

        direct_import = {}
        memory_wildcard = {}
        accelerator_wildcard = {}
        exec("from torch_rs.accelerator.memory import empty_cache", direct_import)
        exec("from torch_rs.accelerator.memory import *", memory_wildcard)
        exec("from torch_rs.accelerator import *", accelerator_wildcard)
        self.assertIs(direct_import["empty_cache"], function)
        self.assertEqual(
            {name for name in memory_wildcard if not name.startswith("__")},
            {"empty_cache"},
        )
        self.assertIs(memory_wildcard["empty_cache"], function)
        self.assertIs(accelerator_wildcard["empty_cache"], function)

        self.assertNotIn("empty_cache", torch.__all__)
        top_level_wildcard = {}
        exec("from torch_rs import *", top_level_wildcard)
        self.assertNotIn("empty_cache", top_level_wildcard)
        self.assertFalse(hasattr(torch, "empty_cache"))

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.accelerator.memory", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_package_and_memory_reloads_keep_calls_stable(self):
        accelerator = torch.accelerator
        memory = accelerator.memory
        old_accelerator_all = accelerator.__all__
        old_memory_all = memory.__all__
        original = memory.empty_cache

        self.assertIs(importlib.reload(accelerator), accelerator)
        self.assertIs(torch.accelerator, accelerator)
        self.assertIs(accelerator.memory, memory)
        self.assertIs(accelerator.empty_cache, original)
        self.assertIsNot(accelerator.__all__, old_accelerator_all)
        self.assertIs(accelerator.empty_cache(), None)

        self.assertIs(importlib.reload(memory), memory)
        replacement = memory.empty_cache
        self.assertIs(accelerator.memory, memory)
        self.assertIsNot(memory.__all__, old_memory_all)
        self.assertIsNot(replacement, original)
        self.assertIs(accelerator.empty_cache, original)
        self.assertIs(original(), None)
        self.assertIs(replacement(), None)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(original)
        self.assertIs(pickle.loads(pickle.dumps(replacement)), replacement)

        self.assertIs(importlib.reload(accelerator), accelerator)
        self.assertIs(accelerator.empty_cache, replacement)
        self.assertIs(accelerator.memory.empty_cache, replacement)
        self.assertIs(accelerator.empty_cache(), None)

    def test_argument_errors_match_python_3_binding_used_by_pytorch_2_13(self):
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


if __name__ == "__main__":
    unittest.main()

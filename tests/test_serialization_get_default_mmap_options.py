import contextlib
import copy
import importlib
import inspect
import mmap
import pickle
import subprocess
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch


FUNCTION_DOC = """
    Get default mmap options for :func:`torch.load` with ``mmap=True``.

    Defaults to ``mmap.MAP_PRIVATE``.


    Returns:
        default_mmap_options: int
    """


def platform_default_mmap_options():
    return getattr(mmap, "MAP_PRIVATE", None)


class SerializationGetDefaultMmapOptionsTests(unittest.TestCase):
    def test_returns_the_platform_default_without_changing_grad_mode(self):
        function = torch.serialization.get_default_mmap_options
        expected = platform_default_mmap_options()
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        def assert_query_preserves_grad_mode(expected_grad_state):
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            result = function()
            self.assertEqual(result, expected)
            self.assertIs(type(result), type(expected))
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)

        assert_query_preserves_grad_mode(True)
        with torch.no_grad():
            assert_query_preserves_grad_mode(False)
            with torch.no_grad():
                assert_query_preserves_grad_mode(False)
            assert_query_preserves_grad_mode(False)
        assert_query_preserves_grad_mode(True)

    def test_platform_default_is_stable_across_threads_and_grad_modes(self):
        function = torch.serialization.get_default_mmap_options
        expected = platform_default_mmap_options()
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    first = function()
                    middle = torch.is_grad_enabled()
                    second = function()
                    results[index] = (
                        middle,
                        first,
                        type(first),
                        torch.is_grad_enabled(),
                        second,
                        type(second),
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
                    expected,
                    type(expected),
                    expected_grad_state,
                    expected,
                    type(expected),
                    expected_grad_state,
                ),
            )

    def test_signature_annotations_documentation_and_module_identity(self):
        serialization = importlib.import_module("torch_rs.serialization")
        function = serialization.get_default_mmap_options

        self.assertIs(torch.serialization, serialization)
        self.assertIs(sys.modules["torch_rs.serialization"], serialization)
        self.assertIsNone(serialization.__doc__)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> int | None")
        self.assertEqual(function.__annotations__, {"return": int | None})
        self.assertEqual(typing.get_type_hints(function), {"return": int | None})
        self.assertEqual(function.__name__, "get_default_mmap_options")
        self.assertEqual(function.__qualname__, "get_default_mmap_options")
        self.assertEqual(function.__module__, "torch_rs.serialization")
        self.assertIs(inspect.getmodule(function), serialization)
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(FUNCTION_DOC),
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_exports_copy_and_pickle_use_the_canonical_module(self):
        serialization = torch.serialization
        function = serialization.get_default_mmap_options

        self.assertEqual(
            serialization.__all__,
            [
                "get_crc32_options",
                "set_crc32_options",
                "get_default_mmap_options",
            ],
        )

        package_import = {}
        exec("from torch_rs import serialization", package_import)
        self.assertIs(package_import["serialization"], serialization)

        direct_import = {}
        exec(
            "from torch_rs.serialization import get_default_mmap_options",
            direct_import,
        )
        self.assertIs(direct_import["get_default_mmap_options"], function)

        serialization_namespace = {}
        exec("from torch_rs.serialization import *", serialization_namespace)
        self.assertIs(serialization_namespace["get_default_mmap_options"], function)

        self.assertFalse(hasattr(torch, "get_default_mmap_options"))
        self.assertNotIn("get_default_mmap_options", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("get_default_mmap_options", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.serialization", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_argument_errors_match_pytorch_2_13(self):
        function = torch.serialization.get_default_mmap_options
        expected = function()
        cases = (
            (
                lambda: function(None),
                "get_default_mmap_options() takes 0 positional arguments but 1 "
                "was given",
            ),
            (
                lambda: function(None, None),
                "get_default_mmap_options() takes 0 positional arguments but 2 "
                "were given",
            ),
            (
                lambda: function(flags=None),
                "get_default_mmap_options() got an unexpected keyword argument "
                "'flags'",
            ),
            (
                lambda: function(None, flags=None),
                "get_default_mmap_options() got an unexpected keyword argument "
                "'flags'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(function(), expected)

    def test_reload_and_reimport_preserve_the_platform_default(self):
        original_module = torch.serialization
        original_getter = original_module.get_default_mmap_options
        module_name = original_module.__name__
        expected = platform_default_mmap_options()

        self.assertIs(importlib.reload(original_module), original_module)
        self.assertIs(torch.serialization, original_module)
        self.assertEqual(original_getter(), expected)
        self.assertEqual(original_module.get_default_mmap_options(), expected)

        try:
            self.assertIs(sys.modules.pop(module_name), original_module)
            replacement_module = importlib.import_module(module_name)

            self.assertIsNot(replacement_module, original_module)
            self.assertIs(sys.modules[module_name], replacement_module)
            self.assertIs(torch.serialization, replacement_module)
            self.assertEqual(original_getter(), expected)
            self.assertEqual(replacement_module.get_default_mmap_options(), expected)
        finally:
            sys.modules[module_name] = original_module
            torch.serialization = original_module

    def test_setter_save_and_load_remain_unsupported(self):
        serialization = torch.serialization
        for name in ("set_default_mmap_options", "save", "load"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(serialization, name))
                self.assertNotIn(name, serialization.__all__)

    def test_missing_platform_option_falls_back_to_none(self):
        script = r"""
import mmap

if hasattr(mmap, "MAP_PRIVATE"):
    del mmap.MAP_PRIVATE

import torch_rs as torch

assert torch.serialization.get_default_mmap_options() is None
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

    def test_importing_reloading_and_calling_does_not_import_pytorch(self):
        script = r"""
import importlib
import mmap
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

expected = getattr(mmap, "MAP_PRIVATE", None)
serialization = torch.serialization
getter = serialization.get_default_mmap_options
assert getter() == expected
assert type(getter()) is type(expected)
assert importlib.import_module("torch_rs.serialization") is serialization
assert importlib.reload(serialization) is serialization
assert serialization.get_default_mmap_options() == expected
del sys.modules["torch_rs.serialization"]
replacement = importlib.import_module("torch_rs.serialization")
assert replacement is not serialization
assert torch.serialization is replacement
assert getter() == expected
assert replacement.get_default_mmap_options() == expected
assert replacement.__all__ == [
    "get_crc32_options",
    "set_crc32_options",
    "get_default_mmap_options",
]
assert not hasattr(replacement, "set_default_mmap_options")
assert not hasattr(replacement, "save")
assert not hasattr(replacement, "load")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
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

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


GETTER_DOC = """
    Get default mmap options for :func:`torch.load` with ``mmap=True``.

    Defaults to ``mmap.MAP_PRIVATE``.


    Returns:
        default_mmap_options: int
    """


class SerializationDefaultMmapOptionsTests(unittest.TestCase):
    def test_platform_default_is_exact_and_preserves_grad_mode(self):
        function = torch.serialization.get_default_mmap_options
        expected = getattr(mmap, "MAP_PRIVATE", None)

        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        def assert_query_preserves_grad_mode(expected_grad_state):
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertIs(function(), expected)
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)

        assert_query_preserves_grad_mode(True)
        with torch.no_grad():
            assert_query_preserves_grad_mode(False)
            with torch.no_grad():
                assert_query_preserves_grad_mode(False)
            assert_query_preserves_grad_mode(False)
        assert_query_preserves_grad_mode(True)

        if expected is None:
            self.assertIsNone(function())
        else:
            self.assertIs(type(function()), int)
            self.assertEqual(function(), mmap.MAP_PRIVATE)

    def test_default_is_stable_across_threads_and_grad_modes(self):
        function = torch.serialization.get_default_mmap_options
        expected = getattr(mmap, "MAP_PRIVATE", None)
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
                        function() is expected,
                        torch.is_grad_enabled(),
                        function() is expected,
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
                    True,
                    expected_grad_state,
                    True,
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
        self.assertEqual(function.__module__, "torch_rs.serialization")
        self.assertIs(inspect.getmodule(function), serialization)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(str(inspect.signature(function)), "() -> int | None")
        self.assertEqual(function.__annotations__, {"return": int | None})
        self.assertEqual(typing.get_type_hints(function), {"return": int | None})
        self.assertEqual(function.__name__, "get_default_mmap_options")
        self.assertEqual(function.__qualname__, "get_default_mmap_options")
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(GETTER_DOC),
        )

    def test_imports_exports_copy_and_pickle_use_the_canonical_module(self):
        serialization = torch.serialization
        function = serialization.get_default_mmap_options
        exported_names = [
            "LoadEndianness",
            "get_crc32_options",
            "set_crc32_options",
            "get_default_load_endianness",
            "set_default_load_endianness",
            "get_default_mmap_options",
            "set_default_mmap_options",
            "clear_safe_globals",
            "get_safe_globals",
            "add_safe_globals",
            "safe_globals",
        ]

        self.assertEqual(serialization.__all__, exported_names)

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
        self.assertEqual(
            {
                name
                for name in serialization_namespace
                if not name.startswith("__")
            },
            set(exported_names),
        )
        self.assertIs(serialization_namespace["get_default_mmap_options"], function)

        self.assertNotIn("serialization", torch.__all__)
        self.assertNotIn("get_default_mmap_options", torch.__all__)
        self.assertFalse(hasattr(torch, "get_default_mmap_options"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("serialization", top_level_namespace)
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
                lambda: function(enabled=True),
                "get_default_mmap_options() got an unexpected keyword argument "
                "'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "get_default_mmap_options() got an unexpected keyword argument "
                "'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                before = function()
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(function(), before)

        self.assertIs(function(**{}), getattr(mmap, "MAP_PRIVATE", None))

    def test_reload_and_reimport_preserve_the_platform_default(self):
        original_module = torch.serialization
        original_function = original_module.get_default_mmap_options
        module_name = original_module.__name__
        expected = getattr(mmap, "MAP_PRIVATE", None)

        self.assertIs(importlib.reload(original_module), original_module)
        self.assertIs(torch.serialization, original_module)
        self.assertIs(original_function(), expected)
        self.assertIs(original_module.get_default_mmap_options(), expected)

        try:
            self.assertIs(sys.modules.pop(module_name), original_module)
            replacement_module = importlib.import_module(module_name)

            self.assertIsNot(replacement_module, original_module)
            self.assertIs(sys.modules[module_name], replacement_module)
            self.assertIs(torch.serialization, replacement_module)
            self.assertIs(original_function(), expected)
            self.assertIs(replacement_module.get_default_mmap_options(), expected)
        finally:
            sys.modules[module_name] = original_module
            torch.serialization = original_module

    def test_save_and_load_remain_unsupported(self):
        serialization = torch.serialization

        self.assertTrue(hasattr(serialization, "get_default_mmap_options"))
        self.assertTrue(hasattr(serialization, "set_default_mmap_options"))
        for name in ("save", "load"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(serialization, name))
                self.assertNotIn(name, serialization.__all__)

    def test_unavailable_map_private_returns_none_without_importing_pytorch(self):
        script = r"""
import importlib
import mmap
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

if hasattr(mmap, "MAP_PRIVATE"):
    del mmap.MAP_PRIVATE
sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

serialization = torch.serialization
assert serialization.get_default_mmap_options() is None
assert importlib.reload(serialization) is serialization
assert serialization.get_default_mmap_options() is None
assert hasattr(serialization, "set_default_mmap_options")
try:
    serialization.set_default_mmap_options(None)
except ValueError:
    pass
else:
    raise AssertionError("None must not stand in for an unavailable mmap flag")
if hasattr(mmap, "MAP_SHARED"):
    with serialization.set_default_mmap_options(mmap.MAP_SHARED):
        assert serialization.get_default_mmap_options() == mmap.MAP_SHARED
    assert serialization.get_default_mmap_options() is None
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

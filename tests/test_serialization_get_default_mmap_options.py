import contextlib
import importlib
import inspect
import mmap
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


class SerializationGetDefaultMmapOptionsTests(unittest.TestCase):
    def test_returns_the_exact_platform_default_and_preserves_grad_mode(self):
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

    def test_platform_default_is_stable_across_threads_and_grad_modes(self):
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
        self.assertEqual(str(inspect.signature(function)), "() -> int | None")
        self.assertEqual(function.__annotations__, {"return": int | None})
        self.assertEqual(typing.get_type_hints(function), {"return": int | None})
        self.assertEqual(function.__name__, "get_default_mmap_options")
        self.assertEqual(function.__qualname__, "get_default_mmap_options")
        self.assertEqual(function.__module__, "torch_rs.serialization")
        self.assertIs(inspect.getmodule(function), serialization)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_argument_errors_match_pytorch_2_13(self):
        function = torch.serialization.get_default_mmap_options
        expected = getattr(mmap, "MAP_PRIVATE", None)
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
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(function(), expected)

    def test_reload_and_reimport_keep_the_platform_default(self):
        original_module = torch.serialization
        original_function = original_module.get_default_mmap_options
        module_name = original_module.__name__
        expected = getattr(mmap, "MAP_PRIVATE", None)

        self.assertIs(importlib.reload(original_module), original_module)
        self.assertIs(original_module.get_default_mmap_options(), expected)
        self.assertIs(original_function(), expected)

        try:
            self.assertIs(sys.modules.pop(module_name), original_module)
            replacement_module = importlib.import_module(module_name)
            self.assertIsNot(replacement_module, original_module)
            self.assertIs(torch.serialization, replacement_module)
            self.assertIs(original_function(), expected)
            self.assertIs(replacement_module.get_default_mmap_options(), expected)
        finally:
            sys.modules[module_name] = original_module
            torch.serialization = original_module

    def test_missing_map_private_uses_the_none_default_without_pytorch(self):
        script = r"""
import mmap
import sys

if hasattr(mmap, "MAP_PRIVATE"):
    del mmap.MAP_PRIVATE

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

assert torch.serialization.get_default_mmap_options() is None
assert not hasattr(torch.serialization, "set_default_mmap_options")
assert not hasattr(torch.serialization, "save")
assert not hasattr(torch.serialization, "load")
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

    def test_mmap_mutation_save_and_load_remain_unsupported(self):
        serialization = torch.serialization
        for name in ("set_default_mmap_options", "save", "load"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(serialization, name))
                self.assertNotIn(name, serialization.__all__)

        self.assertFalse(hasattr(torch, "get_default_mmap_options"))
        self.assertNotIn("get_default_mmap_options", torch.__all__)


if __name__ == "__main__":
    unittest.main()

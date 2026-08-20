import contextlib
import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch


FUNCTION_DOC = """
    Get whether :func:`torch.save` computes and writes crc32 for each record.

    Defaults to ``True``.
    """


class SerializationGetCrc32OptionsTests(unittest.TestCase):
    def test_default_true_is_exact_and_preserves_grad_mode(self):
        function = torch.serialization.get_crc32_options
        self.assertEqual(function.__code__.co_names, ())
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        def assert_query_preserves_grad_mode(expected_grad_state):
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertIs(function(), True)
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)

        assert_query_preserves_grad_mode(True)
        with torch.no_grad():
            assert_query_preserves_grad_mode(False)
            with torch.no_grad():
                assert_query_preserves_grad_mode(False)
            assert_query_preserves_grad_mode(False)
        assert_query_preserves_grad_mode(True)

    def test_default_true_is_stable_across_threads_and_grad_modes(self):
        function = torch.serialization.get_crc32_options
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
                        function(),
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
            self.assertIs(result[1], True)
            self.assertIs(result[3], True)

    def test_signature_annotations_documentation_and_module_identity(self):
        serialization = importlib.import_module("torch_rs.serialization")
        function = serialization.get_crc32_options

        self.assertIs(torch.serialization, serialization)
        self.assertIs(sys.modules["torch_rs.serialization"], serialization)
        self.assertIsNone(serialization.__doc__)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> bool")
        self.assertEqual(function.__annotations__, {"return": bool})
        self.assertEqual(typing.get_type_hints(function), {"return": bool})
        self.assertEqual(function.__name__, "get_crc32_options")
        self.assertEqual(function.__qualname__, "get_crc32_options")
        self.assertEqual(function.__module__, "torch_rs.serialization")
        self.assertIs(inspect.getmodule(function), serialization)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_exports_copy_and_pickle_use_the_canonical_module(self):
        serialization = torch.serialization
        function = serialization.get_crc32_options

        self.assertEqual(serialization.__all__, ["get_crc32_options"])

        package_import = {}
        exec("from torch_rs import serialization", package_import)
        self.assertIs(package_import["serialization"], serialization)

        direct_import = {}
        exec("from torch_rs.serialization import get_crc32_options", direct_import)
        self.assertIs(direct_import["get_crc32_options"], function)

        serialization_namespace = {}
        exec("from torch_rs.serialization import *", serialization_namespace)
        self.assertEqual(
            {
                name
                for name in serialization_namespace
                if not name.startswith("__")
            },
            {"get_crc32_options"},
        )
        self.assertIs(serialization_namespace["get_crc32_options"], function)

        self.assertNotIn("serialization", torch.__all__)
        self.assertNotIn("get_crc32_options", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("serialization", top_level_namespace)
        self.assertNotIn("get_crc32_options", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.serialization", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.serialization.get_crc32_options
        cases = (
            (
                lambda: function(None),
                "get_crc32_options() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: function(None, None),
                "get_crc32_options() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: function(enabled=True),
                "get_crc32_options() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "get_crc32_options() got an unexpected keyword argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_mutation_save_load_and_other_serialization_apis_remain_unsupported(self):
        serialization = torch.serialization

        self.assertEqual(
            {name for name in vars(serialization) if not name.startswith("_")},
            {"get_crc32_options"},
        )
        for name in (
            "set_crc32_options",
            "get_default_load_endianness",
            "set_default_load_endianness",
            "get_default_mmap_options",
            "set_default_mmap_options",
            "save",
            "load",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(serialization, name))
                self.assertNotIn(name, serialization.__all__)

        for name in ("get_crc32_options", "set_crc32_options", "save", "load"):
            with self.subTest(top_level_name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

function = torch.serialization.get_crc32_options
assert function.__code__.co_names == ()
assert function() is True
assert torch.serialization.__all__ == ["get_crc32_options"]
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

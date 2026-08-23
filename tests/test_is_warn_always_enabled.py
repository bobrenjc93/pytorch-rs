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


FUNCTION_DOC = """Returns True if the global warn_always flag is turned on. Refer to
    :func:`torch.set_warn_always` documentation for more details.
    """


class IsWarnAlwaysEnabledTests(unittest.TestCase):
    def setUp(self):
        self.original = torch.is_warn_always_enabled()
        torch.set_warn_always(False)

    def tearDown(self):
        torch.set_warn_always(self.original)

    def test_mutable_state_is_exact_and_preserves_grad_mode(self):
        function = torch.is_warn_always_enabled
        self.assertEqual(function.__code__.co_names, ("_C", "_get_warnAlways"))
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        def assert_query_preserves_grad_mode(expected_grad_state, expected_warn_state):
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertIs(function(), expected_warn_state)
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)

        for warn_state in (False, True, False):
            with self.subTest(warn_state=warn_state):
                torch.set_warn_always(warn_state)
                assert_query_preserves_grad_mode(True, warn_state)
                with torch.no_grad():
                    assert_query_preserves_grad_mode(False, warn_state)
                    with torch.no_grad():
                        assert_query_preserves_grad_mode(False, warn_state)
                    assert_query_preserves_grad_mode(False, warn_state)
                assert_query_preserves_grad_mode(True, warn_state)

    def test_default_false_is_stable_across_threads_and_grad_modes(self):
        function = torch.is_warn_always_enabled
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
                    False,
                    expected_grad_state,
                    False,
                    expected_grad_state,
                ),
            )
            self.assertIs(result[1], False)
            self.assertIs(result[3], False)

    def test_signature_annotations_documentation_and_module_identity(self):
        package = importlib.import_module("torch_rs")
        function = package.is_warn_always_enabled

        self.assertIs(torch, package)
        self.assertIs(sys.modules["torch_rs"], package)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> bool")
        self.assertEqual(function.__annotations__, {"return": bool})
        self.assertEqual(typing.get_type_hints(function), {"return": bool})
        self.assertEqual(function.__name__, "is_warn_always_enabled")
        self.assertEqual(function.__qualname__, "is_warn_always_enabled")
        self.assertEqual(function.__module__, "torch_rs")
        self.assertIs(inspect.getmodule(function), package)
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(FUNCTION_DOC),
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_copy_and_pickle_use_the_canonical_module(self):
        function = torch.is_warn_always_enabled

        self.assertEqual(torch.__all__.count("is_warn_always_enabled"), 1)
        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["is_warn_always_enabled"], function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.is_warn_always_enabled
        cases = (
            (
                lambda: function(None),
                "is_warn_always_enabled() takes 0 positional arguments but 1 "
                "was given",
            ),
            (
                lambda: function(None, None),
                "is_warn_always_enabled() takes 0 positional arguments but 2 "
                "were given",
            ),
            (
                lambda: function(enabled=True),
                "is_warn_always_enabled() got an unexpected keyword argument "
                "'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "is_warn_always_enabled() got an unexpected keyword argument "
                "'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_warn_always_setter_and_private_native_state_are_exposed(self):
        self.assertTrue(hasattr(torch, "set_warn_always"))
        self.assertEqual(torch.__all__.count("set_warn_always"), 1)
        self.assertTrue(hasattr(torch._C, "_set_warnAlways"))
        self.assertTrue(hasattr(torch._C, "_get_warnAlways"))
        self.assertFalse(hasattr(torch, "_set_warnAlways"))
        self.assertFalse(hasattr(torch, "_get_warnAlways"))
        self.assertNotIn("_set_warnAlways", torch._C.__all__)
        self.assertNotIn("_get_warnAlways", torch._C.__all__)

    def test_importing_the_package_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

assert torch.is_warn_always_enabled() is False
assert torch.set_warn_always(True) is None
assert torch.is_warn_always_enabled() is True
assert torch.set_warn_always(False) is None
assert torch.is_warn_always_enabled() is False
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

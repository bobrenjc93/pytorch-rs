import contextlib
import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import threading
import types
import unittest

import numpy as np
import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature


FUNCTION_DOC = "Returns True if multithreading is currently enabled."


class IsMultithreadingEnabledTests(unittest.TestCase):
    def tearDown(self):
        torch._C._set_multithreading_enabled(True)

    def test_default_and_direct_mutation_preserve_grad_mode(self):
        query = torch.autograd.is_multithreading_enabled
        setter = torch._C._set_multithreading_enabled

        self.assertIs(query(), True)
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                self.assertIs(setter(enabled), None)
                self.assertIs(query(), enabled)
                self.assertIs(torch._C._is_multithreading_enabled(), enabled)
                self.assertIs(torch.is_grad_enabled(), True)
                with torch.no_grad():
                    self.assertIs(query(), enabled)
                    self.assertIs(torch.is_grad_enabled(), False)
                self.assertIs(query(), enabled)
                self.assertIs(torch.is_grad_enabled(), True)

    def test_state_is_thread_local_and_defaults_true_in_new_threads(self):
        query = torch.autograd.is_multithreading_enabled
        setter = torch._C._set_multithreading_enabled
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        setter(False)

        def worker(index):
            try:
                initial = query()
                enabled = index % 2 == 0
                setter(enabled)
                grad_context = (
                    contextlib.nullcontext() if enabled else torch.no_grad()
                )
                with grad_context:
                    barrier.wait(timeout=10)
                    results[index] = (
                        initial,
                        query(),
                        torch._C._is_multithreading_enabled(),
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
            enabled = index % 2 == 0
            self.assertEqual(result, (True, enabled, enabled, enabled))
            self.assertIs(result[0], True)
            self.assertIs(result[1], enabled)
            self.assertIs(result[2], enabled)
        self.assertIs(query(), False)

    def test_native_builtins_metadata_imports_copying_and_pickling(self):
        query = torch.autograd.is_multithreading_enabled
        setter = torch._C._set_multithreading_enabled

        self.assertIs(type(query), types.BuiltinFunctionType)
        self.assertEqual(query.__name__, "_is_multithreading_enabled")
        self.assertEqual(query.__qualname__, "_is_multithreading_enabled")
        self.assertEqual(query.__module__, torch.tensor.__module__)
        self.assertEqual(query.__doc__, FUNCTION_DOC)
        self.assertFalse(hasattr(query, "__annotations__"))
        self.assertEqual(repr(query), "<built-in function _is_multithreading_enabled>")
        self.assertIs(query.__self__, torch._C)
        self.assertIs(torch._C._is_multithreading_enabled, query)
        assert_no_argument_signature(self, query, "()")

        self.assertIs(type(setter), types.BuiltinFunctionType)
        self.assertEqual(setter.__name__, "_set_multithreading_enabled")
        self.assertEqual(setter.__qualname__, "_set_multithreading_enabled")
        self.assertEqual(setter.__module__, torch.tensor.__module__)
        self.assertIsNone(setter.__doc__)
        self.assertIsNone(setter.__text_signature__)
        self.assertFalse(hasattr(setter, "__annotations__"))
        self.assertEqual(repr(setter), "<built-in function _set_multithreading_enabled>")
        self.assertIs(setter.__self__, torch._C)

        for function in (query, setter):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=function, protocol=protocol):
                    restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                    self.assertIs(restored, function)

        self.assertIs(importlib.import_module("torch_rs.autograd"), torch.autograd)
        self.assertIs(importlib.import_module("torch_rs._C"), torch._C)
        native_namespace = {}
        exec(
            "from torch_rs._C import "
            "_is_multithreading_enabled, _set_multithreading_enabled",
            native_namespace,
        )
        self.assertIs(native_namespace["_is_multithreading_enabled"], query)
        self.assertIs(native_namespace["_set_multithreading_enabled"], setter)

    def test_public_context_aliases_and_exports(self):
        autograd = importlib.import_module("torch_rs.autograd")
        grad_mode = importlib.import_module("torch_rs.autograd.grad_mode")
        context_type = autograd.set_multithreading_enabled

        self.assertIs(torch.autograd, autograd)
        self.assertIs(autograd.grad_mode, grad_mode)
        self.assertIs(context_type, grad_mode.set_multithreading_enabled)
        self.assertEqual(context_type.__name__, "set_multithreading_enabled")
        self.assertEqual(context_type.__qualname__, "set_multithreading_enabled")
        self.assertEqual(context_type.__module__, "torch_rs.autograd.grad_mode")
        self.assertIs(inspect.getmodule(context_type), grad_mode)
        self.assertEqual(str(inspect.signature(context_type)), "(mode: bool) -> None")

        self.assertNotIn("is_multithreading_enabled", autograd.__all__)
        self.assertIn("set_multithreading_enabled", autograd.__all__)
        self.assertIn("set_multithreading_enabled", grad_mode.__all__)
        self.assertNotIn("_is_multithreading_enabled", torch._C.__all__)
        self.assertNotIn("_set_multithreading_enabled", torch._C.__all__)
        self.assertFalse(hasattr(torch, "is_multithreading_enabled"))
        self.assertFalse(hasattr(torch, "set_multithreading_enabled"))

        autograd_namespace = {}
        exec("from torch_rs.autograd import *", autograd_namespace)
        self.assertNotIn("is_multithreading_enabled", autograd_namespace)
        self.assertIs(
            autograd_namespace["set_multithreading_enabled"], context_type
        )
        grad_mode_namespace = {}
        exec("from torch_rs.autograd.grad_mode import *", grad_mode_namespace)
        self.assertIs(
            grad_mode_namespace["set_multithreading_enabled"], context_type
        )

    def test_query_and_strict_boolean_setter_errors_match_pytorch_2_13(self):
        query = torch.autograd.is_multithreading_enabled
        query_cases = (
            (
                lambda: query(None),
                "torch._C._is_multithreading_enabled() takes no arguments (1 given)",
            ),
            (
                lambda: query(None, None),
                "torch._C._is_multithreading_enabled() takes no arguments (2 given)",
            ),
            (
                lambda: query(enabled=True),
                "torch._C._is_multithreading_enabled() takes no keyword arguments",
            ),
        )
        for call, message in query_cases:
            with self.subTest(query_error=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        setter = torch._C._set_multithreading_enabled
        setter_cases = (
            (
                lambda: setter(),
                'set_multithreading_enabled() missing 1 required positional arguments: "enabled"',
            ),
            (
                lambda: setter(True, False),
                "set_multithreading_enabled() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: setter(True, enabled=False),
                "set_multithreading_enabled() got multiple values for argument 'enabled'",
            ),
            (
                lambda: setter(True, unexpected=False),
                "set_multithreading_enabled() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: setter(None),
                "set_multithreading_enabled(): argument 'enabled' (position 1) must be bool, not NoneType",
            ),
            (
                lambda: setter(enabled=None),
                "set_multithreading_enabled(): argument 'enabled' must be bool, not NoneType",
            ),
            (
                lambda: setter(enabled=None, unexpected=True),
                "set_multithreading_enabled(): argument 'enabled' must be bool, not NoneType",
            ),
            (
                lambda: setter(None, enabled=True),
                "set_multithreading_enabled(): argument 'enabled' (position 1) must be bool, not NoneType",
            ),
            (
                lambda: setter(None, unexpected=True),
                "set_multithreading_enabled(): argument 'enabled' (position 1) must be bool, not NoneType",
            ),
            (
                lambda: setter(1),
                "set_multithreading_enabled(): argument 'enabled' (position 1) must be bool, not int",
            ),
            (
                lambda: setter(enabled=1),
                "set_multithreading_enabled(): argument 'enabled' must be bool, not int",
            ),
            (
                lambda: setter(np.bool_(True)),
                "set_multithreading_enabled(): argument 'enabled' (position 1) must be bool, not numpy.bool",
            ),
        )
        for call, message in setter_cases:
            with self.subTest(setter_error=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(query(), True)

        self.assertIs(query(**{}), True)
        self.assertIs(setter(enabled=False), None)
        self.assertIs(query(), False)

    def test_context_mutates_immediately_and_restores_nested_and_exception_states(self):
        query = torch.autograd.is_multithreading_enabled
        context_type = torch.autograd.set_multithreading_enabled

        outer = context_type(False)
        self.assertEqual(outer.__dict__, {"prev": True, "mode": False})
        self.assertIs(query(), False)
        with outer as entered:
            self.assertIsNone(entered)
            self.assertIs(query(), False)
            inner = context_type(True)
            self.assertIs(query(), True)
            with inner as nested_entered:
                self.assertIsNone(nested_entered)
                self.assertIs(query(), True)
            self.assertIs(query(), False)
        self.assertIs(query(), True)

        error = RuntimeError("restore multithreading state")
        with self.assertRaises(RuntimeError) as raised:
            with context_type(False):
                self.assertIs(query(), False)
                raise error
        self.assertIs(raised.exception, error)
        self.assertIs(query(), True)

    def test_context_decorator_and_generator_restore_callers(self):
        query = torch.autograd.is_multithreading_enabled
        context_type = torch.autograd.set_multithreading_enabled

        decorator = context_type(False)
        self.assertIs(query(), False)

        @decorator
        def decorated():
            return query()

        @decorator
        def generate():
            request = yield query()
            yield request, query()

        torch._C._set_multithreading_enabled(True)
        self.assertIs(decorated(), False)
        self.assertIs(query(), True)

        generator = generate()
        self.assertIs(next(generator), False)
        self.assertIs(query(), True)
        self.assertEqual(generator.send("resume"), ("resume", False))
        self.assertIs(query(), True)

    def test_backward_results_are_identical_in_both_states(self):
        def backward_outcome(enabled):
            values = torch.tensor([-2.0, 0.5, 3.0], requires_grad=True)
            with torch.autograd.set_multithreading_enabled(enabled):
                self.assertIs(
                    torch.autograd.is_multithreading_enabled(), enabled
                )
                loss = (values * values).sum()
                self.assertTrue(loss.requires_grad)
                loss.backward()
            self.assertIs(torch.autograd.is_multithreading_enabled(), True)
            return np.asarray(values.grad).copy()

        enabled_gradient = backward_outcome(True)
        disabled_gradient = backward_outcome(False)
        np.testing.assert_array_equal(enabled_gradient, [-4.0, 1.0, 6.0])
        np.testing.assert_array_equal(disabled_gradient, enabled_gradient)

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
from torch_rs.autograd import (
    is_multithreading_enabled,
    set_multithreading_enabled,
)

assert is_multithreading_enabled is torch._C._is_multithreading_enabled
assert set_multithreading_enabled is torch.autograd.grad_mode.set_multithreading_enabled
assert is_multithreading_enabled() is True
assert torch._C._set_multithreading_enabled(False) is None
assert is_multithreading_enabled() is False
with set_multithreading_enabled(True):
    assert is_multithreading_enabled() is True
assert is_multithreading_enabled() is False
torch._C._set_multithreading_enabled(True)
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

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


def assert_one_argument_signature(test_case, callable_object):
    """Assert CPython's versioned signature for a METH_O-style callable."""
    if sys.version_info >= (3, 13):
        test_case.assertEqual(
            callable_object.__text_signature__, "($self, object, /)"
        )
        test_case.assertEqual(str(inspect.signature(callable_object)), "(object, /)")
    else:
        test_case.assertIsNone(callable_object.__text_signature__)
        with test_case.assertRaises(ValueError):
            inspect.signature(callable_object)


class IsAutocastCacheEnabledTests(unittest.TestCase):
    def preserve_state(self):
        previous = torch.is_autocast_cache_enabled()
        self.addCleanup(torch.set_autocast_cache_enabled, previous)
        return previous

    def test_default_true_and_mutation_preserve_grad_and_execution(self):
        self.assertIs(self.preserve_state(), True)

        def assert_state_preserves_grad(expected_cache, expected_grad):
            self.assertIs(torch.is_grad_enabled(), expected_grad)
            self.assertIs(torch.is_autocast_cache_enabled(), expected_cache)
            values = torch.tensor([1.0, -2.0, 3.0])
            self.assertEqual((values * 2.0).sum().item(), 4.0)
            self.assertIs(torch.is_autocast_cache_enabled(), expected_cache)
            self.assertIs(torch.is_grad_enabled(), expected_grad)

        self.assertIs(torch.set_autocast_cache_enabled(False), None)
        assert_state_preserves_grad(False, True)
        with torch.no_grad():
            assert_state_preserves_grad(False, False)
            self.assertIs(torch.set_autocast_cache_enabled(True), None)
            assert_state_preserves_grad(True, False)
        assert_state_preserves_grad(True, True)

        values = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
        result = (values * 2.0).sum()
        result.backward()
        self.assertEqual(values.grad.tolist(), [2.0, 2.0, 2.0])
        self.assertIs(torch.is_autocast_cache_enabled(), True)

    def test_state_is_thread_local_with_true_defaults_and_isolated_mutation(self):
        self.preserve_state()
        torch.set_autocast_cache_enabled(False)

        worker_count = 8
        ready = threading.Barrier(worker_count + 1)
        release = threading.Barrier(worker_count + 1)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    initial = torch.is_autocast_cache_enabled()
                    selected = index % 3 != 0
                    returned = torch.set_autocast_cache_enabled(selected)
                    after_set = torch.is_autocast_cache_enabled()
                    ready.wait(timeout=10)
                    release.wait(timeout=10)
                    results[index] = (
                        initial,
                        returned,
                        after_set,
                        torch.is_autocast_cache_enabled(),
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

        ready.wait(timeout=10)
        self.assertIs(torch.is_autocast_cache_enabled(), False)
        torch.set_autocast_cache_enabled(True)
        release.wait(timeout=10)
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertIs(torch.is_autocast_cache_enabled(), True)
        for index, result in enumerate(results):
            selected = index % 3 != 0
            self.assertEqual(
                result,
                (True, None, selected, selected, index % 2 == 0),
            )
            self.assertIs(result[0], True)
            self.assertIs(result[2], selected)
            self.assertIs(result[3], selected)

    def test_same_thread_state_survives_native_and_package_reload(self):
        self.preserve_state()
        torch.set_autocast_cache_enabled(False)
        old_getter = torch.is_autocast_cache_enabled
        old_setter = torch.set_autocast_cache_enabled
        native = torch._C

        self.assertIs(importlib.reload(native), native)
        self.assertIs(torch.is_autocast_cache_enabled(), False)
        self.assertIs(native.is_autocast_cache_enabled, old_getter)
        self.assertIs(native.set_autocast_cache_enabled, old_setter)

        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.is_autocast_cache_enabled(), False)
        self.assertIs(torch.is_autocast_cache_enabled, old_getter)
        self.assertIs(torch.set_autocast_cache_enabled, old_setter)

    def test_builtin_metadata_exports_copying_and_pickling(self):
        getter = torch.is_autocast_cache_enabled
        setter = torch.set_autocast_cache_enabled
        for function, name in (
            (getter, "is_autocast_cache_enabled"),
            (setter, "set_autocast_cache_enabled"),
        ):
            with self.subTest(name=name):
                self.assertIs(type(function), types.BuiltinFunctionType)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(function.__module__, torch.tensor.__module__)
                self.assertIsNone(function.__doc__)
                self.assertFalse(hasattr(function, "__annotations__"))
                self.assertEqual(repr(function), f"<built-in function {name}>")
                self.assertIs(function.__self__, torch._C)
                self.assertIs(getattr(torch._C, name), function)
                self.assertEqual(function.__reduce__(), name)
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                self.assertEqual(torch.__all__.count(name), 1)

                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(name=name, protocol=protocol):
                        restored = pickle.loads(
                            pickle.dumps(function, protocol=protocol)
                        )
                        self.assertIs(restored, function)

        assert_no_argument_signature(self, getter, "()")
        assert_one_argument_signature(self, setter)

        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["is_autocast_cache_enabled"], getter)
        self.assertIs(wildcard_namespace["set_autocast_cache_enabled"], setter)

        native_module = importlib.import_module("torch_rs._C")
        self.assertIs(native_module, torch._C)
        explicit_namespace = {}
        exec(
            "from torch_rs._C import "
            "is_autocast_cache_enabled, set_autocast_cache_enabled",
            explicit_namespace,
        )
        self.assertIs(explicit_namespace["is_autocast_cache_enabled"], getter)
        self.assertIs(explicit_namespace["set_autocast_cache_enabled"], setter)

    def test_argument_errors_are_exact_and_do_not_mutate_state(self):
        getter = torch.is_autocast_cache_enabled
        setter = torch.set_autocast_cache_enabled
        self.preserve_state()

        getter_cases = (
            (
                lambda: getter(None),
                "torch.is_autocast_cache_enabled() takes no arguments (1 given)",
            ),
            (
                lambda: getter(None, None),
                "torch.is_autocast_cache_enabled() takes no arguments (2 given)",
            ),
            (
                lambda: getter(enabled=True),
                "torch.is_autocast_cache_enabled() takes no keyword arguments",
            ),
            (
                lambda: getter(None, enabled=True),
                "torch.is_autocast_cache_enabled() takes no keyword arguments",
            ),
        )
        for call, message in getter_cases:
            with self.subTest(api="getter", message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        class BoolLike:
            calls = 0

            def __bool__(self):
                type(self).calls += 1
                return True

        setter_cases = (
            (
                lambda: setter(),
                "torch.set_autocast_cache_enabled() takes exactly one argument "
                "(0 given)",
            ),
            (
                lambda: setter(True, False),
                "torch.set_autocast_cache_enabled() takes exactly one argument "
                "(2 given)",
            ),
            (
                lambda: setter(enabled=True),
                "torch.set_autocast_cache_enabled() takes no keyword arguments",
            ),
            (
                lambda: setter(True, enabled=False),
                "torch.set_autocast_cache_enabled() takes no keyword arguments",
            ),
            (lambda: setter(1), "enabled must be a bool (got int)"),
            (lambda: setter(0), "enabled must be a bool (got int)"),
            (lambda: setter(None), "enabled must be a bool (got NoneType)"),
            (lambda: setter("true"), "enabled must be a bool (got str)"),
            (lambda: setter([]), "enabled must be a bool (got list)"),
            (
                lambda: setter(np.bool_(True)),
                "enabled must be a bool (got numpy.bool)",
            ),
            (
                lambda: setter(np.int64(1)),
                "enabled must be a bool (got numpy.int64)",
            ),
            (
                lambda: setter(torch.tensor([1.0])),
                "enabled must be a bool (got Tensor)",
            ),
            (
                lambda: setter(torch.float32),
                "enabled must be a bool (got torch.dtype)",
            ),
            (
                lambda: setter(torch.device("cpu")),
                "enabled must be a bool (got torch.device)",
            ),
            (
                lambda: setter(torch.preserve_format),
                "enabled must be a bool (got torch.memory_format)",
            ),
            (
                lambda: setter(torch.Size([1])),
                "enabled must be a bool (got torch.Size)",
            ),
            (
                lambda: setter(torch.strided),
                "enabled must be a bool (got torch.layout)",
            ),
            (
                lambda: setter(torch.finfo()),
                "enabled must be a bool (got torch.finfo)",
            ),
            (lambda: setter(BoolLike()), "enabled must be a bool (got BoolLike)"),
        )
        for initial in (True, False):
            for call, message in setter_cases:
                with self.subTest(initial=initial, message=message):
                    setter(initial)
                    with self.assertRaises(TypeError) as raised:
                        call()
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertIs(getter(), initial)

        self.assertEqual(BoolLike.calls, 0)
        self.assertIs(getter(**{}), False)
        self.assertIs(setter(True, **{}), None)
        self.assertIs(getter(), True)

    def test_autocast_contexts_and_mixed_precision_remain_unsupported(self):
        self.assertIs(
            torch._C.set_autocast_cache_enabled,
            torch.set_autocast_cache_enabled,
        )
        self.assertIn("set_autocast_cache_enabled", torch.__all__)
        self.assertFalse(hasattr(torch, "autocast"))
        self.assertFalse(hasattr(torch, "amp"))
        self.assertFalse(hasattr(torch.cpu, "amp"))

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

getter = torch.is_autocast_cache_enabled
setter = torch.set_autocast_cache_enabled
assert getter is torch._C.is_autocast_cache_enabled
assert setter is torch._C.set_autocast_cache_enabled
assert getter() is True
assert setter(False) is None
assert getter() is False
assert setter(True) is None
assert getter() is True
assert not hasattr(torch, "autocast")
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

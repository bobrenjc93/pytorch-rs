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

import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("set_autocast_cache_enabled must not request truthiness")


class AutocastCacheEnabledTests(unittest.TestCase):
    def setUp(self):
        self.original = torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(True)

    def tearDown(self):
        torch.set_autocast_cache_enabled(self.original)

    def test_exact_bool_updates_state_without_changing_grad_or_cpu_execution(self):
        getter = torch.is_autocast_cache_enabled
        setter = torch.set_autocast_cache_enabled

        for state in (False, True, False):
            with self.subTest(state=state):
                before_grad = torch.is_grad_enabled()
                self.assertIs(setter(state), None)
                self.assertIs(getter(), state)

                values = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
                result = (values * 2.0).sum()
                self.assertIs(getter(), state)
                result.backward()
                self.assertEqual(values.grad.tolist(), [2.0, 2.0, 2.0])
                self.assertIs(torch.is_grad_enabled(), before_grad)

                with torch.no_grad():
                    self.assertIs(torch.is_grad_enabled(), False)
                    self.assertIs(setter(not state), None)
                    self.assertIs(getter(), not state)
                    self.assertEqual((values.detach() + 1.0).sum().item(), 5.0)
                    self.assertIs(setter(state), None)

                self.assertIs(torch.is_grad_enabled(), before_grad)
                self.assertIs(getter(), state)

    def test_rejects_non_bool_values_without_coercion_or_state_change(self):
        invalid_values = (
            (None, "NoneType"),
            (0, "int"),
            (1, "int"),
            (0.0, "float"),
            ("", "str"),
            ([], "list"),
            (object(), "object"),
            (_RejectTruthiness(), "_RejectTruthiness"),
            (torch.tensor(True), "Tensor"),
            (torch.float32, "torch.dtype"),
            (torch.device("cpu"), "torch.device"),
        )

        for state in (False, True):
            torch.set_autocast_cache_enabled(state)
            for value, type_name in invalid_values:
                with self.subTest(state=state, value=repr(value)):
                    message = f"enabled must be a bool (got {type_name})"
                    with self.assertRaises(TypeError) as raised:
                        torch.set_autocast_cache_enabled(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertIs(torch.is_autocast_cache_enabled(), state)

    def test_state_defaults_true_and_is_isolated_between_threads(self):
        getter = torch.is_autocast_cache_enabled
        setter = torch.set_autocast_cache_enabled
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        setter(False)

        def worker(index):
            try:
                target = index % 2 == 0
                context = torch.no_grad() if index % 3 == 0 else contextlib.nullcontext()
                with context:
                    before = getter()
                    result = setter(target)
                    barrier.wait(timeout=10)
                    values = torch.tensor([float(index), 1.0])
                    results[index] = (
                        before,
                        result,
                        getter(),
                        torch.is_grad_enabled(),
                        (values + 1.0).sum().item(),
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
            self.assertEqual(
                result,
                (
                    True,
                    None,
                    index % 2 == 0,
                    index % 3 != 0,
                    float(index + 3),
                ),
            )
            self.assertIs(result[0], True)
            self.assertIs(result[2], index % 2 == 0)
        self.assertIs(getter(), False)

    def test_same_thread_state_and_builtin_identity_survive_reloads(self):
        package = importlib.import_module("torch_rs")
        native = package._C
        old_getter = package.is_autocast_cache_enabled
        old_setter = package.set_autocast_cache_enabled

        self.assertIs(old_setter(False), None)
        self.assertIs(importlib.reload(native), native)
        self.assertIs(package._C, native)
        self.assertIs(native.is_autocast_cache_enabled, old_getter)
        self.assertIs(native.set_autocast_cache_enabled, old_setter)
        self.assertIs(old_getter(), False)

        self.assertIs(importlib.reload(package), package)
        self.assertIs(torch, package)
        self.assertIs(package.is_autocast_cache_enabled, old_getter)
        self.assertIs(package.set_autocast_cache_enabled, old_setter)
        self.assertIs(package.is_autocast_cache_enabled(), False)

        self.assertIs(package.set_autocast_cache_enabled(True), None)
        self.assertIs(old_getter(), True)
        self.assertIs(old_setter(False), None)
        self.assertIs(package.is_autocast_cache_enabled(), False)

    def test_builtin_metadata_exports_copying_and_pickling(self):
        functions = (
            (torch.is_autocast_cache_enabled, "is_autocast_cache_enabled", "()"),
            (
                torch.set_autocast_cache_enabled,
                "set_autocast_cache_enabled",
                "(object, /)",
            ),
        )

        for function, name, expected_signature in functions:
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

                if name == "is_autocast_cache_enabled":
                    assert_no_argument_signature(self, function, expected_signature)
                elif sys.version_info >= (3, 13):
                    self.assertEqual(
                        function.__text_signature__, "($self, object, /)"
                    )
                    self.assertEqual(
                        str(inspect.signature(function)), expected_signature
                    )
                else:
                    self.assertIsNone(function.__text_signature__)
                    with self.assertRaises(ValueError):
                        inspect.signature(function)

                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                self.assertEqual(torch.__all__.count(name), 1)
                self.assertEqual(torch._C.__all__.count(name), 1)

                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(name=name, protocol=protocol):
                        restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                        self.assertIs(restored, function)

        public_namespace = {}
        exec("from torch_rs import *", public_namespace)
        native_namespace = {}
        exec("from torch_rs._C import *", native_namespace)
        explicit_namespace = {}
        exec(
            "from torch_rs._C import "
            "is_autocast_cache_enabled, set_autocast_cache_enabled",
            explicit_namespace,
        )
        for function, name, _ in functions:
            self.assertIs(public_namespace[name], function)
            self.assertIs(native_namespace[name], function)
            self.assertIs(explicit_namespace[name], function)

    def test_getter_argument_errors_match_pytorch_2_13(self):
        function = torch.is_autocast_cache_enabled
        cases = (
            (
                lambda: function(None),
                "torch.is_autocast_cache_enabled() takes no arguments (1 given)",
            ),
            (
                lambda: function(None, None),
                "torch.is_autocast_cache_enabled() takes no arguments (2 given)",
            ),
            (
                lambda: function(enabled=True),
                "torch.is_autocast_cache_enabled() takes no keyword arguments",
            ),
            (
                lambda: function(None, enabled=True),
                "torch.is_autocast_cache_enabled() takes no keyword arguments",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        self.assertIs(function(**{}), True)

    def test_setter_binding_errors_do_not_change_state(self):
        function = torch.set_autocast_cache_enabled
        cases = (
            (
                lambda: function(),
                "torch.set_autocast_cache_enabled() takes exactly one argument "
                "(0 given)",
            ),
            (
                lambda: function(False, True),
                "torch.set_autocast_cache_enabled() takes exactly one argument "
                "(2 given)",
            ),
            (
                lambda: function(enabled=True),
                "torch.set_autocast_cache_enabled() takes no keyword arguments",
            ),
            (
                lambda: function(True, enabled=False),
                "torch.set_autocast_cache_enabled() takes no keyword arguments",
            ),
        )

        for state in (False, True):
            torch.set_autocast_cache_enabled(state)
            for call, message in cases:
                with self.subTest(state=state, message=message):
                    with self.assertRaises(TypeError) as raised:
                        call()
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertIs(torch.is_autocast_cache_enabled(), state)

        self.assertIs(function(False, **{}), None)
        self.assertIs(torch.is_autocast_cache_enabled(), False)

    def test_autocast_contexts_and_execution_surfaces_remain_unsupported(self):
        self.assertIs(
            torch.set_autocast_cache_enabled,
            torch._C.set_autocast_cache_enabled,
        )
        self.assertFalse(hasattr(torch, "autocast"))
        self.assertFalse(hasattr(torch, "amp"))
        self.assertFalse(hasattr(torch.cpu, "amp"))

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r"""
import importlib
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
assert importlib.reload(torch._C) is torch._C
assert getter() is False
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

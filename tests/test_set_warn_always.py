import copy
import importlib
import inspect
import json
import pickle
import subprocess
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch


SETTER_DOC = """When this flag is False (default) then some PyTorch warnings may only
    appear once per process. This helps avoid excessive warning information.
    Setting it to True causes these warnings to always appear, which may be
    helpful when debugging.

    Args:
        b (:class:`bool`): If True, force warnings to always be emitted
                           If False, set to the default behaviour
    """


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("set_warn_always must not request truthiness")


class SetWarnAlwaysTests(unittest.TestCase):
    def setUp(self):
        self.original = torch.is_warn_always_enabled()
        torch.set_warn_always(False)

    def tearDown(self):
        torch.set_warn_always(self.original)

    def test_exact_bool_updates_native_state_and_preserves_grad_mode(self):
        for state in (False, True, False):
            with self.subTest(state=state):
                before = torch.is_grad_enabled()
                self.assertIs(torch.set_warn_always(state), None)
                self.assertIs(torch.is_warn_always_enabled(), state)
                self.assertIs(torch.is_grad_enabled(), before)

                with torch.no_grad():
                    self.assertIs(torch.is_grad_enabled(), False)
                    self.assertIs(torch.set_warn_always(not state), None)
                    self.assertIs(torch.is_warn_always_enabled(), not state)
                    self.assertIs(torch.is_grad_enabled(), False)
                    self.assertIs(torch.set_warn_always(state), None)

                self.assertIs(torch.is_grad_enabled(), before)
                self.assertIs(torch.is_warn_always_enabled(), state)

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
            torch.set_warn_always(state)
            for value, type_name in invalid_values:
                with self.subTest(state=state, value=repr(value)):
                    message = (
                        "setWarnOnlyOnce expects a bool, but got "
                        f"{type_name}"
                    )
                    with self.assertRaises(RuntimeError) as raised:
                        torch.set_warn_always(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertIs(torch.is_warn_always_enabled(), state)

    def test_updates_are_process_global_across_threads(self):
        getter = torch.is_warn_always_enabled
        setter = torch.set_warn_always
        worker_ready = threading.Event()
        main_updated = threading.Event()
        worker_reset = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                observations.append(getter())
                worker_ready.set()
                if not main_updated.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(getter())
                observations.append(setter(False))
                observations.append(getter())
                worker_reset.set()
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_ready.wait(timeout=10))
        self.assertIs(setter(True), None)
        self.assertIs(getter(), True)
        main_updated.set()
        self.assertTrue(worker_reset.wait(timeout=10))
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [False, True, None, False])
        self.assertIs(getter(), False)

    def test_reload_preserves_native_state_for_old_and_new_wrappers(self):
        package = importlib.import_module("torch_rs")
        old_getter = package.is_warn_always_enabled
        old_setter = package.set_warn_always

        self.assertIs(old_setter(True), None)
        self.assertIs(importlib.reload(package), package)
        self.assertIs(torch, package)
        self.assertIs(old_getter(), True)
        self.assertIs(package.is_warn_always_enabled(), True)

        self.assertIs(package.set_warn_always(False), None)
        self.assertIs(old_getter(), False)
        self.assertIs(old_setter(True), None)
        self.assertIs(package.is_warn_always_enabled(), True)

    def test_signature_annotations_documentation_and_module_identity(self):
        package = importlib.import_module("torch_rs")
        function = package.set_warn_always

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(b: bool, /) -> None")
        self.assertEqual(function.__annotations__, {"b": bool, "return": None})
        self.assertEqual(
            typing.get_type_hints(function),
            {"b": bool, "return": type(None)},
        )
        self.assertEqual(function.__name__, "set_warn_always")
        self.assertEqual(function.__qualname__, "set_warn_always")
        self.assertEqual(function.__module__, "torch_rs")
        self.assertIs(inspect.getmodule(function), package)
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(SETTER_DOC),
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__code__.co_names, ("_C", "_set_warnAlways"))
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

    def test_exports_copy_and_pickle_use_the_canonical_module(self):
        function = torch.set_warn_always

        self.assertEqual(torch.__all__.count("set_warn_always"), 1)
        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["set_warn_always"], function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_positional_only_binding_errors_match_pytorch_2_13(self):
        function = torch.set_warn_always
        cases = (
            (
                lambda: function(),
                "set_warn_always() missing 1 required positional argument: 'b'",
            ),
            (
                lambda: function(False, True),
                "set_warn_always() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: function(b=True),
                "set_warn_always() got some positional-only arguments passed as "
                "keyword arguments: 'b'",
            ),
            (
                lambda: function(enabled=True),
                "set_warn_always() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: function(False, enabled=True),
                "set_warn_always() got an unexpected keyword argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(torch.is_warn_always_enabled(), False)

    def test_native_warn_once_policy_and_python_warnings_are_independent(self):
        script = r'''
import json
import warnings

import torch_rs as torch


def warning_count(callback, count):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(count):
            callback()
    return len(caught)


scalar = torch.tensor(1.0)
torch.set_warn_always(False)
false_first = warning_count(lambda: scalar.T, 3)
torch.set_warn_always(True)
true_after_consumed = warning_count(lambda: scalar.T, 3)
torch.set_warn_always(False)
false_after_consumed = warning_count(lambda: scalar.T, 3)

torch.set_warn_always(True)
true_before_consumed = warning_count(lambda: scalar.H, 3)
torch.set_warn_always(False)
false_after_true_only = warning_count(lambda: scalar.H, 3)

torch.set_warn_always(True)
python_true = warning_count(lambda: warnings.warn("ordinary", UserWarning), 3)
torch.set_warn_always(False)
python_false = warning_count(lambda: warnings.warn("ordinary", UserWarning), 3)

print(json.dumps({
    "counts": [
        false_first,
        true_after_consumed,
        false_after_consumed,
        true_before_consumed,
        false_after_true_only,
        python_true,
        python_false,
    ],
    "state": torch.is_warn_always_enabled(),
}))
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
        self.assertEqual(
            json.loads(completed.stdout),
            {"counts": [1, 3, 0, 3, 1, 3, 3], "state": False},
        )


if __name__ == "__main__":
    unittest.main()

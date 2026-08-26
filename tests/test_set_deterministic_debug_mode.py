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


FUNCTION_DOC = """Sets the debug mode for deterministic operations.

    .. note:: This is an alternative interface for
        :func:`torch.use_deterministic_algorithms`. Refer to that function's
        documentation for details about affected operations.

    Args:
        debug_mode(str or int): If "default" or 0, don't error or warn on
            nondeterministic operations. If "warn" or 1, warn on
            nondeterministic operations. If "error" or 2, error on
            nondeterministic operations.
    """


class _DefaultInt(int):
    pass


class _DefaultString(str):
    pass


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("the setter must not request truthiness")


class SetDeterministicDebugModeTests(unittest.TestCase):
    def assert_default_state(self):
        debug_mode = torch.get_deterministic_debug_mode()
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

        self.assertIs(type(debug_mode), int)
        self.assertEqual(debug_mode, 0)
        self.assertIs(enabled, False)
        self.assertIs(warn_only, False)

    def test_default_forms_are_idempotent_noops_across_grad_modes(self):
        calls = (
            lambda: torch.set_deterministic_debug_mode(0),
            lambda: torch.set_deterministic_debug_mode(debug_mode=0),
            lambda: torch.set_deterministic_debug_mode(False),
            lambda: torch.set_deterministic_debug_mode("default"),
            lambda: torch.set_deterministic_debug_mode(_DefaultInt(0)),
            lambda: torch.set_deterministic_debug_mode(
                _DefaultString("default")
            ),
        )

        for context in (contextlib.nullcontext(), torch.no_grad()):
            with context:
                expected_grad_mode = torch.is_grad_enabled()
                for case, call in enumerate(calls):
                    with self.subTest(grad=expected_grad_mode, case=case):
                        self.assertIs(call(), None)
                        self.assert_default_state()
                        self.assertIs(torch.is_grad_enabled(), expected_grad_mode)

        self.assertIs(torch.is_grad_enabled(), True)
        self.assert_default_state()

    def test_default_state_is_coherent_across_threads(self):
        modes = (0, False, "default", _DefaultInt(0), _DefaultString("default"))
        worker_count = 10
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    expected_grad_mode = torch.is_grad_enabled()
                    barrier.wait(timeout=10)
                    returned = torch.set_deterministic_debug_mode(
                        modes[index % len(modes)]
                    )
                    results[index] = (
                        returned,
                        torch.get_deterministic_debug_mode(),
                        torch.are_deterministic_algorithms_enabled(),
                        torch.is_deterministic_algorithms_warn_only_enabled(),
                        torch.is_grad_enabled(),
                        expected_grad_mode,
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
            expected_grad_mode = index % 2 == 0
            self.assertEqual(
                result,
                (
                    None,
                    0,
                    False,
                    False,
                    expected_grad_mode,
                    expected_grad_mode,
                ),
            )
        self.assert_default_state()

    def test_reload_preserves_the_default_state_for_old_and_new_functions(self):
        package = importlib.import_module("torch_rs")
        old_setter = package.set_deterministic_debug_mode
        old_debug_query = package.get_deterministic_debug_mode
        old_enabled_query = package.are_deterministic_algorithms_enabled
        old_warn_only_query = package.is_deterministic_algorithms_warn_only_enabled

        self.assertIs(old_setter("default"), None)
        self.assertIs(importlib.reload(package), package)
        self.assertIs(torch, package)
        self.assertIsNot(package.set_deterministic_debug_mode, old_setter)

        for setter, mode in (
            (old_setter, 0),
            (package.set_deterministic_debug_mode, False),
            (old_setter, "default"),
        ):
            with self.subTest(setter=setter.__name__, mode=mode):
                self.assertIs(setter(mode), None)
                self.assertEqual(old_debug_query(), 0)
                self.assertIs(old_enabled_query(), False)
                self.assertIs(old_warn_only_query(), False)
                self.assert_default_state()

    def test_nondefault_modes_are_rejected_without_state_changes(self):
        for mode in (1, True, 2, "warn", "error"):
            with self.subTest(mode=mode):
                expected_grad_mode = torch.is_grad_enabled()
                message = (
                    "set_deterministic_debug_mode(): debug_mode "
                    f"{mode!r} is not supported; only 0, False, and 'default' "
                    "are implemented"
                )
                with self.assertRaises(NotImplementedError) as raised:
                    torch.set_deterministic_debug_mode(mode)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assert_default_state()
                self.assertIs(torch.is_grad_enabled(), expected_grad_mode)

                with torch.no_grad():
                    with self.assertRaises(NotImplementedError):
                        torch.set_deterministic_debug_mode(mode)
                    self.assert_default_state()
                    self.assertIs(torch.is_grad_enabled(), False)

    def test_malformed_input_errors_match_pytorch_2_13_spelling(self):
        invalid_types = (
            (None, "<class 'NoneType'>"),
            (0.0, "<class 'float'>"),
            (b"default", "<class 'bytes'>"),
            (bytearray(b"default"), "<class 'bytearray'>"),
            (memoryview(b"default"), "<class 'memoryview'>"),
            ([], "<class 'list'>"),
            (object(), "<class 'object'>"),
            (_RejectTruthiness(), f"{_RejectTruthiness}"),
            (torch.tensor(0.0), "<class 'torch.Tensor'>"),
            (torch.float32, "<class 'torch.dtype'>"),
            (torch.device("cpu"), "<class 'torch.device'>"),
            (torch.contiguous_format, "<class 'torch.memory_format'>"),
            (torch.strided, "<class 'torch.layout'>"),
            (torch.Size([]), "<class 'torch.Size'>"),
            (torch.finfo(torch.float32), "<class 'torch.finfo'>"),
        )
        for value, type_name in invalid_types:
            with self.subTest(value=repr(value)):
                message = (
                    "debug_mode must be str or int, but got "
                    f"{type_name}"
                )
                with self.assertRaises(TypeError) as raised:
                    torch.set_deterministic_debug_mode(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assert_default_state()

        for mode in (-1, 3, 10**100):
            with self.subTest(mode=mode):
                message = (
                    "invalid value of debug_mode, expected 0, 1, or 2, but got "
                    f"{mode}"
                )
                with self.assertRaises(RuntimeError) as raised:
                    torch.set_deterministic_debug_mode(mode)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assert_default_state()

        for mode in ("", "DEFAULT", " default", "warning"):
            with self.subTest(mode=mode):
                message = (
                    "invalid value of debug_mode, expected one of `default`, "
                    f"`warn`, `error`, but got {mode}"
                )
                with self.assertRaises(RuntimeError) as raised:
                    torch.set_deterministic_debug_mode(mode)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assert_default_state()

    def test_callable_metadata_matches_pytorch_2_13(self):
        package = importlib.import_module("torch_rs")
        function = package.set_deterministic_debug_mode
        annotation = int | str

        self.assertIs(torch, package)
        self.assertIs(sys.modules["torch_rs"], package)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(debug_mode: int | str) -> None",
        )
        self.assertEqual(
            function.__annotations__,
            {"debug_mode": annotation, "return": None},
        )
        self.assertEqual(
            typing.get_type_hints(function),
            {"debug_mode": annotation, "return": type(None)},
        )
        self.assertEqual(function.__name__, "set_deterministic_debug_mode")
        self.assertEqual(function.__qualname__, "set_deterministic_debug_mode")
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
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

    def test_exports_copy_and_pickle_use_the_canonical_module(self):
        function = torch.set_deterministic_debug_mode

        self.assertEqual(torch.__all__.count("set_deterministic_debug_mode"), 1)
        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["set_deterministic_debug_mode"], function)
        self.assertFalse(hasattr(torch._C, "_set_deterministic_algorithms"))

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_argument_binding_errors_match_pytorch_2_13(self):
        function = torch.set_deterministic_debug_mode
        cases = (
            (
                lambda: function(),
                "set_deterministic_debug_mode() missing 1 required positional "
                "argument: 'debug_mode'",
            ),
            (
                lambda: function(0, 0),
                "set_deterministic_debug_mode() takes 1 positional argument but 2 "
                "were given",
            ),
            (
                lambda: function(mode=0),
                "set_deterministic_debug_mode() got an unexpected keyword argument "
                "'mode'",
            ),
            (
                lambda: function(0, debug_mode=0),
                "set_deterministic_debug_mode() got multiple values for argument "
                "'debug_mode'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assert_default_state()

    def test_importing_calling_and_reloading_do_not_import_pytorch(self):
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

for mode in (0, False, "default"):
    assert torch.set_deterministic_debug_mode(mode) is None
    assert torch.get_deterministic_debug_mode() == 0
    assert torch.are_deterministic_algorithms_enabled() is False
    assert torch.is_deterministic_algorithms_warn_only_enabled() is False

for mode in (1, True, 2, "warn", "error"):
    try:
        torch.set_deterministic_debug_mode(mode)
    except NotImplementedError:
        pass
    else:
        raise AssertionError(f"unsupported mode was accepted: {mode!r}")

assert importlib.reload(torch) is torch
assert torch.set_deterministic_debug_mode("default") is None
assert torch.get_deterministic_debug_mode() == 0
assert torch.are_deterministic_algorithms_enabled() is False
assert torch.is_deterministic_algorithms_warn_only_enabled() is False
assert "set_deterministic_debug_mode" in torch.__all__
assert not hasattr(torch._C, "_set_deterministic_algorithms")
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

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


FUNCTION_DOC = """Sets whether operations must use deterministic algorithms.

    This compatibility entrypoint currently accepts only ``mode=False`` with
    ``warn_only=False``. Deterministic algorithm enforcement and warning-only
    enforcement remain unsupported.

    Args:
        mode (:class:`bool`): If False, leave deterministic enforcement
            disabled. True is not supported.

    Keyword args:
        warn_only (:class:`bool`, optional): Must be False. Default: ``False``
    """


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("the setter must not request truthiness")


class UseDeterministicAlgorithmsTests(unittest.TestCase):
    def assert_default_state(self):
        debug_mode = torch.get_deterministic_debug_mode()
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

        self.assertIs(type(debug_mode), int)
        self.assertEqual(debug_mode, 0)
        self.assertIs(enabled, False)
        self.assertIs(warn_only, False)

    def test_disabled_default_forms_are_idempotent_noops_across_grad_modes(self):
        calls = (
            lambda: torch.use_deterministic_algorithms(False),
            lambda: torch.use_deterministic_algorithms(mode=False),
            lambda: torch.use_deterministic_algorithms(False, warn_only=False),
            lambda: torch.use_deterministic_algorithms(
                mode=False,
                warn_only=False,
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
        calls = (
            lambda: torch.use_deterministic_algorithms(False),
            lambda: torch.use_deterministic_algorithms(mode=False),
            lambda: torch.use_deterministic_algorithms(False, warn_only=False),
            lambda: torch.use_deterministic_algorithms(
                mode=False,
                warn_only=False,
            ),
        )
        worker_count = 12
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    expected_grad_mode = torch.is_grad_enabled()
                    barrier.wait(timeout=10)
                    returned = calls[index % len(calls)]()
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

    def test_interacts_with_deterministic_debug_mode_without_state_changes(self):
        self.assertIs(torch.use_deterministic_algorithms(False), None)
        self.assertIs(torch.set_deterministic_debug_mode(0), None)
        self.assertIs(torch.use_deterministic_algorithms(mode=False), None)
        self.assertIs(torch.set_deterministic_debug_mode("default"), None)
        self.assert_default_state()

        for debug_mode in (1, True, 2, "warn", "error"):
            with self.subTest(debug_mode=debug_mode):
                with self.assertRaises(NotImplementedError):
                    torch.set_deterministic_debug_mode(debug_mode)
                self.assertIs(
                    torch.use_deterministic_algorithms(False, warn_only=False),
                    None,
                )
                self.assert_default_state()

    def test_enabled_and_warn_only_modes_are_explicitly_rejected(self):
        enabled_message = (
            "use_deterministic_algorithms(): deterministic algorithm "
            "enforcement is not supported; only mode=False with warn_only=False "
            "is implemented"
        )
        warn_only_message = (
            "use_deterministic_algorithms(): warning-only deterministic "
            "enforcement is not supported; only mode=False with warn_only=False "
            "is implemented"
        )
        cases = (
            (lambda: torch.use_deterministic_algorithms(True), enabled_message),
            (
                lambda: torch.use_deterministic_algorithms(mode=True),
                enabled_message,
            ),
            (
                lambda: torch.use_deterministic_algorithms(True, warn_only=False),
                enabled_message,
            ),
            (
                lambda: torch.use_deterministic_algorithms(True, warn_only=True),
                enabled_message,
            ),
            (
                lambda: torch.use_deterministic_algorithms(
                    False,
                    warn_only=True,
                ),
                warn_only_message,
            ),
            (
                lambda: torch.use_deterministic_algorithms(
                    mode=False,
                    warn_only=True,
                ),
                warn_only_message,
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                expected_grad_mode = torch.is_grad_enabled()
                with self.assertRaises(NotImplementedError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assert_default_state()
                self.assertIs(torch.is_grad_enabled(), expected_grad_mode)

                with torch.no_grad():
                    with self.assertRaises(NotImplementedError):
                        call()
                    self.assert_default_state()
                    self.assertIs(torch.is_grad_enabled(), False)

    def test_non_bool_arguments_match_pytorch_2_13_backend_errors(self):
        invalid_values = (
            (None, "NoneType"),
            (0, "int"),
            (1, "int"),
            (0.0, "float"),
            (b"", "bytes"),
            (bytearray(b""), "bytearray"),
            (memoryview(b""), "memoryview"),
            ([], "list"),
            (object(), "object"),
            (_RejectTruthiness(), "_RejectTruthiness"),
            (torch.tensor(0.0), "Tensor"),
            (torch.float32, "torch.dtype"),
            (torch.device("cpu"), "torch.device"),
            (torch.contiguous_format, "torch.memory_format"),
            (torch.strided, "torch.layout"),
            (torch.Size([]), "torch.Size"),
            (torch.finfo(torch.float32), "torch.finfo"),
        )
        for value, type_name in invalid_values:
            with self.subTest(argument="mode", value=repr(value)):
                message = (
                    "_set_deterministic_algorithms(): argument 'mode' "
                    f"(position 1) must be bool, not {type_name}"
                )
                with self.assertRaises(TypeError) as raised:
                    torch.use_deterministic_algorithms(value, warn_only=False)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assert_default_state()

            with self.subTest(argument="warn_only", value=repr(value)):
                message = (
                    "_set_deterministic_algorithms(): argument 'warn_only' "
                    f"must be bool, not {type_name}"
                )
                with self.assertRaises(TypeError) as raised:
                    torch.use_deterministic_algorithms(False, warn_only=value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assert_default_state()

        with self.assertRaises(TypeError) as raised:
            torch.use_deterministic_algorithms(True, warn_only=0)
        self.assertEqual(
            str(raised.exception),
            "_set_deterministic_algorithms(): argument 'warn_only' "
            "must be bool, not int",
        )
        self.assert_default_state()

    def test_callable_metadata_documents_default_only_scope(self):
        package = importlib.import_module("torch_rs")
        function = package.use_deterministic_algorithms

        self.assertIs(torch, package)
        self.assertIs(sys.modules["torch_rs"], package)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(mode: bool, *, warn_only: bool = False) -> None",
        )
        self.assertEqual(
            function.__annotations__,
            {"mode": bool, "warn_only": bool, "return": None},
        )
        self.assertEqual(
            typing.get_type_hints(function),
            {"mode": bool, "warn_only": bool, "return": type(None)},
        )
        self.assertEqual(function.__name__, "use_deterministic_algorithms")
        self.assertEqual(function.__qualname__, "use_deterministic_algorithms")
        self.assertEqual(function.__module__, "torch_rs")
        self.assertIs(inspect.getmodule(function), package)
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(FUNCTION_DOC),
        )
        self.assertIsNone(function.__defaults__)
        self.assertEqual(function.__kwdefaults__, {"warn_only": False})
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

    def test_exports_reload_copy_and_pickle_use_the_canonical_module(self):
        package = importlib.import_module("torch_rs")
        old_function = package.use_deterministic_algorithms

        self.assertEqual(torch.__all__.count("use_deterministic_algorithms"), 1)
        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["use_deterministic_algorithms"], old_function)
        self.assertFalse(hasattr(torch._C, "_set_deterministic_algorithms"))

        self.assertIs(copy.copy(old_function), old_function)
        self.assertIs(copy.deepcopy(old_function), old_function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(old_function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIs(pickle.loads(payload), old_function)

        self.assertIs(importlib.reload(package), package)
        self.assertIs(torch, package)
        self.assertIsNot(package.use_deterministic_algorithms, old_function)

        for function in (old_function, package.use_deterministic_algorithms):
            with self.subTest(function=function):
                self.assertIs(function(False, warn_only=False), None)
                self.assert_default_state()

    def test_argument_binding_errors_match_pytorch_2_13(self):
        function = torch.use_deterministic_algorithms
        cases = (
            (
                lambda: function(),
                "use_deterministic_algorithms() missing 1 required positional "
                "argument: 'mode'",
            ),
            (
                lambda: function(False, False),
                "use_deterministic_algorithms() takes 1 positional argument "
                "but 2 were given",
            ),
            (
                lambda: function(False, mode=False),
                "use_deterministic_algorithms() got multiple values for "
                "argument 'mode'",
            ),
            (
                lambda: function(False, foo=False),
                "use_deterministic_algorithms() got an unexpected keyword "
                "argument 'foo'",
            ),
            (
                lambda: function(False, warn_only=False, extra=False),
                "use_deterministic_algorithms() got an unexpected keyword "
                "argument 'extra'",
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

for call in (
    lambda: torch.use_deterministic_algorithms(False),
    lambda: torch.use_deterministic_algorithms(mode=False),
    lambda: torch.use_deterministic_algorithms(False, warn_only=False),
):
    assert call() is None
    assert torch.get_deterministic_debug_mode() == 0
    assert torch.are_deterministic_algorithms_enabled() is False
    assert torch.is_deterministic_algorithms_warn_only_enabled() is False

for call in (
    lambda: torch.use_deterministic_algorithms(True),
    lambda: torch.use_deterministic_algorithms(False, warn_only=True),
):
    try:
        call()
    except NotImplementedError:
        pass
    else:
        raise AssertionError("unsupported deterministic state was accepted")

assert importlib.reload(torch) is torch
assert torch.use_deterministic_algorithms(False, warn_only=False) is None
assert torch.set_deterministic_debug_mode("default") is None
assert torch.get_deterministic_debug_mode() == 0
assert torch.are_deterministic_algorithms_enabled() is False
assert torch.is_deterministic_algorithms_warn_only_enabled() is False
assert "use_deterministic_algorithms" in torch.__all__
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

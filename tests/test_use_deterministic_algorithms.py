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


FUNCTION_DOC = """Sets whether PyTorch operations must use "deterministic" algorithms.

    This implementation only supports requests that leave deterministic
    algorithms disabled: ``mode=False`` or integer ``0`` with
    ``warn_only=False``. Enabling deterministic algorithms, warning-only mode,
    fill-uninitialized-memory controls, CUDA determinism, and deterministic
    kernel enforcement are not implemented.

    Args:
        mode (:class:`bool`): If ``False``, allows nondeterministic operations.
            Integer ``0`` is accepted as a default-equivalent disabled request.
        warn_only (:class:`bool`, optional): Must be ``False``. Default:
            ``False``
    """


class _DefaultInt(int):
    pass


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

    def test_disabled_forms_are_idempotent_noops_across_grad_modes(self):
        calls = (
            lambda: torch.use_deterministic_algorithms(False),
            lambda: torch.use_deterministic_algorithms(mode=False),
            lambda: torch.use_deterministic_algorithms(False, warn_only=False),
            lambda: torch.use_deterministic_algorithms(
                mode=False,
                warn_only=False,
            ),
            lambda: torch.use_deterministic_algorithms(0),
            lambda: torch.use_deterministic_algorithms(mode=0),
            lambda: torch.use_deterministic_algorithms(0, warn_only=False),
            lambda: torch.use_deterministic_algorithms(_DefaultInt(0)),
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
        modes = (False, 0, _DefaultInt(0))
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
                    returned = torch.use_deterministic_algorithms(
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

    def test_reload_preserves_default_state_for_old_and_new_functions(self):
        package = importlib.import_module("torch_rs")
        old_setter = package.use_deterministic_algorithms
        old_debug_query = package.get_deterministic_debug_mode
        old_enabled_query = package.are_deterministic_algorithms_enabled
        old_warn_only_query = package.is_deterministic_algorithms_warn_only_enabled

        self.assertIs(old_setter(False), None)
        self.assertIs(importlib.reload(package), package)
        self.assertIs(torch, package)
        self.assertIsNot(package.use_deterministic_algorithms, old_setter)

        for setter, mode in (
            (old_setter, False),
            (package.use_deterministic_algorithms, 0),
            (old_setter, _DefaultInt(0)),
        ):
            with self.subTest(setter=setter.__name__, mode=mode):
                self.assertIs(setter(mode), None)
                self.assertEqual(old_debug_query(), 0)
                self.assertIs(old_enabled_query(), False)
                self.assertIs(old_warn_only_query(), False)
                self.assert_default_state()

    def test_unsupported_modes_and_warn_only_are_rejected_without_state_changes(self):
        cases = (
            (
                lambda: torch.use_deterministic_algorithms(True),
                "use_deterministic_algorithms(): mode True is not supported; "
                "only False and 0 are implemented",
            ),
            (
                lambda: torch.use_deterministic_algorithms(1),
                "use_deterministic_algorithms(): mode 1 is not supported; "
                "only False and 0 are implemented",
            ),
            (
                lambda: torch.use_deterministic_algorithms(_DefaultInt(1)),
                "use_deterministic_algorithms(): mode 1 is not supported; "
                "only False and 0 are implemented",
            ),
            (
                lambda: torch.use_deterministic_algorithms(2),
                "use_deterministic_algorithms(): mode 2 is not supported; "
                "only False and 0 are implemented",
            ),
            (
                lambda: torch.use_deterministic_algorithms(-1),
                "use_deterministic_algorithms(): mode -1 is not supported; "
                "only False and 0 are implemented",
            ),
            (
                lambda: torch.use_deterministic_algorithms(False, warn_only=True),
                "use_deterministic_algorithms(): warn_only=True is not supported; "
                "only warn_only=False is implemented",
            ),
            (
                lambda: torch.use_deterministic_algorithms(0, warn_only=True),
                "use_deterministic_algorithms(): warn_only=True is not supported; "
                "only warn_only=False is implemented",
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

    def test_invalid_mode_types_are_rejected_without_state_changes(self):
        invalid_modes = (
            (None, "NoneType"),
            (0.0, "float"),
            (b"", "bytes"),
            (bytearray(b""), "bytearray"),
            (memoryview(b""), "memoryview"),
            ([], "list"),
            (object(), "object"),
            (_RejectTruthiness(), "_RejectTruthiness"),
            ("default", "str"),
            (torch.tensor(0.0), "Tensor"),
            (torch.float32, "torch.dtype"),
            (torch.device("cpu"), "torch.device"),
            (torch.contiguous_format, "torch.memory_format"),
            (torch.strided, "torch.layout"),
            (torch.Size([]), "torch.Size"),
            (torch.finfo(torch.float32), "torch.finfo"),
        )
        for value, type_name in invalid_modes:
            with self.subTest(value=repr(value)):
                message = (
                    "_set_deterministic_algorithms(): argument 'mode' "
                    f"(position 1) must be bool, not {type_name}"
                )
                with self.assertRaises(TypeError) as raised:
                    torch.use_deterministic_algorithms(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assert_default_state()

    def test_invalid_warn_only_types_are_rejected_without_state_changes(self):
        invalid_warn_only_values = (
            (0, "int"),
            (1, "int"),
            (None, "NoneType"),
            ("", "str"),
            (object(), "object"),
            (_RejectTruthiness(), "_RejectTruthiness"),
            (torch.tensor(0.0), "Tensor"),
            (torch.float32, "torch.dtype"),
            (torch.device("cpu"), "torch.device"),
        )
        for value, type_name in invalid_warn_only_values:
            with self.subTest(value=repr(value)):
                message = (
                    "_set_deterministic_algorithms(): argument 'warn_only' "
                    f"must be bool, not {type_name}"
                )
                with self.assertRaises(TypeError) as raised:
                    torch.use_deterministic_algorithms(False, warn_only=value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assert_default_state()

    def test_callable_metadata_matches_pytorch_2_13_signature(self):
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

    def test_exports_imports_copy_and_pickle_use_the_canonical_module(self):
        function = torch.use_deterministic_algorithms

        self.assertEqual(torch.__all__.count("use_deterministic_algorithms"), 1)
        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["use_deterministic_algorithms"], function)
        direct_namespace = {}
        exec("from torch_rs import use_deterministic_algorithms", direct_namespace)
        self.assertIs(direct_namespace["use_deterministic_algorithms"], function)
        self.assertFalse(hasattr(torch._C, "_set_deterministic_algorithms"))
        self.assertNotIn("use_deterministic_algorithms", torch._C.__all__)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_argument_binding_errors_match_pytorch_2_13(self):
        function = torch.use_deterministic_algorithms
        cases = (
            (
                lambda: function(),
                "use_deterministic_algorithms() missing 1 required positional "
                "argument: 'mode'",
            ),
            (
                lambda: function(warn_only=False),
                "use_deterministic_algorithms() missing 1 required positional "
                "argument: 'mode'",
            ),
            (
                lambda: function(False, False),
                "use_deterministic_algorithms() takes 1 positional argument but 2 "
                "were given",
            ),
            (
                lambda: function(False, warn_only=False, extra=False),
                "use_deterministic_algorithms() got an unexpected keyword "
                "argument 'extra'",
            ),
            (
                lambda: function(False, mode=False),
                "use_deterministic_algorithms() got multiple values for "
                "argument 'mode'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assert_default_state()

    def test_unsupported_fill_memory_cuda_and_enforcement_boundaries(self):
        self.assertFalse(hasattr(torch.utils, "deterministic"))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.utils.deterministic")

        self.assertFalse(hasattr(torch, "cuda"))
        self.assertNotIn("torch_rs.cuda", sys.modules)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.cuda")
        self.assertIs(torch.backends.cuda.is_built(), False)
        self.assertIs(torch.version.cuda, None)

        self.assertFalse(hasattr(torch._C, "_set_deterministic_algorithms"))
        with self.assertRaises(NotImplementedError):
            torch.use_deterministic_algorithms(True)
        self.assert_default_state()

        self.assertIs(torch.use_deterministic_algorithms(False), None)
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda:0' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda:0")
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
from torch_rs import use_deterministic_algorithms

assert use_deterministic_algorithms is torch.use_deterministic_algorithms
for mode in (False, 0):
    assert torch.use_deterministic_algorithms(mode) is None
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
        raise AssertionError("unsupported deterministic mode was accepted")

namespace = {}
exec("from torch_rs import *", namespace)
assert namespace["use_deterministic_algorithms"] is torch.use_deterministic_algorithms
assert importlib.reload(torch) is torch
assert torch.use_deterministic_algorithms(False) is None
assert torch.get_deterministic_debug_mode() == 0
assert torch.are_deterministic_algorithms_enabled() is False
assert torch.is_deterministic_algorithms_warn_only_enabled() is False
assert "use_deterministic_algorithms" in torch.__all__
assert not hasattr(torch._C, "_set_deterministic_algorithms")
assert not hasattr(torch.utils, "deterministic")
assert not hasattr(torch, "cuda")
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

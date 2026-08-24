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


class UseDeterministicAlgorithmsTests(unittest.TestCase):
    def setUp(self):
        self.original_enabled = torch.are_deterministic_algorithms_enabled()
        self.original_warn_only = (
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
        torch.use_deterministic_algorithms(False)

    def tearDown(self):
        torch.use_deterministic_algorithms(
            self.original_enabled,
            warn_only=self.original_warn_only,
        )

    def assert_state(self, enabled, warn_only, debug_mode):
        self.assertIs(torch.are_deterministic_algorithms_enabled(), enabled)
        self.assertIs(
            torch.is_deterministic_algorithms_warn_only_enabled(),
            warn_only,
        )
        actual_debug_mode = torch.get_deterministic_debug_mode()
        self.assertIs(type(actual_debug_mode), int)
        self.assertEqual(actual_debug_mode, debug_mode)

    def operation_outcome(self):
        leaf = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
        output = ((leaf + 2.0) * leaf).sum()
        output.backward()
        return output.item(), leaf.grad.tolist()

    def test_all_flag_combinations_preserve_grad_mode_and_operations(self):
        baseline = self.operation_outcome()
        states = (
            (False, False, 0),
            (False, True, 0),
            (True, True, 1),
            (True, False, 2),
            (False, False, 0),
        )

        for enabled, warn_only, debug_mode in states:
            with self.subTest(
                enabled=enabled,
                warn_only=warn_only,
                debug_mode=debug_mode,
            ):
                self.assertIs(torch.is_grad_enabled(), True)
                self.assertIs(
                    torch.use_deterministic_algorithms(
                        enabled,
                        warn_only=warn_only,
                    ),
                    None,
                )
                self.assertIs(torch.is_grad_enabled(), True)
                self.assert_state(enabled, warn_only, debug_mode)
                self.assertEqual(self.operation_outcome(), baseline)

                with torch.no_grad():
                    self.assertIs(torch.is_grad_enabled(), False)
                    self.assertIs(
                        torch.use_deterministic_algorithms(
                            enabled,
                            warn_only=warn_only,
                        ),
                        None,
                    )
                    self.assertIs(torch.is_grad_enabled(), False)
                    self.assert_state(enabled, warn_only, debug_mode)
                self.assertIs(torch.is_grad_enabled(), True)

    def test_state_is_process_global_across_threads(self):
        states = (
            (False, True, 0),
            (True, True, 1),
            (True, False, 2),
            (False, False, 0),
        )

        for enabled, warn_only, debug_mode in states:
            with self.subTest(enabled=enabled, warn_only=warn_only):
                worker_result = []

                def set_from_worker():
                    with torch.no_grad():
                        before = torch.is_grad_enabled()
                        result = torch.use_deterministic_algorithms(
                            enabled,
                            warn_only=warn_only,
                        )
                        after = torch.is_grad_enabled()
                        worker_result.append((before, result, after))

                thread = threading.Thread(target=set_from_worker)
                thread.start()
                thread.join(timeout=10)
                self.assertFalse(thread.is_alive())
                self.assertEqual(worker_result, [(False, None, False)])
                self.assert_state(enabled, warn_only, debug_mode)

                reader_result = []

                def read_from_worker():
                    reader_result.append(
                        (
                            torch.are_deterministic_algorithms_enabled(),
                            torch.is_deterministic_algorithms_warn_only_enabled(),
                            torch.get_deterministic_debug_mode(),
                        )
                    )

                thread = threading.Thread(target=read_from_worker)
                thread.start()
                thread.join(timeout=10)
                self.assertFalse(thread.is_alive())
                self.assertEqual(
                    reader_result,
                    [(enabled, warn_only, debug_mode)],
                )

    def test_state_survives_package_reload(self):
        original_function = torch.use_deterministic_algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)

        reloaded = importlib.reload(torch)

        self.assertIs(reloaded, torch)
        self.assertIsNot(reloaded.use_deterministic_algorithms, original_function)
        self.assert_state(True, True, 1)

        torch.use_deterministic_algorithms(False, warn_only=True)
        importlib.reload(torch)
        self.assert_state(False, True, 0)

    def test_warn_only_defaults_to_false_on_each_call(self):
        torch.use_deterministic_algorithms(False, warn_only=True)
        self.assert_state(False, True, 0)

        torch.use_deterministic_algorithms(True)
        self.assert_state(True, False, 2)

    def test_signature_annotations_documentation_and_module_identity(self):
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
        self.assertIn(
            'Sets whether PyTorch operations must use "deterministic"',
            function.__doc__,
        )
        self.assertIsNone(function.__defaults__)
        self.assertEqual(function.__kwdefaults__, {"warn_only": False})
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(
            function.__code__.co_names,
            ("_C", "_set_deterministic_algorithms"),
        )
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

    def test_exports_copy_and_pickle_use_the_canonical_module(self):
        function = torch.use_deterministic_algorithms

        self.assertEqual(torch.__all__.count("use_deterministic_algorithms"), 1)
        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["use_deterministic_algorithms"], function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_python_binding_errors_match_pytorch_2_13(self):
        function = torch.use_deterministic_algorithms
        cases = (
            (
                lambda: function(),
                "use_deterministic_algorithms() missing 1 required positional "
                "argument: 'mode'",
            ),
            (
                lambda: function(True, False),
                "use_deterministic_algorithms() takes 1 positional argument but "
                "2 were given",
            ),
            (
                lambda: function(True, unknown=False),
                "use_deterministic_algorithms() got an unexpected keyword "
                "argument 'unknown'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        self.assertIs(function(mode=True, warn_only=True), None)
        self.assert_state(True, True, 1)

    def test_mode_and_warn_only_require_exact_bools_without_mutating_state(self):
        torch.use_deterministic_algorithms(True, warn_only=True)
        cases = (
            (
                lambda: torch.use_deterministic_algorithms(None),
                "_set_deterministic_algorithms(): argument 'mode' (position 1) "
                "must be bool, not NoneType",
            ),
            (
                lambda: torch.use_deterministic_algorithms(1),
                "_set_deterministic_algorithms(): argument 'mode' (position 1) "
                "must be bool, not int",
            ),
            (
                lambda: torch.use_deterministic_algorithms("true"),
                "_set_deterministic_algorithms(): argument 'mode' (position 1) "
                "must be bool, not str",
            ),
            (
                lambda: torch.use_deterministic_algorithms(True, warn_only=None),
                "_set_deterministic_algorithms(): argument 'warn_only' must be "
                "bool, not NoneType",
            ),
            (
                lambda: torch.use_deterministic_algorithms(True, warn_only=0),
                "_set_deterministic_algorithms(): argument 'warn_only' must be "
                "bool, not int",
            ),
            (
                lambda: torch.use_deterministic_algorithms(
                    True,
                    warn_only="false",
                ),
                "_set_deterministic_algorithms(): argument 'warn_only' must be "
                "bool, not str",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assert_state(True, True, 1)

    def test_importing_and_using_the_api_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

baseline = ((torch.tensor([1.0, -2.0, 3.0]) + 2.0) * 3.0).tolist()
for enabled, warn_only, debug_mode in (
    (False, False, 0),
    (False, True, 0),
    (True, True, 1),
    (True, False, 2),
):
    assert torch.use_deterministic_algorithms(enabled, warn_only=warn_only) is None
    assert torch.are_deterministic_algorithms_enabled() is enabled
    assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only
    assert torch.get_deterministic_debug_mode() == debug_mode
    assert ((torch.tensor([1.0, -2.0, 3.0]) + 2.0) * 3.0).tolist() == baseline
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

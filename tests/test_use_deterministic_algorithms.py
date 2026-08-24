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

import numpy as np
import torch_rs as torch


class _RejectTruthiness:
    calls = 0

    def __bool__(self):
        type(self).calls += 1
        raise AssertionError("deterministic mode must not request truthiness")


class UseDeterministicAlgorithmsTests(unittest.TestCase):
    def setUp(self):
        self.original_state = self.state()
        torch.use_deterministic_algorithms(False, warn_only=False)

    def tearDown(self):
        torch.use_deterministic_algorithms(
            self.original_state[0],
            warn_only=self.original_state[1],
        )

    def state(self):
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        debug_mode = torch.get_deterministic_debug_mode()
        self.assertIs(type(enabled), bool)
        self.assertIs(type(warn_only), bool)
        self.assertIs(type(debug_mode), int)
        return enabled, warn_only, debug_mode

    def deterministic_operation_outcome(self):
        values = torch.tensor([-2.0, 1.0, 3.0], requires_grad=True)
        result = (values * 2.0).relu()
        total = result.sum()
        total.backward()
        matrix = torch.tensor([[1.0, 2.0], [-3.0, 4.0]])
        product = torch.matmul(matrix, torch.ones((2, 2)))
        return result.tolist(), total.item(), values.grad.tolist(), product.tolist()

    def test_all_states_preserve_grad_mode_and_supported_deterministic_results(self):
        expected_operation = (
            [0.0, 2.0, 6.0],
            8.0,
            [0.0, 2.0, 2.0],
            [[3.0, 3.0], [1.0, 1.0]],
        )
        cases = (
            (False, False, 0),
            (False, True, 0),
            (True, True, 1),
            (True, False, 2),
        )

        for mode, warn_only, debug_mode in cases:
            with self.subTest(mode=mode, warn_only=warn_only):
                self.assertIs(torch.is_grad_enabled(), True)
                self.assertIs(
                    torch.use_deterministic_algorithms(
                        mode,
                        warn_only=warn_only,
                    ),
                    None,
                )
                self.assertEqual(self.state(), (mode, warn_only, debug_mode))
                self.assertIs(torch.is_grad_enabled(), True)
                self.assertEqual(
                    self.deterministic_operation_outcome(),
                    expected_operation,
                )
                self.assertEqual(self.state(), (mode, warn_only, debug_mode))

                with torch.no_grad():
                    self.assertIs(torch.is_grad_enabled(), False)
                    self.assertIs(
                        torch.use_deterministic_algorithms(
                            mode,
                            warn_only=warn_only,
                        ),
                        None,
                    )
                    matrix = torch.tensor([[1.0, 2.0], [-3.0, 4.0]])
                    self.assertEqual(
                        torch.relu(matrix + 1.0).tolist(),
                        [[2.0, 3.0], [0.0, 5.0]],
                    )
                    self.assertEqual(self.state(), (mode, warn_only, debug_mode))
                    self.assertIs(torch.is_grad_enabled(), False)

                self.assertIs(torch.is_grad_enabled(), True)

    def test_state_is_process_global_across_threads(self):
        worker_ready = threading.Event()
        main_updated = threading.Event()
        worker_updated = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                with torch.no_grad():
                    observations.append((self.state(), torch.is_grad_enabled()))
                    worker_ready.set()
                    if not main_updated.wait(timeout=10):
                        raise RuntimeError("timed out waiting for main-thread update")
                    observations.append((self.state(), torch.is_grad_enabled()))
                    observations.append(
                        (
                            torch.use_deterministic_algorithms(
                                True,
                                warn_only=False,
                            ),
                            torch.is_grad_enabled(),
                        )
                    )
                    observations.append((self.state(), torch.is_grad_enabled()))
                    worker_updated.set()
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_ready.wait(timeout=10))
        self.assertEqual(self.state(), (False, False, 0))
        self.assertIs(torch.use_deterministic_algorithms(True, warn_only=True), None)
        self.assertEqual(self.state(), (True, True, 1))
        main_updated.set()
        self.assertTrue(worker_updated.wait(timeout=10))
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(
            observations,
            [
                ((False, False, 0), False),
                ((True, True, 1), False),
                (None, False),
                ((True, False, 2), False),
            ],
        )
        self.assertEqual(self.state(), (True, False, 2))
        self.assertIs(torch.is_grad_enabled(), True)

    def test_state_survives_native_and_package_reloads(self):
        package = importlib.import_module("torch_rs")
        native = package._C
        old_setter = package.use_deterministic_algorithms
        old_enabled = package.are_deterministic_algorithms_enabled
        old_warn_only = package.is_deterministic_algorithms_warn_only_enabled
        old_debug_mode = package.get_deterministic_debug_mode

        self.assertIs(old_setter(True, warn_only=True), None)
        self.assertIs(importlib.reload(native), native)
        self.assertEqual(self.state(), (True, True, 1))

        self.assertIs(importlib.reload(package), package)
        self.assertIs(torch, package)
        self.assertEqual(self.state(), (True, True, 1))
        self.assertEqual(
            (old_enabled(), old_warn_only(), old_debug_mode()),
            (True, True, 1),
        )

        self.assertIs(package.use_deterministic_algorithms(False, warn_only=True), None)
        self.assertEqual(
            (old_enabled(), old_warn_only(), old_debug_mode()),
            (False, True, 0),
        )
        self.assertIs(old_setter(True, warn_only=False), None)
        self.assertEqual(self.state(), (True, False, 2))

    def test_strict_boolean_binding_does_not_coerce_or_mutate(self):
        invalid_values = (
            (None, "NoneType"),
            (0, "int"),
            (1, "int"),
            (0.0, "float"),
            ("", "str"),
            ([], "list"),
            (object(), "object"),
            (_RejectTruthiness(), "_RejectTruthiness"),
            (np.bool_(True), "numpy.bool"),
            (torch.tensor(True), "Tensor"),
            (torch.float32, "torch.dtype"),
            (torch.device("cpu"), "torch.device"),
        )
        _RejectTruthiness.calls = 0

        for initial_state in ((False, True, 0), (True, False, 2)):
            torch.use_deterministic_algorithms(
                initial_state[0],
                warn_only=initial_state[1],
            )
            for value, type_name in invalid_values:
                with self.subTest(
                    argument="mode",
                    initial_state=initial_state,
                    value=repr(value),
                ):
                    message = (
                        "_set_deterministic_algorithms(): argument 'mode' "
                        f"(position 1) must be bool, not {type_name}"
                    )
                    with self.assertRaises(TypeError) as raised:
                        torch.use_deterministic_algorithms(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertEqual(self.state(), initial_state)

                with self.subTest(
                    argument="warn_only",
                    initial_state=initial_state,
                    value=repr(value),
                ):
                    message = (
                        "_set_deterministic_algorithms(): argument 'warn_only' "
                        f"must be bool, not {type_name}"
                    )
                    with self.assertRaises(TypeError) as raised:
                        torch.use_deterministic_algorithms(True, warn_only=value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertEqual(self.state(), initial_state)

        self.assertEqual(_RejectTruthiness.calls, 0)

    def test_signature_metadata_exports_copying_and_pickling(self):
        package = importlib.import_module("torch_rs")
        function = package.use_deterministic_algorithms

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
        self.assertTrue(
            function.__doc__.startswith(
                'Sets whether PyTorch operations must use "deterministic"'
            )
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

    def test_python_binding_errors_are_exact_and_leave_state_unchanged(self):
        function = torch.use_deterministic_algorithms
        cases = (
            (
                lambda: function(),
                "use_deterministic_algorithms() missing 1 required positional "
                "argument: 'mode'",
            ),
            (
                lambda: function(True, False),
                "use_deterministic_algorithms() takes 1 positional argument "
                "but 2 were given",
            ),
            (
                lambda: function(True, False, None),
                "use_deterministic_algorithms() takes 1 positional argument "
                "but 3 were given",
            ),
            (
                lambda: function(True, enabled=False),
                "use_deterministic_algorithms() got an unexpected keyword "
                "argument 'enabled'",
            ),
            (
                lambda: function(True, mode=False),
                "use_deterministic_algorithms() got multiple values for argument "
                "'mode'",
            ),
            (
                lambda: function(warn_only=True),
                "use_deterministic_algorithms() missing 1 required positional "
                "argument: 'mode'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(self.state(), (False, False, 0))

        self.assertIs(function(mode=True), None)
        self.assertEqual(self.state(), (True, False, 2))

    def test_import_mutation_and_execution_do_not_import_pytorch(self):
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

for mode, warn_only, debug_mode in (
    (False, False, 0),
    (False, True, 0),
    (True, True, 1),
    (True, False, 2),
):
    before = torch.is_grad_enabled()
    assert torch.use_deterministic_algorithms(mode, warn_only=warn_only) is None
    assert torch.is_grad_enabled() is before
    assert torch.are_deterministic_algorithms_enabled() is mode
    assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only
    assert torch.get_deterministic_debug_mode() == debug_mode
    values = torch.tensor([-2.0, 1.0, 3.0], requires_grad=True)
    result = (values * 2.0).relu().sum()
    assert result.item() == 8.0
    result.backward()
    assert values.grad.tolist() == [0.0, 2.0, 2.0]
    with torch.no_grad():
        assert torch.is_grad_enabled() is False
        assert torch.use_deterministic_algorithms(mode, warn_only=warn_only) is None
        assert torch.relu(torch.tensor([-1.0, 2.0])).tolist() == [0.0, 2.0]
        assert torch.is_grad_enabled() is False

assert importlib.reload(torch) is torch
assert torch.are_deterministic_algorithms_enabled() is True
assert torch.is_deterministic_algorithms_warn_only_enabled() is False
assert torch.get_deterministic_debug_mode() == 2
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

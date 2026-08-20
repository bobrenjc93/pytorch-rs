import copy
import inspect
import pickle
import subprocess
import sys
import threading
import types
import unittest
import warnings

import numpy as np

import torch_rs as torch


class UseDeterministicAlgorithmsTests(unittest.TestCase):
    def setUp(self):
        torch.use_deterministic_algorithms(False, warn_only=False)

    def tearDown(self):
        torch.use_deterministic_algorithms(False, warn_only=False)

    def state(self):
        return (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )

    def test_sets_both_process_global_flags_and_returns_none(self):
        cases = (
            (False, False),
            (True, False),
            (True, True),
            (False, True),
        )
        for mode, warn_only in cases:
            with self.subTest(mode=mode, warn_only=warn_only):
                result = torch.use_deterministic_algorithms(
                    mode,
                    warn_only=warn_only,
                )
                self.assertIsNone(result)
                self.assertEqual(self.state(), (mode, warn_only))
                for value in self.state():
                    self.assertIs(type(value), bool)

        torch.use_deterministic_algorithms(False, warn_only=True)
        self.assertEqual(self.state(), (False, True))
        torch.use_deterministic_algorithms(True)
        self.assertEqual(self.state(), (True, False))
        torch.use_deterministic_algorithms(False)
        self.assertEqual(self.state(), (False, False))

    def test_state_changes_are_visible_across_threads(self):
        torch.use_deterministic_algorithms(False, warn_only=True)
        worker_observations = []
        errors = []

        def worker():
            try:
                worker_observations.append(self.state())
                result = torch.use_deterministic_algorithms(
                    True,
                    warn_only=False,
                )
                worker_observations.append((result, self.state()))
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(
            worker_observations,
            [(False, True), (None, (True, False))],
        )
        self.assertEqual(self.state(), (True, False))

    def test_supported_cpu_operations_remain_deterministic_in_every_state(self):
        def outcome():
            left = torch.tensor(
                [[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]],
                requires_grad=True,
            )
            right = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            weights = torch.tensor([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]])
            product = (left + 1.5).relu().matmul(right)
            zeroed = torch.nn.functional.dropout(
                product,
                p=1.0,
                training=True,
                inplace=False,
            )
            (left * weights).sum().backward()
            return product.tolist(), zeroed.tolist(), left.grad.tolist()

        baseline = None
        for mode, warn_only in (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ):
            with self.subTest(mode=mode, warn_only=warn_only):
                torch.use_deterministic_algorithms(mode, warn_only=warn_only)
                with warnings.catch_warnings(record=True) as caught:
                    first = outcome()
                    second = outcome()
                self.assertEqual(caught, [])
                self.assertEqual(first, second)
                if baseline is None:
                    baseline = first
                else:
                    self.assertEqual(first, baseline)

    def test_callable_metadata_exports_and_pickling(self):
        function = torch.use_deterministic_algorithms
        signature = inspect.signature(function)

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "use_deterministic_algorithms")
        self.assertEqual(function.__qualname__, "use_deterministic_algorithms")
        self.assertEqual(function.__module__, "torch_rs")
        self.assertEqual(
            str(signature),
            "(mode: bool, *, warn_only: bool = False) -> None",
        )
        self.assertEqual(
            function.__annotations__,
            {"mode": bool, "warn_only": bool, "return": None},
        )
        self.assertIsNone(function.__defaults__)
        self.assertEqual(function.__kwdefaults__, {"warn_only": False})
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertTrue(
            function.__doc__.startswith(
                'Sets whether PyTorch operations must use "deterministic"\n'
            )
        )
        self.assertTrue(
            function.__doc__.endswith(
                "RuntimeError: avg_pool3d_backward_cuda does not have a "
                "deterministic implementation...\n    "
            )
        )

        self.assertEqual(torch.__all__.count("use_deterministic_algorithms"), 1)
        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["use_deterministic_algorithms"], function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(function, protocol)), function)

        self.assertTrue(hasattr(torch._C, "_set_deterministic_algorithms"))
        self.assertTrue(hasattr(torch._C, "_get_deterministic_algorithms"))
        self.assertTrue(
            hasattr(torch._C, "_get_deterministic_algorithms_warn_only")
        )
        for name in (
            "_set_deterministic_algorithms",
            "_get_deterministic_algorithms",
            "_get_deterministic_algorithms_warn_only",
        ):
            self.assertFalse(hasattr(torch, name))
            self.assertNotIn(name, torch.__all__)
            self.assertNotIn(name, torch._C.__all__)

    def test_python_argument_binding_errors_match_pytorch_2_13(self):
        function = torch.use_deterministic_algorithms
        cases = (
            (
                lambda: function(),
                "use_deterministic_algorithms() missing 1 required positional "
                "argument: 'mode'",
            ),
            (
                lambda: function(True, False),
                "use_deterministic_algorithms() takes 1 positional argument but 2 "
                "were given",
            ),
            (
                lambda: function(True, False, False),
                "use_deterministic_algorithms() takes 1 positional argument but 3 "
                "were given",
            ),
            (
                lambda: function(warn_only=True),
                "use_deterministic_algorithms() missing 1 required positional "
                "argument: 'mode'",
            ),
            (
                lambda: function(True, mode=False),
                "use_deterministic_algorithms() got multiple values for argument "
                "'mode'",
            ),
            (
                lambda: function(True, enabled=False),
                "use_deterministic_algorithms() got an unexpected keyword argument "
                "'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(self.state(), (False, False))

    def test_requires_exact_bool_values_without_mutating_on_error(self):
        invalid_values = (
            (None, "NoneType"),
            (1, "int"),
            (1.0, "float"),
            ("true", "str"),
            (np.bool_(True), "numpy.bool"),
            (torch.tensor(1.0), "Tensor"),
            (torch.float32, "torch.dtype"),
            (torch.device("cpu"), "torch.device"),
        )
        for value, type_name in invalid_values:
            for argument in ("mode", "warn_only"):
                with self.subTest(argument=argument, type_name=type_name):
                    torch.use_deterministic_algorithms(True, warn_only=True)
                    if argument == "mode":
                        call = lambda value=value: torch.use_deterministic_algorithms(
                            value,
                            warn_only=False,
                        )
                        position = " (position 1)"
                    else:
                        call = lambda value=value: torch.use_deterministic_algorithms(
                            False,
                            warn_only=value,
                        )
                        position = ""
                    message = (
                        f"_set_deterministic_algorithms(): argument '{argument}'"
                        f"{position} must be bool, not {type_name}"
                    )
                    with self.assertRaises(TypeError) as raised:
                        call()
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertEqual(self.state(), (True, True))

    def test_debug_mode_apis_remain_unsupported(self):
        for name in (
            "set_deterministic_debug_mode",
            "get_deterministic_debug_mode",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)

    def test_import_reload_and_calls_do_not_import_pytorch(self):
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

assert torch.use_deterministic_algorithms(False, warn_only=True) is None
assert torch.are_deterministic_algorithms_enabled() is False
assert torch.is_deterministic_algorithms_warn_only_enabled() is True
assert importlib.reload(torch) is torch
assert torch.are_deterministic_algorithms_enabled() is False
assert torch.is_deterministic_algorithms_warn_only_enabled() is True
assert torch.use_deterministic_algorithms(True) is None
assert torch.are_deterministic_algorithms_enabled() is True
assert torch.is_deterministic_algorithms_warn_only_enabled() is False
assert not hasattr(torch, "set_deterministic_debug_mode")
assert not hasattr(torch, "get_deterministic_debug_mode")
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

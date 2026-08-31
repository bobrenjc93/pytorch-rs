import contextlib
import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import typing
import unittest

import torch_rs as torch


class _StringPrecision(str):
    def __eq__(self, other):
        raise AssertionError("the setter must not dispatch string equality")

    def __str__(self):
        raise AssertionError("the setter must not dispatch string conversion")

    def encode(self, *args, **kwargs):
        raise AssertionError("the setter must not dispatch string encoding")


class _BytesPrecision(bytes):
    def __eq__(self, other):
        raise AssertionError("the setter must not dispatch bytes equality")

    def decode(self, *args, **kwargs):
        raise AssertionError("the setter must not dispatch bytes decoding")


class SetFloat32MatmulPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.original = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")

    def tearDown(self):
        torch.set_float32_matmul_precision(self.original)

    def test_highest_and_high_update_tf32_preference_without_changing_matmul(self):
        left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        right = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        expected = [[19.0, 22.0], [43.0, 50.0]]

        cases = (
            (lambda: torch.set_float32_matmul_precision("highest"), "highest", False),
            (
                lambda: torch.set_float32_matmul_precision(precision="highest"),
                "highest",
                False,
            ),
            (lambda: torch.set_float32_matmul_precision(b"highest"), "highest", False),
            (
                lambda: torch.set_float32_matmul_precision(
                    _StringPrecision("highest")
                ),
                "highest",
                False,
            ),
            (
                lambda: torch.set_float32_matmul_precision(
                    _BytesPrecision(b"highest")
                ),
                "highest",
                False,
            ),
            (lambda: torch.set_float32_matmul_precision("high"), "high", True),
            (lambda: torch.set_float32_matmul_precision(precision="high"), "high", True),
            (lambda: torch.set_float32_matmul_precision(b"high"), "high", True),
            (
                lambda: torch.set_float32_matmul_precision(_StringPrecision("high")),
                "high",
                True,
            ),
            (
                lambda: torch.set_float32_matmul_precision(_BytesPrecision(b"high")),
                "high",
                True,
            ),
        )
        for context in (contextlib.nullcontext(), torch.no_grad()):
            with context:
                expected_grad_mode = torch.is_grad_enabled()
                for case, (call, precision, allow_tf32) in enumerate(cases):
                    with self.subTest(
                        grad=expected_grad_mode,
                        case=case,
                        precision=precision,
                    ):
                        torch.set_float32_matmul_precision("highest")
                        self.assertIsNone(call())
                        self.assertEqual(torch.get_float32_matmul_precision(), precision)
                        self.assertIs(
                            torch.backends.cuda.matmul.allow_tf32,
                            allow_tf32,
                        )
                        self.assertEqual(torch.matmul(left, right).tolist(), expected)
                        self.assertIs(torch.is_grad_enabled(), expected_grad_mode)

        self.assertTrue(torch.is_grad_enabled())
        self.assertEqual(torch.get_float32_matmul_precision(), "high")

    def test_medium_precision_mode_is_rejected_without_side_effects(self):
        left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        right = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        expected = torch.matmul(left, right).tolist()

        for initial in ("highest", "high"):
            for precision in ("medium", b"medium"):
                with self.subTest(initial=initial, precision=precision):
                    torch.set_float32_matmul_precision(initial)
                    grad_mode = torch.is_grad_enabled()
                    rendered = (
                        precision.decode() if isinstance(precision, bytes) else precision
                    )
                    message = (
                        "set_float32_matmul_precision(): precision "
                        f"{rendered!r} is not supported; only 'highest' and "
                        "'high' are implemented"
                    )
                    with self.assertRaises(NotImplementedError) as raised:
                        torch.set_float32_matmul_precision(precision)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertEqual(torch.get_float32_matmul_precision(), initial)
                    self.assertIs(
                        torch.backends.cuda.matmul.allow_tf32,
                        initial == "high",
                    )
                    self.assertEqual(torch.matmul(left, right).tolist(), expected)
                    self.assertIs(torch.is_grad_enabled(), grad_mode)

    def test_non_string_errors_match_pytorch_2_13_spelling(self):
        class CustomPrecision:
            def __str__(self):
                return "highest"

        values = (
            (None, "NoneType"),
            (True, "bool"),
            (1, "int"),
            (1.0, "float"),
            (object(), "object"),
            ([], "list"),
            (bytearray(b"highest"), "bytearray"),
            (memoryview(b"highest"), "memoryview"),
            (CustomPrecision(), "CustomPrecision"),
            (torch.tensor(1.0), "Tensor"),
            (torch.float32, "torch.dtype"),
            (torch.device("cpu"), "torch.device"),
        )
        for initial in ("highest", "high"):
            for value, type_name in values:
                with self.subTest(initial=initial, type_name=type_name):
                    torch.set_float32_matmul_precision(initial)
                    message = (
                        "set_float32_matmul_precision expects a str, but got "
                        f"{type_name}"
                    )
                    with self.assertRaises(RuntimeError) as raised:
                        torch.set_float32_matmul_precision(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertEqual(torch.get_float32_matmul_precision(), initial)

            torch.set_float32_matmul_precision(initial)
            with self.assertRaisesRegex(
                RuntimeError, "^error unpacking string as utf-8$"
            ):
                torch.set_float32_matmul_precision("\ud800")
            self.assertEqual(torch.get_float32_matmul_precision(), initial)

    def test_callable_metadata_matches_pytorch_2_13(self):
        package = importlib.import_module("torch_rs")
        function = package.set_float32_matmul_precision

        self.assertIs(torch, package)
        self.assertIs(sys.modules["torch_rs"], package)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(precision: str) -> None")
        self.assertEqual(function.__annotations__, {"precision": str, "return": None})
        self.assertEqual(
            typing.get_type_hints(function),
            {"precision": str, "return": type(None)},
        )
        self.assertEqual(function.__name__, "set_float32_matmul_precision")
        self.assertEqual(function.__qualname__, "set_float32_matmul_precision")
        self.assertEqual(function.__module__, "torch_rs")
        self.assertIs(inspect.getmodule(function), package)
        self.assertTrue(
            function.__doc__.startswith(
                "Sets the internal precision of float32 matrix multiplications."
            )
        )
        self.assertIn(
            'precision(str): can be set to "highest" (default), "high", or "medium"',
            function.__doc__,
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

    def test_exports_copy_and_pickle_use_the_canonical_module(self):
        function = torch.set_float32_matmul_precision

        self.assertEqual(torch.__all__.count("set_float32_matmul_precision"), 1)
        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["set_float32_matmul_precision"], function)
        self.assertFalse(hasattr(torch._C, "_set_float32_matmul_precision"))

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_argument_binding_errors_match_pytorch_2_13(self):
        function = torch.set_float32_matmul_precision
        cases = (
            (
                lambda: function(),
                "set_float32_matmul_precision() missing 1 required positional "
                "argument: 'precision'",
            ),
            (
                lambda: function("highest", "highest"),
                "set_float32_matmul_precision() takes 1 positional argument but 2 "
                "were given",
            ),
            (
                lambda: function(value="highest"),
                "set_float32_matmul_precision() got an unexpected keyword argument "
                "'value'",
            ),
            (
                lambda: function("highest", precision="highest"),
                "set_float32_matmul_precision() got multiple values for argument "
                "'precision'",
            ),
        )
        for initial in ("highest", "high"):
            torch.set_float32_matmul_precision(initial)
            for call, message in cases:
                with self.subTest(initial=initial, message=message):
                    with self.assertRaises(TypeError) as raised:
                        call()
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertEqual(torch.get_float32_matmul_precision(), initial)

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

assert torch.set_float32_matmul_precision("highest") is None
assert torch.set_float32_matmul_precision(b"highest") is None
assert torch.set_float32_matmul_precision("high") is None
assert torch.get_float32_matmul_precision() == "high"
assert torch.backends.cuda.matmul.allow_tf32 is True
assert torch.set_float32_matmul_precision(b"high") is None
assert torch.get_float32_matmul_precision() == "high"
assert torch.set_float32_matmul_precision("highest") is None
assert torch.get_float32_matmul_precision() == "highest"
assert "set_float32_matmul_precision" in torch.__all__
assert not hasattr(torch._C, "_set_float32_matmul_precision")
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

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
import warnings

import torch_rs as torch


GETTER_DOC = """Returns the current value of float32 matrix multiplication precision. Refer to
    :func:`torch.set_float32_matmul_precision` documentation for more details.
    """

INVALID_PRECISION_WARNING_SUFFIX = (
    " is not one of 'highest', 'high', or 'medium'; the "
    "currentsetFloat32MatmulPrecision call has no effect. (Triggered internally at "
    "/__w/pytorch/pytorch/aten/src/ATen/Context.cpp:458.)"
)


class Float32MatmulPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.original_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")

    def tearDown(self):
        torch.set_float32_matmul_precision(self.original_precision)

    def test_supported_values_update_the_getter_and_return_none(self):
        for precision in ("highest", "high", "medium", "highest"):
            with self.subTest(precision=precision):
                self.assertIsNone(torch.set_float32_matmul_precision(precision))
                observed = torch.get_float32_matmul_precision()
                self.assertIs(type(observed), str)
                self.assertEqual(observed, precision)

        self.assertIsNone(torch.set_float32_matmul_precision(precision="medium"))
        self.assertEqual(torch.get_float32_matmul_precision(), "medium")

        for precision in (b"highest", b"high", b"medium"):
            with self.subTest(bytes_precision=precision):
                self.assertIsNone(torch.set_float32_matmul_precision(precision))
                self.assertEqual(
                    torch.get_float32_matmul_precision(), precision.decode("ascii")
                )

    def test_state_is_process_global_across_threads_and_grad_modes(self):
        worker_ready = threading.Event()
        read_high = threading.Event()
        wrote_medium = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                with torch.no_grad():
                    worker_ready.set()
                    if not read_high.wait(timeout=10):
                        raise RuntimeError("timed out waiting for high precision")
                    observations.append(
                        (torch.is_grad_enabled(), torch.get_float32_matmul_precision())
                    )
                    observations.append(
                        torch.set_float32_matmul_precision("medium") is None
                    )
                    observations.append(
                        (torch.is_grad_enabled(), torch.get_float32_matmul_precision())
                    )
                    wrote_medium.set()
            except BaseException as error:
                errors.append(error)
                wrote_medium.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_ready.wait(timeout=10))
        self.assertTrue(torch.is_grad_enabled())
        self.assertIsNone(torch.set_float32_matmul_precision("high"))
        read_high.set()
        self.assertTrue(wrote_medium.wait(timeout=10))
        self.assertEqual(torch.get_float32_matmul_precision(), "medium")
        self.assertTrue(torch.is_grad_enabled())
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [(False, "high"), True, (False, "medium")])

    def test_state_survives_package_reload_and_old_function_references(self):
        package = importlib.import_module("torch_rs")
        native = package._C
        old_getter = package.get_float32_matmul_precision
        old_setter = package.set_float32_matmul_precision
        old_setter("medium")

        self.assertIs(importlib.reload(package), package)
        self.assertIs(torch, package)
        self.assertIs(package._C, native)
        self.assertIsNot(package.get_float32_matmul_precision, old_getter)
        self.assertIsNot(package.set_float32_matmul_precision, old_setter)
        self.assertEqual(package.get_float32_matmul_precision(), "medium")
        self.assertEqual(old_getter(), "medium")

        old_setter("high")
        self.assertEqual(package.get_float32_matmul_precision(), "high")
        package.set_float32_matmul_precision("highest")
        self.assertEqual(old_getter(), "highest")

    def test_all_modes_leave_cpu_matmul_values_and_gradient_behavior_unchanged(self):
        baseline_values = None
        for precision in ("highest", "high", "medium"):
            with self.subTest(precision=precision):
                self.assertIsNone(torch.set_float32_matmul_precision(precision))
                grad_mode_before = torch.is_grad_enabled()
                left = torch.tensor(
                    [[1.0001, -2.0003, 3.0007], [0.3333, 0.7777, -1.2345]],
                    requires_grad=True,
                )
                right = torch.tensor(
                    [[-0.1251, 2.5003], [4.1257, -0.0629], [1.3337, 0.2501]],
                    requires_grad=True,
                )
                output = torch.matmul(left, right)
                values = output.tolist()
                if baseline_values is None:
                    baseline_values = values
                else:
                    self.assertEqual(values, baseline_values)

                self.assertEqual(output.device, torch.device("cpu"))
                self.assertIs(output.dtype, torch.float32)
                self.assertFalse(output.requires_grad)
                self.assertTrue(output.is_leaf)
                with self.assertRaises(RuntimeError) as raised:
                    output.sum().backward()
                self.assertEqual(
                    str(raised.exception),
                    "element 0 of tensors does not require grad and does not have a "
                    "grad_fn",
                )
                self.assertIsNone(left.grad)
                self.assertIsNone(right.grad)
                self.assertEqual(torch.is_grad_enabled(), grad_mode_before)

                supported_grad = torch.tensor([1.5, -2.0], requires_grad=True)
                (supported_grad * supported_grad).sum().backward()
                self.assertEqual(supported_grad.grad.tolist(), [3.0, -4.0])

    def test_signature_annotations_documentation_and_module_identity(self):
        package = importlib.import_module("torch_rs")
        getter = package.get_float32_matmul_precision
        setter = package.set_float32_matmul_precision

        self.assertIs(torch, package)
        self.assertIs(sys.modules["torch_rs"], package)
        for function in (getter, setter):
            self.assertIs(type(function), types.FunctionType)
            self.assertEqual(function.__module__, "torch_rs")
            self.assertIs(inspect.getmodule(function), package)
            self.assertIsNone(function.__defaults__)
            self.assertIsNone(function.__kwdefaults__)
            self.assertEqual(function.__dict__, {})
            self.assertFalse(hasattr(function, "__text_signature__"))

        self.assertEqual(str(inspect.signature(getter)), "() -> str")
        self.assertEqual(getter.__annotations__, {"return": str})
        self.assertEqual(typing.get_type_hints(getter), {"return": str})
        self.assertEqual(getter.__name__, "get_float32_matmul_precision")
        self.assertEqual(getter.__qualname__, "get_float32_matmul_precision")
        self.assertEqual(
            inspect.cleandoc(getter.__doc__), inspect.cleandoc(GETTER_DOC)
        )

        self.assertEqual(str(inspect.signature(setter)), "(precision: str) -> None")
        self.assertEqual(
            setter.__annotations__, {"precision": str, "return": None}
        )
        self.assertEqual(
            typing.get_type_hints(setter),
            {"precision": str, "return": type(None)},
        )
        self.assertEqual(setter.__name__, "set_float32_matmul_precision")
        self.assertEqual(setter.__qualname__, "set_float32_matmul_precision")
        self.assertTrue(
            setter.__doc__.startswith(
                "Sets the internal precision of float32 matrix multiplications."
            )
        )
        self.assertIn(
            'precision(str): can be set to "highest" (default), "high", or "medium"',
            setter.__doc__,
        )

    def test_exports_copy_pickle_and_native_hooks_use_canonical_objects(self):
        functions = {
            "get_float32_matmul_precision": torch.get_float32_matmul_precision,
            "set_float32_matmul_precision": torch.set_float32_matmul_precision,
        }
        namespace = {}
        exec("from torch_rs import *", namespace)

        for name, function in functions.items():
            with self.subTest(name=name):
                self.assertEqual(torch.__all__.count(name), 1)
                self.assertIs(namespace[name], function)
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(b"torch_rs", payload)
                    self.assertIs(pickle.loads(payload), function)

        self.assertTrue(hasattr(torch._C, "_get_float32_matmul_precision"))
        self.assertTrue(hasattr(torch._C, "_set_float32_matmul_precision"))
        self.assertNotIn("_get_float32_matmul_precision", torch._C.__all__)
        self.assertNotIn("_set_float32_matmul_precision", torch._C.__all__)
        self.assertFalse(hasattr(torch, "_get_float32_matmul_precision"))
        self.assertFalse(hasattr(torch, "_set_float32_matmul_precision"))

    def test_python_argument_binding_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.get_float32_matmul_precision(None),
                "get_float32_matmul_precision() takes 0 positional arguments but 1 "
                "was given",
            ),
            (
                lambda: torch.get_float32_matmul_precision(precision=None),
                "get_float32_matmul_precision() got an unexpected keyword argument "
                "'precision'",
            ),
            (
                lambda: torch.set_float32_matmul_precision(),
                "set_float32_matmul_precision() missing 1 required positional "
                "argument: 'precision'",
            ),
            (
                lambda: torch.set_float32_matmul_precision("high", "medium"),
                "set_float32_matmul_precision() takes 1 positional argument but 2 "
                "were given",
            ),
            (
                lambda: torch.set_float32_matmul_precision(value="high"),
                "set_float32_matmul_precision() got an unexpected keyword argument "
                "'value'",
            ),
            (
                lambda: torch.set_float32_matmul_precision(
                    "high", precision="medium"
                ),
                "set_float32_matmul_precision() got multiple values for argument "
                "'precision'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                torch.set_float32_matmul_precision("medium")
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(torch.get_float32_matmul_precision(), "medium")

    def test_invalid_types_raise_runtime_error_without_changing_state(self):
        class CustomPrecision:
            pass

        invalid_values = (
            None,
            True,
            1,
            1.0,
            object(),
            [],
            bytearray(b"high"),
            memoryview(b"high"),
            CustomPrecision(),
        )
        for value in invalid_values:
            with self.subTest(value_type=type(value).__name__):
                torch.set_float32_matmul_precision("medium")
                message = (
                    "set_float32_matmul_precision expects a str, but got "
                    f"{type(value).__name__}"
                )
                with self.assertRaises(RuntimeError) as raised:
                    torch.set_float32_matmul_precision(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(torch.get_float32_matmul_precision(), "medium")

        with self.assertRaisesRegex(
            RuntimeError, "^error unpacking string as utf-8$"
        ):
            torch.set_float32_matmul_precision("\ud800")
        self.assertEqual(torch.get_float32_matmul_precision(), "medium")

    def test_invalid_strings_warn_and_are_noops(self):
        cases = (
            ("HIGH", "HIGH" + INVALID_PRECISION_WARNING_SUFFIX),
            ("", INVALID_PRECISION_WARNING_SUFFIX),
            (" medium ", " medium " + INVALID_PRECISION_WARNING_SUFFIX),
            (b"low", "low" + INVALID_PRECISION_WARNING_SUFFIX),
            ("highest\0ignored", "highest"),
            ("\0medium", ""),
        )
        for value, message in cases:
            with self.subTest(value=value):
                torch.set_float32_matmul_precision("medium")
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = torch.set_float32_matmul_precision(value)
                self.assertIsNone(result)
                self.assertEqual(torch.get_float32_matmul_precision(), "medium")
                self.assertEqual(len(caught), 1)
                self.assertIs(caught[0].category, UserWarning)
                self.assertIs(type(caught[0].message), UserWarning)
                self.assertEqual(str(caught[0].message), message)

        torch.set_float32_matmul_precision("high")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(UserWarning) as raised:
                torch.set_float32_matmul_precision("invalid")
        self.assertEqual(
            str(raised.exception), "invalid" + INVALID_PRECISION_WARNING_SUFFIX
        )
        self.assertEqual(torch.get_float32_matmul_precision(), "high")

    def test_importing_reloading_and_calling_does_not_import_pytorch(self):
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

assert torch.get_float32_matmul_precision() == "highest"
assert torch.set_float32_matmul_precision("high") is None
assert torch.get_float32_matmul_precision() == "high"
assert importlib.reload(torch) is torch
assert torch.get_float32_matmul_precision() == "high"
assert torch.set_float32_matmul_precision("medium") is None
assert torch._C._get_float32_matmul_precision() == "medium"
assert "set_float32_matmul_precision" in torch.__all__
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

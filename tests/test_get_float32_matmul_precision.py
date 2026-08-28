import contextlib
import copy
import importlib
import inspect
import os
import pickle
import subprocess
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch


FUNCTION_DOC = """Returns the current value of float32 matrix multiplication precision. Refer to
    :func:`torch.set_float32_matmul_precision` documentation for more details.
    """


class GetFloat32MatmulPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.original_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.set_float32_matmul_precision("highest")

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.original_allow_tf32

    def test_returns_exact_shared_precision_without_runtime_probes(self):
        function = torch.get_float32_matmul_precision
        self.assertEqual(
            function.__code__.co_names,
            ("_C", "_get_cublas_allow_tf32"),
        )
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        for allow_tf32, precision in ((False, "highest"), (True, "high")):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            for _ in range(4):
                result = function()
                self.assertIs(type(result), str)
                self.assertEqual(result, precision)

    def test_query_preserves_native_matmul_and_grad_mode(self):
        left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        right = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

        def query_and_multiply():
            grad_before = torch.is_grad_enabled()
            precision = torch.get_float32_matmul_precision()
            product = torch.matmul(left, right)
            grad_after = torch.is_grad_enabled()
            return grad_before, precision, product.tolist(), grad_after

        expected_product = [[19.0, 22.0], [43.0, 50.0]]
        for allow_tf32, precision in ((False, "highest"), (True, "high")):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            self.assertEqual(
                query_and_multiply(),
                (True, precision, expected_product, True),
            )
            with torch.no_grad():
                self.assertEqual(
                    query_and_multiply(),
                    (False, precision, expected_product, False),
                )
            self.assertEqual(
                query_and_multiply(),
                (True, precision, expected_product, True),
            )

    def test_shared_precision_is_stable_across_threads_and_grad_modes(self):
        function = torch.get_float32_matmul_precision
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    first = function()
                    middle_grad_state = torch.is_grad_enabled()
                    second = function()
                    results[index] = (
                        torch.is_grad_enabled(),
                        type(first) is str,
                        first,
                        middle_grad_state,
                        type(second) is str,
                        second,
                        torch.is_grad_enabled(),
                    )
            except BaseException as error:
                errors.append(error)

        for allow_tf32, precision in ((False, "highest"), (True, "high")):
            with self.subTest(allow_tf32=allow_tf32):
                torch.backends.cuda.matmul.allow_tf32 = allow_tf32
                results[:] = [None] * worker_count
                errors.clear()
                barrier = threading.Barrier(worker_count)
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
                    expected_grad_state = index % 2 == 0
                    self.assertEqual(
                        result,
                        (
                            expected_grad_state,
                            True,
                            precision,
                            expected_grad_state,
                            True,
                            precision,
                            expected_grad_state,
                        ),
                    )

    def test_signature_annotations_documentation_and_module_identity(self):
        package = importlib.import_module("torch_rs")
        function = package.get_float32_matmul_precision

        self.assertIs(torch, package)
        self.assertIs(sys.modules["torch_rs"], package)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> str")
        self.assertEqual(function.__annotations__, {"return": str})
        self.assertEqual(typing.get_type_hints(function), {"return": str})
        self.assertEqual(function.__name__, "get_float32_matmul_precision")
        self.assertEqual(function.__qualname__, "get_float32_matmul_precision")
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

    def test_exports_copy_and_pickle_use_the_canonical_module(self):
        function = torch.get_float32_matmul_precision

        self.assertEqual(torch.__all__.count("get_float32_matmul_precision"), 1)
        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["get_float32_matmul_precision"], function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.get_float32_matmul_precision
        cases = (
            (
                lambda: function(None),
                "get_float32_matmul_precision() takes 0 positional arguments "
                "but 1 was given",
            ),
            (
                lambda: function(None, None),
                "get_float32_matmul_precision() takes 0 positional arguments "
                "but 2 were given",
            ),
            (
                lambda: function(precision=None),
                "get_float32_matmul_precision() got an unexpected keyword "
                "argument 'precision'",
            ),
            (
                lambda: function(None, precision=None),
                "get_float32_matmul_precision() got an unexpected keyword "
                "argument 'precision'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_highest_only_setter_is_exposed_at_the_python_layer(self):
        self.assertTrue(hasattr(torch, "set_float32_matmul_precision"))
        self.assertEqual(torch.__all__.count("set_float32_matmul_precision"), 1)
        self.assertFalse(hasattr(torch._C, "_set_float32_matmul_precision"))

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

assert torch.get_float32_matmul_precision() == "highest"
torch.backends.cuda.matmul.allow_tf32 = True
assert torch.get_float32_matmul_precision() == "high"
assert torch.set_float32_matmul_precision("highest") is None
assert torch.backends.cuda.matmul.allow_tf32 is False
assert torch.get_float32_matmul_precision() == "highest"
assert "set_float32_matmul_precision" in torch.__all__
assert not hasattr(torch._C, "_set_float32_matmul_precision")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
        environment = os.environ.copy()
        environment.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()

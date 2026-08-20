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


FUNCTION_DOC = """Returns the current value of float32 matrix multiplication precision. Refer to
    :func:`torch.set_float32_matmul_precision` documentation for more details.
    """


class GetFloat32MatmulPrecisionTests(unittest.TestCase):
    def test_returns_exact_highest_and_preserves_grad_mode(self):
        function = torch.get_float32_matmul_precision
        self.assertEqual(function.__code__.co_names, ())
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        def assert_query_preserves_grad_mode(expected_grad_state):
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            result = function()
            self.assertIs(type(result), str)
            self.assertEqual(result, "highest")
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)

        assert_query_preserves_grad_mode(True)
        with torch.no_grad():
            assert_query_preserves_grad_mode(False)
            with torch.no_grad():
                assert_query_preserves_grad_mode(False)
            assert_query_preserves_grad_mode(False)
        assert_query_preserves_grad_mode(True)

    def test_highest_is_stable_across_threads_and_grad_modes(self):
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
                    results[index] = (
                        torch.is_grad_enabled(),
                        function(),
                        torch.is_grad_enabled(),
                        function(),
                        torch.is_grad_enabled(),
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
            expected_grad_state = index % 2 == 0
            self.assertEqual(
                result,
                (
                    expected_grad_state,
                    "highest",
                    expected_grad_state,
                    "highest",
                    expected_grad_state,
                ),
            )
            self.assertIs(type(result[1]), str)
            self.assertIs(type(result[3]), str)

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

    def test_setter_remains_unsupported_and_matmul_is_unchanged(self):
        self.assertFalse(hasattr(torch, "set_float32_matmul_precision"))
        self.assertNotIn("set_float32_matmul_precision", torch.__all__)

        left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        right = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        before = torch.matmul(left, right)
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        after = torch.matmul(left, right)
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(after.tolist(), [[19.0, 22.0], [43.0, 50.0]])

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

function = torch.get_float32_matmul_precision
assert function.__code__.co_names == ()
result = function()
assert type(result) is str
assert result == "highest"
assert not hasattr(torch, "set_float32_matmul_precision")
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

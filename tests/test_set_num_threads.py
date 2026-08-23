import contextlib
import copy
import inspect
import pickle
import subprocess
import sys
import threading
import types
import unittest

import numpy as np

import torch_rs as torch


FUNCTION_DOC = """
set_num_threads(int)

Sets the number of threads used for intraop parallelism on CPU.

.. warning::
    To ensure that the correct number of threads is used, set_num_threads
    must be called before running eager, JIT or autograd code.
"""

UNSUPPORTED_WORKER_COUNT = (
    "set_num_threads only supports the single-worker value 1"
)


class _IntSubclass(int):
    pass


class _IndexOne:
    def __index__(self):
        return 1


class _IntOne:
    def __int__(self):
        return 1


class SetNumThreadsTests(unittest.TestCase):
    def computation_outcome(self):
        grad_enabled = torch.is_grad_enabled()
        leaf = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)

        first = torch.set_num_threads(1)
        output = (leaf * leaf).sum()
        second = torch.set_num_threads(1)
        if grad_enabled:
            output.backward()
        third = torch.set_num_threads(1)

        return (
            grad_enabled,
            first,
            second,
            third,
            torch.is_grad_enabled(),
            type(torch.get_num_threads()) is int,
            torch.get_num_threads(),
            output.item(),
            output.requires_grad,
            None if leaf.grad is None else leaf.grad.tolist(),
        )

    def test_one_is_a_repeatable_single_worker_noop(self):
        self.assertEqual(
            self.computation_outcome(),
            (True, None, None, None, True, True, 1, 14.0, True, [2.0, -4.0, 6.0]),
        )
        with torch.no_grad():
            self.assertEqual(
                self.computation_outcome(),
                (False, None, None, None, False, True, 1, 14.0, False, None),
            )
        self.assertIs(torch.is_grad_enabled(), True)
        self.assertIs(torch.get_num_threads(), 1)

        for value in (1, _IntSubclass(1), np.int32(1), np.int64(1), np.uint64(1)):
            with self.subTest(value=repr(value)):
                self.assertIs(torch.set_num_threads(value), None)
                self.assertIs(type(torch.get_num_threads()), int)
                self.assertIs(torch.get_num_threads(), 1)

    def test_repeated_cross_thread_calls_preserve_computation_and_grad_state(self):
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = self.computation_outcome()
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
            if index % 2:
                expected = (
                    False,
                    None,
                    None,
                    None,
                    False,
                    True,
                    1,
                    14.0,
                    False,
                    None,
                )
            else:
                expected = (
                    True,
                    None,
                    None,
                    None,
                    True,
                    True,
                    1,
                    14.0,
                    True,
                    [2.0, -4.0, 6.0],
                )
            self.assertEqual(result, expected)

        self.assertIs(torch.is_grad_enabled(), True)
        self.assertIs(torch.get_num_threads(), 1)

    def test_builtin_metadata_exports_copy_and_pickling(self):
        function = torch.set_num_threads
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "set_num_threads")
        self.assertEqual(function.__qualname__, "set_num_threads")
        self.assertEqual(function.__module__, torch.tensor.__module__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(repr(function), "<built-in function set_num_threads>")
        self.assertIs(function.__self__, torch._C)
        self.assertIs(torch._C.set_num_threads, function)
        with self.assertRaisesRegex(
            ValueError,
            r"^no signature found for builtin <built-in function set_num_threads>$",
        ):
            inspect.signature(function)

        self.assertEqual(torch.__all__.count("set_num_threads"), 1)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["set_num_threads"], function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                self.assertIs(restored, function)

    def test_positional_only_binding_errors_match_pytorch_2_13(self):
        function = torch.set_num_threads
        cases = (
            (
                lambda: function(),
                "torch.set_num_threads() takes exactly one argument (0 given)",
            ),
            (
                lambda: function(1, 1),
                "torch.set_num_threads() takes exactly one argument (2 given)",
            ),
            (
                lambda: function(num_threads=1),
                "torch.set_num_threads() takes no keyword arguments",
            ),
            (
                lambda: function(thread_count=1),
                "torch.set_num_threads() takes no keyword arguments",
            ),
            (
                lambda: function(1, num_threads=1),
                "torch.set_num_threads() takes no keyword arguments",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(torch.get_num_threads(), 1)

    def test_rejects_bools_and_nonintegers_with_pytorch_2_13_errors(self):
        invalid_values = (
            (True, "bool"),
            (False, "bool"),
            (None, "NoneType"),
            (1.0, "float"),
            ("1", "str"),
            (b"1", "bytes"),
            ([], "list"),
            (object(), "object"),
            (_IndexOne(), "_IndexOne"),
            (_IntOne(), "_IntOne"),
            (np.bool_(True), "numpy.bool"),
            (np.float32(1), "numpy.float32"),
            (torch.tensor(1.0), "Tensor"),
            (torch.float32, "torch.dtype"),
            (torch.device("cpu"), "torch.device"),
        )
        for value, type_name in invalid_values:
            with self.subTest(value=repr(value)):
                message = f"set_num_threads expects an int, but got {type_name}"
                with self.assertRaises(RuntimeError) as raised:
                    torch.set_num_threads(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(torch.get_num_threads(), 1)

    def test_rejects_nonpositive_and_overflowing_integers(self):
        for value in (0, -1, -(2**31), _IntSubclass(0), np.int64(-1)):
            with self.subTest(value=repr(value)):
                message = "set_num_threads expects a positive integer"
                with self.assertRaises(RuntimeError) as raised:
                    torch.set_num_threads(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(torch.get_num_threads(), 1)

        for value in (
            2**31,
            -(2**31) - 1,
            np.int64(2**31),
            np.int64(-(2**31) - 1),
            np.uint32(2**31),
        ):
            with self.subTest(value=repr(value)):
                message = "Overflow when unpacking long"
                with self.assertRaises(ValueError) as raised:
                    torch.set_num_threads(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(torch.get_num_threads(), 1)

        for value in (2**100, -(2**100), np.uint64(2**63)):
            with self.subTest(value=repr(value)):
                message = "Overflow when unpacking long long"
                with self.assertRaises(ValueError) as raised:
                    torch.set_num_threads(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(torch.get_num_threads(), 1)

    def test_rejects_worker_counts_above_one_without_state_change(self):
        for value in (2, 8, 2**31 - 1, _IntSubclass(2), np.int64(2)):
            with self.subTest(value=repr(value)):
                with self.assertRaises(RuntimeError) as raised:
                    torch.set_num_threads(value)
                self.assertEqual(str(raised.exception), UNSUPPORTED_WORKER_COUNT)
                self.assertEqual(
                    raised.exception.args,
                    (UNSUPPORTED_WORKER_COUNT,),
                )
                self.assertIs(torch.get_num_threads(), 1)

    def test_numpy_integer_classification_ignores_mutable_module_attributes(self):
        original_integer = np.integer
        numpy_one = np.int64(1)
        try:
            np.integer = int
            self.assertIs(torch.set_num_threads(numpy_one), None)
            self.assertIs(torch.get_num_threads(), 1)

            np.integer = _IndexOne
            message = "set_num_threads expects an int, but got _IndexOne"
            with self.assertRaises(RuntimeError) as raised:
                torch.set_num_threads(_IndexOne())
            self.assertEqual(str(raised.exception), message)
            self.assertEqual(raised.exception.args, (message,))
            self.assertIs(torch.get_num_threads(), 1)
        finally:
            np.integer = original_integer

    def test_set_num_interop_threads_remains_unsupported(self):
        self.assertFalse(hasattr(torch, "set_num_interop_threads"))
        self.assertFalse(hasattr(torch._C, "set_num_interop_threads"))
        self.assertNotIn("set_num_interop_threads", torch.__all__)

        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertNotIn("set_num_interop_threads", wildcard_namespace)

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

assert torch.set_num_threads(1) is None
assert torch.get_num_threads() == 1
assert torch.get_num_interop_threads() == 1
assert not hasattr(torch, "set_num_interop_threads")
assert "numpy" not in sys.modules
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

import copy
import importlib
import inspect
import os
import pickle
import subprocess
import sys
import threading
import types
import unittest

import numpy as np

import torch_rs as torch


SETTER_DOCS = {
    "set_num_threads": """
set_num_threads(int)

Sets the number of threads used for intraop parallelism on CPU.

.. warning::
    To ensure that the correct number of threads is used, set_num_threads
    must be called before running eager, JIT or autograd code.
""",
    "set_num_interop_threads": """
set_num_interop_threads(int)

Sets the number of threads used for interop parallelism
(e.g. in JIT interpreter) on CPU.

.. warning::
    Can only be called once and before any inter-op parallel work
    is started (e.g. JIT execution).
""",
}


class _IntSubclass(int):
    def __int__(self):
        raise AssertionError("int subclass conversion must not dispatch __int__")


class _IndexOnly:
    def __index__(self):
        raise AssertionError("thread setters must reject index-only objects")


class _IntOnly:
    def __int__(self):
        raise AssertionError("thread setters must reject int-only objects")


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("thread setters must not request truthiness")


class ThreadSetterTests(unittest.TestCase):
    def assert_no_builtin_signature(self, function):
        self.assertIsNone(function.__text_signature__)
        with self.assertRaises(ValueError) as raised:
            inspect.signature(function)
        self.assertEqual(
            str(raised.exception),
            f"no signature found for builtin {function!r}",
        )

    def assert_thread_counts_unchanged(self):
        self.assertIs(type(torch.get_num_threads()), int)
        self.assertIs(type(torch.get_num_interop_threads()), int)
        self.assertIs(torch.get_num_threads(), 1)
        self.assertIs(torch.get_num_interop_threads(), 1)

    def test_builtin_metadata_imports_reload_copy_and_pickle(self):
        package = importlib.import_module("torch_rs")
        native = importlib.import_module("torch_rs._C")
        wildcard_namespace = {}
        native_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        exec(
            "from torch_rs._C import set_num_threads, set_num_interop_threads",
            native_namespace,
        )

        old_functions = {}
        for name, doc in SETTER_DOCS.items():
            with self.subTest(name=name):
                function = getattr(package, name)
                old_functions[name] = function
                self.assertIs(type(function), types.BuiltinFunctionType)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(function.__module__, torch.tensor.__module__)
                self.assertEqual(function.__doc__, doc)
                self.assertEqual(repr(function), f"<built-in function {name}>")
                self.assertIs(function.__self__, package._C)
                self.assertIs(getattr(package._C, name), function)
                self.assertIs(getattr(native, name), function)
                self.assertEqual(function.__reduce__(), name)
                self.assert_no_builtin_signature(function)

                self.assertEqual(package.__all__.count(name), 1)
                self.assertIs(wildcard_namespace[name], function)
                self.assertIs(native_namespace[name], function)
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(name=name, protocol=protocol):
                        restored = pickle.loads(
                            pickle.dumps(function, protocol=protocol)
                        )
                        self.assertIs(restored, function)

        self.assertIs(importlib.reload(native), native)
        for name, old_function in old_functions.items():
            self.assertIs(getattr(native, name), old_function)
            self.assertIs(getattr(package, name), old_function)

        self.assertIs(importlib.reload(package), package)
        self.assertIs(package._C, native)
        for name, old_function in old_functions.items():
            self.assertIs(getattr(package, name), old_function)
            self.assertIs(getattr(package._C, name), old_function)

    def test_accepts_singleton_integer_values_and_preserves_getters(self):
        accepted_values = (1, _IntSubclass(1), np.int8(1), np.int64(1), np.uint32(1))
        for name in SETTER_DOCS:
            setter = getattr(torch, name)
            for value in accepted_values:
                with self.subTest(name=name, value=repr(value)):
                    self.assertIs(setter(value), None)
                    self.assert_thread_counts_unchanged()

    def test_rejects_argument_binding_errors_with_pytorch_2_13_messages(self):
        for name in SETTER_DOCS:
            setter = getattr(torch, name)
            cases = (
                (
                    lambda setter=setter: setter(),
                    f"torch.{name}() takes exactly one argument (0 given)",
                ),
                (
                    lambda setter=setter: setter(1, 1),
                    f"torch.{name}() takes exactly one argument (2 given)",
                ),
                (
                    lambda setter=setter: setter(threads=1),
                    f"torch.{name}() takes no keyword arguments",
                ),
                (
                    lambda setter=setter: setter(1, threads=1),
                    f"torch.{name}() takes no keyword arguments",
                ),
            )
            for call, message in cases:
                with self.subTest(name=name, message=message):
                    with self.assertRaises(TypeError) as raised:
                        call()
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assert_thread_counts_unchanged()

    def test_rejects_non_integer_values_with_pytorch_2_13_messages(self):
        invalid_values = (
            (None, "NoneType"),
            (True, "bool"),
            (False, "bool"),
            (np.bool_(True), "numpy.bool"),
            (1.0, "float"),
            (np.float64(1.0), "numpy.float64"),
            ("1", "str"),
            ([], "list"),
            (object(), "object"),
            (_IndexOnly(), "_IndexOnly"),
            (_IntOnly(), "_IntOnly"),
            (_RejectTruthiness(), "_RejectTruthiness"),
            (torch.tensor(1.0), "Tensor"),
            (torch.float32, "torch.dtype"),
            (torch.device("cpu"), "torch.device"),
            (torch.strided, "torch.layout"),
            (torch.Size([1]), "torch.Size"),
            (torch.finfo(torch.float32), "torch.finfo"),
        )
        for name in SETTER_DOCS:
            setter = getattr(torch, name)
            for value, type_name in invalid_values:
                with self.subTest(name=name, value_type=type_name):
                    message = f"{name} expects an int, but got {type_name}"
                    with self.assertRaises(RuntimeError) as raised:
                        setter(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assert_thread_counts_unchanged()

    def test_rejects_invalid_counts_without_state_change(self):
        non_positive_values = (0, -1, np.int8(0), np.int64(-1))
        overflow_cases = (
            (2**31, "Overflow when unpacking long"),
            (-(2**31) - 1, "Overflow when unpacking long"),
            (2**63, "Overflow when unpacking long long"),
            (-(2**63) - 1, "Overflow when unpacking long long"),
            (np.uint64(2**63), "Overflow when unpacking long long"),
        )
        unsupported_values = (2, _IntSubclass(2), np.int16(2), np.uint32(2), 2**31 - 1)

        for name in SETTER_DOCS:
            setter = getattr(torch, name)
            for value in non_positive_values:
                with self.subTest(name=name, value=repr(value)):
                    message = f"{name} expects a positive integer"
                    with self.assertRaises(RuntimeError) as raised:
                        setter(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assert_thread_counts_unchanged()

            for value, message in overflow_cases:
                with self.subTest(name=name, value=repr(value)):
                    with self.assertRaises(ValueError) as raised:
                        setter(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assert_thread_counts_unchanged()

            for value in unsupported_values:
                with self.subTest(name=name, value=repr(value)):
                    with self.assertRaises(NotImplementedError) as raised:
                        setter(value)
                    self.assertEqual(
                        str(raised.exception),
                        f"torch.{name}() only supports the singleton thread count 1",
                    )
                    self.assert_thread_counts_unchanged()

    def test_singleton_setters_are_visible_across_threads(self):
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=10)
                if index % 2:
                    result = torch.set_num_threads(np.int64(1))
                else:
                    result = torch.set_num_interop_threads(_IntSubclass(1))
                results[index] = (
                    result,
                    torch.get_num_threads(),
                    torch.get_num_interop_threads(),
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
        self.assertEqual(results, [(None, 1, 1)] * worker_count)
        self.assert_thread_counts_unchanged()

    def test_importing_calling_and_reloading_do_not_import_pytorch(self):
        script = r"""
import importlib
import os
import pickle
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
os.environ.update(
    OMP_NUM_THREADS="64",
    MKL_NUM_THREADS="32",
    TORCH_NUM_INTEROP_THREADS="16",
)
import torch_rs as torch
from torch_rs import *
from torch_rs._C import set_num_threads as native_set_num_threads

assert set_num_threads is torch.set_num_threads
assert set_num_interop_threads is torch.set_num_interop_threads
assert native_set_num_threads is torch.set_num_threads
assert torch.set_num_threads(1) is None
assert torch.set_num_interop_threads(1) is None
assert torch.get_num_threads() == 1
assert torch.get_num_interop_threads() == 1
try:
    torch.set_num_threads(2)
except NotImplementedError:
    pass
else:
    raise AssertionError("non-singleton intra-op count was accepted")
assert torch.get_num_threads() == 1
assert torch.get_num_interop_threads() == 1
assert pickle.loads(pickle.dumps(torch.set_num_threads)) is torch.set_num_threads
assert importlib.reload(torch._C) is torch._C
assert torch.set_num_threads(1) is None
assert importlib.reload(torch) is torch
assert torch.set_num_interop_threads(1) is None
assert "set_num_threads" in torch.__all__
assert "set_num_interop_threads" in torch.__all__
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

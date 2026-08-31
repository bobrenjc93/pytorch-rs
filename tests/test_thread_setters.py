import copy
import contextlib
import importlib
import inspect
import os
import pickle
import subprocess
import sys
import threading
import types
import unittest
from unittest import mock

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


class _IndexOnly:
    def __index__(self):
        raise AssertionError("thread setters must not call __index__")


class _IntSubclass(int):
    def __index__(self):
        raise AssertionError("thread setters must not call __index__")


class ThreadSetterTests(unittest.TestCase):
    def assert_single_argument_builtin_signature(self, function):
        if sys.version_info >= (3, 13):
            self.assertEqual(function.__text_signature__, "($self, object, /)")
            self.assertEqual(str(inspect.signature(function)), "(object, /)")
        else:
            self.assertIsNone(function.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(function)

    def assert_thread_counts_are_one(self):
        intraop = torch.get_num_threads()
        interop = torch.get_num_interop_threads()
        self.assertIs(type(intraop), int)
        self.assertIs(type(interop), int)
        self.assertIs(intraop, 1)
        self.assertIs(interop, 1)

    def test_exact_one_is_an_idempotent_noop_for_grad_and_threads(self):
        accepted_ones = (1, _IntSubclass(1), np.int64(1), np.uint64(1))
        setters = (torch.set_num_threads, torch.set_num_interop_threads)

        for context in (None, torch.no_grad()):
            if context is None:
                expected_grad_mode = torch.is_grad_enabled()
                manager = contextlib.nullcontext()
            else:
                manager = context
                expected_grad_mode = False
            with manager:
                if context is not None:
                    self.assertIs(torch.is_grad_enabled(), expected_grad_mode)
                for setter in setters:
                    for value in accepted_ones:
                        with self.subTest(setter=setter.__name__, value=type(value)):
                            with mock.patch(
                                "os.cpu_count",
                                side_effect=AssertionError("CPU hardware was probed"),
                            ):
                                self.assertIsNone(setter(value))
                            self.assert_thread_counts_are_one()
                            self.assertIs(
                                torch.is_grad_enabled(),
                                expected_grad_mode,
                            )

        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                manager = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with manager:
                    barrier.wait(timeout=10)
                    before = torch.is_grad_enabled()
                    intraop_result = torch.set_num_threads(1)
                    interop_result = torch.set_num_interop_threads(1)
                    after = torch.is_grad_enabled()
                    results[index] = (
                        before,
                        intraop_result is None,
                        interop_result is None,
                        torch.get_num_threads(),
                        torch.get_num_interop_threads(),
                        after,
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
                (expected_grad_mode, True, True, 1, 1, expected_grad_mode),
            )

    def test_builtin_metadata_exports_copying_pickling_and_reload(self):
        package = importlib.import_module("torch_rs")
        native = importlib.import_module("torch_rs.torch_rs")

        for name, doc in SETTER_DOCS.items():
            with self.subTest(name=name):
                function = getattr(package, name)
                self.assertIs(type(function), types.BuiltinFunctionType)
                self.assertIs(function, getattr(torch, name))
                self.assertIs(function, getattr(native, name))
                self.assertIs(function, getattr(torch._C, name))
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(function.__module__, torch.tensor.__module__)
                self.assertEqual(function.__doc__, doc)
                self.assertEqual(repr(function), f"<built-in function {name}>")
                self.assertIs(function.__self__, torch._C)
                self.assert_single_argument_builtin_signature(function)

                self.assertEqual(torch.__all__.count(name), 1)
                namespace = {}
                exec("from torch_rs import *", namespace)
                self.assertIs(namespace[name], function)

                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(name=name, protocol=protocol):
                        restored = pickle.loads(
                            pickle.dumps(function, protocol=protocol)
                        )
                        self.assertIs(restored, function)

        reloaded = importlib.reload(package)
        self.assertIs(reloaded, package)
        for name in SETTER_DOCS:
            with self.subTest(reloaded=name):
                self.assertIs(getattr(reloaded, name), getattr(native, name))
                self.assertEqual(reloaded.__all__.count(name), 1)
                self.assertIsNone(getattr(reloaded, name)(1))
                self.assert_thread_counts_are_one()

    def test_argument_binding_errors_match_pytorch_2_13_spelling(self):
        for name in SETTER_DOCS:
            function = getattr(torch, name)
            cases = (
                (
                    lambda function=function: function(),
                    f"torch.{name}() takes exactly one argument (0 given)",
                ),
                (
                    lambda function=function: function(1, 1),
                    f"torch.{name}() takes exactly one argument (2 given)",
                ),
                (
                    lambda function=function: function(threads=1),
                    f"torch.{name}() takes no keyword arguments",
                ),
                (
                    lambda function=function: function(1, threads=1),
                    f"torch.{name}() takes no keyword arguments",
                ),
            )
            for call, message in cases:
                with self.subTest(name=name, message=message):
                    with self.assertRaises(TypeError) as raised:
                        call()
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assert_thread_counts_are_one()

    def test_invalid_type_and_nonpositive_errors_match_pytorch_2_13_spelling(self):
        invalid_values = (
            (None, "NoneType"),
            (True, "bool"),
            (False, "bool"),
            (1.0, "float"),
            ("1", "str"),
            (b"1", "bytes"),
            (object(), "object"),
            (_IndexOnly(), "_IndexOnly"),
            (np.bool_(True), "numpy.bool"),
        )
        for name in SETTER_DOCS:
            function = getattr(torch, name)
            for value, type_name in invalid_values:
                with self.subTest(name=name, value=type_name):
                    message = f"{name} expects an int, but got {type_name}"
                    with self.assertRaises(RuntimeError) as raised:
                        function(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assert_thread_counts_are_one()

            for value in (0, -1, np.int64(0), np.int64(-1)):
                with self.subTest(name=name, value=repr(value)):
                    message = f"{name} expects a positive integer"
                    with self.assertRaises(RuntimeError) as raised:
                        function(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assert_thread_counts_are_one()

            for value in (10**100, -(10**100)):
                with self.subTest(name=name, value=value):
                    message = "Overflow when unpacking long long"
                    with self.assertRaises(ValueError) as raised:
                        function(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assert_thread_counts_are_one()

    def test_unsupported_positive_values_are_rejected_without_side_effects(self):
        unsupported_values = (2, _IntSubclass(2), np.int64(2), np.uint64(2))
        for name in SETTER_DOCS:
            function = getattr(torch, name)
            for value in unsupported_values:
                with self.subTest(name=name, value=type(value)):
                    message = (
                        f"{name}(): threads 2 is not supported; "
                        "only 1 is implemented"
                    )
                    with self.assertRaises(NotImplementedError) as raised:
                        function(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assert_thread_counts_are_one()

    def test_importing_and_calling_does_not_import_pytorch(self):
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

for name in ("set_num_threads", "set_num_interop_threads"):
    function = getattr(torch, name)
    assert getattr(torch._C, name) is function
    assert function(1) is None
    assert pickle.loads(pickle.dumps(function)) is function
    assert name in torch.__all__

assert torch.get_num_threads() == 1
assert torch.get_num_interop_threads() == 1
reloaded = importlib.reload(torch)
assert reloaded is torch
assert torch.set_num_threads(1) is None
assert torch.set_num_interop_threads(1) is None
assert torch.get_num_threads() == 1
assert torch.get_num_interop_threads() == 1
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

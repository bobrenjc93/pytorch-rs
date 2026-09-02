import copy
import importlib
import inspect
import os
import pickle
import subprocess
import sys
import types
import unittest
from unittest import mock

import numpy as np

import torch_rs as torch


SET_NUM_THREADS_DOC = """
set_num_threads(int)

Sets the number of threads used for intraop parallelism on CPU.

.. warning::
    To ensure that the correct number of threads is used, set_num_threads
    must be called before running eager, JIT or autograd code.
"""

SET_NUM_INTEROP_THREADS_DOC = """
set_num_interop_threads(int)

Sets the number of threads used for interop parallelism
(e.g. in JIT interpreter) on CPU.

.. warning::
    Can only be called once and before any inter-op parallel work
    is started (e.g. JIT execution).
"""

SETTER_DOCS = {
    "set_num_threads": SET_NUM_THREADS_DOC,
    "set_num_interop_threads": SET_NUM_INTEROP_THREADS_DOC,
}


class ThreadSetterTests(unittest.TestCase):
    def assert_thread_getters_are_one(self):
        intraop = torch.get_num_threads()
        interop = torch.get_num_interop_threads()
        self.assertIs(type(intraop), int)
        self.assertIs(intraop, 1)
        self.assertIs(type(interop), int)
        self.assertIs(interop, 1)

    def test_accepts_integer_scalar_one_as_noop(self):
        class OneInt(int):
            pass

        cases = (
            1,
            OneInt(1),
            np.int8(1),
            np.int32(1),
            np.int64(1),
            np.uint8(1),
            np.uint64(1),
        )
        environments = (
            {},
            {"OMP_NUM_THREADS": "64", "MKL_NUM_THREADS": "32"},
            {"TORCH_NUM_INTEROP_THREADS": "16", "CUDA_VISIBLE_DEVICES": "0"},
        )

        for environment in environments:
            with mock.patch.dict(os.environ, environment, clear=True):
                with mock.patch(
                    "os.cpu_count",
                    side_effect=AssertionError("CPU hardware was probed"),
                ):
                    for name in SETTER_DOCS:
                        function = getattr(torch, name)
                        for value in cases:
                            with self.subTest(name=name, value=repr(value)):
                                self.assertIs(function(value), None)
                                self.assert_thread_getters_are_one()

    def test_rejects_non_integer_values_without_state_changes(self):
        class IndexLike:
            def __index__(self):
                return 1

        invalid_values = (
            (True, RuntimeError, "set_num_threads expects an int, but got bool"),
            (False, RuntimeError, "set_num_threads expects an int, but got bool"),
            (1.0, RuntimeError, "set_num_threads expects an int, but got float"),
            ("1", RuntimeError, "set_num_threads expects an int, but got str"),
            (None, RuntimeError, "set_num_threads expects an int, but got NoneType"),
            (object(), RuntimeError, "set_num_threads expects an int, but got object"),
            (
                IndexLike(),
                RuntimeError,
                "set_num_threads expects an int, but got IndexLike",
            ),
            (
                np.array(1),
                RuntimeError,
                "set_num_threads expects an int, but got numpy.ndarray",
            ),
            (
                np.bool_(True),
                RuntimeError,
                "set_num_threads expects an int, but got numpy.bool",
            ),
            (
                np.float64(1.0),
                RuntimeError,
                "set_num_threads expects an int, but got numpy.float64",
            ),
        )

        for name in SETTER_DOCS:
            function = getattr(torch, name)
            for value, error_type, base_message in invalid_values:
                message = base_message.replace("set_num_threads", name)
                with self.subTest(name=name, value=repr(value)):
                    with self.assertRaises(error_type) as raised:
                        function(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assert_thread_getters_are_one()

    def test_rejects_zero_negative_and_multiworker_counts(self):
        cases = (
            (
                0,
                RuntimeError,
                "set_num_threads expects a positive integer",
            ),
            (
                -1,
                RuntimeError,
                "set_num_threads expects a positive integer",
            ),
            (
                np.int64(0),
                RuntimeError,
                "set_num_threads expects a positive integer",
            ),
            (
                np.int64(-1),
                RuntimeError,
                "set_num_threads expects a positive integer",
            ),
            (
                2,
                NotImplementedError,
                "set_num_threads(): mutable thread pools are not supported; "
                "only 1 worker is implemented",
            ),
            (
                np.int64(2),
                NotImplementedError,
                "set_num_threads(): mutable thread pools are not supported; "
                "only 1 worker is implemented",
            ),
            (
                2**63,
                ValueError,
                "Overflow when unpacking long long",
            ),
        )

        for name in SETTER_DOCS:
            function = getattr(torch, name)
            for value, error_type, base_message in cases:
                message = base_message.replace("set_num_threads", name)
                with self.subTest(name=name, value=repr(value)):
                    with self.assertRaises(error_type) as raised:
                        function(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assert_thread_getters_are_one()

    def test_argument_errors_match_pytorch_2_13_shape(self):
        cases = (
            (
                lambda function: function(),
                "torch.set_num_threads() takes exactly one argument (0 given)",
            ),
            (
                lambda function: function(1, 1),
                "torch.set_num_threads() takes exactly one argument (2 given)",
            ),
            (
                lambda function: function(threads=1),
                "torch.set_num_threads() takes no keyword arguments",
            ),
            (
                lambda function: function(1, threads=1),
                "torch.set_num_threads() takes no keyword arguments",
            ),
        )

        for name in SETTER_DOCS:
            function = getattr(torch, name)
            for call, base_message in cases:
                message = base_message.replace("set_num_threads", name)
                with self.subTest(name=name, message=message):
                    with self.assertRaises(TypeError) as raised:
                        call(function)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assert_thread_getters_are_one()

    def test_builtin_metadata_exports_copying_and_pickling(self):
        for name, doc in SETTER_DOCS.items():
            function = getattr(torch, name)
            with self.subTest(name=name):
                self.assertIs(type(function), types.BuiltinFunctionType)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(function.__module__, torch.tensor.__module__)
                self.assertEqual(function.__doc__, doc)
                self.assertFalse(hasattr(function, "__annotations__"))
                self.assertEqual(repr(function), f"<built-in function {name}>")
                self.assertIs(function.__self__, torch._C)
                self.assertIs(getattr(torch._C, name), function)
                self.assertEqual(function.__reduce__(), name)
                self.assertIsNone(function.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(function)

                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                self.assertEqual(torch.__all__.count(name), 1)

                explicit_namespace = {}
                exec(f"from torch_rs import {name}", explicit_namespace)
                self.assertIs(explicit_namespace[name], function)

                native_namespace = {}
                exec(f"from torch_rs._C import {name}", native_namespace)
                self.assertIs(native_namespace[name], function)

                wildcard_namespace = {}
                exec("from torch_rs import *", wildcard_namespace)
                self.assertIs(wildcard_namespace[name], function)

                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(name=name, protocol=protocol):
                        restored = pickle.loads(
                            pickle.dumps(function, protocol=protocol)
                        )
                        self.assertIs(restored, function)

    def test_native_and_package_reload_preserve_identity_and_noop_behavior(self):
        original_functions = {
            name: getattr(torch, name)
            for name in SETTER_DOCS
        }
        native = torch._C

        self.assertIs(importlib.reload(native), native)
        for name, function in original_functions.items():
            with self.subTest(name=name, reload_kind="native"):
                self.assertIs(getattr(native, name), function)
                self.assertIs(getattr(torch, name), function)
                self.assertIs(function(1), None)
                self.assert_thread_getters_are_one()

        namespace = torch.__dict__
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.__dict__, namespace)
        self.assertIs(torch._C, native)
        for name, function in original_functions.items():
            with self.subTest(name=name, reload_kind="package"):
                self.assertIs(getattr(torch, name), function)
                self.assertIs(getattr(torch._C, name), function)
                self.assertIs(function(1), None)
                self.assert_thread_getters_are_one()

    def test_subprocess_import_noops_are_isolated_from_pytorch_and_environment(self):
        script = r"""
import os
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
    CUDA_VISIBLE_DEVICES="0",
)
import torch_rs as torch

assert torch.get_num_threads() == 1
assert torch.get_num_interop_threads() == 1
for name in ("set_num_threads", "set_num_interop_threads"):
    function = getattr(torch, name)
    assert getattr(torch._C, name) is function
    assert function(1) is None
    try:
        function(2)
    except NotImplementedError as error:
        assert "only 1 worker is implemented" in str(error)
    else:
        raise AssertionError(f"{name} unexpectedly accepted >1 worker")
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

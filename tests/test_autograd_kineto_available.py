import copy
import importlib
import inspect
import os
import pickle
import re
import subprocess
import sys
import threading
import types
import unittest
from unittest import mock

import torch_rs as torch


PYBIND_FUNCTION_RECORD_NAMES = {
    "linux": (
        "pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_"
        "use_cxx11_abi_1"
    ),
    "darwin": "pybind11_detail_function_record_v1_system_libcpp_abi1",
    "win32": "pybind11_detail_function_record_v1_msvc_md_mscver19",
}


class KinetoAvailableTests(unittest.TestCase):
    def test_returns_exact_immutable_false_native_capability(self):
        function = torch.autograd.kineto_available
        native = torch._C._autograd

        self.assertIs(function, native.kineto_available)
        for environment in (
            {},
            {"CUDA_VISIBLE_DEVICES": ""},
            {"CUDA_VISIBLE_DEVICES": "0"},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "KINETO_LOG_LEVEL": "5",
                "NVIDIA_VISIBLE_DEVICES": "all",
            },
        ):
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    result = function()
                    self.assertIs(type(result), bool)
                    self.assertIs(result, False)

        with self.assertRaises(AttributeError):
            function.capability = True

        native.kineto_available = lambda: True
        try:
            self.assertIs(function(), False)
            self.assertIs(torch.autograd.kineto_available, function)
        finally:
            native.kineto_available = function

    def test_native_callable_metadata_matches_pytorch_2_13(self):
        function = torch.autograd.kineto_available
        native = torch._C._autograd
        function_record = function.__self__
        record_type = type(function_record)
        expected_record_name = PYBIND_FUNCTION_RECORD_NAMES.get(
            sys.platform,
            PYBIND_FUNCTION_RECORD_NAMES["linux"],
        )

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "kineto_available")
        self.assertEqual(
            function.__qualname__, f"{expected_record_name}.kineto_available"
        )
        self.assertEqual(function.__module__, "torch_rs._C._autograd")
        self.assertEqual(function.__doc__, "kineto_available() -> bool\n")
        self.assertIsNone(function.__text_signature__)
        self.assertFalse(hasattr(function, "__annotations__"))
        self.assertFalse(hasattr(function, "__defaults__"))
        self.assertFalse(hasattr(function, "__kwdefaults__"))
        self.assertFalse(hasattr(function, "__dict__"))
        self.assertEqual(record_type.__module__, "pybind11_builtins")
        self.assertEqual(record_type.__name__, expected_record_name)
        self.assertIsNot(function_record, native)
        self.assertIs(inspect.getmodule(function), native)
        self.assertRegex(
            repr(function),
            rf"^<built-in method kineto_available of pybind11_builtins\."
            rf"{re.escape(expected_record_name)} object at 0x[0-9a-fA-F]+>$",
        )
        with self.assertRaisesRegex(
            ValueError,
            r"^no signature found for builtin <built-in method "
            r"kineto_available of pybind11_builtins\.",
        ):
            inspect.signature(function)
        self.assertEqual(inspect.get_annotations(function), {})

        reduction = function.__reduce__()
        self.assertIs(reduction[0], getattr)
        self.assertIs(reduction[1][0], function_record)
        self.assertEqual(reduction[1][1], "kineto_available")
        record_reduction = function_record.__reduce_ex__(pickle.HIGHEST_PROTOCOL)
        self.assertIs(record_reduction[0], eval)
        self.assertEqual(
            record_reduction[1],
            ("__import__('importlib').import_module('torch_rs._C._autograd')",),
        )

    def test_direct_imports_are_canonical_but_wildcards_stay_narrow(self):
        autograd = importlib.import_module("torch_rs.autograd")
        native = torch._C._autograd
        function = autograd.kineto_available

        self.assertIs(torch.autograd, autograd)
        self.assertIs(torch._C._autograd, native)
        self.assertIs(sys.modules["torch_rs._C._autograd"], native)
        self.assertIs(sys.modules["torch_rs.torch_rs._autograd"], native)
        self.assertEqual(native.__name__, "torch_rs._C._autograd")
        self.assertEqual(native.__doc__, "autograd bindings")
        self.assertIsNone(native.__package__)
        self.assertIsNone(native.__loader__)
        self.assertIsNone(native.__spec__)
        self.assertFalse(hasattr(native, "__all__"))
        self.assertEqual(
            {name for name in vars(native) if not name.startswith("_")},
            {"kineto_available"},
        )

        public_import = {}
        native_import = {}
        public_wildcard = {}
        native_wildcard = {}
        top_level_wildcard = {}
        exec("from torch_rs.autograd import kineto_available", public_import)
        exec("from torch_rs._C._autograd import kineto_available", native_import)
        exec("from torch_rs.autograd import *", public_wildcard)
        exec("from torch_rs._C._autograd import *", native_wildcard)
        exec("from torch_rs import *", top_level_wildcard)

        self.assertIs(public_import["kineto_available"], function)
        self.assertIs(native_import["kineto_available"], function)
        self.assertNotIn("kineto_available", autograd.__all__)
        self.assertNotIn("kineto_available", public_wildcard)
        self.assertIs(native_wildcard["kineto_available"], function)
        self.assertFalse(hasattr(torch, "kineto_available"))
        self.assertNotIn("kineto_available", torch.__all__)
        self.assertNotIn("kineto_available", top_level_wildcard)
        self.assertNotIn("_autograd", torch.__all__)
        self.assertNotIn("_autograd", torch._C.__all__)

    def test_copying_and_pickling_preserve_the_native_identity(self):
        function = torch.autograd.kineto_available

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs._C._autograd", payload)
                self.assertIn(b"kineto_available", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.autograd.kineto_available
        prefix = (
            "kineto_available(): incompatible function arguments. "
            "The following argument types are supported:\n"
            "    1. () -> bool\n\n"
            "Invoked with: "
        )
        cases = (
            (lambda: function(None), prefix + "None"),
            (lambda: function(None, None), prefix + "None, None"),
            (lambda: function(enabled=True), prefix + "kwargs: enabled=True"),
            (
                lambda: function(None, enabled=True),
                prefix + "None; kwargs: enabled=True",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        self.assertIs(function(*()), False)
        self.assertIs(function(**{}), False)

    def test_threading_and_supported_reloads_preserve_identity(self):
        package = torch
        autograd = torch.autograd
        native = torch._C._autograd
        function = autograd.kineto_available
        barrier = threading.Barrier(17)
        results = [None] * 16
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=10)
                values = tuple(function() for _ in range(1000))
                results[index] = (
                    function is torch.autograd.kineto_available,
                    function is torch._C._autograd.kineto_available,
                    all(type(value) is bool for value in values),
                    all(value is False for value in values),
                )
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(16)]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=10)
        for thread in threads:
            thread.join(timeout=10)

        self.assertEqual([thread.is_alive() for thread in threads], [False] * 16)
        self.assertEqual(errors, [])
        self.assertEqual(results, [(True, True, True, True)] * 16)

        autograd_namespace = autograd.__dict__
        self.assertIs(importlib.reload(torch._C), torch._C)
        self.assertIs(importlib.reload(autograd), autograd)
        self.assertIs(autograd.__dict__, autograd_namespace)
        self.assertIs(importlib.reload(package), package)
        self.assertIs(torch.autograd, autograd)
        self.assertIs(torch._C._autograd, native)
        self.assertIs(torch.autograd.kineto_available, function)
        self.assertIs(torch._C._autograd.kineto_available, function)
        self.assertIs(pickle.loads(pickle.dumps(function)), function)

        with self.assertRaises(AttributeError) as raised:
            importlib.reload(native)
        self.assertRegex(
            str(raised.exception),
            r"^module 'torch_rs\.(?:torch_rs|_C)' has no attribute '__path__'$",
        )

    def test_profiler_surface_remains_unsupported(self):
        for owner in (torch.autograd, torch._C._autograd):
            for name in (
                "ProfilerActivity",
                "ProfilerConfig",
                "ProfilerEvent",
                "ProfilerState",
                "_KinetoEvent",
                "_ProfilerResult",
                "_disable_profiler",
                "_enable_profiler",
                "_kineto_step",
                "_supported_activities",
                "profiler",
                "profiler_legacy",
                "record_function",
            ):
                with self.subTest(owner=owner.__name__, name=name):
                    self.assertFalse(hasattr(owner, name))

        self.assertFalse(hasattr(torch, "profiler"))
        self.assertNotIn("torch_rs.profiler", sys.modules)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.profiler")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.autograd.profiler")

    def test_importing_and_calling_does_not_probe_external_runtimes(self):
        script = r'''
import os
import sys

class RejectExternalRuntimeImport:
    blocked = {"cupy", "kineto", "libkineto", "nvidia", "numpy", "torch"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    CUDA_VISIBLE_DEVICES="0",
    KINETO_LOG_LEVEL="5",
    NVIDIA_VISIBLE_DEVICES="all",
)

import torch_rs as torch
from torch_rs.autograd import kineto_available
from torch_rs._C._autograd import kineto_available as native_kineto_available

assert kineto_available is native_kineto_available
assert kineto_available is torch.autograd.kineto_available
assert kineto_available is torch._C._autograd.kineto_available
assert kineto_available() is False
assert not hasattr(torch, "profiler")
assert not hasattr(torch.autograd, "profiler")
assert not hasattr(torch._C._autograd, "_supported_activities")
assert not any(
    name.split(".", 1)[0] in RejectExternalRuntimeImport.blocked
    for name in sys.modules
)
'''
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

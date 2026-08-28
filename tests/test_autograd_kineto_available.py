import contextlib
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


FUNCTION_DOC = "kineto_available() -> bool\n"
if sys.platform == "darwin":
    FUNCTION_RECORD_NAME = "pybind11_detail_function_record_v1_system_libcpp_abi1"
elif sys.platform == "win32":
    FUNCTION_RECORD_NAME = "pybind11_detail_function_record_v1_msvc_md_mscver19"
else:
    FUNCTION_RECORD_NAME = (
        "pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_"
        "use_cxx11_abi_1"
    )


class AutogradKinetoAvailableTests(unittest.TestCase):
    def test_reports_an_invariant_exact_false_native_capability(self):
        function = torch.autograd.kineto_available
        native = torch._C._autograd
        environments = (
            {},
            {"USE_KINETO": "1"},
            {"KINETO_LIBRARY": "/not/a/library"},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "KINETO_USE_DAEMON": "1",
                "LIBKINETO_PATH": "/not/a/library",
                "USE_KINETO": "1",
            },
        )

        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    self.assertIs(function(), False)
                    self.assertIs(native.kineto_available(), False)
                    self.assertIs(type(function()), bool)

        with mock.patch.object(torch._C, "_has_cuda", True):
            self.assertIs(function(), False)

        self.assertIs(function, native.kineto_available)

    def test_false_and_callable_identity_are_stable_across_threads(self):
        function = torch.autograd.kineto_available
        native = torch._C._autograd
        worker_count = 16
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    before = torch.is_grad_enabled()
                    barrier.wait(timeout=10)
                    results[index] = (
                        before,
                        function(),
                        native.kineto_available(),
                        torch.is_grad_enabled(),
                        function is torch.autograd.kineto_available,
                        function is native.kineto_available,
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
            thread.join(timeout=15)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        for index, result in enumerate(results):
            expected_grad_state = index % 2 == 0
            self.assertEqual(
                result,
                (expected_grad_state, False, False, expected_grad_state, True, True),
            )

    def test_callable_and_native_module_metadata_match_pytorch_2_13(self):
        native = importlib.import_module("torch_rs._C._autograd")
        function = torch.autograd.kineto_available

        self.assertIs(torch._C._autograd, native)
        self.assertIs(sys.modules["torch_rs._C._autograd"], native)
        self.assertIs(type(native), types.ModuleType)
        self.assertEqual(native.__name__, "torch_rs._C._autograd")
        self.assertEqual(native.__doc__, "autograd bindings")
        self.assertIsNone(native.__package__)
        self.assertIsNone(native.__loader__)
        self.assertIsNone(native.__spec__)
        self.assertEqual(native.__file__, torch._C.__file__)
        self.assertEqual(native.__annotations__, {})
        self.assertFalse(hasattr(native, "__all__"))

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertIs(function, native.kineto_available)
        self.assertEqual(function.__name__, "kineto_available")
        self.assertEqual(
            function.__qualname__,
            f"{FUNCTION_RECORD_NAME}.kineto_available",
        )
        self.assertEqual(function.__module__, "torch_rs._C._autograd")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertFalse(hasattr(function, "__annotations__"))
        self.assertFalse(hasattr(function, "__defaults__"))
        self.assertFalse(hasattr(function, "__kwdefaults__"))
        self.assertFalse(hasattr(function, "__dict__"))
        self.assertIs(inspect.getmodule(function), native)
        self.assertEqual(type(function.__self__).__module__, "pybind11_builtins")
        self.assertEqual(type(function.__self__).__name__, FUNCTION_RECORD_NAME)
        self.assertRegex(
            repr(function),
            rf"^<built-in method kineto_available of pybind11_builtins\."
            rf"{FUNCTION_RECORD_NAME} "
            r"object at 0x[0-9a-f]+>$",
        )
        with self.assertRaisesRegex(ValueError, "^no signature found for builtin "):
            inspect.signature(function)

    def test_direct_imports_wildcards_copying_and_pickling_are_canonical(self):
        public = torch.autograd
        native = torch._C._autograd
        function = public.kineto_available
        public_import = {}
        native_import = {}
        public_wildcard = {}
        native_wildcard = {}
        package_wildcard = {}

        exec("from torch_rs.autograd import kineto_available", public_import)
        exec("from torch_rs._C._autograd import kineto_available", native_import)
        exec("from torch_rs.autograd import *", public_wildcard)
        exec("from torch_rs._C._autograd import *", native_wildcard)
        exec("from torch_rs import *", package_wildcard)

        self.assertIs(public_import["kineto_available"], function)
        self.assertIs(native_import["kineto_available"], function)
        self.assertNotIn("kineto_available", public.__all__)
        self.assertNotIn("kineto_available", public_wildcard)
        self.assertIs(native_wildcard["kineto_available"], function)
        self.assertNotIn("_autograd", torch._C.__all__)
        self.assertFalse(hasattr(torch, "_autograd"))
        self.assertNotIn("_autograd", torch.__all__)
        self.assertNotIn("_autograd", package_wildcard)
        self.assertFalse(hasattr(torch, "kineto_available"))

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs._C._autograd", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_reloads_preserve_the_native_callable_identity(self):
        package = torch
        public = package.autograd
        native_extension = package._C
        native = native_extension._autograd
        function = public.kineto_available
        public_namespace = public.__dict__

        self.assertIs(importlib.reload(package), package)
        self.assertIs(package.autograd, public)
        self.assertIs(package._C, native_extension)
        self.assertIs(package._C._autograd, native)
        self.assertIs(package.autograd.kineto_available, function)

        self.assertIs(importlib.reload(native_extension), native_extension)
        self.assertIs(package._C._autograd, native)
        self.assertIs(native.kineto_available, function)

        self.assertIs(importlib.reload(public), public)
        self.assertIs(public.__dict__, public_namespace)
        self.assertIs(package.autograd, public)
        self.assertIs(sys.modules[public.__name__], public)
        self.assertIs(public.kineto_available, function)
        self.assertIs(pickle.loads(pickle.dumps(function)), function)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.autograd.kineto_available
        cases = (
            (
                (None,),
                {},
                "kineto_available(): incompatible function arguments. The following "
                "argument types are supported:\n    1. () -> bool\n\n"
                "Invoked with: None",
            ),
            (
                (None, None),
                {},
                "kineto_available(): incompatible function arguments. The following "
                "argument types are supported:\n    1. () -> bool\n\n"
                "Invoked with: None, None",
            ),
            (
                (),
                {"enabled": True},
                "kineto_available(): incompatible function arguments. The following "
                "argument types are supported:\n    1. () -> bool\n\n"
                "Invoked with: kwargs: enabled=True",
            ),
            (
                (None,),
                {"enabled": True},
                "kineto_available(): incompatible function arguments. The following "
                "argument types are supported:\n    1. () -> bool\n\n"
                "Invoked with: None; kwargs: enabled=True",
            ),
        )
        for args, kwargs, message in cases:
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaises(TypeError) as raised:
                    function(*args, **kwargs)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        self.assertIs(function(**{}), False)

    def test_invalid_call_repr_failures_use_pytorch_placeholder(self):
        class RaisingRepr:
            def __init__(self, error):
                self.error = error

            def __repr__(self):
                raise self.error

        class NonStringRepr:
            def __repr__(self):
                return 1

        cases = (
            ((RaisingRepr(RuntimeError("boom")),), {}),
            ((RaisingRepr(KeyboardInterrupt()),), {}),
            ((RaisingRepr(SystemExit(7)),), {}),
            ((NonStringRepr(),), {}),
            ((), {"enabled": RaisingRepr(RuntimeError("boom"))}),
            ((), {"enabled": RaisingRepr(KeyboardInterrupt())}),
            ((), {"enabled": RaisingRepr(SystemExit(7))}),
            ((), {"enabled": NonStringRepr()}),
        )
        prefix = (
            "kineto_available(): incompatible function arguments. The following "
            "argument types are supported:\n    1. () -> bool\n\nInvoked with: "
        )
        for args, kwargs in cases:
            with self.subTest(args_type=type(args[0]).__name__ if args else None):
                with self.assertRaises(TypeError) as raised:
                    torch.autograd.kineto_available(*args, **kwargs)
                location = "" if args else "kwargs: enabled="
                message = f"{prefix}{location}<repr raised Error>"
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_profiler_surface_remains_unsupported(self):
        public = torch.autograd
        native = torch._C._autograd
        for name in (
            "ProfilerActivity",
            "ProfilerEvent",
            "ProfilerConfig",
            "ProfilerState",
            "profiler",
        ):
            self.assertFalse(hasattr(public, name))
        for name in (
            "ProfilerActivity",
            "ProfilerEvent",
            "_enable_profiler",
            "_kineto_step",
            "_supported_activities",
        ):
            self.assertFalse(hasattr(native, name))
        self.assertFalse(hasattr(torch, "profiler"))

    def test_importing_and_calling_does_not_probe_or_import_pytorch(self):
        script = r'''
import os
import sys

class RejectExternalRuntimeImport:
    blocked = {"cupy", "kineto", "libkineto", "nvidia", "pynvml", "torch"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    CUDA_VISIBLE_DEVICES="0",
    KINETO_LIBRARY="/not/a/library",
    KINETO_USE_DAEMON="1",
    LIBKINETO_PATH="/not/a/library",
    USE_KINETO="1",
)

import torch_rs as torch
from torch_rs.autograd import kineto_available
from torch_rs._C._autograd import kineto_available as native_kineto_available

modules_before_calls = set(sys.modules)
assert kineto_available is torch.autograd.kineto_available
assert kineto_available is native_kineto_available
assert kineto_available is torch._C._autograd.kineto_available
assert kineto_available() is False
assert native_kineto_available() is False
assert set(sys.modules) == modules_before_calls
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

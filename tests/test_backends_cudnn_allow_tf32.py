import copy
import importlib
import inspect
import json
import pickle
import subprocess
import sys
import threading
import types
import unittest

import numpy as np

import torch_rs as torch


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("cudnn.allow_tf32 must not request truthiness")


def fresh_cudnn_module():
    module_name = "torch_rs.backends.cudnn"
    sys.modules.pop(module_name, None)
    if hasattr(torch.backends, "cudnn"):
        del torch.backends.cudnn
    module = importlib.import_module(module_name)
    torch.backends.cudnn = module
    return module


class CudnnAllowTF32Tests(unittest.TestCase):
    def setUp(self):
        self.cudnn = fresh_cudnn_module()
        self.original_allow_tf32 = self.cudnn.allow_tf32
        self.original_benchmark = self.cudnn.benchmark
        self.original_deterministic = self.cudnn.deterministic
        self.original_enabled = self.cudnn.enabled
        self.cudnn.allow_tf32 = True
        self.cudnn.benchmark = False
        self.cudnn.deterministic = False
        self.cudnn.enabled = True

    def tearDown(self):
        cudnn = fresh_cudnn_module()
        cudnn.allow_tf32 = self.original_allow_tf32
        cudnn.benchmark = self.original_benchmark
        cudnn.deterministic = self.original_deterministic
        cudnn.enabled = self.original_enabled

    def test_fresh_process_defaults_to_exact_true_without_cudnn_support(self):
        script = r'''
import json

import torch_rs as torch

cudnn = torch.backends.cudnn
initial = cudnn.allow_tf32
cudnn.allow_tf32 = False
disabled = cudnn.allow_tf32
cudnn.allow_tf32 = True
print(json.dumps({
    "initial": initial,
    "initial_type": type(initial).__name__,
    "disabled": disabled,
    "restored": cudnn.allow_tf32,
    "cudnn_enabled": cudnn.enabled,
    "benchmark": cudnn.benchmark,
    "deterministic": cudnn.deterministic,
    "available": cudnn.is_available(),
    "version": cudnn.version(),
    "cuda": hasattr(torch, "cuda"),
    "execution": hasattr(torch, "cudnn_convolution"),
}))
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
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "initial": True,
                "initial_type": "bool",
                "disabled": False,
                "restored": True,
                "cudnn_enabled": True,
                "benchmark": False,
                "deterministic": False,
                "available": False,
                "version": None,
                "cuda": False,
                "execution": False,
            },
        )

    def test_repeated_exact_bool_assignments_are_independent_preferences(self):
        cudnn = self.cudnn

        self.assertIs(cudnn.allow_tf32, True)
        self.assertIs(type(cudnn.allow_tf32), bool)
        cudnn.enabled = False
        cudnn.benchmark = True
        cudnn.deterministic = True
        for allow_tf32 in (True, False, False, True, True, False):
            with self.subTest(allow_tf32=allow_tf32):
                cudnn.allow_tf32 = allow_tf32
                self.assertIs(cudnn.allow_tf32, allow_tf32)
                self.assertIs(
                    torch._C._get_cudnn_allow_tf32(),
                    allow_tf32,
                )
                self.assertIs(cudnn.enabled, False)
                self.assertIs(cudnn.benchmark, True)
                self.assertIs(cudnn.deterministic, True)
                self.assertIs(cudnn.is_available(), False)
                self.assertIs(cudnn.version(), None)

    def test_non_bool_values_are_rejected_without_coercion_or_state_change(self):
        invalid_values = (
            (None, "NoneType"),
            (0, "int"),
            (1, "int"),
            (0.0, "float"),
            (np.bool_(True), "numpy.bool"),
            ("", "str"),
            ([], "list"),
            (object(), "object"),
            (_RejectTruthiness(), "_RejectTruthiness"),
            (torch.tensor(True), "Tensor"),
            (torch.float32, "torch.dtype"),
            (torch.device("cpu"), "torch.device"),
            (torch.strided, "torch.layout"),
            (torch.Size([1]), "torch.Size"),
            (torch.finfo(torch.float32), "torch.finfo"),
        )
        for state in (False, True):
            self.cudnn.allow_tf32 = state
            self.cudnn.enabled = not state
            self.cudnn.benchmark = state
            self.cudnn.deterministic = not state
            for value, type_name in invalid_values:
                with self.subTest(state=state, value_type=type_name):
                    message = (
                        "set_allow_tf32_cublas expects a bool, but got "
                        f"{type_name}"
                    )
                    for setter in (
                        lambda value=value: setattr(
                            self.cudnn,
                            "allow_tf32",
                            value,
                        ),
                        lambda value=value: torch._C._set_cudnn_allow_tf32(
                            value
                        ),
                    ):
                        with self.assertRaises(RuntimeError) as raised:
                            setter()
                        self.assertEqual(str(raised.exception), message)
                        self.assertEqual(raised.exception.args, (message,))
                        self.assertIs(self.cudnn.allow_tf32, state)
                        self.assertIs(
                            torch._C._get_cudnn_allow_tf32(),
                            state,
                        )
                        self.assertIs(self.cudnn.enabled, not state)
                        self.assertIs(self.cudnn.benchmark, state)
                        self.assertIs(self.cudnn.deterministic, not state)
                        self.assertIs(self.cudnn.is_available(), False)
                        self.assertIs(self.cudnn.version(), None)

    def test_state_is_process_global_across_threads_and_module_aliases(self):
        cudnn = self.cudnn
        imported = importlib.import_module("torch_rs.backends.cudnn")
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []

        self.assertIs(imported, cudnn)
        cudnn.allow_tf32 = False

        def worker():
            try:
                observations.append(cudnn.allow_tf32)
                imported.allow_tf32 = True
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(imported.allow_tf32)
                cudnn.allow_tf32 = True
            except BaseException as error:
                errors.append(error)
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_changed.wait(timeout=10))
        self.assertEqual(errors, [])
        self.assertIs(cudnn.allow_tf32, True)
        cudnn.allow_tf32 = False
        main_changed.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [False, False])
        self.assertIs(cudnn.allow_tf32, True)
        self.assertIs(torch._C._get_cudnn_allow_tf32(), True)

    def test_reload_and_fresh_import_preserve_shared_state(self):
        backends = torch.backends
        cudnn = self.cudnn
        namespace = cudnn.__dict__
        cudnn.allow_tf32 = True

        reloaded = importlib.reload(cudnn)

        self.assertIsNot(reloaded, cudnn)
        self.assertIs(cudnn.__dict__, namespace)
        self.assertIs(backends.cudnn, cudnn)
        self.assertIs(sys.modules[cudnn.__name__], reloaded)
        self.assertIs(reloaded.m, cudnn)
        self.assertIs(cudnn.allow_tf32, True)
        self.assertIs(reloaded.allow_tf32, True)

        reloaded.allow_tf32 = False
        self.assertIs(cudnn.allow_tf32, False)
        cudnn.allow_tf32 = True
        self.assertIs(reloaded.allow_tf32, True)

        fresh = fresh_cudnn_module()
        self.assertIs(torch.backends.cudnn, fresh)
        self.assertIs(fresh.allow_tf32, True)
        fresh.allow_tf32 = False
        self.assertIs(cudnn.allow_tf32, False)
        self.assertIs(reloaded.allow_tf32, False)

    def test_proxy_metadata_imports_deletion_copying_and_pickling(self):
        cudnn = self.cudnn
        cudnn.allow_tf32 = False
        descriptor = vars(type(cudnn))["allow_tf32"]

        self.assertIs(torch.backends.cudnn, cudnn)
        self.assertIs(sys.modules["torch_rs.backends.cudnn"], cudnn)
        self.assertIsInstance(cudnn, types.ModuleType)
        self.assertEqual(type(cudnn).__name__, "CudnnModule")
        self.assertEqual(type(cudnn).__module__, "torch_rs.backends.cudnn")
        self.assertIsNone(cudnn.__doc__)
        self.assertFalse(hasattr(cudnn, "__all__"))
        self.assertEqual(
            {name for name in vars(cudnn) if not name.startswith("_")},
            {"m"},
        )
        self.assertIs(type(cudnn.m), types.ModuleType)
        self.assertIs(cudnn.m.__annotations__["allow_tf32"], bool)
        self.assertNotIn("allow_tf32", vars(cudnn))
        self.assertNotIn("allow_tf32", vars(cudnn.m))
        self.assertNotIn("allow_tf32", dir(cudnn))
        self.assertIsNone(descriptor.__doc__)
        self.assertEqual(set(vars(descriptor)), {"getter", "setter"})
        self.assertIs(descriptor.getter, torch._C._get_cudnn_allow_tf32)
        self.assertIs(descriptor.setter, torch._C._set_cudnn_allow_tf32)
        self.assertIs(descriptor.__get__(cudnn, type(cudnn)), False)

        direct_import = {}
        wildcard_import = {}
        exec(
            "from torch_rs.backends.cudnn import allow_tf32",
            direct_import,
        )
        exec("from torch_rs.backends.cudnn import *", wildcard_import)
        self.assertIs(direct_import["allow_tf32"], False)
        self.assertNotIn("allow_tf32", wildcard_import)
        cudnn.allow_tf32 = True
        self.assertIs(direct_import["allow_tf32"], False)
        exec(
            "from torch_rs.backends.cudnn import allow_tf32",
            direct_import,
        )
        self.assertIs(direct_import["allow_tf32"], True)

        for state in (False, True):
            cudnn.allow_tf32 = state
            self.assertIs(copy.copy(cudnn.allow_tf32), state)
            self.assertIs(copy.deepcopy(cudnn.allow_tf32), state)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(kind="value", state=state, protocol=protocol):
                    self.assertIs(
                        pickle.loads(
                            pickle.dumps(cudnn.allow_tf32, protocol=protocol)
                        ),
                        state,
                    )

        for copier in (copy.copy, copy.deepcopy):
            copied = copier(descriptor)
            self.assertIsNot(copied, descriptor)
            self.assertIs(type(copied), type(descriptor))
            self.assertIs(copied.getter, descriptor.getter)
            self.assertIs(copied.setter, descriptor.setter)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="descriptor", protocol=protocol):
                restored = pickle.loads(pickle.dumps(descriptor, protocol))
                self.assertIsNot(restored, descriptor)
                self.assertIs(type(restored), type(descriptor))
                self.assertIs(restored.getter, descriptor.getter)
                self.assertIs(restored.setter, descriptor.setter)

        for copier in (copy.copy, copy.deepcopy):
            with self.assertRaisesRegex(
                TypeError,
                "^cannot pickle 'CudnnModule' object$",
            ):
                copier(cudnn)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="module", protocol=protocol):
                with self.assertRaises(TypeError):
                    pickle.dumps(cudnn, protocol)

        cudnn.allow_tf32 = True
        with self.assertRaises(AttributeError) as raised:
            del cudnn.allow_tf32
        self.assertEqual(str(raised.exception), "__delete__")
        self.assertEqual(raised.exception.args, ("__delete__",))
        self.assertIs(cudnn.allow_tf32, True)

    def test_private_accessor_metadata_exports_copying_and_pickling(self):
        getter = torch._C._get_cudnn_allow_tf32
        setter = torch._C._set_cudnn_allow_tf32
        self.cudnn.allow_tf32 = False

        for function, name in (
            (getter, "_get_cudnn_allow_tf32"),
            (setter, "_set_cudnn_allow_tf32"),
        ):
            with self.subTest(name=name):
                self.assertIs(type(function), types.BuiltinFunctionType)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(function.__module__, torch.tensor.__module__)
                self.assertIsNone(function.__doc__)
                self.assertFalse(hasattr(function, "__annotations__"))
                self.assertEqual(repr(function), f"<built-in function {name}>")
                self.assertIs(function.__self__, torch._C)
                self.assertEqual(function.__reduce__(), name)
                if sys.version_info >= (3, 13):
                    signature = (
                        "($self, /)"
                        if function is getter
                        else "($self, object, /)"
                    )
                    self.assertEqual(function.__text_signature__, signature)
                    self.assertEqual(
                        str(inspect.signature(function)),
                        "()" if function is getter else "(object, /)",
                    )
                else:
                    self.assertIsNone(function.__text_signature__)
                    with self.assertRaises(ValueError):
                        inspect.signature(function)
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(name=name, protocol=protocol):
                        payload = pickle.dumps(function, protocol=protocol)
                        self.assertIn(b"torch_rs.torch_rs", payload)
                        self.assertIs(pickle.loads(payload), function)

        self.assertIs(getter(), False)
        self.assertIs(setter(True), None)
        self.assertIs(getter(), True)

        native_import = {}
        native_wildcard = {}
        exec(
            "from torch_rs._C import "
            "_get_cudnn_allow_tf32, _set_cudnn_allow_tf32",
            native_import,
        )
        exec("from torch_rs._C import *", native_wildcard)
        self.assertIs(native_import["_get_cudnn_allow_tf32"], getter)
        self.assertIs(native_import["_set_cudnn_allow_tf32"], setter)
        self.assertNotIn("_get_cudnn_allow_tf32", native_wildcard)
        self.assertNotIn("_set_cudnn_allow_tf32", native_wildcard)
        self.assertFalse(hasattr(torch, "_get_cudnn_allow_tf32"))
        self.assertFalse(hasattr(torch, "_set_cudnn_allow_tf32"))
        self.assertNotIn("_get_cudnn_allow_tf32", torch.__all__)
        self.assertNotIn("_set_cudnn_allow_tf32", torch.__all__)
        self.assertNotIn("_get_cudnn_allow_tf32", torch._C.__all__)
        self.assertNotIn("_set_cudnn_allow_tf32", torch._C.__all__)

    def test_private_accessor_binding_errors_preserve_state(self):
        getter = torch._C._get_cudnn_allow_tf32
        setter = torch._C._set_cudnn_allow_tf32
        self.cudnn.allow_tf32 = True
        cases = (
            (
                lambda: getter(None),
                "torch_rs.torch_rs._get_cudnn_allow_tf32() "
                "takes no arguments (1 given)",
            ),
            (
                lambda: getter(value=None),
                "torch_rs.torch_rs._get_cudnn_allow_tf32() "
                "takes no keyword arguments",
            ),
            (
                lambda: setter(),
                "torch_rs.torch_rs._set_cudnn_allow_tf32() "
                "takes exactly one argument (0 given)",
            ),
            (
                lambda: setter(True, False),
                "torch_rs.torch_rs._set_cudnn_allow_tf32() "
                "takes exactly one argument (2 given)",
            ),
            (
                lambda: setter(object=False),
                "torch_rs.torch_rs._set_cudnn_allow_tf32() "
                "takes no keyword arguments",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(self.cudnn.allow_tf32, True)


if __name__ == "__main__":
    unittest.main()

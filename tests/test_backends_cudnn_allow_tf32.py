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


class CudnnAllowTf32Tests(unittest.TestCase):
    def setUp(self):
        self.cudnn = fresh_cudnn_module()
        self.original = (
            self.cudnn.allow_tf32,
            self.cudnn.enabled,
            self.cudnn.benchmark,
            self.cudnn.deterministic,
        )
        self.cudnn.allow_tf32 = True
        self.cudnn.enabled = True
        self.cudnn.benchmark = False
        self.cudnn.deterministic = False

    def tearDown(self):
        cudnn = fresh_cudnn_module()
        (
            cudnn.allow_tf32,
            cudnn.enabled,
            cudnn.benchmark,
            cudnn.deterministic,
        ) = self.original

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
    "enabled": cudnn.enabled,
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
                "enabled": True,
                "benchmark": False,
                "deterministic": False,
                "available": False,
                "version": None,
                "cuda": True,
                "execution": False,
            },
        )

    def test_repeated_exact_bool_assignments_are_independent_preferences(self):
        cudnn = self.cudnn
        cudnn.enabled = False
        cudnn.benchmark = True
        cudnn.deterministic = True

        self.assertIs(cudnn.allow_tf32, True)
        self.assertIs(type(cudnn.allow_tf32), bool)
        for allow_tf32 in (False, True, True, False, False, True):
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

        cudnn.allow_tf32 = False
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = False
        self.assertIs(cudnn.allow_tf32, False)

    def test_non_bool_values_are_rejected_without_state_change(self):
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

    def test_state_is_process_global_across_threads_and_module_aliases(self):
        cudnn = self.cudnn
        imported = importlib.import_module("torch_rs.backends.cudnn")
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []

        self.assertIs(imported, cudnn)

        def worker():
            try:
                observations.append(cudnn.allow_tf32)
                imported.allow_tf32 = False
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(imported.allow_tf32)
                cudnn.allow_tf32 = False
            except BaseException as error:
                errors.append(error)
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_changed.wait(timeout=10))
        self.assertEqual(errors, [])
        self.assertIs(cudnn.allow_tf32, False)
        cudnn.allow_tf32 = True
        main_changed.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [True, True])
        self.assertIs(cudnn.allow_tf32, False)
        self.assertIs(torch._C._get_cudnn_allow_tf32(), False)
        self.assertIs(cudnn.enabled, True)
        self.assertIs(cudnn.benchmark, False)
        self.assertIs(cudnn.deterministic, False)

    def test_reload_and_fresh_import_preserve_shared_state(self):
        backends = torch.backends
        cudnn = self.cudnn
        namespace = cudnn.__dict__
        cudnn.allow_tf32 = False

        reloaded = importlib.reload(cudnn)

        self.assertIsNot(reloaded, cudnn)
        self.assertIs(cudnn.__dict__, namespace)
        self.assertIs(backends.cudnn, cudnn)
        self.assertIs(sys.modules[cudnn.__name__], reloaded)
        self.assertIs(reloaded.m, cudnn)
        self.assertIs(cudnn.allow_tf32, False)
        self.assertIs(reloaded.allow_tf32, False)

        reloaded.allow_tf32 = True
        self.assertIs(cudnn.allow_tf32, True)
        cudnn.allow_tf32 = False
        self.assertIs(reloaded.allow_tf32, False)

        fresh = fresh_cudnn_module()
        self.assertIs(torch.backends.cudnn, fresh)
        self.assertIs(fresh.allow_tf32, False)
        fresh.allow_tf32 = True
        self.assertIs(cudnn.allow_tf32, True)
        self.assertIs(reloaded.allow_tf32, True)

    def test_proxy_deletion_and_private_accessor_contract(self):
        cudnn = self.cudnn
        descriptor = vars(type(cudnn))["allow_tf32"]
        getter = torch._C._get_cudnn_allow_tf32
        setter = torch._C._set_cudnn_allow_tf32

        self.assertIsInstance(cudnn, types.ModuleType)
        self.assertIs(cudnn.m.__annotations__["allow_tf32"], bool)
        self.assertNotIn("allow_tf32", vars(cudnn))
        self.assertNotIn("allow_tf32", vars(cudnn.m))
        self.assertNotIn("allow_tf32", dir(cudnn))
        self.assertIs(descriptor.getter, getter)
        self.assertIs(descriptor.setter, setter)
        self.assertIs(descriptor.__get__(cudnn, type(cudnn)), True)

        imported = {}
        wildcard = {}
        exec("from torch_rs.backends.cudnn import allow_tf32", imported)
        exec("from torch_rs.backends.cudnn import *", wildcard)
        self.assertIs(imported["allow_tf32"], True)
        self.assertNotIn("allow_tf32", wildcard)
        cudnn.allow_tf32 = False
        self.assertIs(imported["allow_tf32"], True)

        with self.assertRaises(AttributeError) as raised:
            del cudnn.allow_tf32
        self.assertEqual(str(raised.exception), "__delete__")
        self.assertEqual(raised.exception.args, ("__delete__",))
        self.assertIs(cudnn.allow_tf32, False)

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
                self.assertIs(function.__self__, torch._C)
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                self.assertIs(pickle.loads(pickle.dumps(function)), function)
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

        direct_import = {}
        native_wildcard = {}
        exec(
            "from torch_rs._C import "
            "_get_cudnn_allow_tf32, _set_cudnn_allow_tf32",
            direct_import,
        )
        exec("from torch_rs._C import *", native_wildcard)
        self.assertIs(direct_import["_get_cudnn_allow_tf32"], getter)
        self.assertIs(direct_import["_set_cudnn_allow_tf32"], setter)
        self.assertNotIn("_get_cudnn_allow_tf32", native_wildcard)
        self.assertNotIn("_set_cudnn_allow_tf32", native_wildcard)
        self.assertFalse(hasattr(torch, "_get_cudnn_allow_tf32"))
        self.assertFalse(hasattr(torch, "_set_cudnn_allow_tf32"))
        self.assertNotIn("_get_cudnn_allow_tf32", torch.__all__)
        self.assertNotIn("_set_cudnn_allow_tf32", torch.__all__)
        self.assertNotIn("_get_cudnn_allow_tf32", torch._C.__all__)
        self.assertNotIn("_set_cudnn_allow_tf32", torch._C.__all__)

        self.assertIs(setter(True), None)
        self.assertIs(getter(), True)

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

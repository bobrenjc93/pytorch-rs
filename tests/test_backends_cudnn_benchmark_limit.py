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


class _IntSubclass(int):
    pass


class _RejectIndex:
    def __index__(self):
        raise AssertionError("cudnn.benchmark_limit must reject index-only objects")


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("cudnn.benchmark_limit must not request truthiness")


def fresh_cudnn_module():
    module_name = "torch_rs.backends.cudnn"
    sys.modules.pop(module_name, None)
    if hasattr(torch.backends, "cudnn"):
        del torch.backends.cudnn
    module = importlib.import_module(module_name)
    torch.backends.cudnn = module
    return module


class CudnnBenchmarkLimitTests(unittest.TestCase):
    def setUp(self):
        self.cudnn = fresh_cudnn_module()
        self.original = self.states(self.cudnn)
        self.set_states(self.cudnn, (10, True, False, False, True))

    def tearDown(self):
        self.set_states(fresh_cudnn_module(), self.original)

    def states(self, module):
        return (
            module.benchmark_limit,
            module.enabled,
            module.benchmark,
            module.deterministic,
            module.allow_tf32,
        )

    def set_states(self, module, states):
        (
            module.benchmark_limit,
            module.enabled,
            module.benchmark,
            module.deterministic,
            module.allow_tf32,
        ) = states

    def test_fresh_process_defaults_to_exact_ten_without_cudnn_execution(self):
        script = r'''
import json

import torch_rs as torch

cudnn = torch.backends.cudnn
initial = cudnn.benchmark_limit
cudnn.benchmark_limit = 2**31
narrowed = cudnn.benchmark_limit
cudnn.benchmark_limit = 10
print(json.dumps({
    "initial": initial,
    "initial_type": type(initial).__name__,
    "narrowed": narrowed,
    "restored": cudnn.benchmark_limit,
    "enabled": cudnn.enabled,
    "benchmark": cudnn.benchmark,
    "deterministic": cudnn.deterministic,
    "allow_tf32": cudnn.allow_tf32,
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
                "initial": 10,
                "initial_type": "int",
                "narrowed": -(2**31),
                "restored": 10,
                "enabled": True,
                "benchmark": False,
                "deterministic": False,
                "allow_tf32": True,
                "available": False,
                "version": None,
                "cuda": True,
                "execution": False,
            },
        )

    def test_integer_inputs_use_pytorch_int64_to_int32_narrowing(self):
        accepted = (
            (-(2**63), 0),
            (-(2**32) - 1, -1),
            (-(2**32), 0),
            (-(2**31) - 1, 2**31 - 1),
            (-(2**31), -(2**31)),
            (-1, -1),
            (0, 0),
            (10, 10),
            (2**31 - 1, 2**31 - 1),
            (2**31, -(2**31)),
            (2**32 - 1, -1),
            (2**32, 0),
            (2**63 - 1, -1),
            (_IntSubclass(12), 12),
            (np.int8(-3), -3),
            (np.int32(4), 4),
            (np.int64(5), 5),
            (np.uint32(2**32 - 1), -1),
            (np.uint64(2**63 - 1), -1),
        )
        independent = (False, True, True, False)
        for value, expected in accepted:
            for kind, setter in (
                (
                    "proxy",
                    lambda value=value: setattr(
                        self.cudnn,
                        "benchmark_limit",
                        value,
                    ),
                ),
                (
                    "native",
                    lambda value=value: torch._C._cuda_set_cudnn_benchmark_limit(
                        value
                    ),
                ),
            ):
                with self.subTest(kind=kind, value=repr(value)):
                    self.set_states(self.cudnn, (123, *independent))
                    self.assertIs(setter(), None)
                    self.assertEqual(self.cudnn.benchmark_limit, expected)
                    self.assertIs(type(self.cudnn.benchmark_limit), int)
                    self.assertEqual(
                        torch._C._cuda_get_cudnn_benchmark_limit(),
                        expected,
                    )
                    self.assertEqual(self.states(self.cudnn)[1:], independent)
                    self.assertIs(self.cudnn.is_available(), False)
                    self.assertIs(self.cudnn.version(), None)

    def test_bool_non_integer_and_overflow_inputs_preserve_all_state(self):
        invalid_values = (
            (None, "NoneType"),
            (True, "bool"),
            (False, "bool"),
            (1.0, "float"),
            (np.bool_(True), "numpy.bool"),
            (np.float64(1.0), "numpy.float64"),
            ("1", "str"),
            ([], "list"),
            (object(), "object"),
            (_RejectIndex(), "_RejectIndex"),
            (_RejectTruthiness(), "_RejectTruthiness"),
            (torch.tensor(True), "Tensor"),
            (torch.float32, "torch.dtype"),
            (torch.device("cpu"), "torch.device"),
            (torch.strided, "torch.layout"),
            (torch.Size([1]), "torch.Size"),
            (torch.finfo(torch.float32), "torch.finfo"),
        )
        setters = (
            lambda value: setattr(self.cudnn, "benchmark_limit", value),
            torch._C._cuda_set_cudnn_benchmark_limit,
        )
        for value, type_name in invalid_values:
            for setter in setters:
                with self.subTest(value_type=type_name, setter=setter):
                    expected_state = (123, False, True, True, False)
                    self.set_states(self.cudnn, expected_state)
                    message = (
                        "set_benchmark_limit_cudnn expects an int, but got "
                        f"{type_name}"
                    )
                    with self.assertRaises(RuntimeError) as raised:
                        setter(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertEqual(self.states(self.cudnn), expected_state)

        for value in (-(2**100), -(2**63) - 1, 2**63, 2**100, np.uint64(2**63)):
            for setter in setters:
                with self.subTest(value=repr(value), setter=setter):
                    expected_state = (123, False, True, True, False)
                    self.set_states(self.cudnn, expected_state)
                    message = "Overflow when unpacking long long"
                    with self.assertRaises(ValueError) as raised:
                        setter(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertEqual(self.states(self.cudnn), expected_state)

    def test_state_is_process_global_across_threads_and_module_aliases(self):
        cudnn = self.cudnn
        imported = importlib.import_module("torch_rs.backends.cudnn")
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []
        self.set_states(cudnn, (10, False, True, True, False))

        self.assertIs(imported, cudnn)

        def worker():
            try:
                observations.append(self.states(cudnn))
                imported.benchmark_limit = 2**31
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(self.states(imported))
                cudnn.benchmark_limit = 17
            except BaseException as error:
                errors.append(error)
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_changed.wait(timeout=10))
        self.assertEqual(errors, [])
        self.assertEqual(cudnn.benchmark_limit, -(2**31))
        cudnn.benchmark_limit = -9
        main_changed.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(
            observations,
            [
                (10, False, True, True, False),
                (-9, False, True, True, False),
            ],
        )
        self.assertEqual(self.states(cudnn), (17, False, True, True, False))

    def test_reload_and_fresh_import_preserve_shared_state(self):
        backends = torch.backends
        cudnn = self.cudnn
        namespace = cudnn.__dict__
        self.set_states(cudnn, (2**31 - 1, False, True, True, False))

        reloaded = importlib.reload(cudnn)

        self.assertIsNot(reloaded, cudnn)
        self.assertIs(cudnn.__dict__, namespace)
        self.assertIs(backends.cudnn, cudnn)
        self.assertIs(sys.modules[cudnn.__name__], reloaded)
        self.assertIs(reloaded.m, cudnn)
        self.assertEqual(
            self.states(cudnn),
            (2**31 - 1, False, True, True, False),
        )
        self.assertEqual(self.states(reloaded), self.states(cudnn))

        reloaded.benchmark_limit = 2**31
        self.assertEqual(cudnn.benchmark_limit, -(2**31))
        cudnn.benchmark_limit = -7
        self.assertEqual(reloaded.benchmark_limit, -7)

        fresh = fresh_cudnn_module()
        self.assertIs(torch.backends.cudnn, fresh)
        self.assertEqual(fresh.benchmark_limit, -7)
        fresh.benchmark_limit = 31
        self.assertEqual(cudnn.benchmark_limit, 31)
        self.assertEqual(reloaded.benchmark_limit, 31)

    def test_proxy_deletion_imports_and_private_accessors(self):
        cudnn = self.cudnn
        descriptor = vars(type(cudnn))["benchmark_limit"]
        getter = torch._C._cuda_get_cudnn_benchmark_limit
        setter = torch._C._cuda_set_cudnn_benchmark_limit

        self.assertIsInstance(cudnn, types.ModuleType)
        self.assertIs(cudnn.m.__annotations__["benchmark_limit"], int)
        self.assertNotIn("benchmark_limit", vars(cudnn))
        self.assertNotIn("benchmark_limit", vars(cudnn.m))
        self.assertNotIn("benchmark_limit", dir(cudnn))
        self.assertIs(descriptor.getter, getter)
        self.assertIs(descriptor.setter, setter)
        self.assertEqual(descriptor.__get__(cudnn, type(cudnn)), 10)

        imported = {}
        wildcard = {}
        exec("from torch_rs.backends.cudnn import benchmark_limit", imported)
        exec("from torch_rs.backends.cudnn import *", wildcard)
        self.assertEqual(imported["benchmark_limit"], 10)
        self.assertNotIn("benchmark_limit", wildcard)
        cudnn.benchmark_limit = 11
        self.assertEqual(imported["benchmark_limit"], 10)

        with self.assertRaises(AttributeError) as raised:
            del cudnn.benchmark_limit
        self.assertEqual(str(raised.exception), "__delete__")
        self.assertEqual(raised.exception.args, ("__delete__",))
        self.assertEqual(cudnn.benchmark_limit, 11)

        for function, name in (
            (getter, "_cuda_get_cudnn_benchmark_limit"),
            (setter, "_cuda_set_cudnn_benchmark_limit"),
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
            "from torch_rs._C import _cuda_get_cudnn_benchmark_limit, "
            "_cuda_set_cudnn_benchmark_limit",
            direct_import,
        )
        exec("from torch_rs._C import *", native_wildcard)
        self.assertIs(direct_import["_cuda_get_cudnn_benchmark_limit"], getter)
        self.assertIs(direct_import["_cuda_set_cudnn_benchmark_limit"], setter)
        self.assertNotIn("_cuda_get_cudnn_benchmark_limit", native_wildcard)
        self.assertNotIn("_cuda_set_cudnn_benchmark_limit", native_wildcard)
        self.assertFalse(hasattr(torch, "_cuda_get_cudnn_benchmark_limit"))
        self.assertFalse(hasattr(torch, "_cuda_set_cudnn_benchmark_limit"))
        self.assertNotIn("_cuda_get_cudnn_benchmark_limit", torch.__all__)
        self.assertNotIn("_cuda_set_cudnn_benchmark_limit", torch.__all__)
        self.assertNotIn("_cuda_get_cudnn_benchmark_limit", torch._C.__all__)
        self.assertNotIn("_cuda_set_cudnn_benchmark_limit", torch._C.__all__)

        self.assertIs(setter(19), None)
        self.assertEqual(getter(), 19)

    def test_private_accessor_binding_errors_preserve_state(self):
        getter = torch._C._cuda_get_cudnn_benchmark_limit
        setter = torch._C._cuda_set_cudnn_benchmark_limit
        self.cudnn.benchmark_limit = 123
        cases = (
            (
                lambda: getter(None),
                "torch_rs.torch_rs._cuda_get_cudnn_benchmark_limit() "
                "takes no arguments (1 given)",
            ),
            (
                lambda: getter(value=None),
                "torch_rs.torch_rs._cuda_get_cudnn_benchmark_limit() "
                "takes no keyword arguments",
            ),
            (
                lambda: setter(),
                "torch_rs.torch_rs._cuda_set_cudnn_benchmark_limit() "
                "takes exactly one argument (0 given)",
            ),
            (
                lambda: setter(1, 2),
                "torch_rs.torch_rs._cuda_set_cudnn_benchmark_limit() "
                "takes exactly one argument (2 given)",
            ),
            (
                lambda: setter(object=1),
                "torch_rs.torch_rs._cuda_set_cudnn_benchmark_limit() "
                "takes no keyword arguments",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(self.cudnn.benchmark_limit, 123)


if __name__ == "__main__":
    unittest.main()

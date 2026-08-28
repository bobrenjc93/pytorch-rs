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
        raise AssertionError("cudnn.benchmark_limit must reject __index__ providers")


class _RejectNumpyIndex(np.int64):
    def __index__(self):
        raise RuntimeError("numpy index failed")


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
        self.original = self.states()
        self.set_states((10, True, False, False, True))

    def tearDown(self):
        fresh_cudnn_module()
        self.set_states(self.original)

    def states(self):
        return (
            self.cudnn.benchmark_limit,
            self.cudnn.enabled,
            self.cudnn.benchmark,
            self.cudnn.deterministic,
            self.cudnn.allow_tf32,
        )

    def set_states(self, states):
        (
            self.cudnn.benchmark_limit,
            self.cudnn.enabled,
            self.cudnn.benchmark,
            self.cudnn.deterministic,
            self.cudnn.allow_tf32,
        ) = states

    def test_fresh_process_defaults_to_exact_ten_without_cudnn_support(self):
        script = r'''
import json

import torch_rs as torch

cudnn = torch.backends.cudnn
initial = cudnn.benchmark_limit
cudnn.benchmark_limit = 2**32 + 7
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
                "narrowed": 7,
                "restored": 10,
                "enabled": True,
                "benchmark": False,
                "deterministic": False,
                "allow_tf32": True,
                "available": False,
                "version": None,
                "cuda": False,
                "execution": False,
            },
        )

    def test_signed_64_bit_inputs_narrow_to_signed_32_bits_independently(self):
        cases = (
            (0, 0),
            (-1, -1),
            (2**31 - 1, 2**31 - 1),
            (-(2**31), -(2**31)),
            (2**31, -(2**31)),
            (-(2**31) - 1, 2**31 - 1),
            (2**32 + 17, 17),
            (-(2**32) + 17, 17),
            (2**63 - 1, -1),
            (-(2**63), 0),
            (_IntSubclass(23), 23),
            (np.int8(-7), -7),
            (np.int64(29), 29),
            (np.uint64(2**63 - 1), -1),
        )
        self.cudnn.enabled = False
        self.cudnn.benchmark = True
        self.cudnn.deterministic = True
        self.cudnn.allow_tf32 = False
        for value, expected in cases:
            for kind, setter in (
                (
                    "property",
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
                with self.subTest(kind=kind, value=value):
                    self.assertIs(setter(), None)
                    self.assertEqual(self.cudnn.benchmark_limit, expected)
                    self.assertIs(type(self.cudnn.benchmark_limit), int)
                    self.assertEqual(
                        torch._C._cuda_get_cudnn_benchmark_limit(),
                        expected,
                    )
                    self.assertEqual(
                        self.states()[1:],
                        (False, True, True, False),
                    )

    def test_bool_and_non_integer_values_are_rejected_without_state_change(self):
        invalid_values = (
            (True, "bool"),
            (False, "bool"),
            (np.bool_(True), "numpy.bool"),
            (None, "NoneType"),
            (1.0, "float"),
            ("1", "str"),
            ([], "list"),
            (object(), "object"),
            (_RejectIndex(), "_RejectIndex"),
            (torch.tensor(1), "Tensor"),
            (torch.float32, "torch.dtype"),
            (torch.device("cpu"), "torch.device"),
            (torch.strided, "torch.layout"),
            (torch.Size([1]), "torch.Size"),
            (torch.finfo(torch.float32), "torch.finfo"),
        )
        for value, type_name in invalid_values:
            for kind, setter in (
                (
                    "property",
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
                with self.subTest(kind=kind, value_type=type_name):
                    self.set_states((123, False, True, True, False))
                    message = (
                        "set_benchmark_limit_cudnn expects an int, but got "
                        f"{type_name}"
                    )
                    with self.assertRaises(RuntimeError) as raised:
                        setter()
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertEqual(
                        self.states(),
                        (123, False, True, True, False),
                    )

    def test_overflow_and_conversion_errors_preserve_state(self):
        for value in (
            2**63,
            -(2**63) - 1,
            _IntSubclass(2**63),
            np.uint64(2**63),
            np.uint64(2**64 - 1),
        ):
            for kind, setter in (
                (
                    "property",
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
                with self.subTest(kind=kind, value=value):
                    self.set_states((123, False, True, True, False))
                    with self.assertRaises(ValueError) as raised:
                        setter()
                    self.assertEqual(
                        str(raised.exception),
                        "Overflow when unpacking long long",
                    )
                    self.assertEqual(
                        raised.exception.args,
                        ("Overflow when unpacking long long",),
                    )
                    self.assertEqual(
                        self.states(),
                        (123, False, True, True, False),
                    )

        value = _RejectNumpyIndex(31)
        for setter in (
            lambda: setattr(self.cudnn, "benchmark_limit", value),
            lambda: torch._C._cuda_set_cudnn_benchmark_limit(value),
        ):
            self.set_states((123, False, True, True, False))
            with self.assertRaisesRegex(RuntimeError, "^numpy index failed$"):
                setter()
            self.assertEqual(self.states(), (123, False, True, True, False))

    def test_state_is_process_global_across_threads_and_module_aliases(self):
        cudnn = self.cudnn
        imported = importlib.import_module("torch_rs.backends.cudnn")
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []
        self.set_states((10, False, True, True, False))

        self.assertIs(imported, cudnn)

        def worker():
            try:
                observations.append(self.states())
                imported.benchmark_limit = 2**32 + 11
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(self.states())
                cudnn.benchmark_limit = -13
            except BaseException as error:
                errors.append(error)
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_changed.wait(timeout=10))
        self.assertEqual(errors, [])
        self.assertEqual(self.states(), (11, False, True, True, False))
        cudnn.benchmark_limit = 12
        main_changed.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(
            observations,
            [
                (10, False, True, True, False),
                (12, False, True, True, False),
            ],
        )
        self.assertEqual(self.states(), (-13, False, True, True, False))

    def test_reload_and_fresh_import_preserve_shared_state(self):
        backends = torch.backends
        cudnn = self.cudnn
        namespace = cudnn.__dict__
        self.set_states((19, False, True, True, False))

        reloaded = importlib.reload(cudnn)

        self.assertIsNot(reloaded, cudnn)
        self.assertIs(cudnn.__dict__, namespace)
        self.assertIs(backends.cudnn, cudnn)
        self.assertIs(sys.modules[cudnn.__name__], reloaded)
        self.assertIs(reloaded.m, cudnn)
        self.assertEqual(cudnn.benchmark_limit, 19)
        self.assertEqual(reloaded.benchmark_limit, 19)

        reloaded.benchmark_limit = 23
        self.assertEqual(cudnn.benchmark_limit, 23)
        cudnn.benchmark_limit = -29
        self.assertEqual(reloaded.benchmark_limit, -29)

        fresh = fresh_cudnn_module()
        self.assertIs(torch.backends.cudnn, fresh)
        self.assertEqual(fresh.benchmark_limit, -29)
        fresh.benchmark_limit = 31
        self.assertEqual(cudnn.benchmark_limit, 31)
        self.assertEqual(reloaded.benchmark_limit, 31)

    def test_proxy_deletion_and_private_accessor_contract(self):
        cudnn = self.cudnn
        descriptor = vars(type(cudnn))["benchmark_limit"]
        getter = torch._C._cuda_get_cudnn_benchmark_limit
        setter = torch._C._cuda_set_cudnn_benchmark_limit

        self.assertIsInstance(cudnn, types.ModuleType)
        self.assertIs(cudnn.m.__annotations__["benchmark_limit"], int)
        self.assertNotIn("benchmark_limit", vars(cudnn))
        self.assertNotIn("benchmark_limit", vars(cudnn.m))
        self.assertNotIn("benchmark_limit", dir(cudnn))
        self.assertEqual(set(vars(descriptor)), {"getter", "setter"})
        self.assertIsNone(descriptor.__doc__)
        self.assertIs(descriptor.getter, getter)
        self.assertIs(descriptor.setter, setter)
        self.assertEqual(descriptor.__get__(cudnn, type(cudnn)), 10)

        imported = {}
        wildcard = {}
        exec("from torch_rs.backends.cudnn import benchmark_limit", imported)
        exec("from torch_rs.backends.cudnn import *", wildcard)
        self.assertEqual(imported["benchmark_limit"], 10)
        self.assertNotIn("benchmark_limit", wildcard)
        cudnn.benchmark_limit = 17
        self.assertEqual(imported["benchmark_limit"], 10)

        with self.assertRaises(AttributeError) as raised:
            del cudnn.benchmark_limit
        self.assertEqual(str(raised.exception), "__delete__")
        self.assertEqual(raised.exception.args, ("__delete__",))
        self.assertEqual(cudnn.benchmark_limit, 17)

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

        self.assertIs(setter(2**32 + 37), None)
        self.assertEqual(getter(), 37)

    def test_private_accessor_binding_errors_preserve_state(self):
        getter = torch._C._cuda_get_cudnn_benchmark_limit
        setter = torch._C._cuda_set_cudnn_benchmark_limit
        self.cudnn.benchmark_limit = 41
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
                self.assertEqual(self.cudnn.benchmark_limit, 41)


if __name__ == "__main__":
    unittest.main()

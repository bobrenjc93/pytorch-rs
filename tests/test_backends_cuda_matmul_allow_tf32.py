import copy
import importlib
import inspect
import json
import pickle
import re
import subprocess
import sys
import threading
import types
import unittest

import numpy as np

import torch_rs as torch


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("cuda.matmul.allow_tf32 must not request truthiness")


def fresh_cuda_module():
    module_name = "torch_rs.backends.cuda"
    sys.modules.pop(module_name, None)
    if hasattr(torch.backends, "cuda"):
        del torch.backends.cuda
    module = importlib.import_module(module_name)
    torch.backends.cuda = module
    return module


class CudaMatmulAllowTf32Tests(unittest.TestCase):
    def setUp(self):
        self.cuda = fresh_cuda_module()
        self.original = self.states(self.cuda)
        self.set_states(self.cuda, (False, True, True, False, True))

    def tearDown(self):
        cuda = fresh_cuda_module()
        self.set_states(cuda, self.original)

    def states(self, cuda):
        return (
            cuda.matmul.allow_tf32,
            cuda.mem_efficient_sdp_enabled(),
            cuda.math_sdp_enabled(),
            cuda.fp16_bf16_reduction_math_sdp_allowed(),
            torch.backends.cudnn.allow_tf32,
        )

    def set_states(self, cuda, states):
        (
            allow_tf32,
            mem_efficient,
            math,
            reduced_precision_math,
            cudnn_allow_tf32,
        ) = states
        cuda.matmul.allow_tf32 = allow_tf32
        cuda.enable_mem_efficient_sdp(mem_efficient)
        cuda.enable_math_sdp(math)
        cuda.allow_fp16_bf16_reduction_math_sdp(reduced_precision_math)
        torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32

    def test_fresh_process_defaults_to_exact_false_without_cuda_support(self):
        script = r'''
import json

import torch_rs as torch

cuda = torch.backends.cuda
matmul = cuda.matmul
initial = matmul.allow_tf32
matmul.allow_tf32 = True
enabled = matmul.allow_tf32
matmul.allow_tf32 = False
print(json.dumps({
    "initial": initial,
    "initial_type": type(initial).__name__,
    "enabled": enabled,
    "restored": matmul.allow_tf32,
    "cuda_built": cuda.is_built(),
    "cuda_module": hasattr(torch, "cuda"),
    "cuda_tensor": hasattr(torch.Tensor, "cuda"),
    "cuda_transfer": hasattr(torch.Tensor, "to"),
    "cublas_execution": hasattr(torch._C, "_cuda_getCurrentBlasHandle"),
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
                "initial": False,
                "initial_type": "bool",
                "enabled": True,
                "restored": False,
                "cuda_built": False,
                "cuda_module": False,
                "cuda_tensor": False,
                "cuda_transfer": False,
                "cublas_execution": False,
            },
        )

    def test_exact_bool_assignments_are_independent_preferences(self):
        cuda = self.cuda
        expected_other_states = self.states(cuda)[1:]

        self.assertIs(cuda.matmul.allow_tf32, False)
        self.assertIs(type(cuda.matmul.allow_tf32), bool)
        for allow_tf32 in (True, False, False, True, True, False):
            with self.subTest(allow_tf32=allow_tf32):
                cuda.matmul.allow_tf32 = allow_tf32
                self.assertIs(cuda.matmul.allow_tf32, allow_tf32)
                self.assertIs(torch._C._get_cublas_allow_tf32(), allow_tf32)
                self.assertEqual(self.states(cuda)[1:], expected_other_states)
                self.assertIs(cuda.is_built(), False)

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
            other_states = (not state, state, not state, state)
            self.set_states(self.cuda, (state, *other_states))
            for value, type_name in invalid_values:
                with self.subTest(state=state, value_type=type_name):
                    message = (
                        "set_allow_tf32_cublas expects a bool, but got "
                        f"{type_name}"
                    )
                    for setter in (
                        lambda value=value: setattr(
                            self.cuda.matmul,
                            "allow_tf32",
                            value,
                        ),
                        lambda value=value: torch._C._set_cublas_allow_tf32(
                            value
                        ),
                    ):
                        with self.assertRaises(RuntimeError) as raised:
                            setter()
                        self.assertEqual(str(raised.exception), message)
                        self.assertEqual(raised.exception.args, (message,))
                        self.assertEqual(
                            self.states(self.cuda),
                            (state, *other_states),
                        )

    def test_deletion_and_unknown_attributes_preserve_state(self):
        matmul = self.cuda.matmul
        matmul.allow_tf32 = True

        with self.assertRaises(AttributeError) as raised:
            del matmul.allow_tf32
        message = "'cuBLASModule' object has no attribute 'allow_tf32'"
        self.assertEqual(str(raised.exception), message)
        self.assertEqual(raised.exception.args, (message,))
        self.assertIs(matmul.allow_tf32, True)

        for name in ("unknown", "_unknown"):
            with self.subTest(name=name, operation="get"):
                with self.assertRaises(AttributeError) as raised:
                    getattr(matmul, name)
                self.assertEqual(str(raised.exception), f"Unknown attribute {name}")
                self.assertIs(matmul.allow_tf32, True)
            with self.subTest(name=name, operation="set"):
                with self.assertRaises(AttributeError) as raised:
                    setattr(matmul, name, False)
                self.assertEqual(str(raised.exception), f"Unknown attribute {name}")
                self.assertIs(matmul.allow_tf32, True)

    def test_state_is_process_global_across_threads_and_proxy_copies(self):
        cuda = self.cuda
        matmul = cuda.matmul
        copied = copy.copy(matmul)
        restored = pickle.loads(pickle.dumps(matmul))
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []
        matmul.allow_tf32 = True

        def worker():
            try:
                observations.append(restored.allow_tf32)
                copied.allow_tf32 = False
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(restored.allow_tf32)
                restored.allow_tf32 = False
            except BaseException as error:
                errors.append(error)
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_changed.wait(timeout=10))
        self.assertIs(matmul.allow_tf32, False)
        matmul.allow_tf32 = True
        main_changed.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [True, True])
        self.assertIs(matmul.allow_tf32, False)
        self.assertIs(copied.allow_tf32, False)
        self.assertIs(restored.allow_tf32, False)

    def test_reload_and_fresh_import_replace_proxies_but_preserve_state(self):
        cuda = self.cuda
        old_proxy = cuda.matmul
        old_type = type(old_proxy)
        namespace = cuda.__dict__
        old_proxy.allow_tf32 = False

        reloaded = importlib.reload(cuda)
        new_proxy = reloaded.matmul

        self.assertIs(reloaded, cuda)
        self.assertIs(cuda.__dict__, namespace)
        self.assertIs(torch.backends.cuda, cuda)
        self.assertIs(sys.modules[cuda.__name__], cuda)
        self.assertIsNot(new_proxy, old_proxy)
        self.assertIsNot(type(new_proxy), old_type)
        self.assertIs(type(new_proxy), cuda.cuBLASModule)
        self.assertIs(old_proxy.allow_tf32, False)
        self.assertIs(new_proxy.allow_tf32, False)

        new_proxy.allow_tf32 = True
        self.assertIs(old_proxy.allow_tf32, True)
        old_proxy.allow_tf32 = False
        self.assertIs(new_proxy.allow_tf32, False)

        fresh = fresh_cuda_module()
        self.assertIsNot(fresh, cuda)
        self.assertIsNot(fresh.matmul, new_proxy)
        self.assertIs(fresh.matmul.allow_tf32, False)
        fresh.matmul.allow_tf32 = True
        self.assertIs(old_proxy.allow_tf32, True)
        self.assertIs(new_proxy.allow_tf32, True)

        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_proxy)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <class 'torch_rs.backends.cuda.cuBLASModule'>: "
            "it's not the same object as torch_rs.backends.cuda.cuBLASModule",
        )
        restored = pickle.loads(pickle.dumps(fresh.matmul))
        self.assertIs(type(restored), fresh.cuBLASModule)
        self.assertIs(restored.allow_tf32, True)

    def test_proxy_metadata_imports_copying_and_pickling_are_canonical(self):
        cuda = self.cuda
        proxy = cuda.matmul
        proxy_type = cuda.cuBLASModule

        self.assertIs(torch.backends.cuda, cuda)
        self.assertIs(sys.modules["torch_rs.backends.cuda"], cuda)
        self.assertIs(type(cuda), types.ModuleType)
        self.assertIs(type(proxy), proxy_type)
        self.assertEqual(proxy_type.__module__, "torch_rs.backends.cuda")
        self.assertEqual(proxy_type.__name__, "cuBLASModule")
        self.assertEqual(proxy_type.__qualname__, "cuBLASModule")
        self.assertIsNone(proxy_type.__doc__)
        self.assertEqual(str(inspect.signature(proxy_type)), "()")
        self.assertEqual(
            set(vars(proxy_type)),
            {
                "__dict__",
                "__doc__",
                "__getattr__",
                "__module__",
                "__setattr__",
                "__weakref__",
                "_parse_reduction_setting",
            },
        )
        self.assertEqual(proxy_type.__annotations__, {})
        self.assertEqual(inspect.get_annotations(proxy_type), {})
        self.assertEqual(vars(proxy), {})
        self.assertEqual(proxy.__module__, "torch_rs.backends.cuda")
        self.assertIsNone(proxy.__doc__)
        self.assertEqual(proxy.__annotations__, {})
        self.assertNotIn("allow_tf32", dir(proxy))
        with self.assertRaisesRegex(
            TypeError,
            r"^<torch_rs\.backends\.cuda\.cuBLASModule object at 0x[0-9a-f]+> "
            r"is not a callable object$",
        ):
            inspect.signature(proxy)

        self.assertEqual(
            cuda.__all__,
            [
                "is_built",
                "cuBLASModule",
                "is_ck_sdpa_available",
                "matmul",
                "enable_mem_efficient_sdp",
                "mem_efficient_sdp_enabled",
                "math_sdp_enabled",
                "enable_math_sdp",
                "allow_fp16_bf16_reduction_math_sdp",
                "fp16_bf16_reduction_math_sdp_allowed",
                "is_flash_attention_available",
            ],
        )
        direct = {}
        wildcard = {}
        exec(
            "from torch_rs.backends.cuda import cuBLASModule, matmul",
            direct,
        )
        exec("from torch_rs.backends.cuda import *", wildcard)
        self.assertIs(direct["cuBLASModule"], proxy_type)
        self.assertIs(direct["matmul"], proxy)
        self.assertIs(wildcard["cuBLASModule"], proxy_type)
        self.assertIs(wildcard["matmul"], proxy)

        with self.assertRaises(ModuleNotFoundError) as raised:
            importlib.import_module("torch_rs.backends.cuda.matmul")
        self.assertEqual(
            str(raised.exception),
            "No module named 'torch_rs.backends.cuda.matmul'",
        )
        self.assertNotIn("torch_rs.backends.cuda.matmul", sys.modules)

        for copier in (copy.copy, copy.deepcopy):
            with self.subTest(copier=copier.__name__):
                copied = copier(proxy)
                self.assertIsNot(copied, proxy)
                self.assertIs(type(copied), proxy_type)
                self.assertEqual(vars(copied), {})
                self.assertIs(copied.allow_tf32, proxy.allow_tf32)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(proxy, protocol=protocol)
                self.assertIn(b"torch_rs.backends.cuda", payload)
                restored = pickle.loads(payload)
                self.assertIsNot(restored, proxy)
                self.assertIs(type(restored), proxy_type)
                self.assertEqual(vars(restored), {})
                self.assertIs(restored.allow_tf32, proxy.allow_tf32)

    def test_private_accessor_metadata_binding_and_pickle(self):
        getter = torch._C._get_cublas_allow_tf32
        setter = torch._C._set_cublas_allow_tf32

        for function, name in (
            (getter, "_get_cublas_allow_tf32"),
            (setter, "_set_cublas_allow_tf32"),
        ):
            with self.subTest(name=name):
                self.assertIs(type(function), types.BuiltinFunctionType)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(function.__module__, "torch_rs.torch_rs")
                self.assertIsNone(function.__doc__)
                self.assertIsNone(function.__text_signature__)
                with self.assertRaises(ValueError) as raised:
                    inspect.signature(function)
                self.assertEqual(
                    str(raised.exception),
                    f"no signature found for builtin <built-in function {name}>",
                )
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(b"torch_rs.torch_rs", payload)
                    self.assertIs(pickle.loads(payload), function)

        direct_import = {}
        native_wildcard = {}
        exec(
            "from torch_rs._C import "
            "_get_cublas_allow_tf32, _set_cublas_allow_tf32",
            direct_import,
        )
        exec("from torch_rs._C import *", native_wildcard)
        self.assertIs(direct_import["_get_cublas_allow_tf32"], getter)
        self.assertIs(direct_import["_set_cublas_allow_tf32"], setter)
        self.assertNotIn("_get_cublas_allow_tf32", native_wildcard)
        self.assertNotIn("_set_cublas_allow_tf32", native_wildcard)
        self.assertFalse(hasattr(torch, "_get_cublas_allow_tf32"))
        self.assertFalse(hasattr(torch, "_set_cublas_allow_tf32"))
        self.assertNotIn("_get_cublas_allow_tf32", torch.__all__)
        self.assertNotIn("_set_cublas_allow_tf32", torch.__all__)
        self.assertNotIn("_get_cublas_allow_tf32", torch._C.__all__)
        self.assertNotIn("_set_cublas_allow_tf32", torch._C.__all__)

        self.assertIs(setter(True), None)
        self.assertIs(getter(), True)

    def test_private_accessor_binding_errors_preserve_state(self):
        getter = torch._C._get_cublas_allow_tf32
        setter = torch._C._set_cublas_allow_tf32
        self.cuda.matmul.allow_tf32 = True
        cases = (
            (
                lambda: getter(None),
                "torch_rs.torch_rs._get_cublas_allow_tf32() "
                "takes no arguments (1 given)",
            ),
            (
                lambda: getter(value=None),
                "torch_rs.torch_rs._get_cublas_allow_tf32() "
                "takes no keyword arguments",
            ),
            (
                lambda: setter(),
                "torch_rs.torch_rs._set_cublas_allow_tf32() "
                "takes exactly one argument (0 given)",
            ),
            (
                lambda: setter(True, False),
                "torch_rs.torch_rs._set_cublas_allow_tf32() "
                "takes exactly one argument (2 given)",
            ),
            (
                lambda: setter(object=False),
                "torch_rs.torch_rs._set_cublas_allow_tf32() "
                "takes no keyword arguments",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(self.cuda.matmul.allow_tf32, True)

    def test_preference_does_not_add_cuda_or_change_cpu_matmul_execution(self):
        left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        right = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        expected = torch.matmul(left, right).tolist()

        for allow_tf32 in (True, False):
            with self.subTest(allow_tf32=allow_tf32):
                self.cuda.matmul.allow_tf32 = allow_tf32
                self.assertEqual(torch.matmul(left, right).tolist(), expected)
                self.assertEqual(torch.get_float32_matmul_precision(), "highest")
                self.assertIs(self.cuda.is_built(), False)
                self.assertFalse(hasattr(torch, "cuda"))
                self.assertFalse(hasattr(torch.Tensor, "cuda"))
                self.assertFalse(hasattr(torch.Tensor, "to"))
                for name in (
                    "allow_bf16_reduced_precision_reduction",
                    "allow_fp16_accumulation",
                    "allow_fp16_reduced_precision_reduction",
                    "fp32_precision",
                ):
                    self.assertFalse(hasattr(self.cuda.matmul, name))

    def test_importing_and_using_preference_does_not_probe_external_runtimes(self):
        script = r'''
import os
import sys

class RejectExternalRuntimeImport:
    blocked = {"cupy", "nvidia", "numpy", "pynvml", "torch"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    CUDA_VISIBLE_DEVICES="0",
    NVIDIA_VISIBLE_DEVICES="all",
    PYTORCH_NVML_BASED_CUDA_CHECK="1",
)
import torch_rs as torch
from torch_rs.backends.cuda import cuBLASModule, matmul

assert type(matmul) is cuBLASModule
assert matmul.allow_tf32 is False
matmul.allow_tf32 = True
assert matmul.allow_tf32 is torch._C._get_cublas_allow_tf32() is True
matmul.allow_tf32 = False
assert torch.backends.cuda.is_built() is False
assert not hasattr(torch, "cuda")
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

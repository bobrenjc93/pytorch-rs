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
        raise AssertionError("cuda.matmul.allow_tf32 must not request truthiness")


def fresh_matmul_module():
    module_name = "torch_rs.backends.cuda.matmul"
    sys.modules.pop(module_name, None)
    if hasattr(torch.backends.cuda, "matmul"):
        del torch.backends.cuda.matmul
    module = importlib.import_module(module_name)
    torch.backends.cuda.matmul = module
    return module


class CudaMatmulAllowTf32Tests(unittest.TestCase):
    def setUp(self):
        self.matmul = fresh_matmul_module()
        self.original = (
            self.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        )
        self.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        torch.set_float32_matmul_precision("highest")

    def tearDown(self):
        matmul = fresh_matmul_module()
        matmul.allow_tf32 = self.original[0]
        torch.backends.cudnn.allow_tf32 = self.original[1]
        torch.backends.cuda.enable_flash_sdp(self.original[2])
        torch.backends.cuda.enable_math_sdp(self.original[3])
        torch.backends.cuda.enable_mem_efficient_sdp(self.original[4])
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(self.original[5])
        torch.set_float32_matmul_precision("highest")

    def test_fresh_process_defaults_to_exact_false_without_cuda_support(self):
        script = r'''
import json
import sys

import torch_rs as torch

cuda = torch.backends.cuda
matmul = cuda.matmul
imported = __import__("torch_rs.backends.cuda.matmul", fromlist=["matmul"])
initial = matmul.allow_tf32
matmul.allow_tf32 = True
enabled = matmul.allow_tf32
matmul.allow_tf32 = False
print(json.dumps({
    "initial": initial,
    "initial_type": type(initial).__name__,
    "enabled": enabled,
    "restored": matmul.allow_tf32,
    "imported_is_parent": imported is matmul,
    "sys_modules_is_parent": sys.modules["torch_rs.backends.cuda.matmul"] is matmul,
    "cudnn": torch.backends.cudnn.allow_tf32,
    "precision": torch.get_float32_matmul_precision(),
    "flash": cuda.flash_sdp_enabled(),
    "math": cuda.math_sdp_enabled(),
    "mem_efficient": cuda.mem_efficient_sdp_enabled(),
    "reduction": cuda.fp16_bf16_reduction_math_sdp_allowed(),
    "built": cuda.is_built(),
    "cuda": hasattr(torch, "cuda"),
    "tensor_cuda": hasattr(torch.Tensor, "cuda"),
    "execution": hasattr(torch, "cublas"),
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
                "imported_is_parent": True,
                "sys_modules_is_parent": True,
                "cudnn": True,
                "precision": "highest",
                "flash": True,
                "math": True,
                "mem_efficient": True,
                "reduction": False,
                "built": False,
                "cuda": False,
                "tensor_cuda": False,
                "execution": False,
            },
        )

    def test_repeated_exact_bool_assignments_are_independent_preferences(self):
        matmul = self.matmul
        cuda = torch.backends.cuda
        cudnn = torch.backends.cudnn
        cudnn.allow_tf32 = True
        cuda.enable_flash_sdp(False)
        cuda.enable_math_sdp(True)
        cuda.enable_mem_efficient_sdp(False)
        cuda.allow_fp16_bf16_reduction_math_sdp(True)

        self.assertIs(matmul.allow_tf32, False)
        self.assertIs(type(matmul.allow_tf32), bool)
        for allow_tf32 in (True, False, False, True, True, False):
            with self.subTest(allow_tf32=allow_tf32):
                matmul.allow_tf32 = allow_tf32
                self.assertIs(matmul.allow_tf32, allow_tf32)
                self.assertIs(torch._C._get_cublas_allow_tf32(), allow_tf32)
                self.assertIs(cudnn.allow_tf32, True)
                self.assertEqual(torch.get_float32_matmul_precision(), "highest")
                self.assertIs(cuda.flash_sdp_enabled(), False)
                self.assertIs(cuda.math_sdp_enabled(), True)
                self.assertIs(cuda.mem_efficient_sdp_enabled(), False)
                self.assertIs(
                    cuda.fp16_bf16_reduction_math_sdp_allowed(),
                    True,
                )
                self.assertIs(cuda.is_built(), False)
                self.assertIs(cuda.is_ck_sdpa_available(), False)
                self.assertIs(cuda.is_flash_attention_available(), False)

        matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("highest")
        self.assertIs(matmul.allow_tf32, True)
        matmul.allow_tf32 = False
        self.assertEqual(torch.matmul(torch.eye(2), torch.eye(2)).tolist(), [[1.0, 0.0], [0.0, 1.0]])

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
            self.matmul.allow_tf32 = state
            torch.backends.cudnn.allow_tf32 = not state
            torch.backends.cuda.enable_flash_sdp(state)
            torch.backends.cuda.enable_math_sdp(not state)
            torch.backends.cuda.enable_mem_efficient_sdp(state)
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(not state)
            for value, type_name in invalid_values:
                with self.subTest(state=state, value_type=type_name):
                    message = (
                        "set_allow_tf32_cublas expects a bool, but got "
                        f"{type_name}"
                    )
                    for setter in (
                        lambda value=value: setattr(
                            self.matmul,
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
                        self.assertIs(self.matmul.allow_tf32, state)
                        self.assertIs(torch._C._get_cublas_allow_tf32(), state)
                        self.assertIs(torch.backends.cudnn.allow_tf32, not state)
                        self.assertIs(torch.backends.cuda.flash_sdp_enabled(), state)
                        self.assertIs(torch.backends.cuda.math_sdp_enabled(), not state)
                        self.assertIs(
                            torch.backends.cuda.mem_efficient_sdp_enabled(),
                            state,
                        )
                        self.assertIs(
                            torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
                            not state,
                        )

    def test_state_is_process_global_across_threads_and_import_paths(self):
        matmul = self.matmul
        imported = importlib.import_module("torch_rs.backends.cuda.matmul")
        from_parent = torch.backends.cuda.matmul
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []

        self.assertIs(imported, matmul)
        self.assertIs(from_parent, matmul)

        def worker():
            try:
                observations.append(matmul.allow_tf32)
                imported.allow_tf32 = True
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(from_parent.allow_tf32)
                matmul.allow_tf32 = True
            except BaseException as error:
                errors.append(error)
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_changed.wait(timeout=10))
        self.assertEqual(errors, [])
        self.assertIs(matmul.allow_tf32, True)
        from_parent.allow_tf32 = False
        main_changed.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [False, False])
        self.assertIs(matmul.allow_tf32, True)
        self.assertIs(torch._C._get_cublas_allow_tf32(), True)

    def test_reload_and_fresh_import_preserve_shared_state(self):
        cuda = torch.backends.cuda
        matmul = self.matmul
        namespace = matmul.__dict__
        matmul.allow_tf32 = True

        reloaded = importlib.reload(matmul)

        self.assertIsNot(reloaded, matmul)
        self.assertIs(matmul.__dict__, namespace)
        self.assertIs(cuda.matmul, matmul)
        self.assertIs(sys.modules[matmul.__name__], reloaded)
        self.assertIs(reloaded.m, matmul)
        self.assertIs(matmul.allow_tf32, True)
        self.assertIs(reloaded.allow_tf32, True)

        reloaded.allow_tf32 = False
        self.assertIs(matmul.allow_tf32, False)
        matmul.allow_tf32 = True
        self.assertIs(reloaded.allow_tf32, True)

        fresh = fresh_matmul_module()
        self.assertIs(torch.backends.cuda.matmul, fresh)
        self.assertIs(fresh.allow_tf32, True)
        fresh.allow_tf32 = False
        self.assertIs(matmul.allow_tf32, False)
        self.assertIs(reloaded.allow_tf32, False)

    def test_proxy_metadata_deletion_copying_and_private_accessors(self):
        matmul = self.matmul
        descriptor = vars(type(matmul))["allow_tf32"]
        getter = torch._C._get_cublas_allow_tf32
        setter = torch._C._set_cublas_allow_tf32

        self.assertIsInstance(matmul, types.ModuleType)
        self.assertEqual(type(matmul).__name__, "_MatmulModule")
        self.assertEqual(type(matmul).__module__, "torch_rs.backends.cuda.matmul")
        self.assertIs(matmul.m.__annotations__["allow_tf32"], bool)
        self.assertNotIn("allow_tf32", vars(matmul))
        self.assertNotIn("allow_tf32", vars(matmul.m))
        self.assertNotIn("allow_tf32", dir(matmul))
        self.assertEqual(set(vars(descriptor)), {"getter", "setter"})
        self.assertIs(descriptor.getter, getter)
        self.assertIs(descriptor.setter, setter)
        self.assertIs(descriptor.__get__(matmul, type(matmul)), False)

        imported = {}
        wildcard = {}
        exec("from torch_rs.backends.cuda import matmul", imported)
        exec("from torch_rs.backends.cuda.matmul import allow_tf32", imported)
        exec("from torch_rs.backends.cuda.matmul import *", wildcard)
        self.assertIs(imported["matmul"], matmul)
        self.assertIs(imported["allow_tf32"], False)
        self.assertNotIn("allow_tf32", wildcard)
        matmul.allow_tf32 = True
        self.assertIs(imported["allow_tf32"], False)

        with self.assertRaises(AttributeError) as raised:
            del matmul.allow_tf32
        self.assertEqual(str(raised.exception), "__delete__")
        self.assertEqual(raised.exception.args, ("__delete__",))
        self.assertIs(matmul.allow_tf32, True)

        for state in (False, True):
            matmul.allow_tf32 = state
            self.assertIs(copy.copy(matmul.allow_tf32), state)
            self.assertIs(copy.deepcopy(matmul.allow_tf32), state)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(state=state, protocol=protocol):
                    self.assertIs(
                        pickle.loads(
                            pickle.dumps(matmul.allow_tf32, protocol=protocol)
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
            restored = pickle.loads(pickle.dumps(descriptor, protocol))
            self.assertIsNot(restored, descriptor)
            self.assertIs(type(restored), type(descriptor))
            self.assertIs(restored.getter, descriptor.getter)
            self.assertIs(restored.setter, descriptor.setter)

        for function, name in (
            (getter, "_get_cublas_allow_tf32"),
            (setter, "_set_cublas_allow_tf32"),
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

        self.assertIs(setter(False), None)
        self.assertIs(getter(), False)

    def test_private_accessor_binding_errors_preserve_state(self):
        getter = torch._C._get_cublas_allow_tf32
        setter = torch._C._set_cublas_allow_tf32
        self.matmul.allow_tf32 = True
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
                self.assertIs(self.matmul.allow_tf32, True)

    def test_reduced_precision_reduction_and_cublas_execution_stay_unsupported(self):
        matmul = self.matmul
        for name in (
            "allow_fp16_accumulation",
            "allow_fp16_reduced_precision_reduction",
            "allow_fp16_reduced_precision_reduction_split_k",
            "allow_bf16_reduced_precision_reduction",
            "allow_bf16_reduced_precision_reduction_split_k",
            "fp32_precision",
        ):
            with self.subTest(get=name):
                self.assertFalse(hasattr(matmul, name))
            with self.subTest(set=name):
                with self.assertRaises(AttributeError) as raised:
                    setattr(matmul, name, True)
                self.assertEqual(str(raised.exception), "Unknown attribute " + name)
                self.assertIs(matmul.allow_tf32, False)

        for name in (
            "_get_cublas_allow_fp16_reduced_precision_reduction",
            "_set_cublas_allow_fp16_reduced_precision_reduction",
            "_get_cublas_allow_bf16_reduced_precision_reduction",
            "_set_cublas_allow_bf16_reduced_precision_reduction",
            "_get_cublas_allow_fp16_accumulation",
            "_set_cublas_allow_fp16_accumulation",
        ):
            with self.subTest(native=name):
                self.assertFalse(hasattr(torch._C, name))
                self.assertFalse(hasattr(torch, name))

        self.assertFalse(hasattr(torch, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "to"))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda:0' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda:0")

    def test_global_flag_freezing_does_not_block_cuda_matmul_preference(self):
        script = r'''
import json

import torch_rs as torch

matmul = torch.backends.cuda.matmul
matmul.allow_tf32 = False
torch.backends.disable_global_flags()
matmul.allow_tf32 = True
blocked_cudnn = None
try:
    torch.backends.cudnn.allow_tf32 = False
except RuntimeError as error:
    blocked_cudnn = str(error)
print(json.dumps({
    "matmul": matmul.allow_tf32,
    "native": torch._C._get_cublas_allow_tf32(),
    "blocked_cudnn": blocked_cudnn,
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
                "matmul": True,
                "native": True,
                "blocked_cudnn": (
                    "not allowed to set torch_rs.backends.cudnn flags after "
                    "disable_global_flags; please use flags() context manager instead"
                ),
            },
        )


if __name__ == "__main__":
    unittest.main()

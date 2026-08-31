import copy
import importlib
import json
import pickle
import subprocess
import sys
import threading
import types
import unittest

import numpy as np

import torch_rs as torch


CUDA_BACKEND_ALL = [
    "is_built",
    "cuBLASModule",
    "is_ck_sdpa_available",
    "matmul",
    "enable_flash_sdp",
    "flash_sdp_enabled",
    "enable_mem_efficient_sdp",
    "mem_efficient_sdp_enabled",
    "math_sdp_enabled",
    "enable_math_sdp",
    "allow_fp16_bf16_reduction_math_sdp",
    "fp16_bf16_reduction_math_sdp_allowed",
    "is_flash_attention_available",
]
CUDA_BACKEND_PUBLIC = {
    "allow_fp16_bf16_reduction_math_sdp",
    "cuBLASModule",
    "enable_flash_sdp",
    "enable_math_sdp",
    "enable_mem_efficient_sdp",
    "flash_sdp_enabled",
    "fp16_bf16_reduction_math_sdp_allowed",
    "is_built",
    "is_ck_sdpa_available",
    "is_flash_attention_available",
    "math_sdp_enabled",
    "matmul",
    "mem_efficient_sdp_enabled",
}
MATMUL_ATTRIBUTES = (
    "allow_tf32",
    "allow_fp16_reduced_precision_reduction",
    "allow_bf16_reduced_precision_reduction",
)
MATMUL_DEFAULTS = (False, True, True)


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("cuda.matmul preferences must not request truthiness")


def matmul_states(matmul):
    return tuple(getattr(matmul, name) for name in MATMUL_ATTRIBUTES)


def set_matmul_states(matmul, states):
    for name, state in zip(MATMUL_ATTRIBUTES, states):
        setattr(matmul, name, state)


class CudaMatmulPreferenceTests(unittest.TestCase):
    def setUp(self):
        self.cuda = importlib.import_module("torch_rs.backends.cuda")
        self.original = matmul_states(self.cuda.matmul)
        set_matmul_states(self.cuda.matmul, MATMUL_DEFAULTS)

    def tearDown(self):
        set_matmul_states(self.cuda.matmul, self.original)

    def test_fresh_process_defaults_without_external_runtime_imports(self):
        script = r'''
import json
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

cuda = torch.backends.cuda
matmul = cuda.matmul
defaults = [
    matmul.allow_tf32,
    matmul.allow_fp16_reduced_precision_reduction,
    matmul.allow_bf16_reduced_precision_reduction,
]
matmul.allow_tf32 = True
matmul.allow_fp16_reduced_precision_reduction = False
matmul.allow_bf16_reduced_precision_reduction = False
print(json.dumps({
    "defaults": defaults,
    "default_types": [type(value).__name__ for value in defaults],
    "updated": [
        matmul.allow_tf32,
        matmul.allow_fp16_reduced_precision_reduction,
        matmul.allow_bf16_reduced_precision_reduction,
    ],
    "native_tf32": torch._C._get_cublas_allow_tf32(),
    "native_fp16": torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
    "native_bf16": torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
    "backend_built": cuda.is_built(),
    "cuda_module": hasattr(torch, "cuda"),
    "cuda_submodule_loaded": "torch_rs.cuda" in sys.modules,
    "cudnn_sdp": hasattr(cuda, "cudnn_sdp_enabled"),
    "sdp_kernel": hasattr(cuda, "sdp_kernel"),
    "sdpa_execution": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
    "compile": hasattr(torch, "compile"),
    "reference_torch_loaded": "torch" in sys.modules,
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
                "defaults": [False, True, True],
                "default_types": ["bool", "bool", "bool"],
                "updated": [True, False, False],
                "native_tf32": True,
                "native_fp16": [False, True],
                "native_bf16": [False, True],
                "backend_built": False,
                "cuda_module": False,
                "cuda_submodule_loaded": False,
                "cudnn_sdp": False,
                "sdp_kernel": False,
                "sdpa_execution": False,
                "compile": False,
                "reference_torch_loaded": False,
            },
        )

    def test_exact_bool_assignments_are_process_global_and_independent(self):
        cuda = self.cuda
        matmul = cuda.matmul
        other_states = (
            cuda.flash_sdp_enabled(),
            cuda.math_sdp_enabled(),
            cuda.mem_efficient_sdp_enabled(),
            cuda.fp16_bf16_reduction_math_sdp_allowed(),
            torch.backends.cudnn.allow_tf32,
        )

        for states in (
            (True, False, False),
            (False, True, True),
            (False, False, True),
            (True, True, False),
            MATMUL_DEFAULTS,
        ):
            with self.subTest(states=states):
                set_matmul_states(matmul, states)
                self.assertEqual(matmul_states(matmul), states)
                self.assertIs(torch._C._get_cublas_allow_tf32(), states[0])
                self.assertEqual(
                    torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
                    (states[1], True),
                )
                self.assertEqual(
                    torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
                    (states[2], True),
                )
                self.assertEqual(
                    (
                        cuda.flash_sdp_enabled(),
                        cuda.math_sdp_enabled(),
                        cuda.mem_efficient_sdp_enabled(),
                        cuda.fp16_bf16_reduction_math_sdp_allowed(),
                        torch.backends.cudnn.allow_tf32,
                    ),
                    other_states,
                )
                self.assertEqual(torch.get_float32_matmul_precision(), "highest")
                self.assertIs(cuda.is_built(), False)

    def test_invalid_assignments_preserve_state(self):
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
            self.cuda.matmul.allow_tf32 = state
            for value, type_name in invalid_values:
                with self.subTest(attribute="allow_tf32", state=state, value=type_name):
                    message = (
                        "set_allow_tf32_cublas expects a bool, but got "
                        f"{type_name}"
                    )
                    with self.assertRaises(RuntimeError) as raised:
                        self.cuda.matmul.allow_tf32 = value
                    self.assertEqual(str(raised.exception), message)
                    self.assertIs(self.cuda.matmul.allow_tf32, state)

        invalid_reduction_values = (
            [],
            [None],
            [False, 1],
            [False, True, False],
            None,
            0,
            1,
            0.0,
            np.bool_(True),
            "",
            object(),
            _RejectTruthiness(),
            torch.tensor(True),
            torch.float32,
            torch.device("cpu"),
            torch.strided,
            torch.Size([1]),
            torch.finfo(torch.float32),
        )
        for attribute in (
            "allow_fp16_reduced_precision_reduction",
            "allow_bf16_reduced_precision_reduction",
        ):
            for state in (False, True):
                setattr(self.cuda.matmul, attribute, state)
                before = matmul_states(self.cuda.matmul)
                for value in invalid_reduction_values:
                    with self.subTest(
                        attribute=attribute,
                        state=state,
                        value=type(value).__name__,
                    ):
                        with self.assertRaises((RuntimeError, TypeError)):
                            setattr(self.cuda.matmul, attribute, value)
                        self.assertEqual(matmul_states(self.cuda.matmul), before)

    def test_reduction_tuple_assignments_match_pytorch_splitk_compatibility(self):
        matmul = self.cuda.matmul
        for attribute, getter_name in (
            (
                "allow_fp16_reduced_precision_reduction",
                "_get_cublas_allow_fp16_reduced_precision_reduction",
            ),
            (
                "allow_bf16_reduced_precision_reduction",
                "_get_cublas_allow_bf16_reduced_precision_reduction",
            ),
        ):
            getter = getattr(torch._C, getter_name)
            for value, native_state in (
                ((False,), (False, True)),
                ([False], (False, True)),
                ((False, False), (False, False)),
                ([False, True], (False, True)),
                ((True, True), (True, True)),
            ):
                with self.subTest(attribute=attribute, value=value):
                    setattr(matmul, attribute, value)
                    self.assertIs(getattr(matmul, attribute), native_state[0])
                    self.assertEqual(getter(), native_state)

            setattr(matmul, attribute, True)
            with self.subTest(attribute=attribute, value=(True, False)):
                with self.assertRaises(RuntimeError) as raised:
                    setattr(matmul, attribute, (True, False))
                self.assertEqual(
                    str(raised.exception),
                    "allow_splitk=False is not supported when reduced "
                    "precision reductions are enabled",
                )
                self.assertEqual(getter(), (True, True))

    def test_state_is_shared_across_threads_imports_copies_and_pickles(self):
        cuda = self.cuda
        matmul = cuda.matmul
        imported = importlib.import_module("torch_rs.backends.cuda").matmul
        copied = copy.copy(matmul)
        restored = pickle.loads(pickle.dumps(matmul))
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []

        self.assertIs(imported, matmul)
        for alias in (copied, restored):
            self.assertIs(type(alias), type(matmul))
            self.assertIsNot(alias, matmul)
            self.assertEqual(vars(alias), {})

        set_matmul_states(matmul, MATMUL_DEFAULTS)

        def worker():
            try:
                observations.append(matmul_states(imported))
                set_matmul_states(copied, (True, False, False))
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(matmul_states(restored))
                set_matmul_states(restored, (False, False, True))
            except BaseException as error:
                errors.append(error)
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_changed.wait(timeout=10))
        self.assertEqual(errors, [])
        self.assertEqual(matmul_states(matmul), (True, False, False))
        set_matmul_states(matmul, (False, True, False))
        main_changed.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(
            observations,
            [MATMUL_DEFAULTS, (False, True, False)],
        )
        self.assertEqual(matmul_states(matmul), (False, False, True))

    def test_reload_replaces_proxy_class_and_preserves_native_state(self):
        cuda = self.cuda
        old_matmul = cuda.matmul
        old_class = cuda.cuBLASModule
        old_getattr = old_class.__getattr__
        old_setattr = old_class.__setattr__
        namespace = cuda.__dict__

        set_matmul_states(old_matmul, (True, False, False))
        reloaded = importlib.reload(cuda)
        new_matmul = cuda.matmul

        self.assertIs(reloaded, cuda)
        self.assertIs(cuda.__dict__, namespace)
        self.assertIs(torch.backends.cuda, cuda)
        self.assertIs(sys.modules[cuda.__name__], cuda)
        self.assertIsNot(cuda.cuBLASModule, old_class)
        self.assertIsNot(new_matmul, old_matmul)
        self.assertIs(type(new_matmul), cuda.cuBLASModule)
        self.assertEqual(matmul_states(old_matmul), (True, False, False))
        self.assertEqual(matmul_states(new_matmul), (True, False, False))

        set_matmul_states(new_matmul, MATMUL_DEFAULTS)
        self.assertEqual(matmul_states(old_matmul), MATMUL_DEFAULTS)

        for name, stale in (
            ("matmul", old_matmul),
            ("cuBLASModule", old_class),
            ("cuBLASModule.__getattr__", old_getattr),
            ("cuBLASModule.__setattr__", old_setattr),
        ):
            with self.subTest(stale=name):
                with self.assertRaises(pickle.PicklingError):
                    pickle.dumps(stale)

    def test_metadata_exports_copying_and_private_accessors(self):
        cuda = self.cuda
        matmul = cuda.matmul

        self.assertIs(torch.backends.cuda, cuda)
        self.assertIs(sys.modules["torch_rs.backends.cuda"], cuda)
        self.assertIs(type(cuda), types.ModuleType)
        self.assertIsNone(cuda.__doc__)
        self.assertEqual(cuda.__all__, CUDA_BACKEND_ALL)
        self.assertEqual(
            {name for name in vars(cuda) if not name.startswith("_")},
            CUDA_BACKEND_PUBLIC | {"torch"},
        )
        self.assertIs(cuda.torch, torch)
        self.assertEqual(cuda.cuBLASModule.__name__, "cuBLASModule")
        self.assertEqual(cuda.cuBLASModule.__qualname__, "cuBLASModule")
        self.assertEqual(cuda.cuBLASModule.__module__, "torch_rs.backends.cuda")
        self.assertIs(type(matmul), cuda.cuBLASModule)
        self.assertEqual(vars(matmul), {})
        self.assertNotIn("allow_tf32", vars(matmul))
        self.assertNotIn("allow_fp16_reduced_precision_reduction", vars(matmul))
        self.assertNotIn("allow_bf16_reduced_precision_reduction", vars(matmul))
        for name in MATMUL_ATTRIBUTES:
            self.assertNotIn(name, dir(matmul))

        backend_import = {}
        object_import = {}
        wildcard = {}
        exec("from torch_rs.backends import cuda", backend_import)
        exec("from torch_rs.backends.cuda import cuBLASModule, matmul", object_import)
        exec("from torch_rs.backends.cuda import *", wildcard)
        self.assertIs(backend_import["cuda"], cuda)
        self.assertIs(object_import["cuBLASModule"], cuda.cuBLASModule)
        self.assertIs(object_import["matmul"], matmul)
        self.assertIs(wildcard["cuBLASModule"], cuda.cuBLASModule)
        self.assertIs(wildcard["matmul"], matmul)
        self.assertEqual(
            {name for name in wildcard if not name.startswith("__")},
            CUDA_BACKEND_PUBLIC,
        )

        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.backends.cuda.matmul")
        self.assertFalse(hasattr(matmul, "allow_fp16_accumulation"))
        self.assertFalse(hasattr(matmul, "fp32_precision"))
        self.assertFalse(
            hasattr(matmul, "allow_fp16_reduced_precision_reduction_split_k")
        )
        self.assertFalse(
            hasattr(matmul, "allow_bf16_reduced_precision_reduction_split_k")
        )

        for instance in (
            copy.copy(matmul),
            copy.deepcopy(matmul),
            pickle.loads(pickle.dumps(matmul)),
        ):
            self.assertIsNot(instance, matmul)
            self.assertIs(type(instance), cuda.cuBLASModule)
            self.assertEqual(vars(instance), {})
            self.assertEqual(matmul_states(instance), matmul_states(matmul))

        for name in (
            "_get_cublas_allow_tf32",
            "_set_cublas_allow_tf32",
            "_get_cublas_allow_fp16_reduced_precision_reduction",
            "_set_cublas_allow_fp16_reduced_precision_reduction",
            "_get_cublas_allow_bf16_reduced_precision_reduction",
            "_set_cublas_allow_bf16_reduced_precision_reduction",
        ):
            with self.subTest(private_accessor=name):
                function = getattr(torch._C, name)
                self.assertIs(type(function), types.BuiltinFunctionType)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(function.__module__, torch.tensor.__module__)
                self.assertIsNone(function.__doc__)
                self.assertIs(function.__self__, torch._C)
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                self.assertIs(pickle.loads(pickle.dumps(function)), function)
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)
                self.assertNotIn(name, torch._C.__all__)

    def test_private_accessor_binding_errors_preserve_state(self):
        self.cuda.matmul.allow_tf32 = True
        cases = (
            (
                lambda: torch._C._get_cublas_allow_tf32(None),
                "torch_rs.torch_rs._get_cublas_allow_tf32() "
                "takes no arguments (1 given)",
            ),
            (
                lambda: torch._C._get_cublas_allow_tf32(value=None),
                "torch_rs.torch_rs._get_cublas_allow_tf32() "
                "takes no keyword arguments",
            ),
            (
                lambda: torch._C._set_cublas_allow_tf32(),
                "torch_rs.torch_rs._set_cublas_allow_tf32() "
                "takes exactly one argument (0 given)",
            ),
            (
                lambda: torch._C._set_cublas_allow_tf32(True, False),
                "torch_rs.torch_rs._set_cublas_allow_tf32() "
                "takes exactly one argument (2 given)",
            ),
            (
                lambda: torch._C._set_cublas_allow_tf32(object=False),
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

        for setter_name, getter_name in (
            (
                "_set_cublas_allow_fp16_reduced_precision_reduction",
                "_get_cublas_allow_fp16_reduced_precision_reduction",
            ),
            (
                "_set_cublas_allow_bf16_reduced_precision_reduction",
                "_get_cublas_allow_bf16_reduced_precision_reduction",
            ),
        ):
            setter = getattr(torch._C, setter_name)
            getter = getattr(torch._C, getter_name)
            setter(False, False)
            for call, message in (
                (
                    lambda setter=setter: setter(),
                    "function takes at least 1 argument (0 given)",
                ),
                (
                    lambda setter=setter: setter(False, False, False),
                    "function takes at most 2 arguments (3 given)",
                ),
                (
                    lambda setter=setter: setter(allow_reduced_precision=False),
                    f"{setter_name}() takes no keyword arguments",
                ),
                (
                    lambda setter=setter: setter(False, allow_splitk=False),
                    f"{setter_name}() takes no keyword arguments",
                ),
            ):
                with self.subTest(setter=setter_name, message=message):
                    with self.assertRaises(TypeError) as raised:
                        call()
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertEqual(getter(), (False, False))

    def test_unsupported_cuda_execution_surface_remains_absent(self):
        cuda = self.cuda

        for name in (
            "Event",
            "Stream",
            "can_use_cudnn_attention",
            "can_use_efficient_attention",
            "can_use_flash_attention",
            "cublas_workspace_size",
            "cublaslt_workspace_size",
            "cudnn_sdp_enabled",
            "current_stream",
            "device_count",
            "enable_cudnn_sdp",
            "is_available",
            "sdp_kernel",
            "synchronize",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(cuda, name))

        self.assertFalse(hasattr(torch, "cuda"))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch.nn.functional, "scaled_dot_product_attention"))
        self.assertNotIn("torch_rs.cuda", sys.modules)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.cuda")
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda:0' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda:0")

    def test_unknown_attribute_assignment_and_deletion_do_not_mutate_state(self):
        matmul = self.cuda.matmul
        set_matmul_states(matmul, (True, False, True))

        with self.assertRaises(AttributeError) as raised:
            matmul.missing
        self.assertEqual(str(raised.exception), "Unknown attribute missing")

        with self.assertRaises(AttributeError) as raised:
            matmul.missing = False
        self.assertEqual(str(raised.exception), "Unknown attribute missing")

        for name in MATMUL_ATTRIBUTES:
            with self.subTest(delete=name):
                with self.assertRaises(AttributeError) as raised:
                    delattr(matmul, name)
                self.assertEqual(
                    str(raised.exception),
                    f"'cuBLASModule' object has no attribute '{name}'",
                )

        self.assertEqual(matmul_states(matmul), (True, False, True))


if __name__ == "__main__":
    unittest.main()

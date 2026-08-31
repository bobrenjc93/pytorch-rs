import copy
import importlib
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


MATMUL_ATTRS = (
    "allow_tf32",
    "allow_fp16_reduced_precision_reduction",
    "allow_bf16_reduced_precision_reduction",
)

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


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("matmul preference assignment must not request truthiness")


class CudaMatmulPreferenceTests(unittest.TestCase):
    def setUp(self):
        self.cuda = importlib.import_module("torch_rs.backends.cuda")
        self.original = self._native_state()
        self._set_native_state(False, (True, True), (True, True))

    def tearDown(self):
        self._set_native_state(*self.original)

    def _native_state(self):
        return (
            torch._C._get_cublas_allow_tf32(),
            torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
            torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
        )

    def _set_native_state(self, allow_tf32, fp16_state, bf16_state):
        torch._C._set_cublas_allow_tf32(allow_tf32)
        torch._C._set_cublas_allow_fp16_reduced_precision_reduction(*fp16_state)
        torch._C._set_cublas_allow_bf16_reduced_precision_reduction(*bf16_state)

    def _matmul_state(self, matmul):
        return {name: getattr(matmul, name) for name in MATMUL_ATTRS}

    def test_fresh_process_defaults_without_torch_import_or_cuda_probe(self):
        script = r'''
import json
import sys

before = "torch" in sys.modules
import torch_rs as torch

cuda = torch.backends.cuda
matmul = cuda.matmul
defaults = {
    name: getattr(matmul, name)
    for name in (
        "allow_tf32",
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    )
}
setattr(matmul, "allow_tf32", True)
setattr(matmul, "allow_fp16_reduced_precision_reduction", False)
setattr(matmul, "allow_bf16_reduced_precision_reduction", False)
print(json.dumps({
    "before_torch": before,
    "after_torch": "torch" in sys.modules,
    "all": cuda.__all__,
    "defaults": defaults,
    "updated": {
        name: getattr(matmul, name)
        for name in (
            "allow_tf32",
            "allow_fp16_reduced_precision_reduction",
            "allow_bf16_reduced_precision_reduction",
        )
    },
    "built": cuda.is_built(),
    "cuda": hasattr(torch, "cuda"),
    "tensor_cuda": hasattr(torch.Tensor, "cuda"),
    "compile": hasattr(torch, "compile"),
    "sdp_execution": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
    "cudnn_sdp": hasattr(cuda, "enable_cudnn_sdp"),
    "unsupported_cublas": hasattr(matmul, "allow_fp16_accumulation"),
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
                "before_torch": False,
                "after_torch": False,
                "all": CUDA_BACKEND_ALL,
                "defaults": {
                    "allow_tf32": False,
                    "allow_fp16_reduced_precision_reduction": True,
                    "allow_bf16_reduced_precision_reduction": True,
                },
                "updated": {
                    "allow_tf32": True,
                    "allow_fp16_reduced_precision_reduction": False,
                    "allow_bf16_reduced_precision_reduction": False,
                },
                "built": False,
                "cuda": False,
                "tensor_cuda": False,
                "compile": False,
                "sdp_execution": False,
                "cudnn_sdp": False,
                "unsupported_cublas": False,
            },
        )

    def test_exact_bool_assignments_are_process_global_preferences(self):
        matmul = self.cuda.matmul
        other_states = (
            self.cuda.flash_sdp_enabled(),
            self.cuda.math_sdp_enabled(),
            self.cuda.mem_efficient_sdp_enabled(),
            self.cuda.fp16_bf16_reduction_math_sdp_allowed(),
            torch.backends.cudnn.allow_tf32,
        )

        self.assertEqual(
            self._matmul_state(matmul),
            {
                "allow_tf32": False,
                "allow_fp16_reduced_precision_reduction": True,
                "allow_bf16_reduced_precision_reduction": True,
            },
        )
        for name in MATMUL_ATTRS:
            self.assertIs(type(getattr(matmul, name)), bool)
            for enabled in (False, True, True, False):
                with self.subTest(name=name, enabled=enabled):
                    setattr(matmul, name, enabled)
                    self.assertIs(getattr(matmul, name), enabled)
                    self.assertEqual(
                        other_states,
                        (
                            self.cuda.flash_sdp_enabled(),
                            self.cuda.math_sdp_enabled(),
                            self.cuda.mem_efficient_sdp_enabled(),
                            self.cuda.fp16_bf16_reduction_math_sdp_allowed(),
                            torch.backends.cudnn.allow_tf32,
                        ),
                    )
                    self.assertIs(self.cuda.is_built(), False)

        self.assertIs(torch._C._get_cublas_allow_tf32(), False)
        self.assertEqual(
            torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
            (False, True),
        )
        self.assertEqual(
            torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
            (False, True),
        )

    def test_invalid_assignments_preserve_state(self):
        invalid_values = (
            (None, "NoneType", "<class 'NoneType'>"),
            (0, "int", "<class 'int'>"),
            (1, "int", "<class 'int'>"),
            (0.0, "float", "<class 'float'>"),
            (np.bool_(True), "numpy.bool", "<class 'numpy.bool'>"),
            ("", "str", "<class 'str'>"),
            (object(), "object", "<class 'object'>"),
            (
                _RejectTruthiness(),
                "_RejectTruthiness",
                f"<class '{_RejectTruthiness.__module__}._RejectTruthiness'>",
            ),
            (torch.tensor(True), "Tensor", "<class 'torch.Tensor'>"),
            (torch.float32, "torch.dtype", "<class 'torch.dtype'>"),
            (torch.device("cpu"), "torch.device", "<class 'torch.device'>"),
            (torch.strided, "torch.layout", "<class 'torch.layout'>"),
            (torch.finfo(torch.float32), "torch.finfo", "<class 'torch.finfo'>"),
        )
        for state in (False, True):
            for name in MATMUL_ATTRS:
                setattr(self.cuda.matmul, name, state)
            for value, c_type_name, python_type_repr in invalid_values:
                with self.subTest(state=state, value_type=c_type_name):
                    with self.assertRaises(RuntimeError) as raised:
                        self.cuda.matmul.allow_tf32 = value
                    message = (
                        "set_allow_tf32_cublas expects a bool, but got "
                        f"{c_type_name}"
                    )
                    self.assertEqual(str(raised.exception), message)
                    self.assertIs(self.cuda.matmul.allow_tf32, state)

                    for name in MATMUL_ATTRS[1:]:
                        with self.assertRaises(TypeError) as reduced_raised:
                            setattr(self.cuda.matmul, name, value)
                        self.assertEqual(
                            str(reduced_raised.exception),
                            f"{name} expects a bool or a tuple/list of bools, "
                            f"but got {python_type_repr}",
                        )
                        self.assertIs(getattr(self.cuda.matmul, name), state)

        for name in MATMUL_ATTRS:
            with self.subTest(delete=name):
                setattr(self.cuda.matmul, name, True)
                with self.assertRaises(AttributeError) as raised:
                    delattr(self.cuda.matmul, name)
                self.assertEqual(
                    str(raised.exception),
                    f"'cuBLASModule' object has no attribute '{name}'",
                )
                self.assertIs(getattr(self.cuda.matmul, name), True)

    def test_reduced_precision_tuple_and_list_assignments_match_pytorch_2_13(self):
        for name in MATMUL_ATTRS[1:]:
            cases = (
                ((), TypeError, f"{name} expects at least one boolean argument", True),
                ([], TypeError, f"{name} expects at least one boolean argument", True),
                ((True,), None, None, True),
                ([True], None, None, True),
                ((False,), None, None, False),
                ([False], None, None, False),
                ((False, False), None, None, False),
                ([False, True], None, None, False),
                (
                    (True, False),
                    RuntimeError,
                    "allow_splitk=False is not supported when reduced "
                    "precision reductions are enabled",
                    True,
                ),
                (
                    [True, False],
                    RuntimeError,
                    "allow_splitk=False is not supported when reduced "
                    "precision reductions are enabled",
                    True,
                ),
                (
                    (1,),
                    TypeError,
                    f"{name} expects a bool for allow_reduced_precision, "
                    "but got <class 'int'>",
                    True,
                ),
                (
                    (True, 1),
                    TypeError,
                    f"{name} expects a bool for allow_splitk, "
                    "but got <class 'int'>",
                    True,
                ),
                (
                    (False, False, False),
                    TypeError,
                    f"{name} expects at most two boolean arguments",
                    True,
                ),
            )
            for value, error_type, message, initial in cases:
                with self.subTest(name=name, value=value):
                    setattr(self.cuda.matmul, name, initial)
                    if error_type is None:
                        setattr(self.cuda.matmul, name, value)
                        self.assertIs(getattr(self.cuda.matmul, name), value[0])
                    else:
                        with self.assertRaises(error_type) as raised:
                            setattr(self.cuda.matmul, name, value)
                        self.assertEqual(str(raised.exception), message)
                        self.assertIs(getattr(self.cuda.matmul, name), initial)

    def test_state_is_process_global_across_threads_aliases_copies_and_pickles(self):
        matmul = self.cuda.matmul
        imported = importlib.import_module("torch_rs.backends.cuda").matmul
        copied = copy.copy(matmul)
        deep_copied = copy.deepcopy(matmul)
        unpickled = pickle.loads(pickle.dumps(matmul))
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []

        self.assertIs(imported, matmul)
        self.assertIsNot(copied, matmul)
        self.assertIsNot(deep_copied, matmul)
        self.assertIsNot(unpickled, matmul)
        self.assertIs(type(copied), type(matmul))
        self.assertIs(type(deep_copied), type(matmul))
        self.assertIs(type(unpickled), type(matmul))

        def worker():
            try:
                observations.append(self._matmul_state(imported))
                copied.allow_tf32 = True
                copied.allow_fp16_reduced_precision_reduction = False
                copied.allow_bf16_reduced_precision_reduction = False
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(self._matmul_state(deep_copied))
                unpickled.allow_tf32 = False
            except BaseException as error:
                errors.append(error)
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_changed.wait(timeout=10))
        self.assertEqual(errors, [])
        self.assertEqual(
            self._matmul_state(matmul),
            {
                "allow_tf32": True,
                "allow_fp16_reduced_precision_reduction": False,
                "allow_bf16_reduced_precision_reduction": False,
            },
        )
        matmul.allow_fp16_reduced_precision_reduction = True
        matmul.allow_bf16_reduced_precision_reduction = True
        main_changed.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(
            observations,
            [
                {
                    "allow_tf32": False,
                    "allow_fp16_reduced_precision_reduction": True,
                    "allow_bf16_reduced_precision_reduction": True,
                },
                {
                    "allow_tf32": True,
                    "allow_fp16_reduced_precision_reduction": True,
                    "allow_bf16_reduced_precision_reduction": True,
                },
            ],
        )
        self.assertIs(matmul.allow_tf32, False)

    def test_reload_preserves_state_and_replaces_matmul_proxy(self):
        cuda = self.cuda
        old_matmul = cuda.matmul
        old_class = cuda.cuBLASModule
        namespace = cuda.__dict__

        old_matmul.allow_tf32 = True
        old_matmul.allow_fp16_reduced_precision_reduction = False
        old_matmul.allow_bf16_reduced_precision_reduction = False
        reloaded = importlib.reload(cuda)

        self.assertIs(reloaded, cuda)
        self.assertIs(cuda.__dict__, namespace)
        self.assertIs(torch.backends.cuda, cuda)
        self.assertIs(sys.modules[cuda.__name__], cuda)
        self.assertIsNot(cuda.matmul, old_matmul)
        self.assertIsNot(cuda.cuBLASModule, old_class)
        self.assertIs(type(cuda.matmul), cuda.cuBLASModule)
        self.assertEqual(
            self._matmul_state(cuda.matmul),
            {
                "allow_tf32": True,
                "allow_fp16_reduced_precision_reduction": False,
                "allow_bf16_reduced_precision_reduction": False,
            },
        )
        old_matmul.allow_tf32 = False
        self.assertIs(cuda.matmul.allow_tf32, False)

        for stale in (old_matmul, old_class):
            with self.subTest(stale=type(stale).__name__):
                with self.assertRaises(pickle.PicklingError) as raised:
                    pickle.dumps(stale)
                message = re.sub(
                    r"0x[0-9a-fA-F]+",
                    "0x...",
                    str(raised.exception),
                )
                self.assertEqual(
                    message,
                    "Can't pickle <class 'torch_rs.backends.cuda.cuBLASModule'>: "
                    "it's not the same object as "
                    "torch_rs.backends.cuda.cuBLASModule",
                )

    def test_metadata_exports_and_unsupported_boundaries_are_canonical(self):
        cuda = self.cuda
        matmul = cuda.matmul

        self.assertIs(torch.backends.cuda, cuda)
        self.assertIs(sys.modules["torch_rs.backends.cuda"], cuda)
        self.assertIs(type(cuda), types.ModuleType)
        self.assertIs(type(matmul), cuda.cuBLASModule)
        self.assertIsNone(matmul.__doc__)
        self.assertEqual(vars(matmul), {})
        self.assertEqual(cuda.__all__, CUDA_BACKEND_ALL)
        self.assertEqual(
            {name for name in vars(cuda) if not name.startswith("_")},
            CUDA_BACKEND_PUBLIC | {"torch"},
        )

        backend_import = {}
        class_import = {}
        matmul_import = {}
        wildcard = {}
        exec("from torch_rs.backends import cuda", backend_import)
        exec("from torch_rs.backends.cuda import cuBLASModule", class_import)
        exec("from torch_rs.backends.cuda import matmul", matmul_import)
        exec("from torch_rs.backends.cuda import *", wildcard)
        self.assertIs(backend_import["cuda"], cuda)
        self.assertIs(class_import["cuBLASModule"], cuda.cuBLASModule)
        self.assertIs(matmul_import["matmul"], matmul)
        self.assertIs(wildcard["cuBLASModule"], cuda.cuBLASModule)
        self.assertIs(wildcard["matmul"], matmul)
        self.assertEqual(
            {name for name in wildcard if not name.startswith("__")},
            CUDA_BACKEND_PUBLIC,
        )

        with self.assertRaises(ImportError):
            exec("from torch_rs.backends.cuda import allow_tf32", {})
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.backends.cuda.matmul")

        for name in (
            "__name__",
            "__all__",
            "allow_fp16_accumulation",
            "fp32_precision",
            "allow_fp16_reduced_precision_reduction_split_k",
            "allow_bf16_reduced_precision_reduction_split_k",
        ):
            with self.subTest(unsupported_attribute=name):
                with self.assertRaises(AttributeError) as raised:
                    getattr(matmul, name)
                self.assertEqual(str(raised.exception), "Unknown attribute " + name)

        for name in (
            "allow_fp16_accumulation",
            "fp32_precision",
            "allow_fp16_reduced_precision_reduction_split_k",
            "allow_bf16_reduced_precision_reduction_split_k",
        ):
            with self.subTest(unsupported_set=name):
                with self.assertRaises(AttributeError) as raised:
                    setattr(matmul, name, True)
                self.assertEqual(str(raised.exception), "Unknown attribute " + name)

        self.assertFalse(hasattr(torch, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "to"))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
        )
        self.assertFalse(hasattr(cuda, "enable_cudnn_sdp"))
        self.assertFalse(hasattr(cuda, "cudnn_sdp_enabled"))

    def test_private_accessors_are_hidden_and_atomic(self):
        accessors = (
            ("_get_cublas_allow_tf32", "_set_cublas_allow_tf32"),
            (
                "_get_cublas_allow_fp16_reduced_precision_reduction",
                "_set_cublas_allow_fp16_reduced_precision_reduction",
            ),
            (
                "_get_cublas_allow_bf16_reduced_precision_reduction",
                "_set_cublas_allow_bf16_reduced_precision_reduction",
            ),
        )
        for getter_name, setter_name in accessors:
            with self.subTest(accessor=getter_name):
                self.assertTrue(hasattr(torch._C, getter_name))
                self.assertTrue(hasattr(torch._C, setter_name))
                self.assertFalse(hasattr(torch, getter_name))
                self.assertFalse(hasattr(torch, setter_name))
                self.assertNotIn(getter_name, torch._C.__all__)
                self.assertNotIn(setter_name, torch._C.__all__)

        torch._C._set_cublas_allow_tf32(True)
        self.assertIs(self.cuda.matmul.allow_tf32, True)
        with self.assertRaises(RuntimeError):
            torch._C._set_cublas_allow_tf32(1)
        self.assertIs(self.cuda.matmul.allow_tf32, True)

        torch._C._set_cublas_allow_fp16_reduced_precision_reduction(False, False)
        self.assertEqual(
            torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
            (False, False),
        )
        with self.assertRaises(RuntimeError):
            torch._C._set_cublas_allow_fp16_reduced_precision_reduction(True, False)
        self.assertEqual(
            torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
            (False, False),
        )


if __name__ == "__main__":
    unittest.main()

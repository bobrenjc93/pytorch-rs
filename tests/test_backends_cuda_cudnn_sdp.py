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
import typing
import unittest

import numpy as np

import torch_rs as torch


CUDNN_SDP_ENABLED_DOC = """
    .. warning:: This flag is beta and subject to change.

    Returns whether cuDNN scaled dot product attention is enabled or not.
    """

ENABLE_CUDNN_SDP_DOC = """
    .. warning:: This flag is beta and subject to change.

    Enables or disables cuDNN scaled dot product attention.
    """

CUDA_BACKEND_ALL = [
    "is_built",
    "cuBLASModule",
    "is_ck_sdpa_available",
    "matmul",
    "enable_cudnn_sdp",
    "cudnn_sdp_enabled",
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
    "cudnn_sdp_enabled",
    "enable_cudnn_sdp",
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

if sys.version_info >= (3, 13):
    CUDNN_SDP_ENABLED_DOC = "\n" + inspect.cleandoc(CUDNN_SDP_ENABLED_DOC) + "\n"
    ENABLE_CUDNN_SDP_DOC = "\n" + inspect.cleandoc(ENABLE_CUDNN_SDP_DOC) + "\n"


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("enable_cudnn_sdp must not request truthiness")


class CudaCudnnSdpTests(unittest.TestCase):
    def setUp(self):
        self.cuda = importlib.import_module("torch_rs.backends.cuda")
        self.original = torch._C._get_cudnn_sdp_enabled()
        self.original_flash = torch._C._get_flash_sdp_enabled()
        self.original_math = torch._C._get_math_sdp_enabled()
        self.original_mem_efficient = torch._C._get_mem_efficient_sdp_enabled()
        self.original_reduction = (
            torch._C._get_math_sdp_allow_fp16_bf16_reduction()
        )
        self.cuda.enable_cudnn_sdp(True)

    def tearDown(self):
        self.cuda.enable_cudnn_sdp(self.original)
        self.cuda.enable_flash_sdp(self.original_flash)
        self.cuda.enable_math_sdp(self.original_math)
        self.cuda.enable_mem_efficient_sdp(self.original_mem_efficient)
        self.cuda.allow_fp16_bf16_reduction_math_sdp(self.original_reduction)

    def test_fresh_process_defaults_to_exact_true_without_cuda_probing(self):
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
flash_before = cuda.flash_sdp_enabled()
math_before = cuda.math_sdp_enabled()
mem_efficient_before = cuda.mem_efficient_sdp_enabled()
reduction_before = cuda.fp16_bf16_reduction_math_sdp_allowed()
initial = cuda.cudnn_sdp_enabled()
first = cuda.enable_cudnn_sdp(False)
disabled = cuda.cudnn_sdp_enabled()
second = cuda.enable_cudnn_sdp(True)
print(json.dumps({
    "initial": initial,
    "initial_type": type(initial).__name__,
    "first": first,
    "disabled": disabled,
    "second": second,
    "restored": cuda.cudnn_sdp_enabled(),
    "native": torch._C._get_cudnn_sdp_enabled(),
    "flash_unchanged": cuda.flash_sdp_enabled() is flash_before,
    "math_unchanged": cuda.math_sdp_enabled() is math_before,
    "mem_efficient_unchanged": (
        cuda.mem_efficient_sdp_enabled() is mem_efficient_before
    ),
    "reduction_unchanged": (
        cuda.fp16_bf16_reduction_math_sdp_allowed() is reduction_before
    ),
    "built": cuda.is_built(),
    "ck_available": cuda.is_ck_sdpa_available(),
    "flash_available": cuda.is_flash_attention_available(),
    "cudnn_available": torch.backends.cudnn.is_available(),
    "cudnn_version": torch.backends.cudnn.version(),
    "cuda": hasattr(torch, "cuda"),
    "cuda_available": torch.cuda.is_available(),
    "cuda_devices": torch.cuda.device_count(),
    "can_use_cudnn_attention": hasattr(cuda, "can_use_cudnn_attention"),
    "sdp_kernel": hasattr(cuda, "sdp_kernel"),
    "execution": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
    "compile": hasattr(torch, "compile"),
    "reference_torch_loaded": "torch" in sys.modules,
    "blocked_loaded": sorted(
        name for name in sys.modules
        if name.split(".", 1)[0] in RejectExternalRuntimeImport.blocked
    ),
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
                "first": None,
                "disabled": False,
                "second": None,
                "restored": True,
                "native": True,
                "flash_unchanged": True,
                "math_unchanged": True,
                "mem_efficient_unchanged": True,
                "reduction_unchanged": True,
                "built": False,
                "ck_available": False,
                "flash_available": False,
                "cudnn_available": False,
                "cudnn_version": None,
                "cuda": True,
                "cuda_available": False,
                "cuda_devices": 0,
                "can_use_cudnn_attention": False,
                "sdp_kernel": False,
                "execution": False,
                "compile": False,
                "reference_torch_loaded": False,
                "blocked_loaded": [],
            },
        )

    def test_repeated_exact_bool_updates_are_independent_preferences(self):
        cuda = self.cuda
        flash_state = cuda.flash_sdp_enabled()
        math_state = cuda.math_sdp_enabled()
        mem_efficient_state = cuda.mem_efficient_sdp_enabled()
        reduction_state = cuda.fp16_bf16_reduction_math_sdp_allowed()
        cudnn_backend_state = (
            torch.backends.cudnn.enabled,
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.benchmark_limit,
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.allow_tf32,
        )

        self.assertIs(cuda.cudnn_sdp_enabled(), True)
        self.assertIs(type(cuda.cudnn_sdp_enabled()), bool)
        for enabled in (False, True, True, False, False, True):
            with self.subTest(enabled=enabled):
                self.assertIs(cuda.enable_cudnn_sdp(enabled), None)
                self.assertIs(cuda.cudnn_sdp_enabled(), enabled)
                self.assertIs(torch._C._get_cudnn_sdp_enabled(), enabled)
                self.assertIs(cuda.flash_sdp_enabled(), flash_state)
                self.assertIs(cuda.math_sdp_enabled(), math_state)
                self.assertIs(
                    cuda.mem_efficient_sdp_enabled(),
                    mem_efficient_state,
                )
                self.assertIs(
                    cuda.fp16_bf16_reduction_math_sdp_allowed(),
                    reduction_state,
                )
                self.assertEqual(
                    (
                        torch.backends.cudnn.enabled,
                        torch.backends.cudnn.benchmark,
                        torch.backends.cudnn.benchmark_limit,
                        torch.backends.cudnn.deterministic,
                        torch.backends.cudnn.allow_tf32,
                    ),
                    cudnn_backend_state,
                )
                self.assertIs(torch.backends.cudnn.is_available(), False)
                self.assertIsNone(torch.backends.cudnn.version())
                self.assertIs(cuda.is_built(), False)
                self.assertIs(cuda.is_ck_sdpa_available(), False)
                self.assertIs(cuda.is_flash_attention_available(), False)

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
            self.cuda.enable_cudnn_sdp(state)
            for value, type_name in invalid_values:
                with self.subTest(state=state, value_type=type_name):
                    message = (
                        "set_sdp_use_cudnn expects a bool, but got "
                        f"%s{type_name}"
                    )
                    with self.assertRaises(RuntimeError) as raised:
                        self.cuda.enable_cudnn_sdp(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertIs(self.cuda.cudnn_sdp_enabled(), state)
                    self.assertIs(torch._C._get_cudnn_sdp_enabled(), state)
                    self.assertIs(self.cuda.is_built(), False)
                    self.assertIs(self.cuda.is_ck_sdpa_available(), False)
                    self.assertIs(
                        self.cuda.is_flash_attention_available(),
                        False,
                    )

    def test_state_is_process_global_across_threads_and_aliases(self):
        cuda = self.cuda
        imported = importlib.import_module("torch_rs.backends.cuda")
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []

        self.assertIs(imported, cuda)

        def worker():
            try:
                observations.append(cuda.cudnn_sdp_enabled())
                observations.append(imported.enable_cudnn_sdp(False))
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(imported.cudnn_sdp_enabled())
                observations.append(cuda.enable_cudnn_sdp(False))
            except BaseException as error:
                errors.append(error)
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_changed.wait(timeout=10))
        self.assertEqual(errors, [])
        self.assertIs(cuda.cudnn_sdp_enabled(), False)
        self.assertIs(cuda.enable_cudnn_sdp(True), None)
        main_changed.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [True, None, True, None])
        self.assertIs(cuda.cudnn_sdp_enabled(), False)
        self.assertIs(torch._C._get_cudnn_sdp_enabled(), False)

    def test_reload_preserves_state_and_replaces_public_functions(self):
        cuda = self.cuda
        old_getter = cuda.cudnn_sdp_enabled
        old_setter = cuda.enable_cudnn_sdp
        namespace = cuda.__dict__

        self.assertIs(old_setter(False), None)
        reloaded = importlib.reload(cuda)

        self.assertIs(reloaded, cuda)
        self.assertIs(cuda.__dict__, namespace)
        self.assertIs(torch.backends.cuda, cuda)
        self.assertIs(sys.modules[cuda.__name__], cuda)
        self.assertIsNot(cuda.cudnn_sdp_enabled, old_getter)
        self.assertIsNot(cuda.enable_cudnn_sdp, old_setter)
        self.assertIs(cuda.cudnn_sdp_enabled(), False)
        self.assertIs(cuda.enable_cudnn_sdp(True), None)
        self.assertIs(old_getter(), True)
        self.assertIs(old_setter(False), None)
        self.assertIs(cuda.cudnn_sdp_enabled(), False)
        self.assertIs(cuda.enable_cudnn_sdp(True), None)

        for name, old_function in (
            ("cudnn_sdp_enabled", old_getter),
            ("enable_cudnn_sdp", old_setter),
        ):
            with self.subTest(stale_function=name):
                with self.assertRaises(pickle.PicklingError) as raised:
                    pickle.dumps(old_function)
                message = re.sub(
                    r"0x[0-9a-fA-F]+",
                    "0x...",
                    str(raised.exception),
                )
                self.assertEqual(
                    message,
                    f"Can't pickle <function {name} at 0x...>: "
                    "it's not the same object as "
                    f"torch_rs.backends.cuda.{name}",
                )

    def test_metadata_exports_copying_and_pickling_match_public_contract(self):
        cuda = self.cuda
        getter = cuda.cudnn_sdp_enabled
        setter = cuda.enable_cudnn_sdp

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

        cases = (
            (
                getter,
                "cudnn_sdp_enabled",
                "()",
                {},
                CUDNN_SDP_ENABLED_DOC,
                ("torch", "_C", "_get_cudnn_sdp_enabled"),
            ),
            (
                setter,
                "enable_cudnn_sdp",
                "(enabled: bool)",
                {"enabled": bool},
                ENABLE_CUDNN_SDP_DOC,
                ("torch", "_C", "_set_sdp_use_cudnn"),
            ),
        )
        for function, name, signature, annotations, doc, code_names in cases:
            with self.subTest(function=name):
                self.assertIs(type(function), types.FunctionType)
                self.assertEqual(str(inspect.signature(function)), signature)
                self.assertEqual(inspect.get_annotations(function), annotations)
                self.assertEqual(typing.get_type_hints(function), annotations)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(function.__module__, "torch_rs.backends.cuda")
                self.assertIs(inspect.getmodule(function), cuda)
                self.assertEqual(function.__doc__, doc)
                self.assertIsNone(function.__defaults__)
                self.assertIsNone(function.__kwdefaults__)
                self.assertEqual(function.__dict__, {})
                self.assertFalse(hasattr(function, "__text_signature__"))
                self.assertEqual(function.__code__.co_names, code_names)
                self.assertEqual(function.__code__.co_freevars, ())
                self.assertEqual(function.__code__.co_cellvars, ())

        backend_import = {}
        getter_import = {}
        setter_import = {}
        child_wildcard = {}
        exec("from torch_rs.backends import cuda", backend_import)
        exec(
            "from torch_rs.backends.cuda import cudnn_sdp_enabled",
            getter_import,
        )
        exec(
            "from torch_rs.backends.cuda import enable_cudnn_sdp",
            setter_import,
        )
        exec("from torch_rs.backends.cuda import *", child_wildcard)
        self.assertIs(backend_import["cuda"], cuda)
        self.assertIs(getter_import["cudnn_sdp_enabled"], getter)
        self.assertIs(setter_import["enable_cudnn_sdp"], setter)
        self.assertIs(child_wildcard["cudnn_sdp_enabled"], getter)
        self.assertIs(child_wildcard["enable_cudnn_sdp"], setter)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            CUDA_BACKEND_PUBLIC,
        )

        for function in (getter, setter):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=function.__name__, protocol=protocol):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(b"torch_rs.backends.cuda", payload)
                    self.assertIs(pickle.loads(payload), function)

    def test_binding_errors_leave_state_unchanged(self):
        cuda = self.cuda
        cuda.enable_cudnn_sdp(True)
        unexpected_keyword = (
            "enable_cudnn_sdp() got an unexpected keyword argument '_enabled'"
        )
        if sys.version_info >= (3, 13):
            unexpected_keyword += ". Did you mean 'enabled'?"
        cases = (
            (
                lambda: cuda.cudnn_sdp_enabled(None),
                "cudnn_sdp_enabled() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: cuda.cudnn_sdp_enabled(enabled=True),
                "cudnn_sdp_enabled() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: cuda.enable_cudnn_sdp(),
                "enable_cudnn_sdp() missing 1 required positional argument: 'enabled'",
            ),
            (
                lambda: cuda.enable_cudnn_sdp(True, False),
                "enable_cudnn_sdp() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: cuda.enable_cudnn_sdp(_enabled=False),
                unexpected_keyword,
            ),
            (
                lambda: cuda.enable_cudnn_sdp(True, enabled=False),
                "enable_cudnn_sdp() got multiple values for argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(cuda.cudnn_sdp_enabled(), True)

        self.assertIs(cuda.enable_cudnn_sdp(enabled=False), None)
        self.assertIs(cuda.cudnn_sdp_enabled(), False)

    def test_private_accessors_and_execution_boundary(self):
        cuda = self.cuda
        flash_state = cuda.flash_sdp_enabled()
        math_state = cuda.math_sdp_enabled()
        mem_efficient_state = cuda.mem_efficient_sdp_enabled()
        reduction_state = cuda.fp16_bf16_reduction_math_sdp_allowed()

        self.assertTrue(hasattr(torch._C, "_get_cudnn_sdp_enabled"))
        self.assertTrue(hasattr(torch._C, "_set_sdp_use_cudnn"))
        self.assertFalse(hasattr(torch, "_get_cudnn_sdp_enabled"))
        self.assertFalse(hasattr(torch, "_set_sdp_use_cudnn"))
        self.assertNotIn("_get_cudnn_sdp_enabled", torch._C.__all__)
        self.assertNotIn("_set_sdp_use_cudnn", torch._C.__all__)

        self.assertIs(torch._C._set_sdp_use_cudnn(False), None)
        self.assertIs(torch._C._get_cudnn_sdp_enabled(), False)
        self.assertIs(cuda.cudnn_sdp_enabled(), False)
        self.assertIs(cuda.enable_flash_sdp(not flash_state), None)
        self.assertIs(cuda.cudnn_sdp_enabled(), False)
        self.assertIs(cuda.enable_math_sdp(not math_state), None)
        self.assertIs(cuda.cudnn_sdp_enabled(), False)
        self.assertIs(
            cuda.enable_mem_efficient_sdp(not mem_efficient_state),
            None,
        )
        self.assertIs(cuda.cudnn_sdp_enabled(), False)
        self.assertIs(
            cuda.allow_fp16_bf16_reduction_math_sdp(not reduction_state),
            None,
        )
        self.assertIs(cuda.cudnn_sdp_enabled(), False)
        self.assertIs(torch.backends.cudnn.is_available(), False)
        self.assertIsNone(torch.backends.cudnn.version())
        self.assertIs(cuda.is_built(), False)
        self.assertIs(cuda.is_ck_sdpa_available(), False)
        self.assertIs(cuda.is_flash_attention_available(), False)
        for name in (
            "SDPAParams",
            "can_use_cudnn_attention",
            "can_use_efficient_attention",
            "can_use_flash_attention",
            "sdp_kernel",
        ):
            with self.subTest(unsupported_execution_api=name):
                self.assertFalse(hasattr(cuda, name))
        self.assertFalse(
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
        )
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertFalse(hasattr(torch.Tensor, "cuda"))
        self.assertTrue(hasattr(torch.Tensor, "to"))
        with self.assertRaisesRegex(
            NotImplementedError, r"device conversions are not supported"
        ):
            torch.tensor([1.0]).to("cuda:0")
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda:0' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda:0")


if __name__ == "__main__":
    unittest.main()

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


SDP_KERNEL_DOC = """
    .. warning:: This flag is beta and subject to change.

    This context manager can be used to temporarily enable or disable any of the three backends for scaled dot product attention.
    Upon exiting the context manager, the previous state of the flags will be restored.
    """

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
    "sdp_kernel",
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
    "sdp_kernel",
}

if sys.version_info >= (3, 13):
    # CPython 3.13+ cleans function docstring indentation while preserving
    # the leading and terminating newlines.
    SDP_KERNEL_DOC = "\n" + inspect.cleandoc(SDP_KERNEL_DOC) + "\n"


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("sdp_kernel must not request truthiness")


class _ContextBodyError(Exception):
    pass


def sdp_states(cuda):
    return (
        cuda.flash_sdp_enabled(),
        cuda.math_sdp_enabled(),
        cuda.mem_efficient_sdp_enabled(),
    )


def set_sdp_states(cuda, states):
    flash, math, mem_efficient = states
    cuda.enable_flash_sdp(flash)
    cuda.enable_math_sdp(math)
    cuda.enable_mem_efficient_sdp(mem_efficient)


class CudaSdpKernelTests(unittest.TestCase):
    def setUp(self):
        self.cuda = importlib.import_module("torch_rs.backends.cuda")
        self.original = sdp_states(self.cuda)
        set_sdp_states(self.cuda, (True, True, True))

    def tearDown(self):
        set_sdp_states(self.cuda, self.original)

    def test_default_explicit_nested_and_exceptional_restoration(self):
        cuda = self.cuda

        for initial in (
            (True, True, True),
            (False, False, False),
            (True, False, True),
        ):
            with self.subTest(initial=initial, mode="default"):
                set_sdp_states(cuda, initial)
                context = cuda.sdp_kernel()
                self.assertEqual(sdp_states(cuda), initial)
                with context as entered:
                    self.assertEqual(entered, {})
                    self.assertEqual(sdp_states(cuda), (True, True, True))
                    self.assertIs(cuda.is_built(), False)
                    self.assertIs(cuda.is_ck_sdpa_available(), False)
                    self.assertIs(cuda.is_flash_attention_available(), False)
                self.assertEqual(sdp_states(cuda), initial)

            with self.subTest(initial=initial, mode="explicit"):
                set_sdp_states(cuda, initial)
                context = cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=True,
                    enable_mem_efficient=False,
                )
                self.assertEqual(sdp_states(cuda), initial)
                self.assertEqual(context.__enter__(), {})
                self.assertEqual(sdp_states(cuda), (False, True, False))
                self.assertIs(context.__exit__(None, None, None), False)
                self.assertEqual(sdp_states(cuda), initial)

        set_sdp_states(cuda, (True, True, True))
        with cuda.sdp_kernel(False, True, False) as outer:
            self.assertEqual(outer, {})
            self.assertEqual(sdp_states(cuda), (False, True, False))
            with cuda.sdp_kernel(True, False, True) as inner:
                self.assertEqual(inner, {})
                self.assertEqual(sdp_states(cuda), (True, False, True))
            self.assertEqual(sdp_states(cuda), (False, True, False))
        self.assertEqual(sdp_states(cuda), (True, True, True))

        marker = _ContextBodyError("body failed")
        with self.assertRaises(_ContextBodyError) as raised:
            with cuda.sdp_kernel(False, False, False) as entered:
                self.assertEqual(entered, {})
                self.assertEqual(sdp_states(cuda), (False, False, False))
                raise marker
        self.assertIs(raised.exception, marker)
        self.assertEqual(sdp_states(cuda), (True, True, True))

    def test_strict_boolean_validation_is_deferred_until_entry(self):
        cuda = self.cuda
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

        for state in ((False, True, False), (True, False, True)):
            for position in range(3):
                for value, type_name in invalid_values:
                    with self.subTest(
                        state=state,
                        position=position,
                        value_type=type_name,
                    ):
                        set_sdp_states(cuda, state)
                        args = [False, True, False]
                        args[position] = value
                        context = cuda.sdp_kernel(*args)
                        self.assertEqual(sdp_states(cuda), state)
                        message = (
                            "set_sdp_use_math expects a bool, but got "
                            f"{type_name}"
                        )
                        with self.assertRaises(RuntimeError) as raised:
                            context.__enter__()
                        self.assertEqual(str(raised.exception), message)
                        self.assertEqual(raised.exception.args, (message,))
                        self.assertEqual(sdp_states(cuda), state)

    def test_binding_errors_leave_state_unchanged(self):
        cuda = self.cuda
        set_sdp_states(cuda, (True, False, True))
        unexpected_keyword = (
            "sdp_kernel() got an unexpected keyword argument '_enable_flash'"
        )
        if sys.version_info >= (3, 13):
            unexpected_keyword += ". Did you mean 'enable_flash'?"
        cases = (
            (
                lambda: cuda.sdp_kernel(True, True, True, True),
                "sdp_kernel() takes from 0 to 3 positional arguments but 4 were given",
            ),
            (
                lambda: cuda.sdp_kernel(_enable_flash=True),
                unexpected_keyword,
            ),
            (
                lambda: cuda.sdp_kernel(True, enable_flash=False),
                "sdp_kernel() got multiple values for argument 'enable_flash'",
            ),
            (
                lambda: cuda.sdp_kernel(enable_cudnn=True),
                "sdp_kernel() got an unexpected keyword argument 'enable_cudnn'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(sdp_states(cuda), (True, False, True))

    def test_context_is_a_reusable_decorator_factory(self):
        cuda = self.cuda
        observations = []

        @cuda.sdp_kernel(False, True, False)
        def decorated(value):
            observations.append(sdp_states(cuda))
            if value == "raise":
                raise _ContextBodyError("decorated body failed")
            return value

        self.assertEqual(decorated("first"), "first")
        self.assertEqual(sdp_states(cuda), (True, True, True))
        self.assertEqual(decorated("second"), "second")
        self.assertEqual(sdp_states(cuda), (True, True, True))
        with self.assertRaisesRegex(_ContextBodyError, "decorated body failed"):
            decorated("raise")
        self.assertEqual(
            observations,
            [(False, True, False), (False, True, False), (False, True, False)],
        )
        self.assertEqual(sdp_states(cuda), (True, True, True))

    def test_state_changes_are_process_global_across_threads(self):
        cuda = self.cuda
        worker_entered = threading.Event()
        main_context_exited = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                with cuda.sdp_kernel(False, False, False) as entered:
                    observations.append(("worker-enter", entered, sdp_states(cuda)))
                    worker_entered.set()
                    if not main_context_exited.wait(timeout=10):
                        raise RuntimeError("timed out waiting for main context")
                    observations.append(("worker-resume", sdp_states(cuda)))
                observations.append(("worker-exit", sdp_states(cuda)))
            except BaseException as error:
                errors.append(error)
                worker_entered.set()

        thread = threading.Thread(target=worker)
        thread.start()
        try:
            self.assertTrue(worker_entered.wait(timeout=10))
            self.assertEqual(errors, [])
            self.assertEqual(sdp_states(cuda), (False, False, False))
            with cuda.sdp_kernel(True, True, True) as entered:
                self.assertEqual(entered, {})
                self.assertEqual(sdp_states(cuda), (True, True, True))
            self.assertEqual(sdp_states(cuda), (False, False, False))
        finally:
            main_context_exited.set()
            thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(
            observations,
            [
                ("worker-enter", {}, (False, False, False)),
                ("worker-resume", (False, False, False)),
                ("worker-exit", (True, True, True)),
            ],
        )
        self.assertEqual(sdp_states(cuda), (True, True, True))

    def test_metadata_imports_copying_and_pickling(self):
        cuda = self.cuda
        function = cuda.sdp_kernel
        wrapped = function.__wrapped__

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

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            (
                "(enable_flash: bool = True, enable_math: bool = True, "
                "enable_mem_efficient: bool = True)"
            ),
        )
        self.assertEqual(
            inspect.get_annotations(function),
            {
                "enable_flash": bool,
                "enable_math": bool,
                "enable_mem_efficient": bool,
            },
        )
        self.assertEqual(
            typing.get_type_hints(function),
            {
                "enable_flash": bool,
                "enable_math": bool,
                "enable_mem_efficient": bool,
            },
        )
        self.assertEqual(function.__name__, "sdp_kernel")
        self.assertEqual(function.__qualname__, "sdp_kernel")
        self.assertEqual(function.__module__, "torch_rs.backends.cuda")
        self.assertIs(inspect.getmodule(function), cuda)
        self.assertEqual(function.__doc__, SDP_KERNEL_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {"__wrapped__": wrapped})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__code__.co_freevars, ("func",))
        self.assertEqual(function.__code__.co_cellvars, ())

        self.assertIs(type(wrapped), types.FunctionType)
        self.assertEqual(inspect.signature(wrapped), inspect.signature(function))
        self.assertEqual(
            inspect.get_annotations(wrapped),
            {
                "enable_flash": bool,
                "enable_math": bool,
                "enable_mem_efficient": bool,
            },
        )
        self.assertEqual(wrapped.__name__, "sdp_kernel")
        self.assertEqual(wrapped.__qualname__, "sdp_kernel")
        self.assertEqual(wrapped.__module__, "torch_rs.backends.cuda")
        self.assertEqual(wrapped.__doc__, SDP_KERNEL_DOC)
        self.assertEqual(wrapped.__defaults__, (True, True, True))
        self.assertIsNone(wrapped.__kwdefaults__)
        self.assertEqual(wrapped.__dict__, {})

        backend_import = {}
        function_import = {}
        child_wildcard = {}
        exec("from torch_rs.backends import cuda", backend_import)
        exec("from torch_rs.backends.cuda import sdp_kernel", function_import)
        exec("from torch_rs.backends.cuda import *", child_wildcard)
        self.assertIs(backend_import["cuda"], cuda)
        self.assertIs(function_import["sdp_kernel"], function)
        self.assertIs(child_wildcard["sdp_kernel"], function)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            CUDA_BACKEND_PUBLIC,
        )

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="function", protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.backends.cuda", payload)
                self.assertIs(pickle.loads(payload), function)

        context = function(False, True, False)
        self.assertEqual(type(context).__module__, "contextlib")
        self.assertEqual(type(context).__qualname__, "_GeneratorContextManager")
        self.assertEqual(context.__doc__, SDP_KERNEL_DOC)
        self.assertIs(context.func, wrapped)
        self.assertEqual(context.args, (False, True, False))
        self.assertEqual(context.kwds, {})
        copied = copy.copy(context)
        self.assertIsNot(copied, context)
        self.assertIs(copied.gen, context.gen)
        with self.assertRaisesRegex(TypeError, "cannot pickle 'generator' object"):
            copy.deepcopy(context)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(kind="context", protocol=protocol):
                with self.assertRaisesRegex(
                    TypeError,
                    "cannot pickle 'generator' object",
                ):
                    pickle.dumps(context, protocol=protocol)

    def test_reload_preserves_state_and_existing_contexts(self):
        cuda = self.cuda
        old_function = cuda.sdp_kernel
        old_wrapped = old_function.__wrapped__
        namespace = cuda.__dict__
        active_context = old_function(False, True, False)

        self.assertEqual(active_context.__enter__(), {})
        self.assertEqual(sdp_states(cuda), (False, True, False))
        reloaded = importlib.reload(cuda)

        self.assertIs(reloaded, cuda)
        self.assertIs(cuda.__dict__, namespace)
        self.assertIs(torch.backends.cuda, cuda)
        self.assertIs(sys.modules[cuda.__name__], cuda)
        self.assertIsNot(cuda.sdp_kernel, old_function)
        self.assertIsNot(cuda.sdp_kernel.__wrapped__, old_wrapped)
        self.assertEqual(sdp_states(cuda), (False, True, False))
        self.assertIs(active_context.__exit__(None, None, None), False)
        self.assertEqual(sdp_states(cuda), (True, True, True))

        for function in (old_function, cuda.sdp_kernel):
            with self.subTest(function=function):
                with function(False, False, True) as entered:
                    self.assertEqual(entered, {})
                    self.assertEqual(sdp_states(cuda), (False, False, True))
                self.assertEqual(sdp_states(cuda), (True, True, True))

        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_function)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function sdp_kernel at 0x...>: "
            "it's not the same object as torch_rs.backends.cuda.sdp_kernel",
        )
        self.assertIs(pickle.loads(pickle.dumps(cuda.sdp_kernel)), cuda.sdp_kernel)

    def test_subprocess_import_does_not_probe_cuda_or_import_pytorch(self):
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
from torch_rs.backends import cuda
from torch_rs.backends.cuda import sdp_kernel

with sdp_kernel(False, True, False) as entered:
    inside = [
        cuda.flash_sdp_enabled(),
        cuda.math_sdp_enabled(),
        cuda.mem_efficient_sdp_enabled(),
    ]

print(json.dumps({
    "same_module": torch.backends.cuda is cuda,
    "same_function": cuda.sdp_kernel is sdp_kernel,
    "entered": entered,
    "inside": inside,
    "after": [
        cuda.flash_sdp_enabled(),
        cuda.math_sdp_enabled(),
        cuda.mem_efficient_sdp_enabled(),
    ],
    "cudnn_sdp": hasattr(cuda, "cudnn_sdp_enabled"),
    "sdpa_params": hasattr(cuda, "SDPAParams"),
    "can_use_flash": hasattr(cuda, "can_use_flash_attention"),
    "can_use_efficient": hasattr(cuda, "can_use_efficient_attention"),
    "can_use_cudnn": hasattr(cuda, "can_use_cudnn_attention"),
    "sdpa_execution": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
    "cuda": hasattr(torch, "cuda"),
    "compile": hasattr(torch, "compile"),
    "reference_torch_loaded": "torch" in sys.modules,
    "blocked_runtime_loaded": any(
        name.split(".", 1)[0] in RejectExternalRuntimeImport.blocked
        for name in sys.modules
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
                "same_module": True,
                "same_function": True,
                "entered": {},
                "inside": [False, True, False],
                "after": [True, True, True],
                "cudnn_sdp": False,
                "sdpa_params": False,
                "can_use_flash": False,
                "can_use_efficient": False,
                "can_use_cudnn": False,
                "sdpa_execution": False,
                "cuda": False,
                "compile": False,
                "reference_torch_loaded": False,
                "blocked_runtime_loaded": False,
            },
        )

    def test_sdp_execution_and_cuda_controls_remain_unsupported(self):
        cuda = self.cuda

        for name in (
            "SDPAParams",
            "can_use_cudnn_attention",
            "can_use_efficient_attention",
            "can_use_flash_attention",
            "cudnn_sdp_enabled",
            "enable_cudnn_sdp",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(cuda, name))

        self.assertFalse(hasattr(torch, "cuda"))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch.nn.functional, "scaled_dot_product_attention"))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda:0' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda:0")


if __name__ == "__main__":
    unittest.main()

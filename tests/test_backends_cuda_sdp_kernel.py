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
import warnings

import numpy as np

import torch_rs as torch


SDP_KERNEL_DOC = """
    .. warning:: This flag is beta and subject to change.

    This context manager can be used to temporarily enable or disable any of the three backends for scaled dot product attention.
    Upon exiting the context manager, the previous state of the flags will be restored.
    """

SDP_KERNEL_DEPRECATION = (
    "`torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this "
    "context manager will be removed. Please see "
    "`torch.nn.attention.sdpa_kernel()` for the new context manager, with "
    "updated signature."
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


class _BoolProbe:
    def __init__(self, label, result=True, record=None, error=None):
        self.label = label
        self.result = result
        self.record = record
        self.error = error

    def __bool__(self):
        if self.record is not None:
            self.record.append(self.label)
        if self.error is not None:
            raise self.error
        return self.result


class _TruthinessError(Exception):
    pass


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
        self.original_reduction = (
            self.cuda.fp16_bf16_reduction_math_sdp_allowed()
        )
        set_sdp_states(self.cuda, (True, True, True))
        self.cuda.allow_fp16_bf16_reduction_math_sdp(False)

    def tearDown(self):
        set_sdp_states(self.cuda, self.original)
        self.cuda.allow_fp16_bf16_reduction_math_sdp(self.original_reduction)

    def context(self, *args, **kwargs):
        return self.context_for(self.cuda.sdp_kernel, *args, **kwargs)

    def context_for(self, function, *args, **kwargs):
        with self.assertWarnsRegex(
            FutureWarning,
            re.escape(SDP_KERNEL_DEPRECATION),
        ):
            return function(*args, **kwargs)

    def context_ignoring_deprecation(self, function, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return function(*args, **kwargs)

    def test_default_explicit_nested_and_exceptional_restoration(self):
        cuda = self.cuda

        for initial in (
            (False, False, False),
            (True, False, True),
        ):
            with self.subTest(initial=initial, mode="default"):
                set_sdp_states(cuda, initial)
                context = self.context()
                self.assertEqual(sdp_states(cuda), initial)
                with context as entered:
                    self.assertEqual(entered, {})
                    self.assertEqual(sdp_states(cuda), (True, True, True))
                    self.assertIs(cuda.is_built(), False)
                self.assertEqual(sdp_states(cuda), initial)

            with self.subTest(initial=initial, mode="explicit"):
                target = (False, True, False)
                set_sdp_states(cuda, initial)
                context = self.context(*target)
                self.assertEqual(sdp_states(cuda), initial)
                self.assertEqual(context.__enter__(), {})
                self.assertEqual(sdp_states(cuda), target)
                self.assertIs(context.__exit__(None, None, None), False)
                self.assertEqual(sdp_states(cuda), initial)

        set_sdp_states(cuda, (True, True, True))
        with self.context(False, False, True) as outer:
            self.assertEqual(outer, {})
            self.assertEqual(sdp_states(cuda), (False, False, True))
            with self.context(True, False, False) as inner:
                self.assertEqual(inner, {})
                self.assertEqual(sdp_states(cuda), (True, False, False))
            self.assertEqual(sdp_states(cuda), (False, False, True))
        self.assertEqual(sdp_states(cuda), (True, True, True))

        marker = _ContextBodyError("body failed")
        with self.assertRaises(_ContextBodyError) as raised:
            with self.context(False, True, False) as entered:
                self.assertEqual(entered, {})
                self.assertEqual(sdp_states(cuda), (False, True, False))
                raise marker
        self.assertIs(raised.exception, marker)
        self.assertEqual(sdp_states(cuda), (True, True, True))

        with self.context(False, False, False):
            set_sdp_states(cuda, (True, False, True))
        self.assertEqual(sdp_states(cuda), (True, True, True))

    def test_enable_cudnn_argument_is_truth_evaluated_but_does_not_add_cudnn_state(
        self,
    ):
        cuda = self.cuda
        set_sdp_states(cuda, (False, False, False))

        with self.context(enable_cudnn=False) as entered:
            self.assertEqual(entered, {})
            self.assertEqual(sdp_states(cuda), (True, True, True))
            self.assertFalse(hasattr(cuda, "cudnn_sdp_enabled"))
            self.assertFalse(hasattr(cuda, "enable_cudnn_sdp"))
        self.assertEqual(sdp_states(cuda), (False, False, False))

        with self.context(enable_cudnn=None) as entered:
            self.assertEqual(entered, {})
            self.assertEqual(sdp_states(cuda), (True, True, True))
        self.assertEqual(sdp_states(cuda), (False, False, False))

    def test_arguments_use_truthiness_on_entry(self):
        cuda = self.cuda
        cases = (
            ({"enable_flash": 1}, (True, True, True)),
            ({"enable_flash": 0}, (False, True, True)),
            ({"enable_flash": None}, (False, True, True)),
            ({"enable_math": 1}, (True, True, True)),
            ({"enable_math": 0}, (True, False, True)),
            ({"enable_math": None}, (True, False, True)),
            ({"enable_mem_efficient": object()}, (True, True, True)),
            ({"enable_mem_efficient": []}, (True, True, False)),
            ({"enable_mem_efficient": None}, (True, True, False)),
            ({"enable_cudnn": object()}, (True, True, True)),
            ({"enable_cudnn": 0}, (True, True, True)),
            (
                {
                    "enable_flash": np.bool_(False),
                    "enable_math": "",
                    "enable_mem_efficient": (1,),
                    "enable_cudnn": None,
                },
                (False, False, True),
            ),
        )

        for kwargs, requested in cases:
            with self.subTest(kwargs=kwargs):
                before = (False, True, False)
                set_sdp_states(cuda, before)
                context = self.context(**kwargs)
                self.assertEqual(sdp_states(cuda), before)
                with context as entered:
                    self.assertEqual(entered, {})
                    self.assertEqual(sdp_states(cuda), requested)
                self.assertEqual(sdp_states(cuda), before)

    def test_truthiness_order_and_errors_leave_state_unchanged(self):
        cuda = self.cuda
        order = []
        before = (False, True, False)
        set_sdp_states(cuda, before)
        context = self.context(
            enable_flash=_BoolProbe("flash", result=False, record=order),
            enable_math=_BoolProbe("math", result=False, record=order),
            enable_mem_efficient=_BoolProbe("mem_efficient", result=True, record=order),
            enable_cudnn=_BoolProbe("cudnn", result=False, record=order),
        )
        self.assertEqual(sdp_states(cuda), before)
        with context:
            self.assertEqual(order, ["flash", "mem_efficient", "math", "cudnn"])
            self.assertEqual(sdp_states(cuda), (False, False, True))
        self.assertEqual(sdp_states(cuda), before)

        parameters = (
            "enable_flash",
            "enable_math",
            "enable_mem_efficient",
            "enable_cudnn",
        )

        for parameter in parameters:
            with self.subTest(parameter=parameter):
                error = _TruthinessError(f"{parameter} truthiness failed")
                before = (False, True, False)
                set_sdp_states(cuda, before)
                context = self.context(**{parameter: _BoolProbe(parameter, error=error)})
                self.assertEqual(sdp_states(cuda), before)
                with self.assertRaises(_TruthinessError) as raised:
                    context.__enter__()
                self.assertIs(raised.exception, error)
                self.assertEqual(sdp_states(cuda), before)

    def test_binding_errors_leave_state_unchanged(self):
        cuda = self.cuda
        before = (True, False, True)
        set_sdp_states(cuda, before)
        unexpected_keyword = (
            "sdp_kernel() got an unexpected keyword argument '_enabled'"
        )
        if sys.version_info >= (3, 13):
            unexpected_keyword += ". Did you mean 'enable_flash'?"
        cases = (
            (
                lambda: self.context_ignoring_deprecation(
                    cuda.sdp_kernel,
                    True,
                    True,
                    True,
                    True,
                    True,
                ),
                "sdp_kernel() takes from 0 to 4 positional arguments but 5 were given",
            ),
            (
                lambda: self.context_ignoring_deprecation(
                    cuda.sdp_kernel,
                    _enabled=False,
                ),
                unexpected_keyword,
            ),
            (
                lambda: self.context_ignoring_deprecation(
                    cuda.sdp_kernel,
                    True,
                    enable_flash=False,
                ),
                "sdp_kernel() got multiple values for argument 'enable_flash'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(sdp_states(cuda), before)

        context = self.context(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
            enable_cudnn=False,
        )
        self.assertEqual(sdp_states(cuda), before)
        with context:
            self.assertEqual(sdp_states(cuda), (False, True, False))
        self.assertEqual(sdp_states(cuda), before)

    def test_deprecation_warning_is_emitted_when_context_is_created(self):
        cuda = self.cuda
        before = (False, True, False)
        set_sdp_states(cuda, before)

        with self.assertWarnsRegex(
            FutureWarning,
            re.escape(SDP_KERNEL_DEPRECATION),
        ) as captured:
            context = cuda.sdp_kernel()
        self.assertEqual(str(captured.warning), SDP_KERNEL_DEPRECATION)
        self.assertEqual(sdp_states(cuda), before)

        with context:
            self.assertEqual(sdp_states(cuda), (True, True, True))
        self.assertEqual(sdp_states(cuda), before)

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            with self.assertRaises(FutureWarning) as raised:
                cuda.sdp_kernel()
        self.assertEqual(str(raised.exception), SDP_KERNEL_DEPRECATION)
        self.assertEqual(sdp_states(cuda), before)

    def test_state_is_process_global_across_threads_and_aliases(self):
        cuda = self.cuda
        imported = importlib.import_module("torch_rs.backends.cuda")
        worker_entered = threading.Event()
        main_context_exited = threading.Event()
        observations = []
        errors = []

        self.assertIs(imported, cuda)
        set_sdp_states(cuda, (True, True, True))

        def worker():
            try:
                with self.context_ignoring_deprecation(
                    imported.sdp_kernel,
                    False,
                    False,
                    False,
                ) as entered:
                    observations.append(("worker-enter", entered, sdp_states(cuda)))
                    worker_entered.set()
                    if not main_context_exited.wait(timeout=10):
                        raise RuntimeError("timed out waiting for main context")
                    observations.append(("worker-resume", sdp_states(imported)))
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
            with self.context(True, False, True) as entered:
                self.assertEqual(entered, {})
                self.assertEqual(sdp_states(cuda), (True, False, True))
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

    def test_metadata_imports_copying_and_pickling_match_public_contract(self):
        cuda = self.cuda
        function = cuda.sdp_kernel
        wrapped = function.__wrapped__
        original = wrapped.__wrapped__

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
            "(enable_flash: bool = True, enable_math: bool = True, "
            "enable_mem_efficient: bool = True, enable_cudnn: bool = True)",
        )
        self.assertEqual(
            inspect.get_annotations(function),
            {
                "enable_flash": bool,
                "enable_math": bool,
                "enable_mem_efficient": bool,
                "enable_cudnn": bool,
            },
        )
        self.assertEqual(typing.get_type_hints(function), inspect.get_annotations(function))
        self.assertEqual(function.__name__, "sdp_kernel")
        self.assertEqual(function.__qualname__, "sdp_kernel")
        self.assertEqual(function.__module__, "torch_rs.backends.cuda")
        self.assertIs(inspect.getmodule(function), cuda)
        self.assertEqual(function.__doc__, SDP_KERNEL_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(
            function.__dict__,
            {"__wrapped__": wrapped, "__deprecated__": SDP_KERNEL_DEPRECATION},
        )
        self.assertEqual(function.__deprecated__, SDP_KERNEL_DEPRECATION)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__code__.co_names, ("_GeneratorContextManager",))
        self.assertEqual(function.__code__.co_freevars, ("func",))
        self.assertEqual(function.__code__.co_cellvars, ())

        self.assertIs(type(wrapped), types.FunctionType)
        self.assertEqual(inspect.signature(wrapped), inspect.signature(function))
        self.assertEqual(inspect.get_annotations(wrapped), inspect.get_annotations(function))
        self.assertEqual(wrapped.__name__, "sdp_kernel")
        self.assertEqual(wrapped.__qualname__, "sdp_kernel")
        self.assertEqual(wrapped.__module__, "torch_rs.backends.cuda")
        self.assertEqual(wrapped.__doc__, SDP_KERNEL_DOC)
        self.assertIsNone(wrapped.__defaults__)
        self.assertIsNone(wrapped.__kwdefaults__)
        self.assertEqual(
            wrapped.__dict__,
            {"__wrapped__": original, "__deprecated__": SDP_KERNEL_DEPRECATION},
        )
        self.assertEqual(wrapped.__deprecated__, SDP_KERNEL_DEPRECATION)

        self.assertIs(type(original), types.FunctionType)
        self.assertEqual(inspect.signature(original), inspect.signature(function))
        self.assertEqual(inspect.get_annotations(original), inspect.get_annotations(function))
        self.assertEqual(original.__name__, "sdp_kernel")
        self.assertEqual(original.__qualname__, "sdp_kernel")
        self.assertEqual(original.__module__, "torch_rs.backends.cuda")
        self.assertEqual(original.__doc__, SDP_KERNEL_DOC)
        self.assertEqual(original.__defaults__, (True, True, True, True))
        self.assertIsNone(original.__kwdefaults__)
        self.assertEqual(original.__dict__, {})
        self.assertIn("enable_flash_sdp", original.__code__.co_names)
        self.assertIn("enable_math_sdp", original.__code__.co_names)
        self.assertIn("enable_mem_efficient_sdp", original.__code__.co_names)

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

        context = self.context_for(function, False, True, False, False)
        self.assertEqual(type(context).__module__, "contextlib")
        self.assertEqual(type(context).__qualname__, "_GeneratorContextManager")
        self.assertEqual(context.__doc__, SDP_KERNEL_DOC)
        self.assertIs(context.func, wrapped)
        self.assertEqual(context.args, (False, True, False, False))
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
        active_context = self.context_for(old_function, False, False, True)

        self.assertEqual(active_context.__enter__(), {})
        self.assertEqual(sdp_states(cuda), (False, False, True))
        reloaded = importlib.reload(cuda)

        self.assertIs(reloaded, cuda)
        self.assertIs(cuda.__dict__, namespace)
        self.assertIs(torch.backends.cuda, cuda)
        self.assertIs(sys.modules[cuda.__name__], cuda)
        self.assertIsNot(cuda.sdp_kernel, old_function)
        self.assertIsNot(cuda.sdp_kernel.__wrapped__, old_wrapped)
        self.assertEqual(sdp_states(cuda), (False, False, True))
        self.assertIs(active_context.__exit__(None, None, None), False)
        self.assertEqual(sdp_states(cuda), (True, True, True))

        for function in (old_function, cuda.sdp_kernel):
            with self.subTest(function=function):
                with self.context_for(function, False, True, False) as entered:
                    self.assertEqual(entered, {})
                    self.assertEqual(sdp_states(cuda), (False, True, False))
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

    def test_subprocess_import_does_not_probe_or_import_external_runtimes(self):
        script = r'''
import json
import os
import sys
import warnings

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

cuda.enable_flash_sdp(False)
cuda.enable_math_sdp(False)
cuda.enable_mem_efficient_sdp(False)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    default_context = sdp_kernel()
with default_context as entered:
    default_state = [
        cuda.flash_sdp_enabled(),
        cuda.math_sdp_enabled(),
        cuda.mem_efficient_sdp_enabled(),
    ]
after_default = [
    cuda.flash_sdp_enabled(),
    cuda.math_sdp_enabled(),
    cuda.mem_efficient_sdp_enabled(),
]
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    explicit_context = sdp_kernel(False, True, False, enable_cudnn=False)
with explicit_context as explicit_entered:
    explicit_state = [
        cuda.flash_sdp_enabled(),
        cuda.math_sdp_enabled(),
        cuda.mem_efficient_sdp_enabled(),
    ]
print(json.dumps({
    "same_module": torch.backends.cuda is cuda,
    "same_function": cuda.sdp_kernel is sdp_kernel,
    "entered": entered,
    "explicit_entered": explicit_entered,
    "default_state": default_state,
    "after_default": after_default,
    "explicit_state": explicit_state,
    "restored": [
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
    "cuda_module": hasattr(torch, "cuda"),
    "compile": hasattr(torch, "compile"),
    "reference_torch_loaded": "torch" in sys.modules,
    "blocked_loaded": [
        name for name in sys.modules
        if name.split(".", 1)[0] in RejectExternalRuntimeImport.blocked
    ],
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
                "explicit_entered": {},
                "default_state": [True, True, True],
                "after_default": [False, False, False],
                "explicit_state": [False, True, False],
                "restored": [False, False, False],
                "cudnn_sdp": False,
                "sdpa_params": False,
                "can_use_flash": False,
                "can_use_efficient": False,
                "can_use_cudnn": False,
                "sdpa_execution": False,
                "cuda_module": False,
                "compile": False,
                "reference_torch_loaded": False,
                "blocked_loaded": [],
            },
        )


if __name__ == "__main__":
    unittest.main()

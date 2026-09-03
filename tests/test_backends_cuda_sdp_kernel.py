import copy
import importlib
import inspect
import itertools
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

import torch_rs as torch


SDP_KERNEL_DOC = """
    .. warning:: This flag is beta and subject to change.

    This context manager can be used to temporarily enable or disable any of the three backends for scaled dot product attention.
    Upon exiting the context manager, the previous state of the flags will be restored.
    """

DEPRECATION_MESSAGE = (
    "`torch.backends.cuda.sdp_kernel()` is deprecated. "
    "In the future, this context manager will be removed. "
    "Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, "
    "with updated signature."
)

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
    "sdp_kernel",
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
    "sdp_kernel",
}

if sys.version_info >= (3, 13):
    SDP_KERNEL_DOC = "\n" + inspect.cleandoc(SDP_KERNEL_DOC) + "\n"


class _ContextBodyError(Exception):
    pass


class _TruthProbe:
    def __init__(self, name, log, result=True, error=None):
        self.name = name
        self.log = log
        self.result = result
        self.error = error

    def __bool__(self):
        self.log.append(self.name)
        if self.error is not None:
            raise self.error
        return self.result


class CudaSdpKernelTests(unittest.TestCase):
    def setUp(self):
        self.cuda = importlib.import_module("torch_rs.backends.cuda")
        self.original = self.states()
        self.original_reduction = (
            torch._C._get_math_sdp_allow_fp16_bf16_reduction()
        )
        self.set_states((True, True, True, True))
        self.cuda.allow_fp16_bf16_reduction_math_sdp(False)

    def tearDown(self):
        self.set_states(self.original)
        self.cuda.allow_fp16_bf16_reduction_math_sdp(self.original_reduction)

    def states(self):
        cuda = self.cuda
        return (
            cuda.flash_sdp_enabled(),
            cuda.math_sdp_enabled(),
            cuda.mem_efficient_sdp_enabled(),
            cuda.cudnn_sdp_enabled(),
        )

    def set_states(self, states):
        flash, math, mem_efficient, cudnn = states
        self.cuda.enable_flash_sdp(flash)
        self.cuda.enable_math_sdp(math)
        self.cuda.enable_mem_efficient_sdp(mem_efficient)
        self.cuda.enable_cudnn_sdp(cudnn)

    def assert_deprecation_warning(self, caught):
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, FutureWarning)
        self.assertEqual(str(caught[0].message), DEPRECATION_MESSAGE)

    def sdp_context(self, *args, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            context = self.cuda.sdp_kernel(*args, **kwargs)
        self.assert_deprecation_warning(caught)
        return context

    def test_fresh_process_defaults_to_all_enabled_without_sdpa_execution(self):
        script = r'''
import json
import warnings

import torch_rs as torch

cuda = torch.backends.cuda
cuda.enable_flash_sdp(False)
cuda.enable_math_sdp(False)
cuda.enable_mem_efficient_sdp(False)
cuda.enable_cudnn_sdp(False)
before = (
    cuda.flash_sdp_enabled(),
    cuda.math_sdp_enabled(),
    cuda.mem_efficient_sdp_enabled(),
    cuda.cudnn_sdp_enabled(),
)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    context = cuda.sdp_kernel()
after_create = (
    cuda.flash_sdp_enabled(),
    cuda.math_sdp_enabled(),
    cuda.mem_efficient_sdp_enabled(),
    cuda.cudnn_sdp_enabled(),
)
with context as entered:
    inside = (
        cuda.flash_sdp_enabled(),
        cuda.math_sdp_enabled(),
        cuda.mem_efficient_sdp_enabled(),
        cuda.cudnn_sdp_enabled(),
    )
    entered_type = type(entered).__name__
    entered_value = entered
after_exit = (
    cuda.flash_sdp_enabled(),
    cuda.math_sdp_enabled(),
    cuda.mem_efficient_sdp_enabled(),
    cuda.cudnn_sdp_enabled(),
)
print(json.dumps({
    "before": before,
    "after_create": after_create,
    "inside": inside,
    "entered_type": entered_type,
    "entered": entered_value,
    "after_exit": after_exit,
    "warning": str(caught[0].message),
    "warning_type": caught[0].category.__name__,
    "built": cuda.is_built(),
    "ck_available": cuda.is_ck_sdpa_available(),
    "flash_available": cuda.is_flash_attention_available(),
    "cuda_available": torch.cuda.is_available(),
    "cuda_count": torch.cuda.device_count(),
    "sdpa_execution": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
    "nn_attention": hasattr(torch.nn, "attention"),
    "torch_compile": hasattr(torch, "compile"),
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
                "before": [False, False, False, False],
                "after_create": [False, False, False, False],
                "inside": [True, True, True, True],
                "entered_type": "dict",
                "entered": {},
                "after_exit": [False, False, False, False],
                "warning": DEPRECATION_MESSAGE,
                "warning_type": "FutureWarning",
                "built": False,
                "ck_available": False,
                "flash_available": False,
                "cuda_available": False,
                "cuda_count": 0,
                "sdpa_execution": False,
                "nn_attention": False,
                "torch_compile": True,
            },
        )

    def test_explicit_boolean_combinations_restore_original_state(self):
        initial_states = (
            (True, True, True, True),
            (False, True, False, True),
        )
        for initial in initial_states:
            for requested in itertools.product((False, True), repeat=4):
                with self.subTest(initial=initial, requested=requested):
                    self.set_states(initial)
                    self.cuda.allow_fp16_bf16_reduction_math_sdp(True)
                    context = self.sdp_context(*requested)
                    self.assertEqual(self.states(), initial)
                    with context as entered:
                        self.assertEqual(entered, {})
                        self.assertIsNot(entered, {})
                        self.assertEqual(self.states(), requested)
                        self.assertIs(
                            self.cuda.fp16_bf16_reduction_math_sdp_allowed(),
                            True,
                        )
                        self.assertIs(self.cuda.is_built(), False)
                        self.assertIs(self.cuda.is_ck_sdpa_available(), False)
                        self.assertIs(
                            self.cuda.is_flash_attention_available(),
                            False,
                        )
                    self.assertEqual(self.states(), initial)
                    self.assertIs(
                        self.cuda.fp16_bf16_reduction_math_sdp_allowed(),
                        True,
                    )

    def test_nested_exceptional_and_mutated_body_restoration(self):
        self.set_states((True, False, True, False))
        outer = (False, False, True, True)
        inner = (True, True, False, False)

        with self.sdp_context(*outer) as outer_entered:
            self.assertEqual(outer_entered, {})
            self.assertEqual(self.states(), outer)
            with self.sdp_context(*inner) as inner_entered:
                self.assertEqual(inner_entered, {})
                self.assertEqual(self.states(), inner)
            self.assertEqual(self.states(), outer)
        self.assertEqual(self.states(), (True, False, True, False))

        marker = _ContextBodyError("body failed")
        with self.assertRaises(_ContextBodyError) as raised:
            with self.sdp_context(False, True, False, True) as entered:
                self.assertEqual(entered, {})
                self.assertEqual(self.states(), (False, True, False, True))
                raise marker
        self.assertIs(raised.exception, marker)
        self.assertEqual(self.states(), (True, False, True, False))

        with self.sdp_context(False, False, False, False):
            self.set_states((True, True, True, True))
        self.assertEqual(self.states(), (True, False, True, False))

    def test_truthy_values_are_accepted_and_truthiness_errors_preserve_state(self):
        self.set_states((True, False, True, False))
        with self.sdp_context(1, 0, "", object()) as entered:
            self.assertEqual(entered, {})
            self.assertEqual(self.states(), (True, False, False, True))
        self.assertEqual(self.states(), (True, False, True, False))

        error = _ContextBodyError("truthiness failed")
        log = []
        context = self.sdp_context(
            _TruthProbe("flash", log),
            _TruthProbe("math", log),
            _TruthProbe("mem_efficient", log, error=error),
            _TruthProbe("cudnn", log),
        )
        with self.assertRaises(_ContextBodyError) as raised:
            context.__enter__()
        self.assertIs(raised.exception, error)
        self.assertEqual(log, ["flash", "mem_efficient"])
        self.assertEqual(self.states(), (True, False, True, False))

    def test_binding_errors_leave_state_unchanged(self):
        self.set_states((True, False, True, False))
        cases = (
            (
                lambda: self.cuda.sdp_kernel(True, False, True, False, True),
                "sdp_kernel() takes from 0 to 4 positional arguments but 5 were given",
            ),
            (
                lambda: self.cuda.sdp_kernel(foo=True),
                "sdp_kernel() got an unexpected keyword argument 'foo'",
            ),
            (
                lambda: self.cuda.sdp_kernel(True, enable_flash=False),
                "sdp_kernel() got multiple values for argument 'enable_flash'",
            ),
            (
                lambda: self.cuda.sdp_kernel(_enabled=False),
                "sdp_kernel() got an unexpected keyword argument '_enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    with self.assertRaises(TypeError) as raised:
                        call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(self.states(), (True, False, True, False))

        with self.sdp_context(enable_flash=False, enable_cudnn=True) as entered:
            self.assertEqual(entered, {})
            self.assertEqual(self.states(), (False, True, True, True))
        self.assertEqual(self.states(), (True, False, True, False))

    def test_context_is_a_reusable_decorator_factory(self):
        self.set_states((True, True, True, True))
        observations = []

        decorator = self.sdp_context(False, True, False, True)

        @decorator
        def decorated(value):
            observations.append(self.states())
            if value == "raise":
                raise _ContextBodyError("decorated body failed")
            return value

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.assertEqual(decorated("first"), "first")
            self.assertEqual(decorated("second"), "second")
            with self.assertRaisesRegex(_ContextBodyError, "decorated body failed"):
                decorated("raise")

        self.assertEqual(
            observations,
            [
                (False, True, False, True),
                (False, True, False, True),
                (False, True, False, True),
            ],
        )
        self.assertEqual(self.states(), (True, True, True, True))

    def test_state_changes_are_process_global_across_threads(self):
        self.set_states((True, True, True, True))
        worker_entered = threading.Event()
        main_context_exited = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                with self.sdp_context(False, False, False, False) as entered:
                    observations.append(("worker-enter", entered, self.states()))
                    worker_entered.set()
                    if not main_context_exited.wait(timeout=10):
                        raise RuntimeError("timed out waiting for main context")
                    observations.append(("worker-resume", self.states()))
                observations.append(("worker-exit", self.states()))
            except BaseException as error:
                errors.append(error)
                worker_entered.set()

        thread = threading.Thread(target=worker)
        thread.start()
        try:
            self.assertTrue(worker_entered.wait(timeout=10))
            self.assertEqual(errors, [])
            self.assertEqual(self.states(), (False, False, False, False))
            with self.sdp_context(True, False, True, False) as entered:
                self.assertEqual(entered, {})
                self.assertEqual(self.states(), (True, False, True, False))
            self.assertEqual(self.states(), (False, False, False, False))
        finally:
            main_context_exited.set()
            thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(
            observations,
            [
                ("worker-enter", {}, (False, False, False, False)),
                ("worker-resume", (False, False, False, False)),
                ("worker-exit", (True, True, True, True)),
            ],
        )
        self.assertEqual(self.states(), (True, True, True, True))

    def test_metadata_imports_copying_pickling_and_reload(self):
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

        signature = (
            "(enable_flash: bool = True, enable_math: bool = True, "
            "enable_mem_efficient: bool = True, enable_cudnn: bool = True)"
        )
        annotations = {
            "enable_flash": bool,
            "enable_math": bool,
            "enable_mem_efficient": bool,
            "enable_cudnn": bool,
        }

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), signature)
        self.assertEqual(inspect.get_annotations(function), annotations)
        self.assertEqual(typing.get_type_hints(function), annotations)
        self.assertEqual(function.__name__, "sdp_kernel")
        self.assertEqual(function.__qualname__, "sdp_kernel")
        self.assertEqual(function.__module__, "torch_rs.backends.cuda")
        self.assertIs(inspect.getmodule(function), cuda)
        self.assertEqual(function.__doc__, SDP_KERNEL_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(
            function.__dict__,
            {"__wrapped__": wrapped, "__deprecated__": DEPRECATION_MESSAGE},
        )
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__code__.co_freevars, ("func",))
        self.assertEqual(function.__code__.co_cellvars, ())

        self.assertIs(type(wrapped), types.FunctionType)
        self.assertEqual(str(inspect.signature(wrapped)), signature)
        self.assertEqual(inspect.get_annotations(wrapped), annotations)
        self.assertEqual(wrapped.__name__, "sdp_kernel")
        self.assertEqual(wrapped.__qualname__, "sdp_kernel")
        self.assertEqual(wrapped.__module__, "torch_rs.backends.cuda")
        self.assertEqual(wrapped.__doc__, SDP_KERNEL_DOC)
        self.assertIsNone(wrapped.__defaults__)
        self.assertIsNone(wrapped.__kwdefaults__)
        self.assertEqual(
            wrapped.__dict__,
            {"__wrapped__": original, "__deprecated__": DEPRECATION_MESSAGE},
        )

        self.assertIs(type(original), types.FunctionType)
        self.assertEqual(str(inspect.signature(original)), signature)
        self.assertEqual(inspect.get_annotations(original), annotations)
        self.assertEqual(original.__defaults__, (True, True, True, True))
        self.assertIsNone(original.__kwdefaults__)
        self.assertEqual(
            original.__dict__,
            {"__deprecated__": DEPRECATION_MESSAGE},
        )
        self.assertEqual(original.__code__.co_freevars, ())
        self.assertEqual(original.__code__.co_cellvars, ())

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

        context = self.sdp_context(False, True, False, True)
        self.assertEqual(type(context).__module__, "contextlib")
        self.assertEqual(type(context).__qualname__, "_GeneratorContextManager")
        self.assertEqual(context.__doc__, SDP_KERNEL_DOC)
        self.assertIs(context.func, wrapped)
        self.assertEqual(context.args, (False, True, False, True))
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

        old_function = function
        old_wrapped = wrapped
        namespace = cuda.__dict__
        self.set_states((True, True, True, True))
        active_context = self.sdp_context(False, False, False, False)

        self.assertEqual(active_context.__enter__(), {})
        self.assertEqual(self.states(), (False, False, False, False))
        reloaded = importlib.reload(cuda)

        self.assertIs(reloaded, cuda)
        self.assertIs(cuda.__dict__, namespace)
        self.assertIs(torch.backends.cuda, cuda)
        self.assertIs(sys.modules[cuda.__name__], cuda)
        self.assertIsNot(cuda.sdp_kernel, old_function)
        self.assertIsNot(cuda.sdp_kernel.__wrapped__, old_wrapped)
        self.assertEqual(self.states(), (False, False, False, False))
        self.assertIs(active_context.__exit__(None, None, None), False)
        self.assertEqual(self.states(), (True, True, True, True))

        for callable_factory in (old_function, cuda.sdp_kernel):
            with self.subTest(function=callable_factory):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    with callable_factory(False, True, False, True) as entered:
                        self.assertEqual(entered, {})
                        self.assertEqual(self.states(), (False, True, False, True))
                self.assertEqual(self.states(), (True, True, True, True))

        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_function)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function sdp_kernel at 0x...>: it's not the same "
            "object as torch_rs.backends.cuda.sdp_kernel",
        )
        self.assertIs(pickle.loads(pickle.dumps(cuda.sdp_kernel)), cuda.sdp_kernel)

    def test_context_management_does_not_add_sdpa_or_cuda_execution(self):
        self.assertIs(self.cuda.is_built(), False)
        self.assertIs(self.cuda.is_ck_sdpa_available(), False)
        self.assertIs(self.cuda.is_flash_attention_available(), False)
        with self.sdp_context(False, False, False, False):
            self.assertFalse(
                hasattr(torch.nn.functional, "scaled_dot_product_attention")
            )
            self.assertFalse(hasattr(torch.nn, "attention"))
            self.assertTrue(callable(torch.compile))
            self.assertIs(torch.cuda.is_available(), False)
            self.assertEqual(torch.cuda.device_count(), 0)
            with self.assertRaisesRegex(
                RuntimeError,
                "^tensor\\(\\): device 'cuda' is not supported; "
                "only 'cpu' is implemented$",
            ):
                torch.tensor([1.0], device="cuda")
        self.assertIs(self.cuda.is_built(), False)


if __name__ == "__main__":
    unittest.main()
